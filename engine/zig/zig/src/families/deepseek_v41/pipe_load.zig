//! TF_DSV41_PREFILL_PIPE: a rank's pipeline stage copies (docs/ZIG-PREFILL-PIPE.md). Its stage's encoder layers planned
//! at world 1 (rank 0 of 1: every head, wo_a group and expert column), each layer's images built (or read from the
//! prepared cache, a key of its own) and cut down to the TP-sharded tensors under "P." (pipe_names.keepSharded), then
//! uploaded beside the TP weights; the stage reads every replicated tensor from the TP copy. Loaded before the boot's
//! memory check, so MemAvailable already holds them.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("cuda_weights.zig");
const prepared = @import("prepared.zig");
const pipe = @import("pipe.zig");
const names = @import("pipe_names.zig");

const log = std.log.scoped(.dsv41);

/// The prepared cache's key variant of the stage copies (a world-1 layer's sharded images only, renamed): never the
/// key of a whole world-1 layer.
pub const cache_variant: u32 = 0x50495045; // "PIPE"

/// Rank `rank`'s stage copies into `w`; returns their world-1 plan (the experts' layouts the pointer tables read).
pub fn loadStage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, pack: *const Pack, cfg: *const Config, s: pipe.Settings, rank: u32, w: *load.Weights) !plan.Plan {
    var lb: [64]u32 = undefined;
    const layers = s.stage(rank).weightLayers(&lb);
    var pl = try plan.build(gpa, cfg, pack, .{ .rank = 0, .world = 1, .blocks = layers });
    errdefer pl.deinit();
    var cache = try prepared.Cache.fromEnv(gpa, io, pack, .{ .rank = 0, .world = 1, .o_groups = cfg.o_groups, .blocks = layers, .dspark = false, .variant = cache_variant });
    defer if (cache) |*c| c.deinit();
    const before = w.bytes;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    for (pl.layers, 0..) |*l, i| {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = pack };
        defer b.deinit();
        var tag_buf: [16]u8 = undefined;
        const tag = try std.fmt.bufPrint(&tag_buf, "L{d}", .{i});
        var schema: named.Builder = .{ .gpa = gpa, .io = io, .pack = pack, .metadata_only = true };
        defer schema.deinit();
        if (cache != null) {
            try schema.layer(l, cfg.o_groups, 0, 1);
            try names.keepSharded(&schema);
        }
        if (cache == null or !try cache.?.get(&b, tag, schema.out.items)) {
            try b.layer(l, cfg.o_groups, 0, 1);
            try names.keepSharded(&b);
            if (cache) |*c| c.put(&b, tag) catch |e| log.warn("dsv41 prepared: pipeline stage {s} not written ({t})", .{ tag, e });
        }
        try w.upload(d, &b);
    }
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    log.info("pipeline: rank {d}'s stage copies (layers {d}-{d} at world 1, sharded tensors only): {d:.1} GiB in {d:.0} ms", .{ rank, layers[0], layers[layers.len - 1], @as(f64, @floatFromInt(w.bytes - before)) / (1 << 30), ms });
    return pl;
}

/// x3gm's pointer table of a stage layer's routed experts ("s.P.L<i>.gm.t{g,u,d}", int64 [E]): each expert's trellis
/// address in the stage copy (forms.zig's "tp" rule over the world-1 layout). `proj`: 0 w1 (gate), 1 w2 (down), 2 w3 (up).
pub fn gmTable(a: std.mem.Allocator, w: *const load.Weights, pl: *const plan.Plan, L: u32, proj: usize) ![]u8 {
    const lp = for (pl.layers) |*l| {
        if (l.index == L) break l;
    } else return error.LayerNotPlanned;
    const lay = &lp.experts[proj].layout;
    const wn = ([_][]const u8{ "w1", "w2", "w3" })[proj];
    const data = w.get(try std.fmt.allocPrint(a, names.prefix ++ "L{d}.moe.{s}.data", .{ L, wn })) orelse uniform: {
        const t = w.get(try std.fmt.allocPrint(a, names.prefix ++ "L{d}.moe.{s}.trellis", .{ L, wn })) orelse return error.MissingWeight;
        if (t.len != lay.trellisBytes()) return error.ExpertStackSize;
        break :uniform t;
    };
    const E = lay.words.len;
    const out = try a.alloc(u8, 8 * E);
    for (0..E) |e| std.mem.writeInt(u64, out[8 * e ..][0..8], data.ptr + lay.byteOffset(e), .little);
    return out;
}
