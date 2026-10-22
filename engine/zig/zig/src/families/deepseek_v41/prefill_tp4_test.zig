//! Host tests of TP=4 prefill planning (the second GPU window's boot failure, 2026-10-10): every rank's 2,048-row
//! segment (TF_DSV41_PF_4K's fallback segments, and the forward's own 2K programs) is on the prefill path at the boot's
//! options (`f.opts`: pf4k off), at the uneven expert widths (640 on ranks 0 / 1, 512 on ranks 2 / 3).
//!
//! The boot refused it on every rank ("a 2048-row segment is not on the prefill path at any position"): x3gm's block
//! must fit the decode-sized expert scratch (block_prefill.moeRows: expert_rows x (topk + 1) x max(8 I, D) floats), which
//! a 2,048-row block's X / Y (P x D, P = rows x topk) does at TP=2 (I 1,152: 8 I > D) but not at TP=4 (8 I <= D).
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const Config = @import("config.zig").Config;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

/// The boot's prefill options at TP=4 (config/tp4-zig.env.example's switches that reach the emitter), pf4k off as in
/// `f.opts`: the 4K segments' options are forward_prefill.k4Options, not these.
fn bootOptions(rank: u32) block.Options {
    const limit: i64 = 420096;
    return .{ .world = 4, .rank = rank, .gm_v2 = .one, .r1 = true, .branches = true, .mhc_defer = true, .expert_topp = 0.85, .limit = limit, .rope_rows = limit + 2048, .prefill_rows = 2048, .index_budget = 64 << 20 };
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

test "TP=4: a 2,048-row segment is on the prefill path on every rank (640- and 512-wide experts), x3gm one block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 }; // Engram, a ratio-2 kv source (compressor + indexer), a plain layer
    var enc: [21]u32 = undefined; // CED's encoder pass: layers 0 .. 20
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const D: i64 = cfg.hidden;
    for (0..4) |r| {
        const o = bootOptions(@intCast(r));
        const I: i64 = o.expertWidth(&cfg);
        try testing.expectEqual(@as(i64, if (r < 2) 640 else 512), I);
        for ([_]i64{ 0, 300000 }) |start| {
            const cs = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 2048, start, true);
            const moes = count(cs, "tf_dsv41_x3gm_v1.rot");
            try testing.expectEqual(@as(usize, layers.len), moes);
            try testing.expectEqual(moes, count(cs, "tf_dsv41_x3gm_v1.gateup2"));
            // the block is the segment: P = 2,048 x 6 pairs, X / Y in s.ex.z sized by these views (the buffer plan)
            var plan: buffers.Plan = .{ .a = aa };
            try plan.add(cs);
            for (cs) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.rot")) try testing.expectEqual(@as(i64, 2048 * 6), c.args[9].arg.i);
            try testing.expect(plan.sizes.get("s.ex.z").? >= 2 * 2048 * 6 * @as(u64, @intCast(D)) * 2);
            try testing.expect(plan.sizes.get("s.ex.xg").? >= 2048 * 6 * @as(u64, @intCast(I)) * 2);
        }
        // CED replay's encoder segment and a two-segment piece at 2,048 rows, the decoder's tails
        _ = try bp.emitReplay(aa, &cfg, &w, o, &enc, 2048, 0, .encoder);
        const sp = [_]bp.Span{ .{ .slot = 0, .start = 0, .n = 2048 - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
        _ = try bp.emitMulti(aa, &cfg, &w, o, &enc, &sp, .encoder);
        // the smaller segments the forward plans (17 and 16 rows) were never short of scratch
        _ = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 17, 0, true);
        _ = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 16, 0, true);
    }
}

/// The largest `_kv_store` RING of a program (the staging ring of a staged segment; 1 on the compressed ratios).
fn ringOf(cs: []const calls.Call) i64 {
    var ring: i64 = -1;
    for (cs) |c| {
        if (!std.mem.eql(u8, c.name, "_kv_store")) continue;
        for (c.args) |x| {
            if (!std.mem.eql(u8, x.name, "RING")) continue;
            switch (x.arg) {
                .i => |v| if (v > ring) {
                    ring = v;
                },
                else => {},
            }
        }
    }
    return ring;
}

test "TP=4: under 4K the boot's f.opts (pf4k off, default 2,048 rows) stage at ring 4,096, the workspace's set at 8,192: the fill enumerates both" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    for ([_]u32{ 0, 2 }) |r| {
        // the boot under TF_DSV41_PF_4K: prod_knobs.apply leaves f.opts at pf4k off and the default prefill_rows
        // (the workspace's own options are forward_prefill.k4Options); a fill that enumerated only the pf4k-on set
        // had no RING 4,096 variant (window 3b's MissingTritonVariant), so aot-needs sweeps both option sets
        const boot = bootOptions(r);
        var k4 = boot;
        k4.pf4k = true;
        k4.prefill_rows = 4096;
        for ([_]i64{ 1500, 17 }) |n| for ([_]i64{ 0, 300000 }) |start| {
            const a = try bp.emitReplay(aa, &cfg, &w, boot, &enc, n, start, .encoder);
            const b = try bp.emitReplay(aa, &cfg, &w, k4, &enc, n, start, .encoder);
            // a staged segment (past 2 x window rows) stages at ceilPow2(prefill_rows + window - 1)
            try testing.expectEqual(@as(i64, if (n > 2 * cfg.window - 1) 4096 else 2 * cfg.window), ringOf(a));
            try testing.expectEqual(@as(i64, if (n > 2 * cfg.window - 1) 8192 else 2 * cfg.window), ringOf(b));
        };
    }
}

test "TP=2 keeps Python's limit: past what the scratch holds without 4K is refused (x3gm's rows would halve)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    var o: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048, .prefill_rows = 4096 };
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, o, &layers, 4096, 0, true));
    // and its 2,048-row segment fits the scratch as it always did
    o.prefill_rows = 2048;
    _ = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 2048, 0, true);
}

/// Every role `need` sizes, at least as large in `have`.
fn covers(have: *const buffers.Plan, need: *const buffers.Plan) !void {
    var it = need.sizes.iterator();
    while (it.next()) |e| {
        const h = have.sizes.get(e.key_ptr.*) orelse {
            std.debug.print("role {s} ({d} bytes) is not in the plan\n", .{ e.key_ptr.*, e.value_ptr.* });
            return error.TestUnexpectedResult;
        };
        if (h < e.value_ptr.*) {
            std.debug.print("role {s}: the plan holds {d} bytes, the call needs {d}\n", .{ e.key_ptr.*, h, e.value_ptr.* });
            return error.TestUnexpectedResult;
        }
    }
}

// The live image window's two plan failures (2026-10-12): the buffer plan sized before the image flags were set (an
// image prompt: `w.out.img` Unbound), then the multi-segment runs planned at the image options, which emitMulti refuses
// (a three-slot run of text tails: `w.mpos64` Unbound). The plan's image options (vision_rows.planOptions: keep, image
// rows -1) must cover an image segment's roles at any split; the runs stay planned at the text options.
test "TP=4 images: the plan's image options cover an image segment at any split; runs are planned at the text options" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    for ([_]u32{ 0, 2 }) |r| {
        const o = bootOptions(r);
        var po = o;
        po.image_keep = true;
        po.image_rows = -1;
        for ([_]i64{ 237, 2048 }) |n| {
            var plan: buffers.Plan = .{ .a = aa };
            try plan.add(try bp.emitPrefill(aa, &cfg, &w, po, &layers, n, 0, true));
            try plan.add(try bp.emitReplay(aa, &cfg, &w, po, &enc, n, 0, .encoder));
            for ([_]i64{ 1, 184, n - 1, n }) |ni| {
                var so = o;
                so.image_keep = true;
                so.image_rows = ni;
                var seg: buffers.Plan = .{ .a = aa };
                try seg.add(try bp.emitPrefill(aa, &cfg, &w, so, &layers, n, 0, true));
                try seg.add(try bp.emitReplay(aa, &cfg, &w, so, &enc, n, 0, .encoder));
                try covers(&plan, &seg);
            }
            // and without the plan's image options the image segment's roles are missing (the first failure)
            var text: buffers.Plan = .{ .a = aa };
            try text.add(try bp.emitPrefill(aa, &cfg, &w, o, &layers, n, 0, true));
            try testing.expect(text.sizes.get("w.out.img") == null and plan.sizes.get("w.out.img") != null);
        }
        // the runs: refused at the image options, planned (their positions' role among them) at the text options
        try testing.expectEqual(@as(usize, 0), (try bp.planRuns(aa, &cfg, &w, po, &enc, null, 2048, .encoder)).items.len);
        const runs = try bp.planRuns(aa, &cfg, &w, o, &enc, null, 2048, .encoder);
        try testing.expect(runs.items.len > 0);
        var rp: buffers.Plan = .{ .a = aa };
        for (runs.items) |cs| try rp.add(cs);
        try testing.expect(rp.sizes.get("w.mpos64") != null);
    }
}
