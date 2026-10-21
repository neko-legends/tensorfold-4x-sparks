//! Host tests of the pipelined prompt reading's stage programs (block_prefill.emitStage, pipe.zig; docs/ZIG-PREFILL-PIPE.md):
//! - the four stages' programs, end to end, ARE the one-rank encoder pass (`emitReplay` at world 1, `.encoder`): call
//!   for call, argument for argument, but for the stages' own hand-off glue (`pipe_in` / `pipe_out`) and the encoder
//!   pass's embedding (the line's round embeds the segment before stage 0). So a pipelined prompt's arithmetic is one
//!   rank computing the encoder (Python's claim for pipe.py, here checked on the emitted programs). Checked at the
//!   served boot's options (`servedOptions`: every emitter switch config/tp4-zig.env.example sets) and at the bare
//!   pipeline set (`bootOptions`), for the sampled (rows, start) of the tests below, at 2,048- and 4,096-row segments:
//!   samples, not a sweep of every row count and start (docs/ZIG-PREFILL-PIPE.md, "What the tests prove");
//! - what a stage hands on is what the next one takes (the same tensors, byte for byte);
//! - a stage program in its namespace with the stage weights' names reads every replicated tensor, form and slot
//!   state exactly as the TP=4 program of its layers does, and has its own copy of every name whose layout differs.
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const pipe = @import("pipe.zig");
const Config = @import("config.zig").Config;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    // the listing is a TP=2 rank's (wo_a groups 0-3): a stage at world 1 runs all eight
    const cfg: Config = .{};
    try pipe.fillGroups(a, &w, &cfg);
    return w;
}

const limit: i64 = 420096; // TP=4's TF_DSV41_CONTEXT (1,641 pages)

/// The boot's prefill options on TP=4 rank `rank` (config/tp4-zig.env.example's switches that reach the emitter), with
/// the served pool (replicated: KV split off, which the pipeline needs).
fn bootOptions(rank: u32) block.Options {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .world = 4, .rank = rank, .gm_v2 = .one, .r1 = true, .branches = true, .mhc_defer = true, .expert_topp = 0.85, .limit = limit, .rope_rows = limit + 2048, .prefill_rows = 2048, .index_budget = 64 << 20, .pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages } };
}

const pfd_table: @import("prod_knobs.zig").PfDense = .{};

/// The served boot's prefill options on TP=4 rank `rank`: `bootOptions` plus every other switch of
/// config/tp4-zig.env.example that prod_knobs.apply / branches.install put on the forward's options and pipe.stageOptions
/// passes through to a stage (TF_DSV41_INDEX_BOUND=1, MHC_PFDEC=17, KV_NORM_STORE=1, ROUTER_GROUP_ROT=1,
/// MHC_DEFER_AT=exchange, PF_DENSE=fused: its default table; the served table picks other tile configs, the same calls)
/// and DRAFTS' taps, which stageOptions clears. PF_4K leaves the forward's own segment rows at 2,048 (prod_knobs.apply).
fn servedOptions(rank: u32) block.Options {
    var o = bootOptions(rank);
    o.index_bound = true;
    o.mhc_pf_rows = 17;
    o.kv_norm_store = true;
    o.router_group_rot = true;
    o.mhc_defer_hold = true;
    o.pfd = &pfd_table;
    o.taps = true;
    return o;
}

fn settings() !pipe.Settings {
    return settingsAt("2048");
}

/// TF_DSV41_PREFILL_PIPE=1024 with TF_DSV41_PIPE_ROWS=`rows`.
fn settingsAt(comptime rows: []const u8) !pipe.Settings {
    const F = struct {
        fn get(name: []const u8) ?[]const u8 {
            if (std.mem.eql(u8, name, "TF_DSV41_PIPE_ROWS")) return rows;
            return if (std.mem.eql(u8, name, "TF_DSV41_PREFILL_PIPE")) "1024" else null;
        }
    };
    const cfg: Config = .{};
    return (try pipe.settings(&F.get, &cfg, 4)).?;
}

fn stageIo(cfg: *const Config, s: *const pipe.Settings, r: u32) bp.StageIo {
    const st = s.stage(r);
    return .{ .entered = r > 0, .last = st.last, .sel_in = st.selIn(cfg), .sel_out = st.selOut(cfg, s.boundsSlice()) };
}

fn eqTensor(x: calls.Tensor, y: calls.Tensor) bool {
    if (std.meta.activeTag(x.role) != std.meta.activeTag(y.role)) return false;
    const rx = switch (x.role) {
        .weight => |v| v,
        .buf => |v| v,
        .empty => "",
    };
    const ry = switch (y.role) {
        .weight => |v| v,
        .buf => |v| v,
        .empty => "",
    };
    return std.mem.eql(u8, rx, ry) and x.dt == y.dt and std.mem.eql(i64, x.shape, y.shape) and std.mem.eql(i64, x.stride, y.stride) and x.offset == y.offset;
}

fn eqArg(x: calls.Arg, y: calls.Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t => |t| eqTensor(t, y.t),
        .opaque_table => |t| eqTensor(t, y.opaque_table),
        .i => |v| v == y.i,
        .f => |v| @as(u64, @bitCast(v)) == @as(u64, @bitCast(y.f)),
        .b => |v| v == y.b,
        .none => true,
        .list => |l| blk: {
            if (l.len != y.list.len) break :blk false;
            for (l, y.list) |p, q| if (!eqArg(p, q)) break :blk false;
            break :blk true;
        },
    };
}

/// The same launch (or glue step): name, kind, grid, streams and marks, every argument (`begin` is a scope marker of
/// the program, not of the launch).
fn sameCall(c: calls.Call, d: calls.Call) !void {
    try testing.expectEqualStrings(c.name, d.name);
    try testing.expectEqual(c.triton, d.triton);
    try testing.expectEqual(c.glue, d.glue);
    try testing.expectEqual(c.grid, d.grid);
    try testing.expectEqual(c.side, d.side);
    try testing.expectEqual(c.fork, d.fork);
    try testing.expectEqual(c.join, d.join);
    try testing.expectEqual(c.mark, d.mark);
    try testing.expectEqual(c.wait, d.wait);
    try testing.expectEqual(c.defer_side, d.defer_side);
    try testing.expectEqual(c.defer_join, d.defer_join);
    try testing.expectEqual(c.args.len, d.args.len);
    for (c.args, d.args) |x, y| {
        if (!eqArg(x.arg, y.arg)) {
            std.debug.print("{s}: argument {s} differs\n", .{ c.name, x.name });
            return error.TestUnexpectedResult;
        }
    }
}

fn isGlue(c: calls.Call, step: []const u8) bool {
    return c.glue and std.mem.eql(u8, c.name["glue.".len..], step);
}

/// The four stages' programs of an `n`-row segment from `start` at the stage options `o1`, minus their hand-off glue,
/// against the one-rank encoder pass minus its embedding: call for call (`sameCall`). Returns the calls compared.
fn expectLine(aa: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, s: *const pipe.Settings, o1: block.Options, n: i64, start: i64) !usize {
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const whole = try bp.emitReplay(aa, cfg, w, o1, &enc, n, start, .encoder);
    var line: std.ArrayList(calls.Call) = .empty;
    var buf: [64]u32 = undefined;
    for (0..4) |r| {
        const st = s.stage(@intCast(r));
        const cs = try bp.emitStage(aa, cfg, w, o1, st.layers(&buf), n, start, stageIo(cfg, s, @intCast(r)));
        // a stage: its positions, its input, its layers, its output (the last: the encoder's end, the stash)
        try testing.expect(isGlue(cs[0], "positions") and isGlue(cs[1], "pipe_in"));
        try testing.expect(if (st.last) isGlue(cs[cs.len - 1], "ced_stash") else isGlue(cs[cs.len - 1], "pipe_out"));
        for (cs, 0..) |c, i| {
            if (isGlue(c, "pipe_in") or isGlue(c, "pipe_out")) continue;
            if (r > 0 and i == 0) continue; // every stage's positions: the encoder pass's one
            try line.append(aa, c);
        }
    }
    var j: usize = 0;
    for (whole) |c| {
        if (isGlue(c, "embed")) continue; // the round's embedding, before stage 0
        if (j >= line.items.len) {
            std.debug.print("{d} rows at {d}: the encoder pass has calls past the stages' end ({s})\n", .{ n, start, c.name });
            return error.TestUnexpectedResult;
        }
        sameCall(c, line.items[j]) catch |e| {
            std.debug.print("{d} rows at {d}: call {d} differs (encoder pass {s}, stages {s})\n", .{ n, start, j, c.name, line.items[j].name });
            return e;
        };
        j += 1;
    }
    try testing.expectEqual(line.items.len, j);
    return j;
}

test "the four stages' programs are the one-rank encoder pass, call for call (2,048 / 1,000 / 33 rows, two starts)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    const o1 = pipe.stageOptions(bootOptions(0), s.rows);
    for ([_]i64{ 2048, 1000, 33 }) |n| for ([_]i64{ 0, 300000 }) |start| {
        _ = try expectLine(aa, &cfg, &w, &s, o1, n, start);
    };
}

test "at the served boot's options (INDEX_BOUND, MHC_PFDEC=17, KV_NORM_STORE, ...) the stages are the encoder pass, call for call" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    // every rank's stage options are the same (stageOptions: world 1, rank 0)
    const o1 = pipe.stageOptions(servedOptions(0), s.rows);
    for (1..4) |r| try testing.expectEqualDeep(o1, pipe.stageOptions(servedOptions(@intCast(r)), s.rows));
    try testing.expect(o1.index_bound and o1.mhc_pf_rows == 17 and o1.kv_norm_store and o1.router_group_rot and o1.mhc_defer_hold and o1.pfd != null and !o1.taps);
    // not the bare set's programs: the fused dense GEMMs (PF_DENSE) reach every stage (the decode-window switches,
    // INDEX_BOUND / MHC_PFDEC / KV_NORM_STORE / ROUTER_GROUP_ROT / MHC_DEFER_AT, emit no stage call of their own at
    // 33-2,048 rows: checked 2026-10-11, the served set's identity stands for them anyway)
    var buf: [64]u32 = undefined;
    const bare = pipe.stageOptions(bootOptions(0), s.rows);
    for (0..4) |r| {
        const x = try bp.emitStage(aa, &cfg, &w, o1, s.stage(@intCast(r)).layers(&buf), 2048, 300000, stageIo(&cfg, &s, @intCast(r)));
        const y = try bp.emitStage(aa, &cfg, &w, bare, s.stage(@intCast(r)).layers(&buf), 2048, 300000, stageIo(&cfg, &s, @intCast(r)));
        var differs = x.len != y.len;
        if (!differs) for (x, y) |c, d| {
            sameCall(c, d) catch {
                differs = true;
                break;
            };
        };
        try testing.expect(differs);
    }
    // whole segments, a piece's or a resumed prompt's ragged ones, the shortest; at the first page, 16-grid starts,
    // deep in the context and at the line's last start (pipe_aot_test: lastEnd is the limit at this config)
    for ([_]i64{ 2048, 2047, 1500, 1000, 100, 64, 34, 33 }) |n| for ([_]i64{ 0, 16, 4112, 154176, 300000, limit - n }) |start| {
        _ = try expectLine(aa, &cfg, &w, &s, o1, n, start);
    };
}

test "4,096-row stages (TF_DSV41_PIPE_ROWS=4096) are the 4K encoder pass, call for call (both option sets)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settingsAt("4096");
    try testing.expectEqual(@as(u32, 4096), s.rows);
    for ([_]block.Options{ bootOptions(0), servedOptions(0) }) |o| {
        const o1 = pipe.stageOptions(o, s.rows);
        try testing.expect(o1.pf4k and o1.prefill_rows == 4096);
        for ([_]i64{ 4096, 2049, 33 }) |n| for ([_]i64{ 0, 8192, 300000, limit - n }) |start| {
            _ = try expectLine(aa, &cfg, &w, &s, o1, n, start);
        };
    }
}

test "a stage hands on exactly what the next takes: pipe_out's tensors are the next pipe_in's, byte for byte" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    const o1 = pipe.stageOptions(bootOptions(0), s.rows);
    var buf: [64]u32 = undefined;
    for ([_]i64{ 2048, 33 }) |n| {
        var outs: [4]?calls.Call = @splat(null);
        var ins: [4]calls.Call = undefined;
        for (0..4) |r| {
            const cs = try bp.emitStage(aa, &cfg, &w, o1, s.stage(@intCast(r)).layers(&buf), n, 4096, stageIo(&cfg, &s, @intCast(r)));
            ins[r] = cs[1];
            if (r < 3) outs[r] = cs[cs.len - 1];
        }
        // stage 0 takes the embedded streams alone; the others the streams, the pending post and the selection
        try testing.expectEqual(@as(usize, 1), ins[0].args.len);
        for (1..4) |r| {
            const o = outs[r - 1].?;
            try testing.expectEqual(o.args.len, ins[r].args.len);
            try testing.expectEqual(@as(usize, 7), o.args.len); // streams, partial, pre / post / comb, sel / cnt
            for (o.args, ins[r].args) |x, y| {
                try testing.expectEqual(x.arg.t.dt, y.arg.t.dt);
                try testing.expectEqualSlices(i64, x.arg.t.shape, y.arg.t.shape);
            }
            // the streams [n, 4 D], the partial at world 1 [1, n, D]
            try testing.expectEqualSlices(i64, &.{ n, 4 * @as(i64, cfg.hidden) }, o.args[0].arg.t.shape);
            try testing.expectEqualSlices(i64, &.{ 1, n, @as(i64, cfg.hidden) }, o.args[1].arg.t.shape);
            // the runtime's packing (pipe.HandoffLayout) is these tensors' bytes, contiguous views each
            const lay = pipe.HandoffLayout.of(&cfg, @intCast(n), true, true);
            try testing.expectEqual(o.args.len, lay.parts);
            for (o.args, 0..) |x, i| {
                const t = x.arg.t;
                var numel: u64 = 1;
                for (t.shape) |d| numel *= @intCast(d);
                try testing.expectEqual(lay.bytes[i], numel * buffers.dtSize(t.dt));
                try testing.expect(std.mem.eql(i64, t.stride, try calls.contiguous(aa, t.shape)));
            }
        }
    }
    // the shortest segment a stage takes (pipe.min_seg): the mHC split programs' out-of-place streams never in a stage
    try testing.expectError(error.Unsupported, bp.emitStage(aa, &cfg, &w, o1, s.stage(1).layers(&buf), 32, 0, stageIo(&cfg, &s, 1)));
}

fn roleName(t: calls.Tensor) []const u8 {
    return switch (t.role) {
        .weight => |v| v,
        .buf => |v| v,
        .empty => "",
    };
}

const View = struct { dt: calls.Dt, shape: []const i64, stride: []const i64, offset: i64 };

fn collect(a: std.mem.Allocator, cs: []const calls.Call, m: *std.StringHashMapUnmanaged(std.ArrayList(View))) !void {
    for (cs) |c| for (c.args) |x| try note(a, x.arg, m);
}

fn note(a: std.mem.Allocator, x: calls.Arg, m: *std.StringHashMapUnmanaged(std.ArrayList(View))) !void {
    switch (x) {
        .t, .opaque_table => |t| {
            if (t.role == .empty) return;
            const g = try m.getOrPut(a, roleName(t));
            if (!g.found_existing) g.value_ptr.* = .empty;
            try g.value_ptr.append(a, .{ .dt = t.dt, .shape = t.shape, .stride = t.stride, .offset = t.offset });
        },
        .list => |l| for (l) |y| try note(a, y, m),
        else => {},
    }
}

fn hasView(vs: []const View, v: View) bool {
    for (vs) |u| if (u.dt == v.dt and std.mem.eql(i64, u.shape, v.shape) and std.mem.eql(i64, u.stride, v.stride) and u.offset == v.offset) return true;
    return false;
}

test "a stage program (its namespace, the P. names) reads every replicated tensor, form and slot state as TP=4 does" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    var buf: [64]u32 = undefined;
    for (0..4) |r| {
        const o4 = bootOptions(@intCast(r));
        const o1 = pipe.stageOptions(o4, s.rows);
        const st = s.stage(@intCast(r));
        const layers = st.layers(&buf);
        const plain = try bp.emitStage(aa, &cfg, &w, o1, layers, 2048, 300000, stageIo(&cfg, &s, @intCast(r)));
        const stage = try pipe.rename(aa, try bp.ownCalls(aa, plain, pipe.tag));
        const tp = try bp.emitReplay(aa, &cfg, &w, o4, layers, 2048, 300000, if (st.last) .encoder else .whole);
        var sm: std.StringHashMapUnmanaged(std.ArrayList(View)) = .empty;
        var tm: std.StringHashMapUnmanaged(std.ArrayList(View)) = .empty;
        try collect(aa, stage, &sm);
        try collect(aa, tp, &tm);
        var own: usize = 0;
        var renamed: usize = 0;
        var shared: usize = 0;
        var it = sm.iterator();
        while (it.next()) |kv| {
            const name = kv.key_ptr.*;
            // the stage's window / layer / scratch roles: its namespace
            if (std.mem.indexOf(u8, name, pipe.tag) != null) {
                own += 1;
                continue;
            }
            // the full-width copies: a tensor TP=4 has too (its shard, or a table of its shard's addresses)
            const bare = if (std.mem.startsWith(u8, name, pipe.prefix)) name[pipe.prefix.len..] else if (std.mem.startsWith(u8, name, "s." ++ pipe.prefix)) try std.fmt.allocPrint(aa, "s.{s}", .{name[2 + pipe.prefix.len ..]}) else null;
            if (bare) |b| {
                // (a TP rank names only its own wo_a groups, 0 .. o_groups / world - 1: the stage runs all of them)
                if (tm.get(b) == null and std.mem.indexOf(u8, b, ".attn.wo_a.") == null) {
                    std.debug.print("rank {d}: {s} renamed, but TP=4 has no {s}\n", .{ r, name, b });
                    return error.TestUnexpectedResult;
                }
                renamed += 1;
                continue;
            }
            // anything else is TP=4's own tensor or state: every view the stage takes, TP=4 takes too
            const tv = tm.get(name) orelse {
                std.debug.print("rank {d}: the stage reads {s}, which the TP=4 program never names\n", .{ r, name });
                return error.TestUnexpectedResult;
            };
            for (kv.value_ptr.items) |v| if (!hasView(tv.items, v)) {
                std.debug.print("rank {d}: {s} viewed as {any} in the stage, not by TP=4 (a sharded layout under a shared name)\n", .{ r, name, v.shape });
                return error.TestUnexpectedResult;
            };
            shared += 1;
        }
        try testing.expect(own > 0 and renamed > 0 and shared > 0);
        // no unprefixed x3gm pointer table: its addresses are the TP shard's
        it = sm.iterator();
        while (it.next()) |kv| try testing.expect(!(std.mem.startsWith(u8, kv.key_ptr.*, "s.L") and std.mem.indexOf(u8, kv.key_ptr.*, ".gm.t") != null and !std.mem.endsWith(u8, kv.key_ptr.*, "k2g") and !std.mem.endsWith(u8, kv.key_ptr.*, "k2d")));
        // the slot's state is the stage's own writes into the slot (rings, carries, the pool, the stash): shared names
        for ([_][]const u8{ "s.kv.pt", "s.rope.comp" }) |x| try testing.expect(sm.get(x) != null);
        if (st.last) try testing.expect(sm.get("s.ced.x") != null);
    }
}

test "the stage programs' buffer plan: a namespace of its own, sized by world-1 views (heads, groups, expert columns)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    var buf: [64]u32 = undefined;
    const o1 = pipe.stageOptions(bootOptions(2), s.rows);
    var plan: buffers.Plan = .{ .a = aa };
    for (0..4) |r| {
        const cs = try bp.emitStage(aa, &cfg, &w, o1, s.stage(@intCast(r)).layers(&buf), 2048, 300000, stageIo(&cfg, &s, @intCast(r)));
        try plan.add(try pipe.rename(aa, try bp.ownCalls(aa, cs, pipe.tag)));
    }
    // the attention's q at 64 heads, x3gm's down input at 2,304 columns (6 pairs a row)
    try testing.expectEqual(@as(u64, 2048 * 64 * 512 * 2), plan.sizes.get("L.pp~q").?);
    try testing.expectEqual(@as(u64, 2048 * 6 * 2304 * 2), plan.sizes.get("s.pp~ex.xg").?);
    try testing.expect(plan.sizes.get("L.q") == null and plan.sizes.get("s.ex.xg") == null);
}

/// The roles of rank `rank`'s stage over an `n`-row segment from `start` that the boot's stage plan (`plan`: pipe.planned's
/// programs, as pipe_run.planInto adds them) lacks or holds smaller; the first few printed. 0: every role bound and big enough.
fn unplanned(cfg: *const Config, w: *const block.Widths, o1: block.Options, s: *const pipe.Settings, plan: *const buffers.Plan, rank: u32, n: i64, start: i64) !usize {
    var pa = std.heap.ArenaAllocator.init(testing.allocator);
    defer pa.deinit();
    const cs = pipe.program(pa.allocator(), cfg, w, o1, s, rank, n, start) catch |e| switch (e) {
        error.Unsupported => return 0, // past the stage's last start: the line never enters it (pipe.lastEnd)
        else => return e,
    };
    var need: buffers.Plan = .{ .a = pa.allocator() };
    try need.add(cs);
    var bad: usize = 0;
    for (need.sizes.keys(), need.sizes.values()) |role, bytes| {
        // the forward's roles (the slot's state, the replicated tensors' forms): its own plan's, not the stage's
        if (std.mem.indexOf(u8, role, pipe.tag) == null and !std.mem.startsWith(u8, role, "s." ++ pipe.prefix)) continue;
        const have = plan.sizes.get(role) orelse 0;
        if (have >= bytes) continue;
        if (bad < 4) std.debug.print("rank {d}, {d} rows at {d}: {s} needs {d} bytes, the stage plan holds {d}\n", .{ rank, n, start, role, bytes, have });
        bad += 1;
    }
    return bad;
}

test "every role a stage names at any segment the line runs is in the boot's stage plan, at least as large (Unbound, window 6)" {
    // Spark 2026-10-11 (window 6): the first pipelined prompt's third segment (rank 0, rows 4,096-6,143) named the
    // materialised indexer's row-block positions "L.pp~ix.pos0/1", in no planned stage program: `the line failed (Unbound)`.
    // pipe.planned held a segment at 0 and at its last start only (the indexer's regimes in between were not planned).
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var prng = std.Random.DefaultPrng.init(0x5eed_0006);
    const rnd = prng.random();
    for ([_][]const u8{ "2048", "4096" }) |rows_knob| {
        const s = if (std.mem.eql(u8, rows_knob, "2048")) try settingsAt("2048") else try settingsAt("4096");
        const R: i64 = s.rows;
        for (0..4) |ri| {
            const r: u32 = @intCast(ri);
            const o1 = pipe.stageOptions(servedOptions(r), s.rows);
            var plan: buffers.Plan = .{ .a = aa };
            for (try pipe.planned(aa, &cfg, &w, o1, s, r)) |x| try plan.add(try pipe.program(aa, &cfg, &w, o1, &s, r, x[0], x[1]));
            var bad: usize = 0;
            // the line's whole segments through the first 16K positions (the window-6 prompt's), a ragged tail's and the
            // shortest, a key split's ends (the stream's row blocks), deep in the context and at the end
            const ns = [_]i64{ R, R - 1, 1547, 1366, 1365, 1024, 100, 64, 33 };
            for (ns) |n| {
                var start: i64 = 0;
                while (start + n <= 16384) : (start += 512) bad += try unplanned(&cfg, &w, o1, &s, &plan, r, n, start);
                for ([_]i64{ 3583, 5683, 6143, 6824, 8191 - n, 32768 * 8 - n, 32768 * 8 + 1 - n, 300000, limit - n }) |st| {
                    if (st >= 0) bad += try unplanned(&cfg, &w, o1, &s, &plan, r, n, st);
                }
            }
            // and a sample of any segment at any start
            for (0..40) |_| {
                const n = rnd.intRangeAtMost(i64, pipe.min_seg, R);
                const st = if (rnd.boolean()) rnd.intRangeAtMost(i64, 0, 16384) else rnd.intRangeAtMost(i64, 0, limit - n);
                bad += try unplanned(&cfg, &w, o1, &s, &plan, r, n, st);
            }
            try testing.expectEqual(@as(usize, 0), bad);
        }
    }
}

test "every role pipe_run looks up by name is one the stage programs name (the commit's carry rows under pp~, never bare)" {
    // window 6's suspect (pipe_run.Line.stage's commit, no bare "w.L<i>.proj" fallback as forward_prefill.commitRoles
    // has for K4.keep): a stage's ratio-2 projection rows are only ever "w.pp~L<i>.proj" (ownCalls keeps no role), so the
    // bare name, which the forward's plan holds for its own segments, would be the TP rows of another segment
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    var nb: [64]u8 = undefined;
    var sb: [64]u32 = undefined;
    for (0..4) |ri| {
        const r: u32 = @intCast(ri);
        const o1 = pipe.stageOptions(servedOptions(r), s.rows);
        var plan: buffers.Plan = .{ .a = aa };
        for (try pipe.planned(aa, &cfg, &w, o1, s, r)) |x| try plan.add(try pipe.program(aa, &cfg, &w, o1, &s, r, x[0], x[1]));
        const st = s.stage(r);
        // Line.stage's commit (pipe_run.zig: src, dst) and Line.rows' pool families
        for (st.sources(&cfg, &sb)) |L| {
            if (cfg.compressRatio(L) == 2) {
                try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "w." ++ pipe.tag ++ "L{d}.proj", .{L})) != null);
                try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "w.L{d}.proj", .{L})) == null);
                try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) != null);
            }
            try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "s.kv.comp.L{d}", .{L})) != null);
            try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "s.kv.ik.L{d}", .{L})) != null);
        }
        // Line.state's drain: each encoder layer's SWA ring, the last stage's stash ring
        for (st.lo..@min(st.hi, st.d)) |L| {
            try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "s.L{d}.swa.v", .{L})) != null);
            try testing.expect(plan.sizes.get(try std.fmt.bufPrint(&nb, "s.L{d}.swa.s", .{L})) != null);
        }
        if (st.last) for (@import("ced.zig").roles) |role| try testing.expect(plan.sizes.get(role) != null);
    }
}

test "the boot refuses a stage whose selection the plan does not model (candidate source, top-k off the powers of two, reindex, GM_V2 != 1)" {
    // pipe.planned (boot: model.zig's residentBytes, forward_prefill.plan) refuses rather than leaving the stage's
    // selection scratch unplanned, which would fail as `Unbound` (or overrun a smaller slot) mid-prompt (window 6)
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const base: Config = .{};
    var w = try widths(aa);
    // the reindex case's layer 3 runs an indexer: its projections' widths (layer 2's; the fixture's layer 3 has none)
    for ([_][]const u8{ "ix_wk", "ix_wq_b" }) |m| try w.dense.put(aa, try std.fmt.allocPrint(aa, "L3.attn.{s}", .{m}), w.dense.get(try std.fmt.allocPrint(aa, "L2.attn.{s}", .{m})).?);
    // the GM_V2=0 / split cases' x3gm v1 / width-bit plans: each routed layer's K2 widths (as the k2 test below)
    for (0..40) |L| try w.gm.put(aa, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    const s = try settings();
    const Case = struct { cfg: Config, gm: @import("gm2pf.zig").Mode = .one, refused: [4]bool };
    var topk = base;
    topk.index_topk = 384;
    var cand = base;
    cand.candidate_source = 8; // rank 1's index source
    var reindex = base;
    reindex.index_sources = @TypeOf(base.index_sources).of(&.{ 2, 3, 8, 14, 20, 24, 28, 32, 36 }); // 3: rank 0's, not a kv source
    try testing.expectEqual(.reindex, reindex.mode(3));
    const cases = [_]Case{
        .{ .cfg = base, .refused = .{ false, false, false, false } },
        .{ .cfg = topk, .refused = .{ true, true, true, false } }, // rank 3 (15..19) selects at no index source
        .{ .cfg = cand, .refused = .{ s.owner(8) == 0, s.owner(8) == 1, s.owner(8) == 2, false } },
        .{ .cfg = reindex, .refused = .{ true, false, false, false } },
        .{ .cfg = base, .gm = .off, .refused = .{ true, true, true, true } },
        .{ .cfg = base, .gm = .split, .refused = .{ true, true, true, true } },
    };
    try testing.expectEqual(@as(u32, 0), s.owner(3));
    for (cases) |c| for (0..4) |ri| {
        const r: u32 = @intCast(ri);
        var o = servedOptions(r);
        o.gm_v2 = c.gm;
        const got = pipe.planned(aa, &c.cfg, &w, pipe.stageOptions(o, s.rows), s, r);
        if (c.refused[ri]) try testing.expectError(error.Unsupported, got) else _ = try got;
    };
}

test "a stage's gm_plan glue reads a K2 table (k2 > 0, the forward's bare name) at TF_DSV41_GM_V2=0 and split, never at 1" {
    // the reach of forward_prefill.glueStep's "s.L<i>.ex.k2_{g,d}" read from a stage (pipe_run.stageGlue): why
    // pipe.planned and pipe_run.check refuse every mode but `.one`
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w = try widths(aa);
    // each routed layer's K2 widths (the run's plan; two bits, as the other prefill tests)
    for (0..40) |L| try w.gm.put(aa, @intCast(L), .{ (1 << 3) | (1 << 4), (1 << 3) | (1 << 4) });
    const s = try settings();
    for ([_]@import("gm2pf.zig").Mode{ .one, .off, .split }) |gm| {
        var o = servedOptions(0);
        o.gm_v2 = gm;
        const cs = try pipe.emit(aa, &cfg, &w, pipe.stageOptions(o, s.rows), &s, 0, 2048, 2048);
        var k2: usize = 0;
        var plans: usize = 0;
        for (cs) |c| if (std.mem.eql(u8, c.name, "glue.gm_plan")) {
            plans += 1;
            if (c.args[6].arg.i > 0) k2 += 1;
        };
        try testing.expect(plans > 0);
        if (gm == .one) try testing.expectEqual(@as(usize, 0), k2) else try testing.expect(k2 > 0);
    }
}

test "the stage plan's indexer scores and sort keys hold a 2,048-row segment from 2,048 (round 1's 16 MiB scores overran an 8 MiB slot)" {
    // the 2,048-row segment from 2,048 (nvis 2,048, materialised whole) writes 16,777,216 bytes of scores and 33,554,432
    // of sort keys; e62f70a0's plan held "L.pp~ix.scores" at 8,388,608 (its (2048, 0) program: 8 MiB past the slot) and
    // "L.pp~ix.keys" at 109,051,904 already (its last start's stream merge names the same role: no keys overrun).
    // Pinned exactly: a plan that drops either program fails here, not as corruption on hardware
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const s = try settings();
    for (0..3) |ri| {
        const r: u32 = @intCast(ri);
        const o1 = pipe.stageOptions(servedOptions(r), s.rows);
        var plan: buffers.Plan = .{ .a = aa };
        for (try pipe.planned(aa, &cfg, &w, o1, s, r)) |x| try plan.add(try pipe.program(aa, &cfg, &w, o1, &s, r, x[0], x[1]));
        var need: buffers.Plan = .{ .a = aa };
        try need.add(try pipe.program(aa, &cfg, &w, o1, &s, r, 2048, 2048));
        try testing.expectEqual(@as(u64, 16777216), need.sizes.get("L.pp~ix.scores").?);
        try testing.expectEqual(@as(u64, 33554432), need.sizes.get("L.pp~ix.keys").?);
        // the plan's: the materialised ceiling n x nvis <= index_budget / 12 at (1547, nvis 3615), f32 scores; the keys
        // the larger of its i64 sort keys (44,739,240) and the stream's merge (block_prefill.selectScratch)
        try testing.expectEqual(@as(u64, 22369620), plan.sizes.get("L.pp~ix.scores").?);
        try testing.expect(plan.sizes.get("L.pp~ix.keys").? >= 44739240);
        try testing.expectEqual(@as(u64, 109051904), plan.sizes.get("L.pp~ix.keys").?);
    }
}

test "an Unbound names its role and the bytes the call needs (run.Runner.tensorAddr's warn; window 6's error named nothing)" {
    const t: calls.Tensor = .{ .role = .{ .buf = "L.pp~ix.pos0" }, .dt = .i32, .shape = &.{1}, .stride = &.{1} };
    var mb: [256]u8 = undefined;
    const m = buffers.unboundMessage(&mb, t);
    try testing.expect(std.mem.indexOf(u8, m, "L.pp~ix.pos0") != null);
    try testing.expect(std.mem.indexOf(u8, m, "needs 4 bytes") != null);
    try testing.expect(std.mem.indexOf(u8, m, "holds 0") != null);
}
