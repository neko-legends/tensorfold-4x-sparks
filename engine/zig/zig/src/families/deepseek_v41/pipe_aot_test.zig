//! Host test of the AOT fill's coverage of the pipeline's stage programs (TF_DSV41_PREFILL_PIPE, aot_needs.enumerateStage):
//! at TP=4's served config (TF_DSV41_CONTEXT=420096: 1,641 pages), on every rank, every Triton launch of the stage
//! programs the boot plans (pipe.planned) and of sampled prompt segments (pipe.min_seg .. 2,048 rows at 16-grid starts,
//! ragged ends and the line's last start, as the prompt pieces and snapshots fall) reduces, through the served Triton
//! signatures (fixtures/triton-sigs.json: `triton_fill.py sigs` of the Python tree the fill compiles), to a variant
//! the fill's regimes list (aot_needs.enumerateRegimes: the regime sweep WITHOUT the planned loop, so neither leg is in
//! the list by construction). The fill itself lists the planned programs too (enumerateStage); what this test proves
//! is the regime argument at these points, not at every (rows, start): the rest is the argument
//! (docs/ZIG-PREFILL-PIPE.md, "What the tests prove"). The boot's plan at this config is frozen below, so a change to
//! pipe.planned / lastEnd shows here, deliberately. A boot-requested variant the host pass did not list cost a window
//! (2026-10-10).
const std = @import("std");
const testing = std.testing;
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("m1_check.zig");
const aot = @import("aot_needs.zig");
const pipe = @import("pipe.zig");
const Config = @import("config.zig").Config;

const limit: i64 = 420096;

/// aot-needs' options for config/tp4-zig.env (Served.fromEnv: TF_TP_WORLD=4, TF_TP_RANK=r, the PF_4K workspace set,
/// x3gm v2, INDEX_BOUND, the 64 MiB index budget, MHC_PFDEC=17, R1, drafts), the pool as it plans it (KV not split).
fn served(rank: u32) block.Options {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .world = 4, .rank = rank, .r1 = true, .expert_topp = 0.85, .limit = limit, .rope_rows = limit + 2048, .prefill_rows = 4096, .pf4k = true, .gm_v2 = .one, .index_bound = true, .index_budget = 64 << 20, .mhc_pf_rows = 17, .taps = true, .pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages } };
}

/// pipe.planned at this config (frozen 2026-10-11 from pipe.planned; update deliberately, never re-derived in the test):
/// the whole segment and the shortest, at 0 and at their last start (lastEnd: the limit), then on ranks 0-2 (index
/// sources 2, 8, 14) the selection's worst segments (block_prefill.selectWorst): the materialised scores' largest launch
/// and the most row blocks (the stream's largest is the whole segment's last start); rank 3's stage selects nowhere
const frozen_planned = [_][]const [2]i64{
    &.{ .{ 2048, 0 }, .{ 2048, 418048 }, .{ 33, 0 }, .{ 33, 420063 }, .{ 1547, 5684 }, .{ 1366, 6825 } },
    &.{ .{ 2048, 0 }, .{ 2048, 418048 }, .{ 33, 0 }, .{ 33, 420063 }, .{ 1547, 5684 }, .{ 1366, 6825 } },
    &.{ .{ 2048, 0 }, .{ 2048, 418048 }, .{ 33, 0 }, .{ 33, 420063 }, .{ 1547, 5684 }, .{ 1366, 6825 } },
    &.{ .{ 2048, 0 }, .{ 2048, 418048 }, .{ 33, 0 }, .{ 33, 420063 } },
};

fn settings() !pipe.Settings {
    const F = struct {
        fn get(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_PREFILL_PIPE")) "1024" else null;
        }
    };
    const cfg: Config = .{};
    return (try pipe.settings(&F.get, &cfg, 4)).?;
}

test "TP=4 (CONTEXT=420096): the boot's planned and sampled prompt stage programs are in the fill's regimes (planned loop excluded), every rank" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(struct { kernels: aot.Sigs }, gpa, @embedFile("fixtures/triton-sigs.json"), .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(aa, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(aa, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    try pipe.fillGroups(aa, &w, &cfg);
    const s = try settings();
    var discard: std.Io.Writer.Discarding = .init(&.{});
    var buf: [64]u32 = undefined;
    for (0..4) |r| {
        const o1 = pipe.stageOptions(served(@intCast(r)), s.rows);
        var needs: aot.Needs = .{ .gpa = gpa, .sigs = &parsed.value.kernels };
        defer needs.deinit();
        // the regimes alone: the planned programs are not in this list by construction
        try aot.enumerateRegimes(gpa, &needs, &cfg, &w, o1, s, @intCast(r), &discard.writer);
        try testing.expect(needs.keys.count() > 0);
        // every stage emits up to the context's end at the served config (the boot's agreed end is the limit)
        try testing.expectEqual(@as(u64, limit), try pipe.lastEnd(&cfg, &w, o1, &s, @intCast(r)));
        const st = s.stage(@intCast(r));
        const io: bp.StageIo = .{ .entered = r > 0, .last = st.last, .sel_in = st.selIn(&cfg), .sel_out = st.selOut(&cfg, s.boundsSlice()) };
        // the boot's plan, then prompt segments: whole ones, a piece's or a resumed prompt's ragged ones, the shortest
        var cases: std.ArrayList([2]i64) = .empty;
        const plan = try pipe.planned(aa, &cfg, &w, o1, s, @intCast(r));
        try testing.expectEqual(frozen_planned[r].len, plan.len);
        for (frozen_planned[r], plan) |f, p| try testing.expectEqual(f, p);
        for (plan) |p| try cases.append(aa, p);
        for ([_]i64{ 2048, 2047, 1500, 1000, 100, 64, 34, 33 }) |n| for ([_]i64{ 0, 16, 2048, 4112, 37984, 154176, 300000, 417984, limit - n }) |start| {
            if (start + n <= limit) try cases.append(aa, .{ n, start });
        };
        for (cases.items) |p| {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            const cs = try bp.emitStage(pa.allocator(), &cfg, &w, o1, st.layers(&buf), p[0], p[1], io);
            for (cs) |*c| if (c.triton) {
                const key = try needs.reduce(pa.allocator(), c);
                if (needs.keys.get(key) == null) {
                    std.debug.print("rank {d}: a {d}-row stage at {d} launches a variant aot-needs did not list: {s}\n", .{ r, p[0], p[1], key });
                    return error.TestUnexpectedResult;
                }
            };
        }
    }
}
