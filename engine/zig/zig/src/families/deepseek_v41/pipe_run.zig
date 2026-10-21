//! TF_DSV41_PREFILL_PIPE's runtime (docs/ZIG-PREFILL-PIPE.md; patches/0003 pipe.Pipeline): one prompt's segments
//! through the TP ranks' stages, every rank running the same rounds from the same op stream (the leader sends the
//! prompt pieces, every follower runs forward_prefill.prompt on the same ids; pipe.route decides from what every rank
//! holds).
//!
//! A round t:
//! 1. the new segment's embedding: every rank's vocabulary rows, gathered (the TP glue's), stage 0's input;
//! 2. each Engram layer's rows of the segment its stage runs this round: every rank's head columns, gathered;
//! 3. rank s runs its stage on segment t - s (pipe.program: a world-1 program in the "pp~" namespace with the stage
//!    copies "P.*"), then keeps its ratio-2 sources' carries (the stage's own commit);
//! 4. the exchange: the stage output to rank s + 1 (Collective.sendRecv: one NCCL group, the hand-off packed as
//!    pipe.HandoffLayout), and one all-gather of the comp rows and index keys every stage's kv sources wrote
//!    (pipe.RowsLayout; pipe.rowsOf's rows through the slot's page table, which every rank holds alike).
//! The drain runs rounds with no new segment until the line is empty, then one all-gather of each stage's SWA rings, its
//! ratio-2 carries and the last stage's CED stash ring (pipe.FinalsLayout): the slot's state is whole on every rank.
//! A segment's host state (position, Engram tail, session ids, stash ids) advances at its entry, on every rank.
//!
//! Fail-fast: an error inside a round or a drain aborts this rank's communicator before it returns, so no other rank
//! waits forever in a collective this rank will not reach; followers exit on the error, the fate channel takes the
//! fleet down. No rank continues past a broken line.

const std = @import("std");
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const pipe = @import("pipe.zig");
const pipe_load = @import("pipe_load.zig");
const longpf = @import("longpf.zig");
const ced = @import("ced.zig");
const hs = @import("host_stage.zig");
const eh = @import("engram_host.zig");
const plan = @import("plan.zig");
const load = @import("cuda_weights.zig");
const vision_rows = @import("vision_rows.zig");
const fwd = @import("forward.zig");
const pf = @import("forward_prefill.zig");
const Forward = fwd.Forward;
const Config = @import("config.zig").Config;

const log = std.log.scoped(.dsv41);

pub const Error = error{ BadGlue, LineOrder, NoEngram, NoLineBuffers };

const Bufs = struct {
    /// stage 0's input: the embedded segment [rows, 4 D] bf16
    x0: ?cuda.DeviceBuffer = null,
    /// the embedding's send [rows, D] and gathered [W, rows, D] bf16, the ids int64 [rows]
    emb_send: ?cuda.DeviceBuffer = null,
    emb_recv: ?cuda.DeviceBuffer = null,
    dev_ids: ?cuda.DeviceBuffer = null,
    /// an Engram layer's full rows [rows, K] bf16 (the layers this rank's stage runs), the send / gathered columns
    eng_rows: [eh.max_layers]?cuda.DeviceBuffer = @splat(null),
    eng_send: ?cuda.DeviceBuffer = null,
    eng_gathered: ?cuda.DeviceBuffer = null,
    /// the stage hand-off, out and in (pipe.HandoffLayout at the line's rows)
    send_f: ?cuda.DeviceBuffer = null,
    recv_f: ?cuda.DeviceBuffer = null,
    /// the rows exchange (a stride a rank) and the drain's hand-off
    rows_send: ?cuda.DeviceBuffer = null,
    rows_recv: ?cuda.DeviceBuffer = null,
    fin_send: ?cuda.DeviceBuffer = null,
    fin_recv: ?cuda.DeviceBuffer = null,

    fn all(b: *Bufs) [12 + eh.max_layers]*?cuda.DeviceBuffer {
        var out: [12 + eh.max_layers]*?cuda.DeviceBuffer = undefined;
        const fixed = [_]*?cuda.DeviceBuffer{ &b.x0, &b.emb_send, &b.emb_recv, &b.dev_ids, &b.eng_send, &b.eng_gathered, &b.send_f, &b.recv_f, &b.rows_send, &b.rows_recv, &b.fin_send, &b.fin_recv };
        @memcpy(out[0..fixed.len], &fixed);
        for (&b.eng_rows, 0..) |*x, i| out[fixed.len + i] = x;
        return out;
    }
};

/// TF_DSV41_PIPE_TRACE: a line's round times (each part synced).
const Stats = struct { rounds: u64 = 0, segments: u64 = 0, pre_ns: u64 = 0, stage_ns: u64 = 0, exchange_ns: u64 = 0, drain_ns: u64 = 0 };

pub const Line = struct {
    gpa: std.mem.Allocator,
    s: pipe.Settings,
    rank: u32,
    world: u32,
    /// the stage programs' options (pipe.stageOptions of the forward's at boot)
    o1: block.Options,
    sch: pipe.Schedule,
    rows_lay: pipe.RowsLayout,
    fin_lay: pipe.FinalsLayout,
    /// the open line's slot (null: none open)
    slot: ?u32 = null,
    /// the prompt's ids from position `base`: the slot's Engram tail when the line opened, then every segment entered
    hist: std.ArrayList(u32) = .empty,
    base: u64 = 0,
    bufs: Bufs = .{},
    ids_stage: hs.Stage = .{},
    eng_stage: [eh.max_layers]hs.Stage = @splat(.{}),
    stats: Stats = .{},
    lines: u64 = 0,
    /// the tp session's fail-fast (model.zig: tp.Session.fatal): a line that fails takes every rank down with its
    /// reason (the fate channel; peers' kernels waiting on this rank are aborted by their own hooks). null (tests, a
    /// session without fail-fast): the communicator aborted and the error returned
    fatal: ?Fatal = null,

    pub const Fatal = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, where: []const u8, err: anyerror) void };

    pub fn create(gpa: std.mem.Allocator, f: *const Forward, s: pipe.Settings) !*Line {
        const ln = try gpa.create(Line);
        ln.* = .{ .gpa = gpa, .s = s, .rank = @intCast(f.comm.rank()), .world = @intCast(f.comm.world()), .o1 = pipe.stageOptions(f.opts, s.rows), .sch = .{ .world = s.world }, .rows_lay = pipe.RowsLayout.of(f.cfg, s), .fin_lay = pipe.FinalsLayout.of(f.cfg, s) };
        return ln;
    }

    /// Installs the line on the forward: its state's pointer and the settle / release hooks.
    pub fn install(ln: *Line, f: *Forward) void {
        f.prefill_state.pipe = ln;
        f.prefill_state.pipe_settle = &pf.settle;
        f.prefill_state.pipe_release = &pf.pipeRelease;
    }

    pub fn deinit(ln: *Line) void {
        for (ln.bufs.all()) |b| if (b.*) |*x| x.free();
        ln.ids_stage.deinit();
        for (&ln.eng_stage) |*x| x.deinit();
        ln.hist.deinit(ln.gpa);
        ln.gpa.destroy(ln);
    }

    /// The device buffers' bytes (`bind` allocates them at boot; the boot check prices them before the Engram host is
    /// loaded: its head columns are the config's, complete heads TP-major as engram_host.headShard splits them).
    pub fn bufferBytes(cfg: *const Config, s: pipe.Settings, rank: u32) u64 {
        const W: u64 = s.world;
        const R: u64 = s.rows;
        const D: u64 = cfg.hidden;
        const hand = pipe.HandoffLayout.of(cfg, s.rows, true, true).total;
        var total: u64 = R * 4 * D * 2 + R * D * 2 + W * R * D * 2 + R * 8 + 2 * hand;
        total += pipe.RowsLayout.of(cfg, s).stride * (W + 1) + pipe.FinalsLayout.of(cfg, s).stride * (W + 1);
        if (cfg.engram_layers.len > 0) {
            const heads: u64 = cfg.hashCols();
            const cols: u64 = (heads + W - 1) / W * cfg.engram_head_dim;
            const K: u64 = heads * cfg.engram_head_dim;
            total += R * cols * 2 + W * R * cols * 2;
            const st = s.stage(rank);
            for (cfg.engram_layers.items()) |L| if (L >= st.lo and L < st.hi) {
                total += R * K * 2;
            };
        }
        return total;
    }

    /// At the boot's end, every rank: one tiny shift along the line, so NCCL makes its point-to-point connections (and
    /// their buffers) at boot, where the memory checks see them, not at the first long prompt.
    pub fn warm(ln: *Line, f: *Forward) !void {
        const to: ?u32 = if (ln.rank + 1 < ln.world) ln.rank + 1 else null;
        const from: ?u32 = if (ln.rank > 0) ln.rank - 1 else null;
        try f.comm.sendRecv(ln.bufs.send_f.?.ptr, 16, to, ln.bufs.recv_f.?.ptr, 16, from, .u8, f.runner.stream.handle);
        try f.runner.stream.synchronize();
    }

    /// After Runner.bind and the forms: the line's buffers, and x3gm's pointer tables of the stage copies (`copies`:
    /// their world-1 plan, pipe_load.loadStage's).
    pub fn bind(ln: *Line, f: *Forward, w: *const load.Weights, copies: *const plan.Plan) !void {
        const d = f.runner.d;
        const cfg = f.cfg;
        const W: usize = ln.world;
        const R: usize = ln.s.rows;
        const D: usize = cfg.hidden;
        const b = &ln.bufs;
        const hand: usize = @intCast(pipe.HandoffLayout.of(cfg, ln.s.rows, true, true).total);
        b.x0 = try cuda.DeviceBuffer.alloc(d, R * 4 * D * 2);
        b.emb_send = try cuda.DeviceBuffer.alloc(d, R * D * 2);
        b.emb_recv = try cuda.DeviceBuffer.alloc(d, W * R * D * 2);
        b.dev_ids = try cuda.DeviceBuffer.alloc(d, R * 8);
        b.send_f = try cuda.DeviceBuffer.alloc(d, hand);
        b.recv_f = try cuda.DeviceBuffer.alloc(d, hand);
        const rs: usize = @intCast(@max(ln.rows_lay.stride, 16));
        b.rows_send = try cuda.DeviceBuffer.alloc(d, rs);
        b.rows_recv = try cuda.DeviceBuffer.alloc(d, W * rs);
        const fs: usize = @intCast(@max(ln.fin_lay.stride, 16));
        b.fin_send = try cuda.DeviceBuffer.alloc(d, fs);
        b.fin_recv = try cuda.DeviceBuffer.alloc(d, W * fs);
        if (f.engram) |h| {
            const cols: usize = h.headShard(ln.rank, ln.s.world)[1] * @as(usize, cfg.engram_head_dim);
            const K: usize = @as(usize, cfg.hashCols()) * cfg.engram_head_dim;
            b.eng_send = try cuda.DeviceBuffer.alloc(d, @max(R * cols * 2, 16));
            b.eng_gathered = try cuda.DeviceBuffer.alloc(d, @max(W * R * cols * 2, 16));
            const st = ln.s.stage(ln.rank);
            for (h.layer_ids, 0..) |L, li| if (L >= st.lo and L < st.hi) {
                b.eng_rows[li] = try cuda.DeviceBuffer.alloc(d, R * K * 2);
            };
        }
        // x3gm's pointer tables of the stage copies (the TP ones hold the shard's addresses): the routed experts' trellis
        // addresses, gate / up / down
        var lb: [64]u32 = undefined;
        var nb: [64]u8 = undefined;
        for (ln.s.stage(ln.rank).weightLayers(&lb)) |L| for ([_]struct { []const u8, usize }{ .{ "tg", 0 }, .{ "tu", 2 }, .{ "td", 1 } }) |t| {
            const dst = f.runner.addressOf(try std.fmt.bufPrint(&nb, "s." ++ pipe.prefix ++ "L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
            var ta = std.heap.ArenaAllocator.init(ln.gpa);
            defer ta.deinit();
            const bytes = try pipe_load.gmTable(ta.allocator(), w, copies, L, t[1]);
            try cuda.DeviceBuffer.upload(.{ .d = d, .ptr = dst, .len = bytes.len }, 0, bytes);
        };
        try f.runner.stream.synchronize();
    }

    /// The open line (pipe.route's view), or null.
    pub fn at(ln: *const Line) ?pipe.LineAt {
        const s = ln.slot orelse return null;
        return .{ .slot = s, .next = ln.base + ln.hist.items.len };
    }

    /// A prompt call on the active slot (forward_prefill.prompt): pipe.route's steps, tensor-parallel rows through
    /// forward_prefill's own segments.
    pub fn call(ln: *Line, f: *Forward, ids: []const u32) !void {
        const start = f.slot.pos;
        const images = vision_rows.segmentOptions(f, ids).image_keep;
        const buf = try ln.gpa.alloc(pipe.Step, ids.len / @max(1, ln.s.rows) + 4);
        defer ln.gpa.free(buf);
        const steps = pipe.route(ln.s, ln.at(), activeSlot(f), start, ids.len, images, buf);
        for (steps, 0..) |st, i| switch (st) {
            .enter => |sg| {
                // the next segment's Engram reads ahead of this round (a wrong guess costs a read, never a bit): the
                // call's next, or past the call's last the next prompt piece's (op 60, Forward.prefillAhead)
                const nxt: ?pipe.Seg = if (i + 1 < steps.len) switch (steps[i + 1]) {
                    .enter => |nx| nx,
                    else => null,
                } else null;
                if (nxt) |nx| {
                    ln.readAhead(f, ids[@intCast(nx.start - start)..][0..nx.n], ids[0..@intCast(nx.start - start)]);
                } else if (i + 1 == steps.len and f.prefill_state.next.items.len > 0) {
                    const piece = f.prefill_state.next.items;
                    ln.readAhead(f, piece[0..@min(piece.len, ln.s.rows)], ids);
                }
                try ln.enter(f, ids[@intCast(sg.start - start)..][0..sg.n]);
            },
            .drain => try ln.drain(f),
            .abandon => ln.abandon(),
            .tp => |sg| try pf.tpRows(f, ids[@intCast(sg.start - start)..][0..sg.n]),
        };
    }

    /// The slot is reset (released, restored over): its segments in the line are dropped (Python's abandon).
    pub fn abandon(ln: *Line) void {
        if (ln.slot == null) return;
        log.info("pipeline: rank {d}: slot {d}'s line abandoned at round {d}", .{ ln.rank, ln.slot.?, ln.sch.t });
        ln.close();
    }

    fn close(ln: *Line) void {
        ln.sch.clear();
        ln.slot = null;
        ln.hist.clearRetainingCapacity();
        ln.base = 0;
    }

    fn idsOf(ln: *const Line, sg: pipe.Seg) []const u32 {
        return ln.hist.items[@intCast(sg.start - ln.base)..][0..sg.n];
    }

    /// Engram's lookback of a segment: the ids before it, at most max_ngram - 1.
    fn lookback(ln: *const Line, sg: pipe.Seg, keep: usize) []const u32 {
        const i: usize = @intCast(sg.start - ln.base);
        return ln.hist.items[i -| keep..i];
    }

    /// A segment enters (every rank): the line opens on its slot if none is; its host state advances; one round. Any
    /// failure past the open is the line's (`fail`): a half-entered segment never stays behind in a running line.
    fn enter(ln: *Line, f: *Forward, ids: []const u32) !void {
        ln.admit(f, ids) catch |e| return ln.fail(f, e);
    }

    fn admit(ln: *Line, f: *Forward, ids: []const u32) !void {
        const start = f.slot.pos;
        if (ln.slot == null) {
            ln.slot = activeSlot(f);
            ln.sch.clear();
            ln.hist.clearRetainingCapacity();
            ln.base = start - f.slot.tail_len;
            try ln.hist.appendSlice(ln.gpa, f.slot.tail[0..f.slot.tail_len]);
            ln.lines += 1;
            if (ln.lines <= 4 or ln.lines % 64 == 0) log.info("pipeline: rank {d}: slot {d}'s prompt from {d} through the stages {any} (line {d})", .{ ln.rank, ln.slot.?, start, ln.s.boundsSlice(), ln.lines });
        } else if (ln.slot.? != activeSlot(f) or ln.base + ln.hist.items.len != start) return error.LineOrder;
        if (f.kv) |kx| try kx.reserve(start + ids.len);
        try ln.hist.appendSlice(ln.gpa, ids);
        try pf.advanceHost(f, start, ids);
        ln.stats.segments += 1;
        try ln.round(f, .{ .start = start, .n = @intCast(ids.len) });
    }

    /// Runs the line empty and hands every stage's state to every rank (every rank, the same op).
    pub fn drain(ln: *Line, f: *Forward) !void {
        if (ln.slot == null) return;
        const t0 = ln.tick(f);
        while (ln.sch.busy()) ln.round(f, null) catch |e| return ln.fail(f, e);
        ln.finals(f) catch |e| return ln.fail(f, e);
        ln.tock(f, t0, &ln.stats.drain_ns);
        if (ln.s.trace) {
            const st = ln.stats;
            const n = @max(st.rounds, 1);
            log.info("pipeline rank {d}: {d} segments, {d} rounds; a round: pre {d:.1} ms, stage {d:.1} ms, exchange {d:.1} ms; drain {d:.1} ms", .{ ln.rank, st.segments, st.rounds, ms(st.pre_ns) / @as(f64, @floatFromInt(n)), ms(st.stage_ns) / @as(f64, @floatFromInt(n)), ms(st.exchange_ns) / @as(f64, @floatFromInt(n)), ms(st.drain_ns) });
            ln.stats = .{};
        }
        ln.close();
    }

    /// A failed line on this rank: every rank goes down (fail-fast's exit, which in serving does not return), never a
    /// rank left waiting in a collective or a point-to-point this rank will not reach, nor one serving on.
    fn fail(ln: *Line, f: *Forward, e: anyerror) anyerror {
        log.err("pipeline: rank {d}: the line failed ({t}): every rank goes down", .{ ln.rank, e });
        ln.close();
        if (ln.fatal) |h| h.run(h.ctx, "the prefill pipeline", e);
        f.comm.abort();
        return e;
    }

    /// At boot, every rank: the end up to which every rank's stage emits (pipe.lastEnd, the least over the ranks, one
    /// all-gather), so no route hands a rank a segment its stage refuses.
    pub fn agreeEnd(ln: *Line, f: *Forward) !void {
        const mine = try pipe.lastEnd(f.cfg, f.widths, ln.o1, &ln.s, ln.rank);
        const send = ln.bufs.fin_send.?.ptr;
        const recv = ln.bufs.fin_recv.?.ptr;
        var b: [8]u8 = undefined;
        std.mem.writeInt(u64, &b, mine, .little);
        try cuda.DeviceBuffer.upload(.{ .d = f.runner.d, .ptr = send, .len = 8 }, 0, &b);
        try f.comm.allGather(send, recv, 8, .u8, f.runner.stream.handle);
        try f.runner.stream.synchronize();
        var all: [pipe.max_world * 8]u8 = undefined;
        try cuda.DeviceBuffer.download(.{ .d = f.runner.d, .ptr = recv, .len = 8 * ln.world }, 0, all[0 .. 8 * ln.world]);
        var end: u64 = std.math.maxInt(u64);
        for (0..ln.world) |q| end = @min(end, std.mem.readInt(u64, all[8 * q ..][0..8], .little));
        ln.s.end = end;
        if (end < @as(u64, @intCast(f.opts.limit))) log.info("pipeline: rank {d}: segments end by {d} (a stage refuses past it; the rest runs tensor parallel)", .{ ln.rank, end });
    }

    fn round(ln: *Line, f: *Forward, new: ?pipe.Seg) !void {
        ln.sch.enter(new);
        ln.stats.rounds += 1;
        var t0 = ln.tick(f);
        if (new) |sg| try ln.embed(f, sg);
        if (f.engram) |h| for (h.layer_ids) |L| {
            const st = ln.s.stage(ln.s.owner(L));
            if (L >= st.d or !(L >= st.lo and L < st.hi)) continue;
            if (ln.sch.segOf(st.rank)) |sg| try ln.engram(f, h, L, sg, st.rank == ln.rank);
        };
        ln.tock(f, t0, &ln.stats.pre_ns);
        t0 = ln.tick(f);
        if (ln.sch.segOf(ln.rank)) |sg| try ln.stage(f, sg);
        ln.tock(f, t0, &ln.stats.stage_ns);
        t0 = ln.tick(f);
        try ln.exchange(f);
        ln.tock(f, t0, &ln.stats.exchange_ns);
    }

    // -- 1. the embedding, 2. the Engram rows ---------------------------------------------------------------------

    fn embed(ln: *Line, f: *Forward, sg: pipe.Seg) !void {
        const r = f.runner;
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        const ids = ln.idsOf(sg);
        const n: usize = sg.n;
        const D: usize = f.cfg.hidden;
        const W: usize = ln.world;
        // the previous copy out of the ids' staging has finished (Forward's embed: a pageable-size window syncs)
        if (n <= hs.chunk) try r.stream.synchronize();
        const h = try ln.ids_stage.take(r.d, 8 * n);
        for (ids, 0..) |id, i| std.mem.writeInt(i64, h[8 * i ..][0..8], id, .little);
        const dev = (ln.bufs.dev_ids orelse return error.NoLineBuffers).ptr;
        try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(dev, h.ptr, 8 * n, s), "cuMemcpyHtoDAsync");
        try ln.ids_stage.issued(r.d, r.stream);
        const table = r.weights.get("embed") orelse return error.MissingWeight;
        const V: usize = f.cfg.vocab / W;
        const send = ln.bufs.emb_send.?.ptr;
        const recv = ln.bufs.emb_recv.?.ptr;
        try k.embedSend(dev, n, table.ptr, @intCast(ln.rank * V), V, D, send);
        try f.comm.allGather(send, recv, n * D, .bf16, s);
        // stage 0 alone reads it (every rank joined the gather)
        if (ln.rank == 0) try k.embedSum(recv, W, n, D, ln.bufs.x0.?.ptr);
    }

    /// Layer L's rows of `sg` at full width: every rank's head columns (its shard of the table), gathered; the stage
    /// that runs L keeps them (glue engram_rows copies them in).
    fn engram(ln: *Line, f: *Forward, h: *const eh.Host, L: u32, sg: pipe.Seg, own: bool) !void {
        const r = f.runner;
        const s = r.stream.handle;
        const li = std.mem.indexOfScalar(u32, h.layer_ids, L) orelse return error.NoEngram;
        const W: usize = ln.world;
        const n: usize = sg.n;
        const cols: usize = h.headShard(ln.rank, ln.s.world)[1] * @as(usize, f.cfg.engram_head_dim);
        const bytes = 2 * n * cols;
        if (n <= hs.chunk) try r.stream.synchronize();
        const host = try ln.eng_stage[li].take(r.d, bytes);
        try h.windowRows(f.gpa, try f.engramSource(h), ln.idsOf(sg), ln.lookback(sg, h.max_ngram - 1), L, ln.rank, ln.s.world, f.cfg.engram_head_dim, @alignCast(std.mem.bytesAsSlice(u16, host)));
        const send = (ln.bufs.eng_send orelse return error.NoLineBuffers).ptr;
        const gathered = ln.bufs.eng_gathered.?.ptr;
        try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(send, host.ptr, bytes, s), "cuMemcpyHtoDAsync");
        try ln.eng_stage[li].issued(r.d, r.stream);
        try f.comm.allGather(send, gathered, n * cols, .bf16, s);
        if (own) try r.kernels.others(r.stream).gatherCols(gathered, W, n, cols, (ln.bufs.eng_rows[li] orelse return error.NoLineBuffers).ptr);
    }

    /// The reads of a coming segment's Engram rows started (I/O only: a wrong guess costs a read, never a bit).
    fn readAhead(ln: *Line, f: *Forward, ids: []const u32, before: []const u32) void {
        const h = f.engram orelse return;
        if (!f.engram_prefetch) return;
        const src = (f.engramSource(h) catch return) orelse return;
        const keep: usize = h.max_ngram - 1;
        var buf: [2 * eh.max_ngram]u32 = undefined;
        const lb: []const u32 = if (before.len >= keep) before[before.len - keep ..] else blk: {
            // fewer of the call's ids than the lookback before it: the slot's committed tail first (TP's readAhead)
            const have = f.slot.tail[0..f.slot.tail_len];
            const old = @min(have.len, keep - before.len);
            @memcpy(buf[0..old], have[have.len - old ..]);
            @memcpy(buf[old .. old + before.len], before);
            break :blk buf[0 .. old + before.len];
        };
        h.prefetch(f.gpa, src, ids, lb, ln.rank, ln.s.world);
    }

    // -- 3. the stage -----------------------------------------------------------------------------------------------

    fn stage(ln: *Line, f: *Forward, sg: pipe.Seg) !void {
        var arena = std.heap.ArenaAllocator.init(ln.gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const cs = try pipe.program(a, f.cfg, f.widths, ln.o1, &ln.s, ln.rank, sg.n, @intCast(sg.start));
        const st = &f.prefill_state;
        const keep_ids = f.ids;
        const keep_pending = f.slot.pending;
        const keep_glue = f.runner.glue;
        defer {
            f.ids = keep_ids;
            f.slot.pending = keep_pending;
            f.runner.glue = keep_glue;
        }
        // the glue reads the segment's ids and window (positions, the stash) on the forward while the stage runs
        f.ids = ln.idsOf(sg);
        f.slot.pending = .{ .start = sg.start, .n = sg.n };
        var lb: [64]u32 = undefined;
        st.layers = try a.dupe(u32, ln.s.stage(ln.rank).layers(&lb));
        st.moe = 0;
        f.runner.glue = .{ .ctx = f, .run = stageGlue };
        try longpf.window(&st.long, f.gpa, f.runner, f.kv, sg.start, cs);
        if (f.runner.prof) |pr| pr.prefillRows(sg.n, false);
        // the stage's own commit: each ratio-2 kv source's carry, its projection's last row (forward_prefill.commit's)
        const r = f.runner;
        const row: usize = 4 * 2 * @as(usize, f.cfg.head_dim);
        var nb: [64]u8 = undefined;
        var sb: [64]u32 = undefined;
        for (ln.s.stage(ln.rank).sources(f.cfg, &sb)) |L| if (f.cfg.compressRatio(L) == 2) {
            const src = try bound(r, try std.fmt.bufPrint(&nb, "w." ++ pipe.tag ++ "L{d}.proj", .{L}));
            const dst = try bound(r, try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L}));
            try copy(r, dst, src + (sg.n - 1) * row, row);
        };
    }

    /// glue pipe_in: the stage's input into its roles (stage 0: the embedded segment; later: the hand-off received).
    fn input(ln: *Line, r: *run.Runner, cfg: *const Config, c: *const calls.Call) !void {
        const sg = ln.sch.segOf(ln.rank) orelse return error.BadGlue;
        if (ln.rank == 0) {
            if (c.args.len != 1) return error.BadGlue;
            const t = c.args[0].arg.t;
            if (bytesOf(t) != @as(usize, sg.n) * 4 * cfg.hidden * 2) return error.BadGlue;
            return copy(r, try r.tensorAddr(t), ln.bufs.x0.?.ptr, bytesOf(t));
        }
        const lay = pipe.HandoffLayout.of(cfg, sg.n, true, c.args.len == 7);
        if (lay.parts != c.args.len) return error.BadGlue;
        const base = ln.bufs.recv_f.?.ptr;
        for (c.args, 0..) |x, i| {
            if (bytesOf(x.arg.t) != lay.bytes[i]) return error.BadGlue;
            try copy(r, try r.tensorAddr(x.arg.t), base + lay.offset[i], lay.bytes[i]);
        }
    }

    /// glue pipe_out: the stage's output packed for the next stage.
    fn output(ln: *Line, r: *run.Runner, cfg: *const Config, c: *const calls.Call) !void {
        const sg = ln.sch.segOf(ln.rank) orelse return error.BadGlue;
        const lay = pipe.HandoffLayout.of(cfg, sg.n, true, c.args.len == 7);
        if (lay.parts != c.args.len) return error.BadGlue;
        const base = ln.bufs.send_f.?.ptr;
        for (c.args, 0..) |x, i| {
            if (bytesOf(x.arg.t) != lay.bytes[i]) return error.BadGlue;
            try copy(r, base + lay.offset[i], try r.tensorAddr(x.arg.t), lay.bytes[i]);
        }
    }

    /// glue engram_rows in a stage: the round's gathered rows of the layer.
    fn engramIn(ln: *Line, f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
        const h = f.engram orelse return error.NoEngram;
        const L: u32 = @intCast(c.args[1].arg.i);
        const li = std.mem.indexOfScalar(u32, h.layer_ids, L) orelse return error.NoEngram;
        const t = c.args[0].arg.t;
        return copy(r, try r.tensorAddr(t), (ln.bufs.eng_rows[li] orelse return error.NoLineBuffers).ptr, bytesOf(t));
    }

    // -- 4. the exchange --------------------------------------------------------------------------------------------

    fn exchange(ln: *Line, f: *Forward) !void {
        const r = f.runner;
        const s = r.stream.handle;
        const cfg = f.cfg;
        const me = ln.s.stage(ln.rank);
        const mine = ln.sch.segOf(ln.rank);
        const prev: ?pipe.Seg = if (ln.rank > 0) ln.sch.segOf(ln.rank - 1) else null;
        // the stage output to the next stage, the previous stage's into this one's input
        const to: ?u32 = if (mine != null and !me.last) ln.rank + 1 else null;
        const from: ?u32 = if (prev != null) ln.rank - 1 else null;
        const sc: usize = if (to != null) @intCast(pipe.HandoffLayout.of(cfg, mine.?.n, true, me.selOut(cfg, ln.s.boundsSlice()) != null).total) else 0;
        const rc: usize = if (from != null) @intCast(pipe.HandoffLayout.of(cfg, prev.?.n, true, me.selIn(cfg) != null).total) else 0;
        try f.comm.sendRecv(ln.bufs.send_f.?.ptr, sc, to, ln.bufs.recv_f.?.ptr, rc, from, .u8, s);
        // the comp rows and index keys every stage's kv sources wrote this round (none: no gather, on every rank alike)
        var sb: [64]u32 = undefined;
        var any = false;
        for (0..ln.world) |q| if (ln.sch.segOf(@intCast(q)) != null and ln.s.stage(@intCast(q)).sources(cfg, &sb).len > 0) {
            any = true;
        };
        if (!any) return;
        const stride: usize = @intCast(ln.rows_lay.stride);
        if (mine) |sg| try ln.rows(f, ln.rank, sg, ln.bufs.rows_send.?.ptr, true);
        try f.comm.allGather(ln.bufs.rows_send.?.ptr, ln.bufs.rows_recv.?.ptr, stride, .u8, s);
        for (0..ln.world) |q| {
            if (q == ln.rank) continue;
            const sg = ln.sch.segOf(@intCast(q)) orelse continue;
            try ln.rows(f, @intCast(q), sg, ln.bufs.rows_recv.?.ptr + q * stride, false);
        }
    }

    /// Stage q's kv sources' rows of segment `sg` between the slot's pool and `part` (q's part of the exchange): out
    /// (`pack`) or in.
    fn rows(ln: *Line, f: *Forward, q: u32, sg: pipe.Seg, part: u64, pack: bool) !void {
        const r = f.runner;
        const kx = f.kv orelse return error.NoPool;
        var sb: [64]u32 = undefined;
        var nb: [64]u8 = undefined;
        for (ln.s.stage(q).sources(f.cfg, &sb)) |L| {
            const ratio = f.cfg.compressRatio(L);
            const span = pipe.rowsOf(ratio, sg.start, sg.n);
            const off = pipe.RowsLayout.at(f.cfg, ln.s, L);
            const comp = try bound(r, try std.fmt.bufPrint(&nb, "s.kv.comp.L{d}", .{L}));
            const ik = try bound(r, try std.fmt.bufPrint(&nb, "s.kv.ik.L{d}", .{L}));
            const shift: u6 = @intCast(block.Pool.shift(ratio));
            const p0 = span[0] >> shift;
            const p1 = (span[1] + (@as(u64, 1) << shift) - 1) >> shift;
            var tb: [64]u32 = undefined;
            if (p1 - p0 > tb.len) return error.TooManyPages;
            for (tb[0..@intCast(p1 - p0)], 0..) |*x, i| x.* = kx.slot.tableAt(@intCast(p0 + i));
            var rb: [64]pipe.Run = undefined;
            const lo = p0 << shift;
            for (pipe.pageRuns(tb[0..@intCast(p1 - p0)], shift, span[0] - lo, span[1] - lo, &rb)) |run_| {
                const j = run_.row + lo - span[0];
                for ([_]struct { u64, u64, u64 }{ .{ comp, off.comp, block.Pool.row_bytes }, .{ ik, off.ik, block.Pool.ik_bytes } }) |fam| {
                    const pool_at = fam[0] + run_.phys * fam[2];
                    const part_at = part + fam[1] + j * fam[2];
                    if (pack) try copy(r, part_at, pool_at, run_.n * fam[2]) else try copy(r, pool_at, part_at, run_.n * fam[2]);
                }
            }
        }
    }

    // -- the drain's hand-off ---------------------------------------------------------------------------------------

    fn finals(ln: *Line, f: *Forward) !void {
        const r = f.runner;
        const s = r.stream.handle;
        const stride: usize = @intCast(ln.fin_lay.stride);
        // the stash ring holds the line slot's bytes on every rank (slots.zig swaps it lazily)
        if (f.pf_stash) |h| try h.run(h.ctx);
        try ln.state(f, ln.rank, ln.bufs.fin_send.?.ptr, true);
        try f.comm.allGather(ln.bufs.fin_send.?.ptr, ln.bufs.fin_recv.?.ptr, stride, .u8, s);
        for (0..ln.world) |q| {
            if (q == ln.rank) continue;
            try ln.state(f, @intCast(q), ln.bufs.fin_recv.?.ptr + q * stride, false);
        }
    }

    /// Stage q's part of the slot's state (pipe.FinalsLayout's order) between the slot's roles and `part`.
    fn state(ln: *Line, f: *Forward, q: u32, part: u64, pack: bool) !void {
        const r = f.runner;
        const st = ln.s.stage(q);
        const ring = pipe.FinalsLayout.ringBytes(f.cfg);
        var nb: [64]u8 = undefined;
        var off: u64 = 0;
        const Mv = struct {
            fn go(rr: *run.Runner, role: u64, p: u64, cursor: *u64, bytes: u64, out: bool) !void {
                if (out) try copy(rr, p + cursor.*, role, bytes) else try copy(rr, role, p + cursor.*, bytes);
                cursor.* += std.mem.alignForward(u64, bytes, 16);
            }
        };
        for (st.lo..@min(st.hi, st.d)) |i| {
            const L: u32 = @intCast(i);
            try Mv.go(r, try bound(r, try std.fmt.bufPrint(&nb, "s.L{d}.swa.v", .{L})), part, &off, ring[0], pack);
            try Mv.go(r, try bound(r, try std.fmt.bufPrint(&nb, "s.L{d}.swa.s", .{L})), part, &off, ring[1], pack);
            if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2)
                try Mv.go(r, try bound(r, try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})), part, &off, pipe.FinalsLayout.carryBytes(f.cfg), pack);
        }
        if (st.last) for (ced.roles, pipe.FinalsLayout.stashBytes(f.cfg)) |role, bytes| {
            try Mv.go(r, try bound(r, role), part, &off, bytes, pack);
        };
    }

    // -- TF_DSV41_PIPE_TRACE ----------------------------------------------------------------------------------------

    fn tick(ln: *const Line, f: *Forward) ?u64 {
        if (!ln.s.trace) return null;
        f.runner.stream.synchronize() catch {};
        return nowNs();
    }

    fn tock(ln: *const Line, f: *Forward, t0: ?u64, into: *u64) void {
        _ = ln;
        const t = t0 orelse return;
        f.runner.stream.synchronize() catch {};
        into.* += nowNs() - t;
    }
};

/// The active slot (the pool's current one; the line runs on its views alone).
pub fn activeSlot(f: *const Forward) u32 {
    return if (f.kv) |kx| kx.slot.id else 0;
}

/// A stage program's glue (pipe.program's calls): its hand-off, the world-1 exchange (the partial is the sum), the
/// round's Engram rows; every other step is a prefill segment's (forward_prefill.glueStep).
fn stageGlue(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const ln = f.prefill_state.pipe orelse return error.BadGlue;
    const step = c.name["glue.".len..];
    const eql = std.mem.eql;
    if (eql(u8, step, "pipe_in")) return ln.input(r, f.cfg, c);
    if (eql(u8, step, "pipe_out")) return ln.output(r, f.cfg, c);
    if (eql(u8, step, "exchange")) {
        const part = c.args[0].arg.t;
        const into = c.args[1].arg.t;
        if (bytesOf(part) != bytesOf(into)) return error.BadGlue;
        return copy(r, try r.tensorAddr(into), try r.tensorAddr(part), bytesOf(part));
    }
    if (eql(u8, step, "engram_rows")) return ln.engramIn(f, r, c);
    if (std.mem.startsWith(u8, step, "exchange") or eql(u8, step, "embed")) return error.BadGlue;
    return pf.glueStep(ctx, r, c);
}

/// The refusals of a boot with TF_DSV41_PREFILL_PIPE (every rank reads the same environment, so every rank refuses).
pub fn check(f: *const Forward, s: pipe.Settings) !void {
    if (f.prefill_state.mode != .replay) {
        log.warn("TF_DSV41_PREFILL_PIPE: pipelines CED's encoder pass; needs TF_DSV41_PREFILL=replay", .{});
        return error.BadKnob;
    }
    const pool = f.opts.pool orelse {
        log.warn("TF_DSV41_PREFILL_PIPE: needs the paged pool (TF_DSV41_POOL_TOKENS)", .{});
        return error.BadKnob;
    };
    if (pool.split) {
        log.warn("TF_DSV41_PREFILL_PIPE: a stage reads its sources' whole history; needs TF_DSV41_KV_SPLIT=0", .{});
        return error.BadKnob;
    }
    if (f.opts.gm_v2 != .one) {
        // 0 / split: gm_plan's width bits read the forward's K2 table by its bare name (pipe.planned refuses it too);
        // 4,096 rows: x3gm one block a segment
        log.warn("TF_DSV41_PREFILL_PIPE: a stage's MoE needs TF_DSV41_GM_V2=1, not {t}", .{f.opts.gm_v2});
        return error.BadKnob;
    }
    if (f.cfg.vocab % s.world != 0) return error.BadKnob;
}

/// The stage programs' roles into `p` (forward_prefill.plan): pipe.planned's programs.
pub fn planInto(f: *Forward, ln: *const Line, p: *buffers.Plan) !void {
    var arena = std.heap.ArenaAllocator.init(f.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for (try pipe.planned(a, f.cfg, f.widths, ln.o1, ln.s, ln.rank)) |x| {
        var pa = std.heap.ArenaAllocator.init(f.gpa);
        defer pa.deinit();
        try p.add(try pipe.program(pa.allocator(), f.cfg, f.widths, ln.o1, &ln.s, ln.rank, x[0], x[1]));
    }
}

/// The bytes the line adds at boot beyond its stage copies (priced by kv_state.bootCheck as resident): the stage
/// programs' own roles (their namespace and the stage copies' run-time forms) and the line's buffers.
pub fn residentBytes(f: *Forward, ln: *const Line) !u64 {
    // priced before the pool exists: a stage's own roles do not depend on its options (pipe_test.zig)
    return try pipe.stageRoleBytes(f.gpa, f.cfg, f.widths, ln.o1, ln.s, ln.rank) + Line.bufferBytes(f.cfg, ln.s, ln.rank);
}

/// A role the line's glue looks up by name; `Unbound` with the name in the log (run.Runner.tensorAddr's for a call's).
fn bound(r: *const run.Runner, role: []const u8) !u64 {
    return r.addressOf(role) orelse {
        log.warn("role {s} is unbound: the line's glue looks it up by name, the plan holds 0", .{role});
        return error.Unbound;
    };
}

fn copy(r: *run.Runner, dst: u64, src: u64, bytes: u64) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, @intCast(bytes), r.stream.handle), "cuMemcpyDtoDAsync");
}

fn bytesOf(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |x| n *= @intCast(x);
    return n * buffers.dtSize(t.dt);
}

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * std.time.ns_per_s + @as(u64, @intCast(ts.nsec));
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}
