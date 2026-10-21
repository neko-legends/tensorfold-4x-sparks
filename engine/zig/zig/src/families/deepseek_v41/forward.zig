//! The decode forward: a slot's windows over the whole model, block.zig's program run by run.zig, TP exchanges.

const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const calls = @import("block_calls.zig");
const block = @import("block.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const Config = @import("config.zig").Config;
const eh = @import("engram_host.zig");
const graphs = @import("graphs.zig");
const round_graph = @import("round_graph.zig");
const pf = @import("forward_prefill.zig");
const egate = @import("engram_gate.zig");
const greedy_host = @import("greedy.zig");
const hooks = @import("forward_hooks.zig");
const ft = @import("forward_types.zig");
const kvs = @import("cuda_kv.zig");
const er = @import("engram_rows.zig");
pub const kops = @import("dsv41_kernels").ops;
const ph = @import("phases.zig");
const hs = @import("host_stage.zig");
const fwd_sample = @import("forward_sample.zig");
const fwd_follow = @import("forward_follow.zig");
const fwd_glue = @import("forward_glue.zig");
const fwd_engram = @import("forward_engram.zig");
const fwd_graph = @import("forward_graph.zig");

pub const Error = error{ GlueNotBuilt, BadWindow };

/// One slot's position, Engram tail and pending window (what keep() commits).
pub const Slot = struct {
    pos: u64 = 0,
    /// the last max_ngram - 1 committed ids (Engram's lookback; fewer at the sequence start)
    tail: [eh.max_ngram]u32 = undefined,
    tail_len: usize = 0,
    /// the window in flight: its first position, its rows
    pending: ?struct { start: u64, n: u32 } = null,
    /// CED replay (TF_DSV41_PREFILL=replay): the positions and ids of the stash ring "s.ced.*" (ced.zig)
    ced: ft.Stash = .{},
};

pub const Forward = struct {
    gpa: std.mem.Allocator,
    cfg: *const Config,
    widths: *const block.Widths,
    opts: block.Options,
    runner: *run.Runner,
    comm: tp.collective.Collective,
    /// the window's calls by row count, emitted once (arena-owned)
    arena: std.heap.ArenaAllocator,
    programs: std.AutoHashMapUnmanaged(u32, []const calls.Call) = .empty,
    slot: Slot = .{},
    transaction: @import("cuda_transaction.zig").Transaction = .{},
    /// pinned staging of the window's token ids, and their device copy (int64, the embedding step reads them)
    ids_stage: hs.Stage = .{},
    uploads: hs.Stage = .{},
    init_owners: std.ArrayList(@import("construction.zig").Owner) = .empty,
    dev_ids: ?cuda.DeviceBuffer = null,
    /// Engram's hasher (recipe/dsv41_engram_host.py's tables) and the rows' source; none: the synthetic rows
    engram: ?*const eh.Host = null,
    gathered: ?cuda.DeviceBuffer = null,
    rows_stage: [eh.max_layers]hs.Stage = @splat(.{}), // Engram's own pinned staging, one a layer (copies may overlap)
    /// a bf16 [rows, hidden] send buffer for the exchanges whose partial is not bf16 already (embedding, MoE)
    send: ?cuda.DeviceBuffer = null,
    /// the window's token ids (the embedding step reads them; window() copies them into ids_own)
    ids: []const u32 = &.{},
    /// TF_DSV41_IMAGES=native (vision_rows.zig): every rank keeps image positions out of Engram; rank 0's image rows
    images: bool = false,
    /// TF_DSV41_BIAS_VL loaded (every rank): image rows route with gate.bias_vl ("L<i>.moe.bias_vl")
    image_bias: bool = false,
    vision: ?*anyopaque = null, // vision_rows.Rows
    ids_own: std.ArrayList(u32) = .empty,
    /// the backbone layers the forward runs (all of them unless a gate runs a prefix)
    layers: ?[]const u32 = null,
    /// TP: rank 0 leads (sends each forward operation to the others over the plan link); the others `follow`
    link: ?*tp.planlink.PlanLink = null,
    /// a follower's drafter (the leader's lanes ask its pass; every rank runs it)
    drafter: ?Drafter = null,
    /// each block's traced streams (block.Options.trace) as SHA-256 hex by layer, for the last window
    trace: ?*std.AutoArrayHashMapUnmanaged(u32, [64]u8) = null,
    /// the largest row bucket plan() sized the buffers for (a graphed window never has more rows)
    planned_rows: u32 = 0,
    graphed: ?*Graphed = null, // decode windows through CUDA graphs (TF_DSV41_GRAPHS=1); null until the knob is read
    graphs_read: bool = false,
    /// the graphs' default when TF_DSV41_GRAPHS is unset (model.zig: on for the served model; the gates' drivers: off)
    graphs_default: bool = false,
    /// Engram's local shards (TF_DSV41_ENGRAM_DIR, engram_rows.zig), opened on first use; unset: synthetic index rows
    engram_rows: ?*er.Rows = null,
    engram_opened: bool = false,
    /// Engram read-ahead at the DSpark pass (TF_DSV41_ENGRAM_PREFETCH=0: rows read when the window needs them)
    engram_prefetch: bool = true,
    gate: ?*anyopaque = null, // TF_DSV41_ENGRAM_GATE: opened with the table (engram_gate.Gate)
    gated: bool = false, // the window being issued armed the gate: its engram_rows steps launch the gate kernel
    /// TF_DSV41_ROUND_GRAPH's `gate`: recorded after each gated window, the gate's arm waits for it (not the stream)
    gated_end: ?cuda.Event = null,
    /// TF_DSV41_BRANCHES / TF_DSV41_MHC_DEFER (branches.zig): the runner's side streams, owned here
    branches: ?*anyopaque = null, // branches.Branches
    /// TF_DSV41_PHASES (phases.zig): the window path's timers
    phases: ?*ph.Phases = null,
    /// greedy's pinned logits / pairs staging and device pairs, kept across windows
    pick_host: ?cuda.HostBuffer = null,
    pick_dev: ?cuda.DeviceBuffer = null,
    greedy_gpu: ?bool = null, // TF_DSV41_GREEDY_GPU=1: each rank's best column on GPU, pairs copied back; null: unread
    greedy_dev: ?cuda.DeviceBuffer = null,
    /// the last GPU pick on the device (`pickDevice`), null when it was merged on the host
    pick_last: ?PickDevice = null,
    after_pick: ?AfterPick = null, // TF_DSV41_DRAFT_GRAPHS + TF_DSV41_SPEC_DRAFT: enqueued after the device pick
    after_sample: ?AfterSample = null,
    pick_done: ?cuda.Event = null, // the pick's copy-back event the host waits on
    /// the prefill's state (forward_prefill.State, forward_prefill.st), made at engine open
    prefill_state: pf.State = .{},
    /// rows of "w.logits" the last prefill left (its last segment's, or its tail's): greedy takes out[0..last_rows]
    last_rows: u32 = 0,
    /// rank 0 inside a prefill: its windows and keeps are the followers' own (they got the whole prompt), not sent
    quiet: bool = false,
    /// the KV families in the paged pool (null: the slot's contiguous buffers); opts.pool is its emitter view
    kv: ?*kvs.Kv = null,
    /// the session store (sessions_gpu.zig): the ids each keep commits, and the operations a follower runs
    sess: ?Sess = null,
    /// a workstream's plan-link operations (slots.zig 16-19), run by a follower in the leader's order
    ext: ?Ext = null,
    /// rank 0's clean stop of every rank (model.zig: the TP session's announceStop), run by `stop`
    on_stop: ?struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) void } = null,
    pf_switch: ?PfSwitch = null, // several live slots: a multi-segment prefill run's slot switch; null: one slot
    pf_stash: ?PfStash = null, // the CED stash ring made the current slot's before a stash step (swapped lazily)
    /// the multi-segment prefill run in flight (forward_prefill.promptMulti; its glue reads it), null outside one
    multi: ?*ft.Multi = null,
    /// the keyed sampler (sampling_gpu.zig): a follower runs rank 0's sampled choices too (their gathers pair up)
    sampler: ?Sampler = null,
    grammar: ?Grammar = null, // TF_DSV41_GRAMMAR=1: grammar masks applied to "w.logits" before a pick
    calib_marks: std.ArrayList(u64) = .empty, // TF_DSV41_CALIB=measure: a follower's clock at each op_calib_mark

    pub fn init(gpa: std.mem.Allocator, cfg: *const Config, widths: *const block.Widths, opts: block.Options, runner: *run.Runner, comm: tp.collective.Collective) Forward {
        // rank and world ride the options (model.zig sets them from the same communicator): the emitters' uneven
        // shards (block.Options.expertWidth / vocabShard) are that rank's, and a test may pass no communicator
        return .{ .gpa = gpa, .cfg = cfg, .widths = widths, .opts = opts, .runner = runner, .comm = comm, .arena = std.heap.ArenaAllocator.init(gpa) };
    }

    /// This rank's head columns: [lo, lo + len) of the vocabulary on whole 128-blocks (shard.zig; uneven at TP=4), and
    /// the widest / narrowest rank's len for gathers that must send the same bytes from every rank.
    pub fn vocabShard(f: *const Forward) block.shard.Shard {
        return block.shard.checked(f.cfg.vocab, @intCast(f.comm.world()), @intCast(f.comm.rank()));
    }

    pub fn upload(f: *Forward, destination: u64, bytes: []const u8) !void {
        const owned = try f.uploads.take(f.runner.d, bytes.len);
        @memcpy(owned, bytes);
        try f.uploads.copyTo(.{ .d = f.runner.d, .ptr = destination, .len = bytes.len }, 0, f.runner.stream);
        try f.uploads.wait();
    }
    pub fn drain(f: *Forward) !void {
        if (f.init_owners.items.len > 0) try f.runner.d.synchronize();
        try f.runner.stream.synchronize();
        try f.ids_stage.wait();
        try f.uploads.wait();
        for (&f.rows_stage) |*x| try x.wait();
        if (f.graphed) |g| try g.drain();
    }
    pub fn close(f: *Forward) !void {
        try f.drain();
        f.deinitAfterDrain();
    }
    /// The fork's plain teardown for the debug drivers (m1/m2a/m2b): a failed drain is the process exit's problem.
    pub fn deinit(f: *Forward) void {
        f.close() catch {};
    }
    pub fn deinitAfterDrain(f: *Forward) void {
        @import("construction.zig").deinit(f);
        f.ids_stage.close() catch unreachable;
        f.uploads.close() catch unreachable;
        for (&f.rows_stage) |*x| x.close() catch unreachable;
        f.transaction.deinit(f.gpa);
        if (f.graphed) |g| {
            g.deinitAfterDrain();
            f.gpa.destroy(g);
        }
        if (f.dev_ids) |*b| b.free();
        if (f.send) |*b| b.free();
        if (f.gathered) |*b| b.free();
        if (hooks.set.prefill != null) hooks.set.prefill.?.deinit(f);
        f.calib_marks.deinit(f.gpa);
        if (f.gate) |x| hooks.set.gate.?.close(x);
        if (f.engram_rows) |x| x.close();
        if (f.branches != null) hooks.set.branches.?.close(f);
        if (f.pick_host) |*b| b.free();
        if (f.pick_dev) |*b| b.free();
        if (f.greedy_dev) |*b| b.free();
        if (f.pick_done) |*e| e.deinit();
        if (f.gated_end) |*e| e.deinit();
        f.programs.deinit(f.gpa);
        f.ids_own.deinit(f.gpa);
        f.arena.deinit();
    }

    /// The backbone's layers, then the head.
    pub fn backbone(f: *Forward, a: std.mem.Allocator) ![]const u32 {
        if (f.layers) |ls| return ls;
        const ls = try a.alloc(u32, f.cfg.layers);
        for (ls, 0..) |*l, i| l.* = @intCast(i);
        return ls;
    }

    /// The buffer plan of every row bucket the forward runs (bind the runner with it once, before the first window).
    pub fn plan(f: *Forward, buckets: []const u32) !buffers.Plan {
        var p: buffers.Plan = .{ .a = f.arena.allocator() };
        for (buckets) |n| try p.add(try f.program(n));
        for (buckets) |n| f.planned_rows = @max(f.planned_rows, n);
        return p;
    }

    /// The calls of an `n`-row window at the slot's last position (the largest every role gets; plans size from it).
    pub fn program(f: *Forward, n: u32) ![]const calls.Call {
        if (f.programs.get(n)) |cs| return cs;
        const a = f.arena.allocator();
        // the emitter's positions only size the indexer's scores at the window's end; the largest the slot reaches
        const cs = try block.emit(a, f.cfg, f.widths, f.opts, try f.backbone(a), n, f.opts.limit - n, true);
        try f.programs.put(f.gpa, n, cs);
        return cs;
    }

    /// Adds the prefill program's roles (segments of up to `rows` rows) to `p` before bind; shared roles: the larger.
    pub fn planPrefill(f: *Forward, p: *buffers.Plan, rows: u32) !void {
        return (try hooks.prefill()).plan(f, p, rows);
    }

    /// The next prompt piece's ids for the Engram read-ahead (TF_DSV41_PREFETCH_AHEAD), every rank; I/O only.
    pub fn prefillAhead(f: *Forward, next: []const u32) !void {
        const ahead = (try hooks.prefill()).ahead(f) orelse return;
        if (next.len == 0) return;
        if (f.leads()) {
            const msg = try f.gpa.alloc(i64, 1 + next.len);
            defer f.gpa.free(msg);
            msg[0] = op_ahead;
            for (next, msg[1..]) |t, *y| y.* = t;
            try f.link.?.send(msg);
        }
        ahead.clearRetainingCapacity();
        try ahead.appendSlice(f.gpa, next);
    }

    /// The prompt `ids` from the slot's position: segments kept whole, a tail of at most 16 rows as decode windows.
    pub fn prefill(f: *Forward, ids: []const u32) !void {
        if (f.slot.pending != null) return error.BadWindow;
        if (ids.len == 0) return;
        f.gated = false; // prompt segments read their rows on the old path
        if (f.leads()) {
            // [op, total, offset, ids...] frames (planlink takes up to 64M words a frame; a prompt is a few)
            const frame: usize = 1 << 20;
            const msg = try f.gpa.alloc(i64, 3 + @min(frame, ids.len));
            defer f.gpa.free(msg);
            var at: usize = 0;
            while (at < ids.len) : (at += frame) {
                const m = @min(frame, ids.len - at);
                msg[0] = op_prefill;
                msg[1] = @intCast(ids.len);
                msg[2] = @intCast(at);
                for (ids[at .. at + m], msg[3 .. 3 + m]) |t, *y| y.* = t;
                try f.link.?.send(msg[0 .. 3 + m]);
            }
        }
        f.quiet = true;
        defer f.quiet = false;
        return (try hooks.prefill()).prompt(f, ids);
    }

    /// One decode window: `ids` at the slot's position. This rank's logits stay in the runner's "w.logits".
    pub fn window(f: *Forward, ids_in: []const u32) !void {
        // the forward's own copy: keep() reads the ids (Engram's tail) after the caller's slice may be gone
        if (ids_in.ptr != f.ids_own.items.ptr) {
            f.ids_own.clearRetainingCapacity();
            try f.ids_own.appendSlice(f.gpa, ids_in);
        }
        const ids = f.ids_own.items;
        if (ids.len == 0 or f.slot.pending != null) return error.BadWindow;
        // no embed of a window fills image rows (graphed: never; eager: no keep or bias), so image ids fail here
        if (hooks.set.vision) |v| try v.checkWindow(ids);
        // the pool's pages for the window's positions, their table entries up before the launches (every rank alike)
        if (f.kv) |kx| try kx.reserve(f.slot.pos + ids.len);
        if (f.leads()) {
            var msg: [2 + 128]i64 = undefined;
            if (ids.len > 128) return error.BadWindow;
            msg[0] = op_window;
            msg[1] = @intCast(ids.len);
            for (ids, 0..) |t, i| msg[2 + i] = t;
            try f.link.?.send(msg[0 .. 2 + ids.len]);
        }
        // TF_DSV41_PREFILL_PIPE: a slot's open line drained before its window reads the slot (every rank, in the op);
        // the verify tail comes after the CED finish, which drains it, so today this finds no line
        if (f.prefill_state.pipe_settle) |settle| try settle(f);
        const tf = ph.start(f.phases, .forward);
        defer tf.stop();
        f.runner.rows = @intCast(ids.len); // l2pf: wider windows skip the prefetch sites
        try f.transaction.begin(f, @intCast(ids.len));
        if (try f.graphedWindow(ids)) return f.gatedEnd();
        // the gate (its worker reads every layer at once) or every layer's reads in flight while the calls are emitted
        f.gated = try f.armEngram(&.{.{ .ids = ids, .tail = f.slot.tail[0..f.slot.tail_len] }}, @intCast(ids.len));
        if (!f.gated) f.issueEngram(ids, f.slot.tail[0..f.slot.tail_len]);
        // the window's calls at its position (the indexer's scores and top-k grow with the cache: NK = end / ratio)
        var wa = std.heap.ArenaAllocator.init(f.gpa);
        defer wa.deinit();
        const cs = try block.emit(wa.allocator(), f.cfg, f.widths, f.opts, try f.backbone(wa.allocator()), @intCast(ids.len), @intCast(f.slot.pos), true);
        f.ids = ids;
        f.slot.pending = .{ .start = f.slot.pos, .n = @intCast(ids.len) };
        f.runner.glue = .{ .ctx = f, .run = glueFn };
        try f.runner.timedWindow(cs);
        try f.gatedEnd();
    }

    /// Greedy choices of the last window's rows (Python's forward.greedy): each rank's max a row (the lowest column on
    /// ties; -0.0 == +0.0), the float64 (value, global id) pairs all-gathered, the higher value (the lower rank on ties).
    pub fn greedy(f: *Forward, out: []u32) !void {
        if (f.leads()) try f.link.?.send(&.{ op_greedy, @intCast(out.len) });
        const tc = ph.start(f.phases, .candidates);
        defer tc.stop();
        const r = f.runner;
        const n = out.len;
        const W: usize = f.comm.world();
        // this rank's head columns: whole 128-blocks (uneven at TP=4); the pairs carry global ids, one a row a rank
        const vs = f.vocabShard();
        const V: usize = vs.len;
        // pinned: the logits [n, V], then the pairs [2n] and every rank's [W][2n] (reused, no allocation a window)
        const lb = std.mem.alignForward(usize, 4 * n * V, 8);
        const need = lb + 16 * n * (W + 1);
        if (f.pick_host == null or f.pick_host.?.bytes.len < need) {
            if (f.pick_host) |*b| b.free();
            f.pick_host = try cuda.HostBuffer.alloc(r.d, @max(need, 1 << 20));
        }
        if (f.pick_dev == null or f.pick_dev.?.len < 16 * n * (W + 1)) {
            if (f.pick_dev) |*b| b.free();
            f.pick_dev = try cuda.DeviceBuffer.alloc(r.d, @max(16 * n * (W + 1), 4096));
        }
        const hb = f.pick_host.?.bytes;
        const host: []f32 = @alignCast(std.mem.bytesAsSlice(f32, hb[0 .. 4 * n * V]));
        const pairs: []f64 = @alignCast(std.mem.bytesAsSlice(f64, hb[lb..need]));
        const dev = f.pick_dev.?;
        const logits = r.addressOf("w.logits") orelse return error.Unbound;
        if (f.grammar) |g| try g.apply(g.ptr, 0, @intCast(n));
        if (f.greedy_gpu == null) f.greedy_gpu = if (std.c.getenv("TF_DSV41_GREEDY_GPU")) |v| std.mem.eql(u8, std.mem.span(v), "1") else false;
        if (f.greedy_gpu.? and V <= 1 << 16) {
            // Python's served pick (decode.cand_gather, count 1): topk_keys' best column a row (value descending, then the
            // lower column), [n] fp32 values and [n] int64 columns back instead of the [n, V] logits
            const vals_dev = try f.greedyScratch(r, n);
            const cols_dev = vals_dev + std.mem.alignForward(u64, 4 * n, 16);
            const k = r.kernels.others(r.stream);
            try k.topKeys(logits, V, n, V, 1, vals_dev, cols_dev);
            f.pick_last = null;
            if (k.f.pick_pack != null and k.f.pick_merge != null and n <= kops.glue.pick_max_rows) {
                // the pairs, their exchange and the merge on the device; one copy back (glue.cu pick_*)
                const ids_dev = cols_dev + 8 * n;
                const out_dev = ids_dev + 8 * n;
                try k.pickPack(vals_dev, cols_dev, n, vs.lo, dev.ptr);
                try f.comm.allGather(dev.ptr, dev.ptr + 16 * n, 2 * n, .f64, r.stream.handle);
                const chain = if (f.slot.pending) |p| p.n == n and f.ids.len >= n else false;
                const start: i64 = if (f.slot.pending) |p| @intCast(p.start) else 0;
                const ih: []i64 = @alignCast(std.mem.bytesAsSlice(i64, hb[0 .. 8 * n]));
                if (chain) {
                    for (ih, f.ids[0..n]) |*x, t| x.* = t;
                    try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(ids_dev, ih.ptr, 8 * n, r.stream.handle), "cuMemcpyHtoDAsync");
                }
                try k.pickMerge(dev.ptr + 16 * n, W, n, if (chain) ids_dev else 0, start, out_dev);
                const oh: []i64 = @alignCast(std.mem.bytesAsSlice(i64, hb[std.mem.alignForward(usize, 8 * n, 16)..][0 .. 8 * (n + 3)]));
                try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(oh.ptr, out_dev, 8 * (n + 3), r.stream.handle), "cuMemcpyDtoHAsync");
                f.pick_last = .{ .out = out_dev, .n = @intCast(n), .chain = chain, .start = @intCast(start), .host = oh.ptr };
                if (f.after_pick) |h| {
                    // work that starts from the device pick (the speculative DSpark pass, spec.py's order) goes on the
                    // stream after the copy back; the host waits for the copy alone
                    if (f.pick_done == null) f.pick_done = try cuda.Event.init(r.d, false);
                    try f.pick_done.?.record(r.stream);
                    try h.run(h.ctx, f.pick_last.?);
                    try f.pick_done.?.synchronize();
                } else try r.stream.synchronize();
                for (out, oh[0..n]) |*o, x| o.* = @intCast(x);
                return;
            }
            const vh: []f32 = @alignCast(std.mem.bytesAsSlice(f32, hb[0 .. 4 * n]));
            const ch: []i64 = @alignCast(std.mem.bytesAsSlice(i64, hb[std.mem.alignForward(usize, 4 * n, 16)..][0 .. 8 * n]));
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(vh.ptr, vals_dev, 4 * n, r.stream.handle), "cuMemcpyDtoHAsync");
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(ch.ptr, cols_dev, 8 * n, r.stream.handle), "cuMemcpyDtoHAsync");
            try r.stream.synchronize();
            for (0..n) |row| {
                pairs[2 * row] = vh[row];
                pairs[2 * row + 1] = @floatFromInt(vs.id(@intCast(ch[row])));
            }
        } else {
            try r.d.check(r.d.api.cuMemcpyDtoHAsync_v2(host.ptr, logits, 4 * n * V, r.stream.handle), "cuMemcpyDtoHAsync");
            try r.stream.synchronize();
            for (0..n) |row| {
                const vals = host[row * V ..][0..V];
                const best = greedy_host.argmax(vals);
                pairs[2 * row] = vals[best];
                pairs[2 * row + 1] = @floatFromInt(vs.id(best));
            }
        }
        try dev.uploadAsync(0, std.mem.sliceAsBytes(pairs[0 .. 2 * n]), r.stream.handle);
        try f.comm.allGather(dev.ptr, dev.ptr + 16 * n, 2 * n, .f64, r.stream.handle);
        const all = pairs[2 * n ..];
        try dev.downloadAsync(16 * n, std.mem.sliceAsBytes(all), r.stream.handle);
        try r.stream.synchronize();
        for (out, 0..) |*o, row| {
            var bk: usize = 0;
            for (1..W) |k| if (all[2 * (k * n + row)] > all[2 * (bk * n + row)]) {
                bk = k;
            };
            o.* = @intFromFloat(all[2 * (bk * n + row) + 1]);
        }
    }

    /// The last window's pick on the device: `out` int64 [n + 3] = picks, then (chain) accepted, bonus, next position.
    pub const PickDevice = struct { out: u64, n: u32, chain: bool, start: u64 = 0, host: ?[*]const i64 = null };

    /// Work enqueued from the device pick, on every rank, after the pick's copy back and before the host waits for it.
    pub const AfterPick = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, pick: PickDevice) anyerror!void };

    /// Work enqueued from sampled device choices (rows [row0, row0 + n) into `chosen` int64 [n]), every rank.
    pub const AfterSample = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, row0: u32, n: u32, chosen: u64) anyerror!void };

    pub fn pickDevice(f: *const Forward) ?PickDevice {
        return f.pick_last;
    }

    /// Commits the pending window's first row + `accepted` drafts.
    pub fn keep(f: *Forward, accepted: u32) !void {
        const p = f.slot.pending orelse return error.BadWindow;
        if (accepted >= p.n) return error.BadWindow;
        if (f.sess) |x| if (x.reserve) |reserve| try reserve(x.ptr, f.ids[0 .. accepted + 1]);
        try f.engramCheck(); // a gated window whose rows failed: its picks are not committed
        if (f.leads()) try f.link.?.send(&.{ op_keep, accepted });
        // the ratio-2 KV sources' carry: fp32 of the accepted row of the window's compressor projection
        const r = f.runner;
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(f.gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
            const w = 2 * @as(usize, f.cfg.head_dim);
            const src = r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.comp", .{L})) orelse return error.Unbound;
            const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
            try r.kernels.others(r.stream).carry(src, w, 0, accepted, w, dst);
        };
        try f.transaction.keep(accepted + 1);
        // Engram's tail: (tail + the committed ids)[-(max_ngram - 1):]
        if (f.engram) |h| {
            const keep_n = h.max_ngram - 1;
            var all: [2 * eh.max_ngram + 128]u32 = undefined;
            var m: usize = 0;
            for (f.slot.tail[0..f.slot.tail_len]) |t| {
                all[m] = t;
                m += 1;
            }
            const take = @min(accepted + 1, all.len - m);
            for (f.ids[0..take]) |t| {
                all[m] = t;
                m += 1;
            }
            const k = @min(m, keep_n);
            @memcpy(f.slot.tail[0..k], all[m - k .. m]);
            f.slot.tail_len = k;
        }
        f.slot.pos = p.start + accepted + 1;
        f.slot.pending = null;
        if (f.sess) |x| try x.commit(x.ptr, f.ids[0 .. accepted + 1]);
    }

    const op_window = fwd_follow.op_window;
    const op_greedy = fwd_follow.op_greedy;
    const op_keep = fwd_follow.op_keep;
    const op_stop = fwd_follow.op_stop;
    const op_ds_ingest = fwd_follow.op_ds_ingest;
    const op_ds_propose = fwd_follow.op_ds_propose;
    const op_prefill = fwd_follow.op_prefill;
    const op_ahead = fwd_follow.op_ahead;
    const op_sess_save = fwd_follow.op_sess_save;
    const op_sess_restore = fwd_follow.op_sess_restore;
    const op_release = fwd_follow.op_release;
    const op_ds_reset = fwd_follow.op_ds_reset;
    pub const op_sample = fwd_follow.op_sample;
    pub const op_sample_many = fwd_follow.op_sample_many;

    /// The sampler's side of a follower (sampling_gpu.zig): rank 0's op_sample message, run on this rank.
    pub const Sampler = struct {
        ptr: *anyopaque,
        follow: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void,
    };

    pub const sendSample = fwd_follow.sendSample;
    pub const dsResetOp = fwd_follow.dsResetOp;

    const op_tree_drop = fwd_follow.op_tree_drop;
    const op_ced_finish = fwd_follow.op_ced_finish;
    const op_sess_prompt = fwd_follow.op_sess_prompt;
    const op_sess_admit = fwd_follow.op_sess_admit;
    const op_pf_multi = fwd_follow.op_pf_multi;
    const op_ced_multi = fwd_follow.op_ced_multi;
    pub const replayMulti = fwd_follow.replayMulti;

    /// A slot's views made current on this rank (slots.zig SlotSet.view), for a multi-segment run's segments.
    pub const PfSwitch = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque, slot: u32) anyerror!void };
    pub const PfStash = struct { ctx: *anyopaque, run: *const fn (ctx: *anyopaque) anyerror!void };

    pub const prefillMulti = fwd_follow.prefillMulti;
    pub const finishPrompt = fwd_follow.finishPrompt;

    pub const op_grammar = fwd_follow.op_grammar;

    /// Structured output's side of the forward: `apply` before a pick reads "w.logits", `follow` an op_grammar message.
    pub const Grammar = struct {
        ptr: *anyopaque,
        follow: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void,
        apply: *const fn (ptr: *anyopaque, row0: u32, n: u32) anyerror!void,
    };

    /// A workstream's follower side: rank 0's message of an operation number this file does not know.
    pub const Ext = struct { ptr: *anyopaque, run: *const fn (ptr: *anyopaque, msg: []const i64) anyerror!void };

    /// The session store's side: `commit` hears every committed id; `save` / `restore` run in the leader's order.
    pub const Sess = struct {
        ptr: *anyopaque,
        commit: *const fn (ptr: *anyopaque, ids: []const u32) anyerror!void,
        reserve: ?*const fn (ptr: *anyopaque, ids: []const u32) anyerror!void = null,
        save: *const fn (ptr: *anyopaque) anyerror!void,
        /// `ids`: an NVMe entry's ids (the leader's prompt prefix: its tokens are not indexed)
        restore: *const fn (ptr: *anyopaque, id: u32, ids: []const i64) anyerror!void,
        /// CED replay's prompt snapshot: an entry at the slot's position, the slot keeps going (sessions_gpu.zig)
        prompt: *const fn (ptr: *anyopaque) anyerror!void,
        /// the slot empty: its pages back, the committed ids cleared
        reset: *const fn (ptr: *anyopaque) anyerror!void,
        /// a request's pool admission (op_sess_admit's message): the leader's spills evicted, its pages reserved
        admit: ?*const fn (ptr: *anyopaque, msg: []const i64) anyerror!void = null,
    };

    /// 64-67: dsq-zig-calib (TF_DSV41_CALIB=measure, calib_gpu.zig). 64: [op] every rank synchronizes its device and
    /// reads its clock (a timed run's start or end); 65: [op, n] every rank's n samples (the marks paired up, ns)
    /// all-gathered, so rank 0 takes the slower rank's statistic an entry (Python calib's gather_max)
    pub const op_calib_mark: i64 = 64;
    pub const op_calib_gather: i64 = 65;

    /// A timed run's start or end on every rank (Python's sync(); perf_counter()): the leader sends the mark, then
    /// this rank's device (every stream) is synchronized and its monotonic clock read.
    pub fn calibMark(f: *Forward) !u64 {
        if (f.leads()) try f.link.?.send(&.{op_calib_mark});
        const d = f.runner.d;
        try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
        return ph.nowNs();
    }

    /// Every rank's samples (`mine`, ns, one a timed run) into `all` [world][mine.len], rank-major; the leader sends
    /// the count first. A follower whose marks do not pair up into as many samples sends -1s (the leader refuses).
    pub fn calibGather(f: *Forward, mine: []const i64, all: []i64) !void {
        const w: usize = f.comm.world();
        if (all.len != w * mine.len) return error.BadWindow;
        if (f.leads()) try f.link.?.send(&.{ op_calib_gather, @intCast(mine.len) });
        if (w == 1 or mine.len == 0) return @memcpy(all[0..mine.len], mine);
        const d = f.runner.d;
        const bytes = 8 * mine.len;
        var dev = try cuda.DeviceBuffer.alloc(d, bytes * (w + 1));
        defer dev.free();
        var host = try cuda.HostBuffer.alloc(d, bytes * (w + 1));
        defer host.free();
        const h = host.slice(i64);
        @memcpy(h[0..mine.len], mine);
        const st = f.runner.stream;
        try dev.uploadAsync(0, std.mem.sliceAsBytes(h[0..mine.len]), st.handle);
        try f.comm.allGather(dev.ptr, dev.ptr + bytes, mine.len, .i64, st.handle);
        try dev.downloadAsync(bytes, std.mem.sliceAsBytes(h[mine.len .. mine.len * (w + 1)]), st.handle);
        try st.synchronize();
        @memcpy(all, h[mine.len .. mine.len * (w + 1)]);
    }

    /// A follower's op_calib_gather: its marks paired up (end - start) as its samples, gathered with the leader's.
    fn followCalib(f: *Forward, n: usize) !void {
        defer f.calib_marks.clearRetainingCapacity();
        const mine = try f.gpa.alloc(i64, n);
        defer f.gpa.free(mine);
        const marks = f.calib_marks.items;
        if (marks.len == 2 * n) {
            for (mine, 0..) |*x, k| x.* = @intCast(marks[2 * k + 1] - marks[2 * k]);
        } else {
            std.log.scoped(.dsv41).err("calibration: {d} marks on this rank for {d} timed runs; sending none", .{ marks.len, n });
            @memset(mine, -1);
        }
        const all = try f.gpa.alloc(i64, n * f.comm.world());
        defer f.gpa.free(all);
        try f.calibGather(mine, all);
    }

    /// Drops the pending window on every rank without committing a row (draft/branches.zig: a tree's chain whose branch
    /// lost; the next chain rewrites its positions before any read, as a rejected draft's).
    pub fn drop(f: *Forward) !void {
        if (f.slot.pending == null) return error.BadWindow;
        if (f.leads()) try f.link.?.send(&.{op_tree_drop});
        f.slot.pending = null;
    }

    /// Frees the slot for the next stream on every rank (the Spark window's fix: a follower kept its position, so the
    /// next request ran rank 1 from a stale slot): position 0, no Engram tail, no pending window, the pool's pages back.
    pub fn release(f: *Forward) !void {
        if (f.leads()) try f.link.?.send(&.{op_release});
        if (f.prefill_state.pipe_release) |h| h(f); // TF_DSV41_PREFILL_PIPE: the slot's segments in the line are dropped
        if (f.sess) |x| return x.reset(x.ptr);
        if (f.kv) |k| try k.release();
        f.slot = .{};
    }

    /// Rank 0: a session operation to the followers (before running it).
    pub fn sendSess(f: *Forward, op: enum { save, restore, prompt }, id: u32) !void {
        if (f.leads()) try f.link.?.send(&.{ switch (op) {
            .save => op_sess_save,
            .restore => op_sess_restore,
            .prompt => op_sess_prompt,
        }, id });
    }

    /// Rank 0: a request's admission to the followers (before running it): its pages and the entries to evict first.
    pub fn sendSessAdmit(f: *Forward, need: u32, spills: []const u32) !void {
        if (!f.leads()) return;
        var msg: std.ArrayList(i64) = .empty;
        defer msg.deinit(f.gpa);
        try msg.appendSlice(f.gpa, &.{ op_sess_admit, need, @intCast(spills.len) });
        for (spills) |id| try msg.append(f.gpa, id);
        try f.link.?.send(msg.items);
    }

    /// The drafter's operations a follower runs (dspark_gpu.zig): its collectives pair up with the leader's.
    pub const Drafter = struct {
        ptr: *anyopaque,
        ingest: *const fn (ptr: *anyopaque, start: u64, row0: u32, n: u32, avail: u32) anyerror!void,
        propose: *const fn (ptr: *anyopaque, anchor: u32, start: u64) anyerror!void,
        /// the slot's draft context restarts (a new request): every rank's, or their candidates differ
        reset: *const fn (ptr: *anyopaque) void,
    };


    /// Rank 0: a drafter operation to the followers (before running it).
    pub fn sendDrafter(f: *Forward, ints: []const i64) !void {
        if (f.leads()) try f.link.?.send(ints);
        // R3: the pass's anchor is the next window's row 0: its Engram rows are read while the pass runs
        if (ints.len == 3 and ints[0] == op_ds_propose) f.prefetchEngram(@intCast(ints[2]), &.{@intCast(ints[1])});
    }

    pub fn dsIngestOp(start: u64, row0: u32, n: u32, avail: u32) [5]i64 {
        return .{ op_ds_ingest, @intCast(start), row0, n, avail };
    }

    pub fn dsProposeOp(anchor: u32, start: u64) [3]i64 {
        return .{ op_ds_propose, anchor, @intCast(start) };
    }

    pub fn leads(f: *const Forward) bool {
        return f.link != null and !f.quiet and f.comm.rank() == 0;
    }

    /// Rank 0: the followers' loops end (after the last operation they were sent).
    pub fn stop(f: *Forward) !void {
        if (f.leads()) try f.link.?.send(&.{op_stop});
        // every rank's clean end, before rank 0 closes its sockets (tp.Session.announceStop: no fail-fast exit 70)
        if (f.leads()) if (f.on_stop) |h| h.run(h.ctx);
    }

    /// Ranks > 0: run the leader's operations as they arrive, in its order (every collective then pairs up), until
    /// `stop`. The window's token ids come with it; greedy's choice is computed here too (its all-gather needs every
    /// rank) and dropped. A prefill's prompt comes in frames; the last one starts it.
    pub fn follow(f: *Forward) !void {
        var buf: std.ArrayList(i64) = .empty;
        defer buf.deinit(f.gpa);
        var ids: std.ArrayList(u32) = .empty;
        defer ids.deinit(f.gpa);
        var prompt: std.ArrayList(u32) = .empty;
        defer prompt.deinit(f.gpa);
        // greedy's rows: a window's, or a prefill's last segment (up to prefill_rows)
        var picks: std.ArrayList(u32) = .empty;
        defer picks.deinit(f.gpa);
        while (true) {
            const msg = try f.link.?.recvGrow(f.gpa, &buf);
            if (msg.len == 0) return error.BadPlan;
            switch (msg[0]) {
                op_window => {
                    const n: usize = @intCast(msg[1]);
                    ids.clearRetainingCapacity();
                    for (msg[2 .. 2 + n]) |t| try ids.append(f.gpa, @intCast(t));
                    try f.window(ids.items);
                },
                op_greedy => {
                    try picks.resize(f.gpa, @intCast(msg[1]));
                    try f.greedy(picks.items);
                },
                op_prefill => {
                    const total: usize = @intCast(msg[1]);
                    const at: usize = @intCast(msg[2]);
                    if (at == 0) prompt.clearRetainingCapacity();
                    if (at != prompt.items.len) return error.BadPlan;
                    for (msg[3..]) |t| try prompt.append(f.gpa, @intCast(t));
                    if (prompt.items.len > total) return error.BadPlan;
                    if (prompt.items.len == total) try f.prefill(prompt.items);
                },
                op_ahead => {
                    var nx: std.ArrayList(u32) = .empty;
                    defer nx.deinit(f.gpa);
                    for (msg[1..]) |t| try nx.append(f.gpa, @intCast(t));
                    try f.prefillAhead(nx.items);
                },
                op_keep => try f.keep(@intCast(msg[1])),
                op_release => try f.release(),
                op_tree_drop => try f.drop(),
                op_calib_mark => {
                    const d = f.runner.d;
                    try d.check(d.api.cuCtxSynchronize(), "cuCtxSynchronize");
                    try f.calib_marks.append(f.gpa, ph.nowNs());
                },
                op_calib_gather => {
                    if (msg.len != 2 or msg[1] < 0) return error.BadPlan;
                    try f.followCalib(@intCast(msg[1]));
                },
                op_stop => return,
                op_sess_save => {
                    const x = f.sess orelse return error.BadPlan;
                    x.save(x.ptr) catch |err| return f.followSess("save", err);
                },
                op_sess_restore => {
                    const x = f.sess orelse return error.BadPlan;
                    const lid: u32 = @intCast(msg[1]);
                    const lids: ?[]const i64 = if (msg.len > 2) msg[2..] else null;
                    x.restore(x.ptr, lid, lids orelse &.{}) catch |err| return f.followSess("restore", err);
                },
                op_sess_prompt => {
                    const x = f.sess orelse return error.BadPlan;
                    x.prompt(x.ptr) catch |err| return f.followSess("prompt", err);
                },
                op_sess_admit => {
                    const x = f.sess orelse return error.BadPlan;
                    const run_ = x.admit orelse return error.BadPlan;
                    run_(x.ptr, msg) catch |err| return f.followSess("admit", err);
                },
                op_pf_multi => {
                    const ns: usize = @intCast(msg[1]);
                    if (ns == 0 or ns > 16 or msg.len < 2 + 3 * ns) return error.BadPlan;
                    var segs: [16]pf.MultiSeg = undefined;
                    prompt.clearRetainingCapacity();
                    for (msg[2 + 3 * ns ..]) |t| try prompt.append(f.gpa, @intCast(t));
                    var at: usize = 0;
                    for (segs[0..ns], 0..) |*sg, j| {
                        const n: usize = @intCast(msg[4 + 3 * j]);
                        if (at + n > prompt.items.len) return error.BadPlan;
                        sg.* = .{ .slot = @intCast(msg[2 + 3 * j]), .start = @intCast(msg[3 + 3 * j]), .ids = prompt.items[at .. at + n] };
                        at += n;
                    }
                    if (at != prompt.items.len) return error.BadPlan;
                    try f.prefillMulti(segs[0..ns]);
                },
                op_ced_multi => {
                    const ns: usize = @intCast(msg[1]);
                    if (ns < 2 or ns > 16 or msg.len != 2 + ns) return error.BadPlan;
                    var slots: [16]u32 = undefined;
                    for (msg[2..], slots[0..ns]) |x, *y| y.* = @intCast(x);
                    var out: [16]pf.Replayed = undefined;
                    try f.replayMulti(slots[0..ns], &out);
                },
                op_ced_finish => {
                    // the leader's slot ends where this one does, and both run replay (the same TF_DSV41_PREFILL)
                    if (f.prefill_state.mode != .replay or msg.len < 2 or msg[1] != f.slot.pos) return error.BadPlan;
                    try pf.finish(f);
                },
                op_ds_ingest => {
                    const d = f.drafter orelse return error.BadPlan;
                    try d.ingest(d.ptr, @intCast(msg[1]), @intCast(msg[2]), @intCast(msg[3]), @intCast(msg[4]));
                },
                op_ds_reset => {
                    const d = f.drafter orelse return error.BadPlan;
                    d.reset(d.ptr);
                },
                op_ds_propose => {
                    const d = f.drafter orelse return error.BadPlan;
                    f.prefetchEngram(@intCast(msg[2]), &.{@intCast(msg[1])});
                    try d.propose(d.ptr, @intCast(msg[1]), @intCast(msg[2]));
                },
                op_sample, op_sample_many => {
                    const x = f.sampler orelse return error.BadPlan;
                    try x.follow(x.ptr, msg);
                },
                op_grammar => {
                    const x = f.grammar orelse return error.BadPlan;
                    try x.follow(x.ptr, msg);
                },
                else => {
                    const x = f.ext orelse return error.BadPlan;
                    try x.run(x.ptr, msg);
                },
            }
        }
    }

    /// A replayed session operation's failure (the w4b window's fleet crash: an evicted session's resume was
    /// a per-request error on rank 0 and process-fatal on every follower). Request-level errors — the classes
    /// rank 0 ends only the request for — are logged and the follower serves on; process, device, transport
    /// and pool-divergence faults stay process-fatal (tp/replay.zig's classes: fail-fast's own).
    fn followSess(f: *Forward, op: []const u8, err: anyerror) anyerror!void {
        if (tp.replay.processFatal(err)) return err;
        std.log.scoped(.dsv41).err("rank {d}: replayed session {s} failed ({t}); a request-level error rank 0 ends the request for, the follower serves on", .{ f.comm.rank(), op, err });
    }

    fn glueMissing(name: []const u8) Error {
        std.log.scoped(.dsv41).warn("glue step {s}: its kernel is not built yet (the CUDA port glue list)", .{name});
        return error.GlueNotBuilt;
    }

    /// The materialised selection's glue (backend._blocked in decode windows, prefill segments): index.top_positions
    /// ("topk": each row's `count` largest int64 keys as positions, ascending, -1 padded), backend.visible_counts
    /// ("counts") and Win.sub's first position ("block_pos": one int32; a row-mode block reads the table instead).
    /// False: not one of them.
    fn selectionGlue(k: anytype, r: *run.Runner, c: *const calls.Call, step: []const u8) !bool {
        const x = c.args;
        if (std.mem.eql(u8, step, "topk")) {
            const keys = x[0].arg.t;
            const out = x[1].arg.t;
            const count: usize = @intCast(x[2].arg.i);
            if (out.shape[0] != keys.shape[0] or out.shape[1] != count) return error.BadGlue;
            try k.topPositions(try r.tensorAddr(keys), @intCast(keys.stride[0]), @intCast(keys.shape[0]), @intCast(keys.shape[1]), count, try r.tensorAddr(out));
            return true;
        }
        if (std.mem.eql(u8, step, "counts")) {
            const sel = x[0].arg.t;
            try k.visibleCounts(try r.tensorAddr(sel), @intCast(sel.shape[1]), try r.tensorAddr(x[2].arg.t), @intCast(x[3].arg.i), @intCast(sel.shape[0]), try r.tensorAddr(x[1].arg.t));
            return true;
        }
        if (std.mem.eql(u8, step, "block_pos")) {
            const v: i32 = @intCast(x[1].arg.i);
            try r.d.check(r.d.api.cuMemsetD32Async(try r.tensorAddr(x[0].arg.t), @bitCast(v), 1, r.stream.handle), "cuMemsetD32Async");
            return true;
        }
        return false;
    }

    fn numel(t: calls.Tensor) usize {
        var n: usize = 1;
        for (t.shape) |x| n *= @intCast(x);
        return n;
    }

    pub fn sendBuffer(f: *Forward, r: *run.Runner, bytes: usize) !u64 {
        if (f.send == null or f.send.?.len < bytes) {
            if (f.send) |*b| b.free();
            f.send = try cuda.DeviceBuffer.alloc(r.d, bytes);
        }
        return f.send.?.ptr;
    }

    /// The decode window's glue (a prefill segment's handler hands it the steps both have).
    pub fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        const step = c.name["glue.".len..];
        if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.GlueNotBuilt).glue(r, c, step);
        const s = r.stream.handle;
        const k = r.kernels.others(r.stream);
        if (try selectionGlue(k, r, c, step)) return;
        const p = f.slot.pending.?;
        const D: usize = f.cfg.hidden;
        const W: usize = f.comm.world();
        const g = f.activeGraph();
        if (std.mem.eql(u8, step, "positions")) {
            const t = [_]calls.Tensor{ c.args[0].arg.t, c.args[1].arg.t, c.args[2].arg.t };
            // graphed: the start staged in the statics (one graph serves every window of its key)
            if (g) |gr| return k.positionsDev(gr.statics.startDev(), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
            try k.positions(@intCast(p.start), p.n, try r.tensorAddr(t[0]), try r.tensorAddr(t[1]), try r.tensorAddr(t[2]));
            // CED's decoder replay: every row's window starts at the pass's first row (replay.finish's ctx.lo)
            if (c.args.len == 4) try r.d.check(r.d.api.cuMemsetD32Async(try r.tensorAddr(t[2]), @bitCast(@as(i32, @intCast(c.args[3].arg.i))), p.n, s), "cuMemsetD32Async");
            return;
        }
        if (g != null and std.mem.eql(u8, step, "embed")) {
            // graphed: the ids staged before the launch; no host step inside the window
            const n: usize = p.n;
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const V: usize = f.cfg.vocab / W;
            const rows = try f.sendBuffer(r, 2 * n * D);
            try k.embedSend(g.?.statics.idsDev(), n, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try f.comm.allGather(rows, recv, n * D, .bf16, s);
            return k.embedSum(recv, W, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "embed")) {
            // this rank's vocabulary rows (others +0), their all-sum (the window's first exchange), x4 into the streams
            const n: usize = p.n;
            if (f.dev_ids == null or f.dev_ids.?.len < 8 * n) {
                if (f.dev_ids) |*b| b.free();
                f.dev_ids = try cuda.DeviceBuffer.alloc(r.d, @max(8 * n, 256));
            }
            // the previous window's copy out of this staging has finished: a prompt segment waits for that copy
            // alone (Python's ids_of: to_device past CHUNK rows), a smaller window for the stream (a pageable copy)
            if (n <= hs.chunk) try r.stream.synchronize();
            const h = try f.ids_stage.take(r.d, 8 * n);
            for (f.ids, 0..) |id, i| std.mem.writeInt(i64, h[8 * i ..][0..8], id, .little);
            try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(f.dev_ids.?.ptr, h.ptr, 8 * n, s), "cuMemcpyHtoDAsync");
            try f.ids_stage.issued(r.d, r.stream);
            const embed = r.weights.get("embed") orelse return error.MissingWeight;
            const V: usize = f.cfg.vocab / W;
            const rows = try f.sendBuffer(r, 2 * n * D);
            try k.embedSend(f.dev_ids.?.ptr, n, embed.ptr, @intCast(f.comm.rank() * V), V, D, rows);
            if (hooks.set.vision) |vr| try vr.fill(f.vision.?, r.stream, f.ids, rows, D); // rank 0: the image positions' rows (vision.fill)
            const recv = try r.tensorAddr(c.args[1].arg.t);
            try f.comm.allGather(rows, recv, n * D, .bf16, s);
            return k.embedSum(recv, W, n, D, try r.tensorAddr(c.args[0].arg.t));
        }
        if (std.mem.eql(u8, step, "exchange")) {
            // attention's bf16 partial gathered into the receive buffer, rank 0's first
            const send = c.args[0].arg.t;
            return f.comm.allGather(try r.tensorAddr(send), try r.tensorAddr(c.args[1].arg.t), numel(send), .bf16, s);
        }
        if (std.mem.eql(u8, step, "exchange_rows")) {
            // TF_DSV41_PF_OVERLAP: rows [r0, r0 + m) of a bf16 partial into the gathered [world, n, D], where the
            // whole exchange's all-gather puts them: an all-gather of the piece into "L.xrows" [world, m, D] (the
            // collective's channels: no NCCL p2p connection, whose buffers no plan or price holds), then each rank's
            // m rows copied to their place. The same bytes as the whole exchange's
            const send = c.args[0].arg.t;
            const into = c.args[1].arg.t;
            const r0: usize = @intCast(c.args[2].arg.i);
            const n: usize = @intCast(into.shape[1]);
            const m = numel(send) / D;
            if (r0 + m > n or c.args.len < 4) return error.BadGlue;
            const tmp = try r.tensorAddr(c.args[3].arg.t);
            try f.comm.allGather(try r.tensorAddr(send), tmp, m * D, .bf16, s);
            const base = try r.tensorAddr(into);
            for (0..W) |q| try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(base + ((q * n + r0) * D) * 2, tmp + q * m * D * 2, m * D * 2, s), "cuMemcpyDtoDAsync");
            return;
        }
        if (std.mem.eql(u8, step, "exchange_f32")) {
            // the MoE's fp32 partial rounded to bf16, then gathered
            const part = c.args[0].arg.t;
            const n = numel(part);
            const half = try f.sendBuffer(r, 2 * n);
            try k.castBf16(try r.tensorAddr(part), half, n);
            return f.comm.allGather(half, try r.tensorAddr(c.args[1].arg.t), n, .bf16, s);
        }
        if (std.mem.eql(u8, step, "kit_weights"))
            return k.kitWeights(try r.tensorAddr(c.args[0].arg.t), try r.tensorAddr(c.args[1].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "kit_logits")) return k.kitLogits(try r.tensorAddr(c.args[0].arg.t), numel(c.args[0].arg.t));
        if (std.mem.eql(u8, step, "engram_rows")) {
            if (g) |gr| return f.engramGathered(gr, r, c);
            return f.engramRows(r, c);
        }
        if (std.mem.eql(u8, step, "trace")) {
            const map = f.trace orelse return;
            const t = c.args[0].arg.t;
            const host = try f.gpa.alloc(u8, 2 * numel(t));
            defer f.gpa.free(host);
            try r.stream.synchronize();
            try cuda.DeviceBuffer.download(.{ .d = r.d, .ptr = try r.tensorAddr(t), .len = host.len }, 0, host);
            var sha: [32]u8 = undefined;
            std.crypto.hash.sha2.Sha256.hash(host, &sha, .{});
            return map.put(f.gpa, @intCast(c.args[1].arg.i), std.fmt.bytesToHex(sha, .lower));
        }
        return glueMissing(step);
    }

    /// A window's Engram rows of layer L: hashes of its ids (with the slot's tail), this rank's heads' rows (M2a: the
    /// synthetic index rows), the all-gather and the column order of forward.gather_cols.
    pub fn engramRows(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
        const h = f.engram orelse return glueMissing("engram_rows (no Engram tables)");
        const L: u32 = @intCast(c.args[1].arg.i);
        const n = f.ids.len; // a multi-segment run: every segment's rows (fillEngramRows takes each with its slot's tail)
        const W = f.comm.world();
        const cols = f.engramCols(h); // this rank's columns of a row
        const bytes = 2 * n * cols;
        if (f.gated) {
            // the gate: the GPU waits for the worker's rows and copies them where the staged copy would land
            const send = try f.sendBuffer(r, bytes);
            try f.gateCopy(r, L, send, n);
            if (f.gathered == null or f.gathered.?.len < W * bytes) {
                if (f.gathered) |*g| g.free();
                f.gathered = try cuda.DeviceBuffer.alloc(r.d, W * bytes);
            }
            try f.comm.allGather(send, f.gathered.?.ptr, n * cols, .bf16, r.stream.handle);
            return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, W, n, cols, try r.tensorAddr(c.args[0].arg.t));
        }
        // an earlier copy out of this layer's staging must have left it before it is rewritten: a prompt segment waits
        // for that copy alone (decode.LazyEngram: to_device past CHUNK rows, no stream sync), a smaller window for the
        // stream (Python's pageable copy)
        const li = std.mem.indexOfScalar(u32, h.layer_ids, L) orelse return error.NotEngram;
        if (n <= hs.chunk) try r.stream.synchronize();
        const stage = &f.rows_stage[li];
        const host = try stage.take(r.d, bytes);
        try f.fillEngramRows(h, L, @alignCast(std.mem.bytesAsSlice(u16, host)));
        const send = try f.sendBuffer(r, bytes);
        const s = r.stream.handle;
        try r.d.check(r.d.api.cuMemcpyHtoDAsync_v2(send, host.ptr, bytes, s), "cuMemcpyHtoDAsync");
        try stage.issued(r.d, r.stream);
        if (f.gathered == null or f.gathered.?.len < W * bytes) {
            if (f.gathered) |*g| g.free();
            f.gathered = try cuda.DeviceBuffer.alloc(r.d, W * bytes);
        }
        try f.comm.allGather(send, f.gathered.?.ptr, n * cols, .bf16, s);
        return r.kernels.others(r.stream).gatherCols(f.gathered.?.ptr, W, n, cols, try r.tensorAddr(c.args[0].arg.t));
    }

    /// This rank's columns of an Engram row (its heads' share).
    pub fn engramCols(f: *const Forward, h: *const eh.Host) usize {
        return h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
    }

    /// Layer L's rows of the window's ids (with the slot's tail): the hashes, this rank's heads' index rows.
    pub fn fillEngramRows(f: *Forward, h: *const eh.Host, L: u32, out: []u16) !void {
        const tw = ph.start(f.phases, .@"engram.wait");
        defer tw.stop();
        if (f.multi) |m| {
            // a multi-segment run: each segment's rows after its own slot's tail, at its rows of the run
            const cols = out.len / f.ids.len;
            for (m.segs, m.row0, m.tails) |sg, r0, t| try h.windowRows(f.gpa, try f.engramSource(h), sg.ids, t.items(), L, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, out[r0 * cols ..][0 .. sg.ids.len * cols]);
            return;
        }
        return h.windowRows(f.gpa, try f.engramSource(h), f.ids, f.slot.tail[0..f.slot.tail_len], L, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, out);
    }

    /// R3: the Engram rows' source, opened on first use: TF_DSV41_ENGRAM_DIR's shards on every rank alike (null:
    /// M2a's index rows, the gates' references). TF_DSV41_ENGRAM_CACHE: records a layer (65,536).
    pub fn engramSource(f: *Forward, h: *const eh.Host) !?eh.Source {
        if (!f.engram_opened) {
            f.engram_opened = true;
            if (std.c.getenv("TF_DSV41_ENGRAM_PREFETCH")) |v| f.engram_prefetch = !std.mem.eql(u8, std.mem.span(v), "0");
            if (std.c.getenv("TF_DSV41_ENGRAM_DIR")) |dir| {
                const cap: u32 = if (std.c.getenv("TF_DSV41_ENGRAM_CACHE")) |v| try std.fmt.parseInt(u32, std.mem.span(v), 10) else 65536;
                f.engram_rows = try er.Rows.open(f.gpa, std.mem.span(dir), f.comm.rank(), f.comm.world(), h.layer_ids, cap);
                std.log.scoped(.dsv41).info("engram: rows from {s} (io_uring {s}, {d} records a layer cached)", .{ std.mem.span(dir), if (f.engram_rows.?.ring != null) "on" else "off", cap });
            } else std.log.scoped(.dsv41).warn("engram: TF_DSV41_ENGRAM_DIR unset: synthetic index rows (M2a's), not the model's table", .{});
            const s = try egate.settings();
            if (s.on) {
                if (f.engram_rows) |x| {
                    var la = std.heap.ArenaAllocator.init(f.gpa);
                    defer la.deinit();
                    var layers: std.ArrayList(u32) = .empty;
                    for (try f.backbone(la.allocator())) |L| if (std.mem.indexOfScalar(u32, h.layer_ids, L) != null) try layers.append(la.allocator(), L);
                    const g = try egate.Gate.open(std.heap.c_allocator, f.runner.d, h, x.source(), layers.items, f.comm.rank(), f.comm.world(), f.cfg.engram_head_dim, s);
                    g.sync = if ((try round_graph.current()).gate) .{ .ctx = f, .run = syncGatedEnd } else .{ .ctx = f.runner, .run = syncRunner };
                    f.gate = g;
                } else std.log.scoped(.dsv41).warn("engram gate off: no packed table (TF_DSV41_ENGRAM_DIR)", .{});
            }
        }
        if (gateOf(f)) |g| return g.source();
        return if (f.engram_rows) |x| x.source() else null;
    }

    /// The Engram gate behind the forward's opaque hook handle (engram_gate.Gate).
    fn gateOf(f: *Forward) ?*egate.Gate {
        const g = f.gate orelse return null;
        return @ptrCast(@alignCast(g));
    }

    /// TF_DSV41_ENGRAM_GATE: arms the gate for a window over `items` (`fill`: its rows with a bucket's padding); false
    /// when the gate is off or the window is too wide (the old path stages its rows).
    pub fn armEngram(f: *Forward, items: []const egate.Item, fill: u32) !bool {
        const h = f.engram orelse return false;
        _ = try f.engramSource(h); // the table and the gate open on the first Engram window
        const g = gateOf(f) orelse return false;
        if (!g.eligible(fill)) return false;
        const ts = ph.start(f.phases, .@"engram.wait");
        defer ts.stop();
        try g.arm(items, fill);
        return true;
    }

    fn syncRunner(ctx: *anyopaque) anyerror!void {
        const r: *run.Runner = @ptrCast(@alignCast(ctx));
        try r.stream.synchronize();
    }

    /// TF_DSV41_ROUND_GRAPH's `gate`: the gate's arm waits for the last gated window's end, not for everything queued
    /// since (a keep's carries, a row table: the GPU drained before the window's launch).
    fn syncGatedEnd(ctx: *anyopaque) anyerror!void {
        const f: *Forward = @ptrCast(@alignCast(ctx));
        if (f.gated_end) |e| return e.synchronize();
        try f.runner.stream.synchronize();
    }

    /// After a window's launches (one-slot or row): a gated window's end recorded for the next arm (`syncGatedEnd`).
    pub fn gatedEnd(f: *Forward) !void {
        if (!f.gated or f.gate == null or !(try round_graph.current()).gate) return;
        if (f.gated_end == null) f.gated_end = try cuda.Event.init(f.runner.d, false);
        try f.gated_end.?.record(f.runner.stream);
    }

    /// The armed round's errors (a failed read, a GPU wait past the timeout); nothing when no window was gated.
    pub fn engramCheck(f: *Forward) !void {
        if (gateOf(f)) |g| if (f.gated) try g.check();
    }

    /// The gate kernel of Engram layer L: `n` rows of the armed round's staging into `dst` once the worker published them.
    pub fn gateCopy(f: *Forward, r: *run.Runner, L: u32, dst: u64, n: usize) !void {
        const g = (gateOf(f) orelse return error.NotEngram);
        const k = g.slotOf(L) orelse return error.NotEngram;
        return r.kernels.others(r.stream).engramGate(g.mem.ctl_dev, @intCast(egate.ctl_ready + k), egate.ctl_err, g.layerBase(k), g.parityStride(), dst, @intCast(2 * n * g.cols), g.timeout_ns, true);
    }

    /// Every Engram layer's reads of a window's rows `ids` (after `tail`) in flight at once, before the first layer
    /// waits for its own (Python's round job: all layers' rows in one batched read); the waits then find them read or
    /// pending (engram_rows), so a window pays one read latency, not one a layer. Off with TF_DSV41_ENGRAM_PREFETCH=0.
    pub fn issueEngram(f: *Forward, ids: []const u32, tail: []const u32) void {
        const h = f.engram orelse return;
        const src = (f.engramSource(h) catch return) orelse return;
        if (!f.engram_prefetch or h.layer_ids.len < 2) return;
        h.prefetch(f.gpa, src, ids, tail, f.comm.rank(), f.comm.world());
    }

    /// R3: the coming window's rows `ids` at `start` (row 0: the pending token) read ahead on this rank; its lookback
    /// is the slot's tail plus the rows of the window in flight that `start` implies are kept. A wrong guess costs a
    /// read at the window, never a bit.
    pub fn prefetchEngram(f: *Forward, start: u64, ids: []const u32) void {
        const h = f.engram orelse return;
        const src = (f.engramSource(h) catch return) orelse return;
        if (!f.engram_prefetch) return;
        const tp_ = ph.start(f.phases, .prefetch);
        defer tp_.stop();
        var buf: [2 * eh.max_ngram + 128]u32 = undefined;
        const pend: ?[]const u32 = if (f.slot.pending) |p| f.ids[0..p.n] else null;
        const lb = eh.lookbackAt(f.slot.tail[0..f.slot.tail_len], f.slot.pos, pend, start, h.max_ngram - 1, &buf) orelse return;
        // the gate's worker owns the table between windows: a warm job (Python's `warm`), not a read on this thread
        if (gateOf(f)) |g| return g.warm(&.{.{ .ids = ids, .tail = lb }});
        h.prefetch(f.gpa, src, ids, lb, f.comm.rank(), f.comm.world());
    }

    pub const greedyScratch = fwd_sample.greedyScratch;
    /// The multi-segment prefill run's segments (forward_prefill.promptMulti; prefillMulti above sends the leader's op)
    pub const finishMulti = pf.finishMulti;
    pub const engramGathered = fwd_graph.engramGathered;
    pub const activeGraph = fwd_graph.activeGraph;
    pub const graphState = fwd_graph.graphState;
    const graphedWindow = fwd_graph.graphedWindow;
    const graphBody = fwd_graph.graphBody;
    pub const fingerprint = fwd_graph.fingerprint;
};

pub const Graphed = fwd_graph.Graphed;


/// M2a's peer: world 2, rank 0 alone on one GPU, the peer's half of every all-gather zeros (dsv41_m1_capture.py's
/// ZeroPeer, the Python reference's). Only the all-gather the forward uses is implemented.
pub const ZeroPeer = struct {
    d: *const cuda.Driver,

    const C = tp.collective;

    fn rank(_: *anyopaque) u32 {
        return 0;
    }
    fn world(_: *anyopaque) u32 {
        return 2;
    }
    fn allGather(ptr: *anyopaque, send: C.DevicePtr, recv: C.DevicePtr, count: usize, t: C.DType, stream: C.Stream) C.Error!void {
        const z: *ZeroPeer = @ptrCast(@alignCast(ptr));
        const n = count * t.size();
        z.d.check(z.d.api.cuMemcpyDtoDAsync_v2(recv, send, n, stream), "cuMemcpyDtoDAsync") catch return error.BackendFailed;
        z.d.check(z.d.api.cuMemsetD8Async(recv + n, 0, n, stream), "cuMemsetD8Async") catch return error.BackendFailed;
    }
    fn unsupported(_: *anyopaque, _: C.DevicePtr, _: C.DevicePtr, _: usize, _: C.DType, _: C.Op, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn exchange(_: *anyopaque, _: C.DevicePtr, _: C.DevicePtr, _: usize, _: C.DType, _: u32, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn p2p(_: *anyopaque, _: C.DevicePtr, _: usize, _: C.DType, _: u32, _: C.Stream) C.Error!void {
        return error.Unsupported;
    }
    fn none(_: *anyopaque) C.Error!void {}
    fn abort(_: *anyopaque) void {}

    const vtable: C.Collective.VTable = .{
        .kind = .host,
        .rank = rank,
        .world = world,
        .all_reduce = unsupported,
        .all_gather = allGather,
        .exchange = exchange,
        .send = p2p,
        .recv = p2p,
        .barrier = none,
        .check = none,
        .abort = abort,
    };

    pub fn collective(z: *ZeroPeer) C.Collective {
        return .{ .ptr = z, .vtable = &vtable };
    }
};

