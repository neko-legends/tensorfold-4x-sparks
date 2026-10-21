//! The forward's own prefill: a prompt in segments through block_prefill.zig's program, each committed as a keep.

const std = @import("std");
const quiet = @import("builtin").is_test; // a test's stderr is the build runner's channel
const cuda = @import("cuda");
const calls = @import("block_calls.zig");
const buffers = @import("buffers.zig");
const run = @import("run.zig");
const block = @import("block.zig");
const block_prefill = @import("block_prefill.zig");
const longpf = @import("longpf.zig");
const image_opts = @import("image_opts.zig");
const fwd = @import("forward.zig");
const eh = @import("engram_host.zig");
const ced = @import("ced.zig");
const ft = @import("forward_types.zig");
const graphs = @import("graphs.zig");
const prod_knobs = @import("prod_knobs.zig");
const serve_knobs = @import("serve_knobs.zig");
const pf_k4 = @import("forward_prefill_k4.zig");
const vision_rows = @import("vision_rows.zig");
const pipe_run = @import("pipe_run.zig");
const Forward = fwd.Forward;

pub const log = std.log.scoped(.dsv41);

pub const Error = error{ NoExpertTable, NoK2Table, BadGlue, StashGap };

/// The prefill's own state on the forward (forward.zig: `prefill_state`).
pub const State = struct {
    /// the persistent tables and grown buffers are in place (ready())
    ready: bool = false,
    /// the prompt's own copy (a caller's slice may alias the window's ids)
    ids: std.ArrayList(u32) = .empty,
    /// gm_plan's width-masked picks (int32 [P])
    masked: ?cuda.DeviceBuffer = null,
    /// the segment's layers and the MoE blocks seen: gm_ticket starts each layer's MoE, gm_plan reads its K2 table
    layers: []const u32 = &.{},
    moe: usize = 0,
    layer: u32 = 0,
    /// TF_DSV41_PREFETCH_AHEAD: the next segment's Engram reads issued before this one runs
    ahead: bool = false,
    /// long prompts' buffers (longpf.zig)
    long: longpf.State = .{},
    /// TF_DSV41_PF_TBO: micro-batch B's own long-prompt buffers, and the pair in flight
    long_b: longpf.State = .{},
    tbo: ?*Pair = null,
    /// TF_DSV41_PF_4K: 4,096-row segments in the prefill's own workspace (`K4`); the forward's plan stays 2,048 rows
    k4: bool = false,
    k4ws: K4 = .{},
    /// the prompt call may use the workspace (every rank's MemAvailable above the floor); false: 2,048-row segments
    ws_ok: bool = true,
    ws_fallbacks: u64 = 0,
    agreement: ?graphs.Agreement = null,
    /// the next prompt piece's ids (op 60): the call's last segment reads their Engram rows ahead; cleared by the call
    next: std.ArrayList(u32) = .empty,
    /// TF_DSV41_PREFILL: `.replay` = CED (segments are encoder passes, `finish` runs the decoder); `.full`: every layer
    mode: serve_knobs.PrefillMode = .full,
    /// the decoder replay's ids (the glue reads them through Forward.ids while it runs)
    tail_ids: [ced.keep]u32 = undefined,
    /// what "w.taps" holds after the prompt: positions [taps_at, taps_at + taps_rows) at rows 0 ..: the DSpark hand-off
    taps_at: u64 = 0,
    taps_rows: u32 = 0,
    /// TF_DSV41_PREFILL_PIPE (pipe_run.zig, model.zig installs it): long prompts' encoder segments through the ranks'
    /// stages; null: off
    pipe: ?*pipe_run.Line = null,
    /// with `pipe`: `settle` / `pipeRelease` for the session store's snapshots and the forward's release, called through
    /// these (no static path from those files into the line's code; null: off)
    pipe_settle: ?*const fn (f: *Forward) anyerror!void = null,
    pipe_release: ?*const fn (f: *Forward) void = null,

    pub fn deinit(s: *State, gpa: std.mem.Allocator) void {
        if (s.pipe) |p| p.deinit();
        s.long.deinit(gpa);
        s.long_b.deinit(gpa);
        if (s.k4ws.buf) |*b| b.free();
        if (s.k4ws.send) |*b| b.free();
        if (s.k4ws.gathered) |*b| b.free();
        if (s.masked) |*b| b.free();
        s.ids.deinit(gpa);
        s.next.deinit(gpa);
        if (s.agreement) |*x| x.deinit();
        var it = s.k4ws.keep.keyIterator();
        while (it.next()) |k| gpa.free(k.*);
        s.k4ws.keep.deinit(gpa);
    }
};

/// The prefill's state on `f` (`Forward.prefill_state`, made by `attach` at engine open).
pub fn state(f: *const Forward) *State {
    return @constCast(&f.prefill_state);
}

/// Engine open: the forward's prefill state, released by Forward.deinit (forward_hooks.Prefill.deinit).
pub fn attach(f: *Forward) !void {
    f.prefill_state = .{};
}

fn deinitFn(f: *Forward) void {
    f.prefill_state.deinit(f.gpa);
    f.prefill_state = .{};
}

fn aheadFn(f: *Forward) ?*std.ArrayList(u32) {
    const s = state(f);
    return if (s.ahead) &s.next else null;
}

fn replayingFn(f: *const Forward) bool {
    return state(f).mode == .replay;
}

fn planFn(f: *Forward, p: *buffers.Plan, rows: u32) anyerror!void {
    return plan(f, p, rows);
}

fn promptFn(f: *Forward, ids: []const u32) anyerror!void {
    return prompt(f, ids);
}

fn promptMultiFn(f: *Forward, segs: []const MultiSeg) anyerror!void {
    return promptMulti(f, segs);
}

fn finishFn(f: *Forward) anyerror!void {
    return finish(f);
}

fn finishMultiFn(f: *Forward, slots: []const u32, out: []Replayed) anyerror!void {
    return finishMulti(f, slots, out);
}

pub const hooks_table: @import("forward_hooks.zig").Prefill = .{ .plan = planFn, .prompt = promptFn, .promptMulti = promptMultiFn, .finish = finishFn, .finishMulti = finishMultiFn, .ahead = aheadFn, .replaying = replayingFn, .deinit = deinitFn };

/// Adds the prefill program's roles at `rows` (the largest segment) to `p`, at every start and segment size it uses.
pub fn plan(f: *Forward, p: *buffers.Plan, rows: u32) !void {
    if (rows <= 16) return error.BadWindow;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const layers = try f.backbone(la.allocator());
    try planPrograms(f, p, rows, layers);
    // TF_DSV41_PREFILL_PIPE: the stage programs' roles (their own namespace, the stage copies' forms)
    if (f.prefill_state.pipe) |ln| try pipe_run.planInto(f, ln, p);
    if (f.prefill_state.k4 or (f.prefill_state.mode == .replay and f.opts.pf_tbo)) {
        const ws = &f.prefill_state.k4ws;
        if (ws.plan == null) {
            ws.plan = .{ .a = f.gpa };
            try planWorkspace(f, &ws.plan.?, rows, layers);
            if (keepsShared(f)) {
                const before = planBytes(&ws.plan.?);
                try k4Trim(f.gpa, &ws.plan.?, p, &ws.keep);
                log.info("prefill 4K: rank {d}: {d} roles shared with the forward (TF_DSV41_PF_WS_SHARE): workspace {d} -> {d} MiB", .{ f.comm.rank(), ws.keep.count(), before >> 20, planBytes(&ws.plan.?) >> 20 });
            }
        }
    }
}

/// The forward's prefill programs at `rows` (`plan` without the workspace).
pub fn planPrograms(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const sizes = [_]u32{ rows, 17, 16 };
    for (sizes[if (rows > 17) 0 else 1..]) |n| {
        const top = try lastStart(f, layers, n, .whole);
        for ([_]i64{ 0, top }) |start| {
            var ea = std.heap.ArenaAllocator.init(f.gpa);
            defer ea.deinit();
            // replay's segments run the encoder (no head): its whole programs plan no head, w.logits not 2,048 rows
            const head = state(f).mode != .replay;
            try p.add(try block_prefill.emitPrefill(ea.allocator(), f.cfg, f.widths, image_opts.planOptions(f), layers, n, start, head));
        }
    }
    if (state(f).mode == .replay) try planReplay(f, p, rows, layers);
    // the multi-segment runs (TF_DSV41_PIECE_RUNS / _REPLAY_RUNS): their own roles and their rows' at `rows`
    var ra = std.heap.ArenaAllocator.init(f.gpa);
    defer ra.deinit();
    const replay = f.prefill_state.mode == .replay;
    // f.opts, not planOptions: a run never holds an image segment (emitMulti refuses image rows; the caller runs those
    // one at a time), so its roles are the text options' (with planOptions' image_rows -1 every run is refused and
    // its roles, w.mpos64 among them, drop out of the plan: Unbound on the first multi-slot run, 2026-10-12)
    const runs = try block_prefill.planRuns(ra.allocator(), f.cfg, f.widths, f.opts, if (replay) try encoderLayers(f, layers) else layers, if (replay) try decoderLayers(f, layers) else null, rows, if (replay) .encoder else .whole);
    for (runs.items) |cs| try p.add(cs);
}

/// TF_DSV41_PF_WS_SHARE=1 with TF_DSV41_PF_4K alone: a 4K segment reuses forward roles that are already large enough.
pub fn keepsShared(f: *const Forward) bool {
    return state(f).k4 and !f.opts.pf_tbo and envOn("TF_DSV41_PF_WS_SHARE");
}

fn envOn(name: [:0]const u8) bool {
    const v = std.mem.span(std.c.getenv(name) orelse return false);
    return std.mem.eql(u8, v, "1");
}

pub const k4Trim = pf_k4.k4Trim;
const planWorkspace = pf_k4.planWorkspace;
pub const workspaceBytes = pf_k4.workspaceBytes;
const planBytes = pf_k4.planBytes;

/// CED replay's programs (ced.zig): an encoder segment (the whole segment's roles up to the decoder's first layer, and
/// the stash ring), and the decoder pass at the row counts where its paths change (a short segment up to 16 rows, the
/// selection's attn_cuda top-k up to 64, the materialised selection past it, the replay's 127), at 0 and its last start.
fn planReplay(f: *Forward, p: *buffers.Plan, rows: u32, layers: []const u32) !void {
    const enc = try encoderLayers(f, layers);
    const dec = try decoderLayers(f, layers);
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    try p.add(try block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, image_opts.planOptions(f), enc, rows, 0, .encoder));
    for ([_]u32{ 16, 17, 32, 33, 64, 65, ced.keep }) |m| {
        if (m > rows) continue;
        const top = try lastStart(f, dec, m, .decoder);
        for ([_]i64{ 0, top }) |start| {
            var da = std.heap.ArenaAllocator.init(f.gpa);
            defer da.deinit();
            try p.add(try block_prefill.emitReplay(da.allocator(), f.cfg, f.widths, image_opts.planOptions(f), dec, m, start, .decoder));
        }
    }
}

/// The encoder pass's layers: the forward's up to the decoder's first (ced.decoderStart), inclusive.
pub fn encoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[0 .. i + 1];
}

/// The decoder pass's layers: the forward's from the decoder's first on.
pub fn decoderLayers(f: *Forward, layers: []const u32) ![]const u32 {
    const d = try ced.decoderStart(f.cfg);
    const i = std.mem.indexOfScalar(u32, layers, d) orelse return error.NoDecoder;
    return layers[i..];
}

/// Whether an `n`-row segment (`part` of it) at `start` is on the prefill path (block_prefill's emitter takes it).
fn emits(f: *Forward, o: block.Options, layers: []const u32, n: u32, start: i64, part: block_prefill.Part) !bool {
    var ea = std.heap.ArenaAllocator.init(f.gpa);
    defer ea.deinit();
    _ = block_prefill.emitReplay(ea.allocator(), f.cfg, f.widths, o, layers, n, start, part) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// The last start an `n`-row segment emits at; the decoder pass's is one lower.
fn lastStart(f: *Forward, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    return lastStartWith(f, f.opts, layers, n, part);
}

/// `lastStart` with options `o` (TF_DSV41_PF_4K's segments: k4Options).
pub fn lastStartWith(f: *Forward, o: block.Options, layers: []const u32, n: u32, part: block_prefill.Part) !i64 {
    var hi: i64 = o.limit - n - @intFromBool(part == .decoder);
    if (hi < 0) return error.BadWindow;
    if (try emits(f, o, layers, n, hi, part)) return hi;
    if (!try emits(f, o, layers, n, 0, part)) {
        if (!quiet) log.warn("prefill: a {d}-row segment is not on the prefill path at any position (block_prefill: Unsupported)", .{n});
        return error.Unsupported;
    }
    var lo: i64 = 0;
    while (hi - lo > 1) {
        const mid = lo + @divFloor(hi - lo, 2);
        if (try emits(f, o, layers, n, mid, part)) lo = mid else hi = mid;
    }
    log.info("prefill: {d}-row segments emit up to start {d}", .{ n, lo });
    return lo;
}

pub const K4 = pf_k4.K4;
pub const segmentRows = pf_k4.segmentRows;
const wsAgreed = pf_k4.wsAgreed;
pub const k4Options = pf_k4.k4Options;
pub const wsPair = pf_k4.wsPair;
const wsArg = pf_k4.wsArg;
const k4Acquire = pf_k4.k4Acquire;
const nowMs = pf_k4.nowMs;
const wsFloorGiB = pf_k4.wsFloorGiB;
const hardFloorGiB = pf_k4.hardFloorGiB;
const k4Release = pf_k4.k4Release;

/// TF_DSV41_PF_TBO: one micro-batch's host side of a pair in flight: ids, window, Engram tail, MoE calls seen.
pub const Micro = struct {
    ids: []const u32,
    pending: @TypeOf(@as(fwd.Slot, undefined).pending),
    tail: Tail,
    moe: usize = 0,
    layer: u32 = 0,
};

pub const Pair = struct { a: Micro, b: Micro };

/// The prompt `ids` from the slot's position (forward.prefill: the leader has sent it to the followers).
pub fn prompt(f: *Forward, ids_in: []const u32) !void {
    const st = state(f);
    if (f.slot.pending != null) return error.BadWindow;
    if (ids_in.len == 0) return;
    if (f.slot.pos + ids_in.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
    if (ids_in.ptr != st.ids.items.ptr) {
        st.ids.clearRetainingCapacity();
        try st.ids.appendSlice(f.gpa, ids_in);
    }
    const ids = st.ids.items;
    st.taps_rows = 0;
    try ready(f);
    defer st.next.clearRetainingCapacity();
    // TF_DSV41_PREFILL_PIPE: the line routes the call (its rows through the stages, or tensor parallel: tpRows)
    if (st.pipe) |ln| if (st.mode == .replay) return ln.call(f, ids);
    return tpRows(f, ids);
}

/// Prompt rows `ids` from the slot's position as the forward's own segments (tensor parallel; `prompt` without the
/// pipeline's route): the workspace's decision, then 4K / 2K segments or TBO pairs.
pub fn tpRows(f: *Forward, ids: []const u32) !void {
    const st = &f.prefill_state;
    // TF_DSV41_PF_4K / _PF_TBO: the workspace for this call, decided on every rank alike (else 2,048-row segments)
    st.ws_ok = true;
    if (st.k4 or (st.mode == .replay and f.opts.pf_tbo)) st.ws_ok = try wsAgreed(f);
    const seg: usize = segmentRows(f);
    var at: usize = 0;
    while (at < ids.len) {
        // TF_DSV41_PF_TBO: two segments at a time (the second at least tbo_min rows), else one
        if (pairOf(f, ids.len - at)) |pr| {
            const n = pr[0] + pr[1];
            if (st.ahead) ahead(f, ids, at, n, seg);
            if (try segmentPair(f, ids[at .. at + pr[0]], ids[at + pr[0] .. at + n])) {
                at += n;
                continue;
            }
        }
        const n = @min(seg, ids.len - at);
        if (st.ahead) ahead(f, ids, at, n, seg);
        try segment(f, ids[at .. at + n], at + n == ids.len and st.mode == .full);
        at += n;
    }
}

/// TF_DSV41_PF_TBO's split of the `left` rows ahead into two segments; null when off, full mode, or under tbo_min.
fn pairOf(f: *const Forward, left: usize) ?[2]usize {
    if (!f.opts.pf_tbo or state(f).mode != .replay or !state(f).ws_ok) return null;
    return pairSplit(segmentRows(f), left);
}

/// `pairOf`'s split of `left` rows at segments of `seg` rows (null: one segment at a time).
pub fn pairSplit(seg: usize, left: usize) ?[2]usize {
    const min: usize = @intCast(block_prefill.tbo_min);
    if (left >= 2 * seg) return .{ seg, seg };
    if (left < 2 * min) return null;
    const na = @min(seg, (left / 2) / 16 * 16);
    if (left - na > seg or na < min) return null;
    return .{ na, left - na };
}

/// Two encoder segments as one two-stream program, committed once; false: refused, run the segments one at a time.
fn segmentPair(f: *Forward, ia: []const u32, ib: []const u32) !bool {
    const st = state(f);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try encoderLayers(f, try f.backbone(a));
    const start = f.slot.pos;
    const na: u32 = @intCast(ia.len);
    const nb: u32 = @intCast(ib.len);
    // under TF_DSV41_PF_4K a micro-batch past the forward's rows is a 4K one (both in the workspace: wsPair)
    const big = st.k4 and @max(na, nb) > f.opts.prefill_rows;
    const o0 = image_opts.segmentOptions(f, ia.ptr[0 .. na + nb]);
    const o = if (big) k4Options(o0) else o0;
    const plain = block_prefill.emitTbo(a, f.cfg, f.widths, o, layers, na, nb, @intCast(start)) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    const cs = try wsPair(a, plain, big);
    try k4Acquire(f);
    if (f.kv) |kx| try kx.reserve(start + na + nb);
    // the micro-batches' host sides: B's Engram tail is the slot's after A's ids (tailOf, as A's commit leaves it)
    var pair: Pair = .{ .a = .{ .ids = ia, .pending = .{ .start = start, .n = na }, .tail = .{} }, .b = .{ .ids = ib, .pending = .{ .start = start + na, .n = nb }, .tail = .{} } };
    pair.a.tail.len = f.slot.tail_len;
    @memcpy(pair.a.tail.ids[0..pair.a.tail.len], f.slot.tail[0..f.slot.tail_len]);
    if (f.engram) |h| {
        var s2 = f.slot;
        tailOf(&s2, ia, h.max_ngram - 1);
        pair.b.tail.len = s2.tail_len;
        @memcpy(pair.b.tail.ids[0..s2.tail_len], s2.tail[0..s2.tail_len]);
    } else pair.b.tail = pair.a.tail;
    f.ids = ia;
    f.slot.pending = pair.a.pending;
    st.tbo = &pair;
    defer st.tbo = null;
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    // the pair's send / gathered buffers are the workspace's while it runs (as a 4K segment's)
    std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
    std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    defer {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    }
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(na + nb), false);
    // the commit: both micro-batches' rows as one segment, the carries from B's last rows (its roles in the workspace)
    const both = ia.ptr[0 .. na + nb];
    f.ids = both;
    f.slot.pending = .{ .start = start, .n = na + nb };
    try commitRoles(f, nb, 0, K4.tag ++ "b~", na + nb);
    f.slot.ced.extend(start, both);
    f.last_rows = 0;
    st.taps_at = start;
    st.taps_rows = 0;
    return true;
}

/// Read-ahead of the Engram rows after segment ids[at .. at + n] (next segment or piece); a wrong guess costs a read.
fn ahead(f: *Forward, ids: []const u32, at: usize, n: usize, seg: usize) void {
    const st = state(f);
    if (at + n < ids.len) return readAhead(f, ids, at, n, ids[at + n .. at + n + @min(seg, ids.len - at - n)]);
    if (st.next.items.len > 0) readAhead(f, ids, at, n, st.next.items);
}

fn readAhead(f: *Forward, ids: []const u32, at: usize, n: usize, next: []const u32) void {
    const h = f.engram orelse return;
    if (!f.engram_prefetch) return;
    const src = (f.engramSource(h) catch return) orelse return;
    const keep: usize = h.max_ngram - 1;
    const b = at + n;
    var buf: [2 * eh.max_ngram]u32 = undefined;
    const lb: []const u32 = if (n >= keep) ids[b - keep .. b] else blk: {
        const old = @min(f.slot.tail_len, keep - n);
        @memcpy(buf[0..old], f.slot.tail[f.slot.tail_len - old .. f.slot.tail_len]);
        @memcpy(buf[old .. old + n], ids[at..b]);
        break :blk buf[0 .. old + n];
    };
    h.prefetch(f.gpa, src, next, lb, f.comm.rank(), f.comm.world());
}

/// One segment at the slot's position, run eagerly, then committed.
fn segment(f: *Forward, ids: []const u32, head: bool) !void {
    const st = state(f);
    const n: u32 = @intCast(ids.len);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    // CED replay: the encoder pass (layers up to the decoder's first, its site and compressor, the stash; no head)
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const start = f.slot.pos;
    if (f.kv) |kx| try kx.reserve(start + n);
    // TF_DSV41_PF_4K: a segment past the forward's rows runs in the workspace (its own roles: block_prefill.ownCalls)
    const big = st.k4 and st.ws_ok and n > f.opts.prefill_rows;
    const o = if (big) k4Options(image_opts.segmentOptions(f, ids)) else image_opts.segmentOptions(f, ids);
    const emitted = if (replay) block_prefill.emitReplay(a, f.cfg, f.widths, o, layers, n, @intCast(start), .encoder) else block_prefill.emitPrefill(a, f.cfg, f.widths, o, layers, n, @intCast(start), head);
    const plain = emitted catch |e| {
        if (e == error.Unsupported) if (!quiet) log.warn("prefill: the {d}-row segment at {d} is not on the prefill path (block_prefill: Unsupported)", .{ n, start });
        return e;
    };
    const cs = if (big) try block_prefill.ownCallsKeep(a, plain, K4.tag, &st.k4ws.keep) else plain;
    if (big) try k4Acquire(f);
    // the glue reads the ids (embedding, Engram) and the pending window (positions)
    f.ids = ids;
    f.slot.pending = .{ .start = start, .n = n };
    errdefer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    // TF_DSV41_PF_4K: the segment's send / gathered buffers are the workspace's while it runs
    if (big) {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    }
    defer if (big) {
        std.mem.swap(?cuda.DeviceBuffer, &f.send, &st.k4ws.send);
        std.mem.swap(?cuda.DeviceBuffer, &f.gathered, &st.k4ws.gathered);
    };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(n), false);
    if (big) try commitRoles(f, n, 0, K4.tag, n) else try commit(f, n);
    // replay.encode: the stash's ids follow its rows (a segment that does not continue it starts it over)
    if (replay) f.slot.ced.extend(start, ids);
    f.last_rows = if (replay) 0 else n;
    st.taps_at = start;
    st.taps_rows = if (replay) 0 else n;
}

pub const MultiSeg = ft.MultiSeg;

pub const Tail = ft.Tail;

pub const Multi = ft.Multi;

/// Several slots' prompt segments in one run (block_prefill.emitMulti; Python's slots.prefill / replay.encode over
/// prefill_runs: GLM 0560's one expert pass over every prefilling slot). Each segment starts at its slot's position;
/// the rows' launches run over all segments, each segment's CSA2 steps on its own slot (glue pf_seg, Forward.pf_switch),
/// then each slot commits its segment as `segment` would (carries from its last row, Engram tail, position, CED's
/// stash ids). error.Unsupported: the caller runs the segments one at a time (a short segment, an image, split KV).
pub fn promptMulti(f: *Forward, segs: []const MultiSeg) !void {
    const st = &f.prefill_state;
    const sw = f.pf_switch orelse return error.Unsupported;
    if (segs.len == 0 or segs.len > 16) return error.BadWindow;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const replay = st.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const spans = try a.alloc(block_prefill.Span, segs.len);
    const row0 = try a.alloc(usize, segs.len);
    const tails = try a.alloc(Tail, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans, row0, tails) |sg, *sp, *r0, *t| {
        try sw.run(sw.ctx, sg.slot);
        if (f.slot.pending != null or f.slot.pos != sg.start or sg.ids.len == 0) return error.BadWindow;
        if (sg.start + sg.ids.len > @as(u64, @intCast(f.opts.limit))) return error.BadWindow;
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        r0.* = all.items.len;
        t.len = f.slot.tail_len;
        @memcpy(t.ids[0..t.len], f.slot.tail[0..t.len]);
        try all.appendSlice(a, sg.ids);
    }
    if (vision_rows.segmentOptions(f, all.items).image_rows != 0) return error.Unsupported;
    const cs = try block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, layers, spans, if (replay) .encoder else .whole);
    try ready(f);
    // the pages of every segment's positions, before any launch (each on its slot's table)
    for (segs) |sg| {
        try sw.run(sw.ctx, sg.slot);
        if (f.kv) |kx| try kx.reserve(sg.start + sg.ids.len);
    }
    var m: Multi = .{ .segs = segs, .row0 = row0, .tails = tails, .ids = all.items };
    f.multi = &m;
    defer f.multi = null;
    try enterRun(f, &m);
    st.layers = layers;
    st.moe = 0;
    st.taps_rows = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, segs[0].start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(all.items.len), false);
    // each slot keeps its segment (Python's keep(slot, n - 1) a segment, replay.encode's stash ids)
    for (segs, row0) |sg, r0| {
        try sw.run(sw.ctx, sg.slot);
        f.ids = sg.ids;
        f.slot.pending = .{ .start = sg.start, .n = @intCast(sg.ids.len) };
        try commitAt(f, @intCast(sg.ids.len), r0);
        if (replay) f.slot.ced.extend(sg.start, sg.ids);
    }
    f.last_rows = 0;
}

/// Whether a multi-segment run takes `segs` (promptMulti's refusals), checked before the leader sends it.
pub fn runnable(f: *Forward, segs: []const MultiSeg) !bool {
    if (f.pf_switch == null or segs.len < 2 or segs.len > 16) return false;
    // TF_DSV41_PREFILL_PIPE: a segment of the open line's slot, or one that would open a line, runs by itself (its route)
    if (f.prefill_state.pipe) |ln| for (segs) |sg| {
        if (ln.slot) |ls| if (ls == sg.slot) return false;
        if (ln.slot == null and sg.ids.len >= ln.s.min_rows) return false;
    };
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const replay = f.prefill_state.mode == .replay;
    const layers = if (replay) try encoderLayers(f, try f.backbone(a)) else try f.backbone(a);
    const spans = try a.alloc(block_prefill.Span, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans) |sg, *sp| {
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        try all.appendSlice(a, sg.ids);
    }
    if (vision_rows.segmentOptions(f, all.items).image_rows != 0) return false;
    _ = block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, layers, spans, if (replay) .encoder else .whole) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// One slot's decoder replay in a batched run (finishMulti): the replayed tail's first position and rows, and its
/// first row in the run (its rows of "w.taps": the drafter's context, ced.handoff).
pub const Replayed = ft.Replayed;

/// Whether a batched decoder replay takes `segs` (each a slot's prefilled tail from its R0), checked before the leader
/// sends it (emitMulti's refusals: a tail of 32 rows or fewer, more rows than a prefill segment, an image, split KV).
pub fn replayRunnable(f: *Forward, segs: []const MultiSeg) !bool {
    if (f.pf_switch == null or f.prefill_state.mode != .replay or segs.len < 2 or segs.len > 16) return false;
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const spans = try a.alloc(block_prefill.Span, segs.len);
    var all: std.ArrayList(u32) = .empty;
    for (segs, spans) |sg, *sp| {
        sp.* = .{ .slot = sg.slot, .start = @intCast(sg.start), .n = @intCast(sg.ids.len) };
        try all.appendSlice(a, sg.ids);
    }
    if (vision_rows.segmentOptions(f, all.items).image_rows != 0) return false;
    _ = block_prefill.emitMulti(a, f.cfg, f.widths, f.opts, try decoderLayers(f, try f.backbone(a)), spans, .decoder) catch |e| switch (e) {
        error.Unsupported => return false,
        else => return e,
    };
    return true;
}

/// Several slots' CED decoder replays in one run (block_prefill.emitMulti `.decoder`; Python's rounds.py replays its
/// finals a slot at a time): each slot's prefilled tail [R0, end) from its stash, the rows' launches over every tail,
/// each tail's CSA2 steps on its slot. Nothing is committed: each slot stays at its prompt's last token, as `finish`
/// leaves it. `out[j]`: slot j's tail and its rows of "w.taps". A tail batched is its one-slot replay row for row
/// (emitMulti's row floor keeps every tail on the one-slot replay's kernel paths).
pub fn finishMulti(f: *Forward, slots: []const u32, out: []Replayed) !void {
    const st = &f.prefill_state;
    const sw = f.pf_switch orelse return error.Unsupported;
    if (st.mode != .replay or slots.len < 2 or slots.len > 16 or out.len < slots.len) return error.BadWindow;
    // TF_DSV41_PREFILL_PIPE: a replayed slot's line drained first (on its own views)
    if (st.pipe) |ln| if (ln.slot) |ls| if (std.mem.indexOfScalar(u32, slots, ls) != null) {
        try sw.run(sw.ctx, ls);
        try ln.drain(f);
    };
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const segs = try a.alloc(MultiSeg, slots.len);
    const spans = try a.alloc(block_prefill.Span, slots.len);
    const row0 = try a.alloc(usize, slots.len);
    const tails = try a.alloc(Tail, slots.len);
    var all: std.ArrayList(u32) = .empty;
    for (slots, segs, spans, row0, tails) |slot, *sg, *sp, *r0, *t| {
        try sw.run(sw.ctx, slot);
        if (f.slot.pending != null) return error.BadWindow;
        const end = f.slot.pos;
        const tl = ced.tailOf(end);
        if (tl.m == 0) return error.BadWindow;
        if (!f.slot.ced.covers(tl.r0, end)) {
            log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ tl.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
            return error.StashGap;
        }
        const ids = try a.alloc(u32, tl.m);
        f.slot.ced.tail(tl.r0, end, ids);
        sg.* = .{ .slot = slot, .start = tl.r0, .ids = ids };
        sp.* = .{ .slot = slot, .start = @intCast(tl.r0), .n = @intCast(tl.m) };
        r0.* = all.items.len;
        t.len = f.slot.tail_len;
        @memcpy(t.ids[0..t.len], f.slot.tail[0..t.len]);
        try all.appendSlice(a, ids);
    }
    const cs = try block_prefill.emitMulti(a, f.cfg, f.widths, vision_rows.segmentOptions(f, all.items), layers, spans, .decoder);
    try ready(f);
    var m: Multi = .{ .segs = segs, .row0 = row0, .tails = tails, .ids = all.items };
    f.multi = &m;
    defer f.multi = null;
    try enterRun(f, &m);
    st.layers = layers;
    st.moe = 0;
    st.taps_rows = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, segs[0].start, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(m.ids.len), true);
    for (segs, row0, out[0..slots.len]) |sg, r0, *o| {
        try sw.run(sw.ctx, sg.slot);
        f.slot.pending = null;
        o.* = .{ .slot = sg.slot, .at = sg.start, .rows = @intCast(sg.ids.len), .row0 = @intCast(r0) };
    }
    f.last_rows = 0;
    try k4Release(f); // TF_DSV41_PF_4K: these prompts are prefilled
}

/// The glue's view of the whole run (Forward.multi): the first segment's slot, every row's ids, the run's rows pending.
fn enterRun(f: *Forward, m: *const Multi) !void {
    const sw = f.pf_switch.?;
    try sw.run(sw.ctx, m.segs[0].slot);
    f.ids = m.ids;
    f.slot.pending = .{ .start = m.segs[0].start, .n = @intCast(m.ids.len) };
}

/// glue pf_seg: segment j's slot current, its rows pending, its positions (`w.pos`, `w.pos64`, `w.lo`); -1: the run's
/// view again (enterRun).
fn segGlue(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const m = f.multi orelse return error.BadGlue;
    const j = c.args[0].arg.i;
    if (j < 0) return enterRun(f, m);
    const sg = m.segs[@intCast(j)];
    const sw = f.pf_switch.?;
    try sw.run(sw.ctx, sg.slot);
    f.ids = sg.ids;
    f.slot.pending = .{ .start = sg.start, .n = @intCast(sg.ids.len) };
    const k = r.kernels.others(r.stream);
    const lo = try r.tensorAddr(c.args[3].arg.t);
    try k.positions(@intCast(sg.start), @intCast(sg.ids.len), try r.tensorAddr(c.args[1].arg.t), try r.tensorAddr(c.args[2].arg.t), lo);
    // a decoder replay: every row's window starts at the segment's first row (R0)
    if (c.args[4].arg.b) try r.d.check(r.d.api.cuMemsetD32Async(lo, @bitCast(@as(i32, @intCast(sg.start))), sg.ids.len, r.stream.handle), "cuMemsetD32Async");
}

/// glue mpositions: every segment's positions into the run's "w.mpos64" (rope over the run's rows); the window starts
/// and the start word land in the scratch the segments' pf_seg rewrites.
fn runPositions(f: *Forward, r: *run.Runner, c: *const calls.Call) !void {
    const m = f.multi orelse return error.BadGlue;
    const k = r.kernels.others(r.stream);
    const mpos = try r.tensorAddr(c.args[0].arg.t);
    const pos = r.addressOf("w.pos") orelse return error.Unbound;
    const lo = r.addressOf("w.lo") orelse return error.Unbound;
    for (m.segs, m.row0) |sg, r0| try k.positions(@intCast(sg.start), @intCast(sg.ids.len), pos, mpos + 8 * r0, lo);
}

/// CED's decoder replay (replay.finish; Forward.finishPrompt on every rank): layers D .. over the prompt's prefilled
/// tail [R0, end) from the stash, end = the slot's position, every window from R0. Nothing is committed: the slot
/// stays at `end`, the decoder's SWA rings hold [R0, end), and the prompt's last token is the next window's row.
pub fn finish(f: *Forward) !void {
    const st = &f.prefill_state;
    if (st.mode != .replay) return;
    try settle(f); // TF_DSV41_PREFILL_PIPE: the stash and the decoder's KV whole on every rank first
    if (f.slot.pending != null) return error.BadWindow;
    const end = f.slot.pos;
    const t = ced.tailOf(end);
    if (t.m == 0) return;
    if (!f.slot.ced.covers(t.r0, end)) {
        log.err("CED replay of [{d}, {d}), but the stash holds [{d}, {d})", .{ t.r0, end, f.slot.ced.lo(), f.slot.ced.hi });
        return error.StashGap;
    }
    try ready(f);
    var sa = std.heap.ArenaAllocator.init(f.gpa);
    defer sa.deinit();
    const a = sa.allocator();
    const layers = try decoderLayers(f, try f.backbone(a));
    const ids = st.tail_ids[0..t.m];
    f.slot.ced.tail(t.r0, end, ids);
    // the tail's image positions route with gate.bias_vl (replay._image); no decoder layer reads Engram
    const cs = try block_prefill.emitReplay(a, f.cfg, f.widths, vision_rows.segmentOptions(f, ids), layers, t.m, @intCast(t.r0), .decoder);
    f.ids = ids;
    f.slot.pending = .{ .start = t.r0, .n = t.m };
    defer f.slot.pending = null;
    st.layers = layers;
    st.moe = 0;
    f.runner.glue = .{ .ctx = f, .run = glueFn };
    try longpf.window(&st.long, f.gpa, f.runner, f.kv, t.r0, cs);
    if (f.runner.prof) |pr| pr.prefillRows(@intCast(t.m), true);
    f.last_rows = 0;
    // TF_DSV41_PF_4K: the prompt is prefilled, its workspace goes back
    try k4Release(f);
    // the drafter's context rows: the tail's taps (replay.finish's, which slots.finish_prompt hands to `replayed`)
    st.taps_at = t.r0;
    st.taps_rows = t.m;
}

/// ced_stash (`store`): the segment's last min(n, ring) rows of each tensor (the streams, the normed input, the
/// coefficients) into the ring, position p at row p % ring; ced_load: the ring's rows of the pass's positions into the
/// tensors' rows 0 .. n.
fn stashCopy(f: *Forward, r: *run.Runner, c: *const calls.Call, store: bool) !void {
    const p = f.slot.pending orelse return error.BadGlue;
    if (f.pf_stash) |h| try h.run(h.ctx); // the ring holds this slot's rows (slots.zig swaps it lazily)
    const x = c.args;
    if (x.len != 2 * ced.roles.len) return error.BadGlue;
    const k: u64 = @min(p.n, ced.ring);
    const a0: u64 = p.start + p.n - k;
    var sp: [2]ced.Span = undefined;
    for (0..ced.roles.len) |j| {
        const t = x[j].arg.t;
        const rg = x[ced.roles.len + j].arg.t;
        const row = dim(t, 1) * buffers.dtSize(t.dt);
        if (dim(rg, 1) * buffers.dtSize(rg.dt) != row or ld(t) != dim(t, 1) or dim(rg, 0) != ced.ring) return error.BadGlue;
        const ta = try r.tensorAddr(t);
        const ra = try r.tensorAddr(rg);
        for (ced.spans(a0, a0 + k, &sp)) |s| {
            const tr = ta + (a0 - p.start + s.at) * row;
            const rr = ra + @as(u64, s.row) * row;
            if (store) try copy(r, rr, tr, s.n * row) else try copy(r, tr, rr, s.n * row);
        }
    }
}

/// Keep of all the pending segment's rows: the ratio-2 carries, the Engram tail, the position.
fn commit(f: *Forward, n: u32) !void {
    return commitAt(f, n, 0);
}

/// `commit` of a segment whose rows are rows [row0, row0 + n) of the run's projections.
fn commitAt(f: *Forward, n: u32, row0: usize) !void {
    return commitRoles(f, n, row0, "", n);
}

/// `commitAt` with the carries from rows of "w.<pre>L<i>.proj" (TF_DSV41_PF_TBO's B: "b~") whose last row is the
/// `n`th; the slot advances by `adv` rows (f.ids' first `adv`).
fn commitRoles(f: *Forward, n: u32, row0: usize, comptime pre: []const u8, adv: u32) !void {
    const p = f.slot.pending orelse return error.BadWindow;
    const r = f.runner;
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    const row: usize = 4 * 2 * @as(usize, f.cfg.head_dim); // fp32 [kv | gate]
    for (try f.backbone(la.allocator())) |L| if (f.cfg.isKvSource(L) and f.cfg.compressRatio(L) == 2) {
        // a role the 4K segment shares with the forward (K4.keep) is under its bare name
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "w." ++ pre ++ "L{d}.proj", .{L})) orelse r.addressOf(try std.fmt.bufPrint(&nb, "w.L{d}.proj", .{L})) orelse return error.Unbound;
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.carry", .{L})) orelse return error.Unbound;
        try copy(r, dst, src + (row0 + n - 1) * row, row);
    };
    if (f.engram) |h| tailOf(&f.slot, f.ids[0..adv], h.max_ngram - 1);
    f.slot.pos = p.start + adv;
    f.slot.pending = null;
    if (f.sess) |x| try x.commit(x.ptr, f.ids[0..adv]);
}

/// TF_DSV41_PREFILL_PIPE: a segment's host commit at its entry into the line (pipe_run.zig), every rank alike: the Engram
/// tail, the position, the session's committed ids, the stash's ids. Its carries are its stage's own commit and its
/// rows the stages' exchanges; the slot's device state is whole once the line drains (`settle`).
pub fn advanceHost(f: *Forward, start: u64, ids: []const u32) !void {
    if (f.slot.pending != null or f.slot.pos != start) return error.BadWindow;
    if (f.engram) |h| tailOf(&f.slot, ids, h.max_ngram - 1);
    f.slot.pos = start + ids.len;
    if (f.sess) |x| try x.commit(x.ptr, ids);
    f.slot.ced.extend(start, ids);
}

/// TF_DSV41_PREFILL_PIPE: the active slot's line drained before its state is read (the CED finish, a snapshot), on every
/// rank at the same op (each runs this inside the op the leader sent).
pub fn settle(f: *Forward) !void {
    const ln = f.prefill_state.pipe orelse return;
    const s = ln.slot orelse return;
    if (s == pipe_run.activeSlot(f)) try ln.drain(f);
}

/// TF_DSV41_PREFILL_PIPE: the active slot is released (or reset): its segments in the line are dropped.
pub fn pipeRelease(f: *Forward) void {
    const ln = f.prefill_state.pipe orelse return;
    const s = ln.slot orelse return;
    if (s == pipe_run.activeSlot(f)) ln.abandon();
}

/// A prefill segment's glue step (the pipeline's stage programs hand theirs here, pipe_run.stageGlue).
pub fn glueStep(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    return glueOne(ctx, r, c);
}

/// Engram's tail after committing `ids`: (tail + ids)[-keep:].
fn tailOf(s: *fwd.Slot, ids: []const u32, keep: usize) void {
    if (ids.len >= keep) {
        @memcpy(s.tail[0..keep], ids[ids.len - keep ..]);
        s.tail_len = keep;
        return;
    }
    const old = @min(s.tail_len, keep - ids.len); // the old tail's last ones that stay
    std.mem.copyForwards(u32, s.tail[0..old], s.tail[s.tail_len - old .. s.tail_len]);
    @memcpy(s.tail[old .. old + ids.len], ids);
    s.tail_len = old + ids.len;
}

/// Once: the tables no loader form builds and the buffers a segment's exchanges / Engram rows need.
fn ready(f: *Forward) !void {
    const st = &f.prefill_state;
    if (st.ready) return;
    const r = f.runner;
    // exl3.prefill's Hadamard: H[i, j] = (-1)^popcount(i & j), bf16
    if (r.addressOf("s.pf.had")) |p| {
        const h = try f.gpa.alloc(u16, 128 * 128);
        defer f.gpa.free(h);
        for (0..128) |i| for (0..128) |j| {
            h[i * 128 + j] = if (@popCount(i & j) & 1 == 1) 0xBF80 else 0x3F80;
        };
        try cuda.DeviceBuffer.upload(.{ .d = r.d, .ptr = p, .len = h.len * 2 }, 0, std.mem.sliceAsBytes(h));
    }
    // x3gm's pointer tables: the routed experts' trellis addresses (forms' tables hold the shared expert's last)
    var nb: [64]u8 = undefined;
    var la = std.heap.ArenaAllocator.init(f.gpa);
    defer la.deinit();
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "tg", "tp_g" }, .{ "tu", "tp_u" }, .{ "td", "tp_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoExpertTable;
        try copy(r, dst, src, 8 * @as(usize, f.cfg.expertsOf(L).count));
    };
    // x3gm v2's width tables (gm2pf.zig, x3gm.Ragged.k2g / k2d): the routed experts' K2s, then a 0 sentinel
    for (try f.backbone(la.allocator())) |L| for ([_][2][]const u8{ .{ "k2g", "k2_g" }, .{ "k2d", "k2_d" } }) |t| {
        const dst = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.gm.{s}", .{ L, t[0] })) orelse continue;
        const src = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.{s}", .{ L, t[1] })) orelse return error.NoK2Table;
        const Et: usize = f.cfg.expertsOf(L).count;
        try copy(r, dst, src, 4 * Et);
        try zero(r, dst + 4 * Et, 4);
    };
    // a segment's exchange send and Engram rows at their largest (decode's would grow them on the first segment)
    const rows: usize = @intCast(f.opts.prefill_rows);
    const W: usize = f.comm.world();
    _ = try f.sendBuffer(r, 2 * rows * f.cfg.hidden);
    if (f.engram) |h| {
        const cols = h.headShard(f.comm.rank(), f.comm.world())[1] * @as(usize, f.cfg.engram_head_dim);
        _ = try grow(&f.gathered, r.d, W * 2 * rows * cols);
        for (0..h.layer_ids.len) |li| _ = try f.rows_stage[li].take(r.d, 2 * rows * cols);
    }
    _ = try f.ids_stage.take(r.d, 8 * rows);
    try r.stream.synchronize();
    st.ready = true;
}

fn grow(b: *?cuda.DeviceBuffer, d: *const cuda.Driver, bytes: usize) !u64 {
    if (b.* == null or b.*.?.len < bytes) {
        if (b.*) |*x| x.free();
        b.* = try cuda.DeviceBuffer.alloc(d, @max(bytes, 256));
    }
    return b.*.?.ptr;
}

fn copy(r: *run.Runner, dst: u64, src: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(dst, src, bytes, r.stream.handle), "cuMemcpyDtoDAsync");
}

fn zero(r: *run.Runner, dst: u64, bytes: usize) !void {
    if (bytes == 0) return;
    try r.d.check(r.d.api.cuMemsetD8Async(dst, 0, bytes, r.stream.handle), "cuMemsetD8Async");
}

fn numel(t: calls.Tensor) usize {
    var n: usize = 1;
    for (t.shape) |x| n *= @intCast(x);
    return n;
}

fn bytesOf(t: calls.Tensor) usize {
    return numel(t) * buffers.dtSize(t.dt);
}

fn dim(t: calls.Tensor, i: usize) usize {
    return @intCast(t.shape[i]);
}

fn ld(t: calls.Tensor) usize {
    return @intCast(t.stride[0]);
}

/// A prefill segment's glue steps (block_prefill.zig's module doc); the ones a decode window has too (positions,
/// embed, exchange, kit_logits, engram_rows, trace, and the materialised selection's topk / counts / block_pos) are the
/// decode glue's.
fn glueFn(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = &f.prefill_state;
    const pr = st.tbo orelse return glueOne(ctx, r, c);
    // TF_DSV41_PF_TBO: the step's micro-batch (B's are the side stream's) on the forward while it runs
    const m = if (c.side) &pr.b else &pr.a;
    const keep_ids = f.ids;
    const keep_pending = f.slot.pending;
    var keep_tail: Tail = .{ .len = f.slot.tail_len };
    @memcpy(keep_tail.ids[0..keep_tail.len], f.slot.tail[0..f.slot.tail_len]);
    f.ids = m.ids;
    f.slot.pending = m.pending;
    f.slot.tail_len = m.tail.len;
    @memcpy(f.slot.tail[0..m.tail.len], m.tail.items());
    st.moe = m.moe;
    st.layer = m.layer;
    if (c.side) std.mem.swap(longpf.State, &st.long, &st.long_b);
    defer {
        if (c.side) std.mem.swap(longpf.State, &st.long, &st.long_b);
        m.moe = st.moe;
        m.layer = st.layer;
        f.ids = keep_ids;
        f.slot.pending = keep_pending;
        f.slot.tail_len = keep_tail.len;
        @memcpy(f.slot.tail[0..keep_tail.len], keep_tail.items());
    }
    return glueOne(ctx, r, c);
}

fn glueOne(ctx: *anyopaque, r: *run.Runner, c: *const calls.Call) anyerror!void {
    const f: *Forward = @ptrCast(@alignCast(ctx));
    const st = &f.prefill_state;
    const step = c.name["glue.".len..];
    const k = r.kernels.others(r.stream);
    const x = c.args;
    const eql = std.mem.eql;
    if (std.mem.startsWith(u8, step, "kx_")) return (f.kv orelse return error.BadGlue).glue(r, c, step);
    if (eql(u8, step, "proj_carry")) {
        // torch.cat([the slot's carry, the projection]): position start - 1's row, then the segment's
        const carry = x[0].arg.t;
        const buf = try r.tensorAddr(x[2].arg.t);
        try copy(r, buf, try r.tensorAddr(carry), bytesOf(carry));
        return copy(r, buf + bytesOf(carry), try r.tensorAddr(x[1].arg.t), bytesOf(x[1].arg.t));
    }
    if (eql(u8, step, "stream_merge")) return longpf.merge(&st.long, f.gpa, r, c);
    // TF_DSV41_PF_TBO: A's ratio-2 carry into the slot at its layer's end, and the pair's closing join (Runner.issue
    // made it: nothing to run)
    if (eql(u8, step, "copy_rows")) return copy(r, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "tbo_join")) return;
    if (eql(u8, step, "pf_seg")) return segGlue(f, r, c);
    if (eql(u8, step, "mpositions")) return runPositions(f, r, c);
    if (eql(u8, step, "ced_stash") or eql(u8, step, "ced_load")) return stashCopy(f, r, c, eql(u8, step, "ced_stash"));
    if (eql(u8, step, "zeros")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    if (eql(u8, step, "image_keep")) return vision_rows.keep(f, r, c);
    if (eql(u8, step, "image_split") or eql(u8, step, "image_merge")) return vision_rows.splitRows(f, r, c, eql(u8, step, "image_merge"));
    if (eql(u8, step, "gm_ticket_again")) return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t)); // vision.moe's text call: the same layer
    if (eql(u8, step, "cand_keys")) {
        const blocks = x[0].arg.t;
        return k.candKeys(try r.tensorAddr(blocks), ld(blocks), dim(blocks, 0), dim(blocks, 1), @intCast(x[2].arg.i), try r.tensorAddr(x[1].arg.t));
    }
    if (eql(u8, step, "proj")) {
        // blocks._projection: f32(kv), or ratio 2 [f32(kv) | f32(gate)]
        const kv = x[0].arg.t;
        const gate: u64 = if (x.len == 3) try r.tensorAddr(x[1].arg.t) else 0;
        return k.widenCat(try r.tensorAddr(kv), gate, dim(kv, 0), dim(kv, 1), try r.tensorAddr(x[x.len - 1].arg.t));
    }
    if (eql(u8, step, "ix_w")) return k.f64Bf16F32(try r.tensorAddr(x[0].arg.t), try r.tensorAddr(x[1].arg.t), numel(x[0].arg.t));
    if (eql(u8, step, "stage_in") or eql(u8, step, "stage_out")) {
        // blocks.stage_rows(dst, dst_ring, src, src_ring, lo, hi), a tensor at a time (values, scales)
        const dst_ring: usize = @intCast(x[2].arg.i);
        const src_ring: usize = @intCast(x[5].arg.i);
        const lo: u64 = @intCast(x[6].arg.i);
        const hi: u64 = @intCast(x[7].arg.i);
        for (0..2) |j| {
            const dst = x[j].arg.t;
            const src = x[3 + j].arg.t;
            const row = dim(dst, 1) * buffers.dtSize(dst.dt);
            if (dim(src, 1) * buffers.dtSize(src.dt) != row) return error.BadGlue;
            try k.ringCopy(try r.tensorAddr(src), src_ring, try r.tensorAddr(dst), dst_ring, row, lo, hi);
        }
        return;
    }
    if (eql(u8, step, "gm_picks")) {
        const pick = x[0].arg.t;
        return k.gmPicks(try r.tensorAddr(pick), try r.tensorAddr(x[1].arg.t), dim(pick, 1), dim(x[2].arg.t, 1), dim(pick, 0), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t));
    }
    if (eql(u8, step, "gm_ticket")) {
        // a layer's MoE starts here: the layer gm_plan's K2 tables are of
        if (st.moe >= st.layers.len) return error.BadGlue;
        st.layer = st.layers[st.moe];
        st.moe += 1;
        return zero(r, try r.tensorAddr(x[0].arg.t), bytesOf(x[0].arg.t));
    }
    if (eql(u8, step, "gm_plan")) {
        // x3gm._run_ragged: the picks of one width (the others past the table), then x3gm.plan's passes
        const pk = x[0].arg.t;
        const P = numel(pk);
        const k2: u32 = @intCast(x[6].arg.i);
        const down = x[7].arg.b;
        const bm: usize = @intCast(x[8].arg.i);
        const Et: usize = @intCast(x[9].arg.i);
        // k2 0 (x3gm._run_v2's one plan, gm2pf.zig): every pick, no width mask
        if (k2 == 0) return r.kernels.experts(r.stream).gmPlan(try r.tensorAddr(pk), P, Et, bm, try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t));
        var nb: [64]u8 = undefined;
        // forms' "s.L<i>.ex.k2_{g,d}": each routed expert's K2 (gate = up), the shared expert's past Et (never read)
        const tab = r.addressOf(try std.fmt.bufPrint(&nb, "s.L{d}.ex.k2_{s}", .{ st.layer, if (down) "d" else "g" })) orelse return error.NoK2Table;
        const masked = try grow(&st.masked, r.d, 4 * P);
        try k.widthMask(try r.tensorAddr(pk), P, tab, Et, k2, masked);
        const t = [5]u64{ try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), try r.tensorAddr(x[3].arg.t), try r.tensorAddr(x[4].arg.t), try r.tensorAddr(x[5].arg.t) };
        return r.kernels.experts(r.stream).gmPlan(masked, P, Et, bm, t[0], t[1], t[2], t[3], t[4]);
    }
    if (eql(u8, step, "swiglu")) {
        // the shared expert's silu(min(g, limit)) * clamp(u, +-limit), fp32
        const g = x[0].arg.t;
        const u = x[1].arg.t;
        const act = x[2].arg.t;
        return k.swiglu(try r.tensorAddr(g), ld(g), try r.tensorAddr(u), ld(u), try r.tensorAddr(act), ld(act), dim(g, 0), dim(g, 1), @floatCast(x[3].arg.f));
    }
    if (eql(u8, step, "moe_sum")) {
        // bf16(routed + shared): one RNE of the fp32 sum; no shared expert: the routed partial's cast
        const routed = x[0].arg.t;
        if (x.len == 3) return k.addBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), try r.tensorAddr(x[2].arg.t), numel(routed));
        return k.castBf16(try r.tensorAddr(routed), try r.tensorAddr(x[1].arg.t), numel(routed));
    }
    return Forward.glueFn(ctx, r, c);
}
