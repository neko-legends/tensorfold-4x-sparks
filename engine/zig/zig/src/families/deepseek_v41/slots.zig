//! Several live slots on the GPU path: each slot's bounded state stacked a role, one slot activated through views.

const std = @import("std");
const cuda = @import("cuda");
const fwd = @import("forward.zig");
const buffers = @import("buffers.zig");
const rowmode = @import("rowmode.zig");

pub const Error = error{ RowsNeedPool, SlotRange, NotBound };

/// Plan-link operations of this workstream (docs: 16-19).
pub const op_select: i64 = 16;
pub const op_rows: i64 = 17;
pub const op_rows_keep: i64 = 18;
pub const op_rows_drop: i64 = 19;

/// TF_DSV41_SLOTS: live slots (1: today's one-slot path, nothing stacked). Every rank must set the same.
pub fn countFromEnv() !u32 {
    const v = std.c.getenv("TF_DSV41_SLOTS") orelse return 1;
    const n = try std.fmt.parseInt(u32, std.mem.span(v), 10);
    if (n < 1 or n > 16) return error.SlotRange;
    return n;
}

const Stacked = struct { base: u64, bytes: u64 };

pub const SlotSet = struct {
    gpa: std.mem.Allocator,
    f: *fwd.Forward,
    n: u32,
    states: []fwd.Slot,
    transactions: []@import("cuda_transaction.zig").Transaction,
    /// each cache-transaction region once for every slot: the stacked backup (slot s at s x stride), see beginWindow
    stack: std.ArrayList(Stack) = .empty,
    /// the slot whose views are bound (its state lives in f.slot meanwhile); null: the stacked bases
    active: ?u32 = null,
    roles: std.StringArrayHashMapUnmanaged(Stacked) = .empty,
    /// roles one buffer holds for one slot at a time, each slot's copy aside (`swap`); `owner`: whose bytes are in
    swapped: std.ArrayList(Swapped) = .empty,
    owner: ?u32 = null,

    const Swapped = struct { addr: u64, len: u64, aside: cuda.DeviceBuffer };
    const Stack = struct { live: u64, stride: u64, bytes: usize, backup: cuda.DeviceBuffer, row_bytes: usize, rows: u32, carry: bool };

    pub fn init(gpa: std.mem.Allocator, f: *fwd.Forward, n: u32) !*SlotSet {
        if (f.kv == null) return error.RowsNeedPool;
        if (f.kv.?.slots.len != n) return error.SlotRange;
        const ss = try gpa.create(SlotSet);
        errdefer gpa.destroy(ss);
        const states = try gpa.alloc(fwd.Slot, n);
        errdefer gpa.free(states);
        @memset(states, .{});
        const transactions = try gpa.alloc(@import("cuda_transaction.zig").Transaction, n);
        @memset(transactions, .{});
        ss.* = .{ .gpa = gpa, .f = f, .n = n, .states = states, .transactions = transactions };
        f.pf_switch = .{ .ctx = ss, .run = viewFn };
        f.pf_stash = .{ .ctx = ss, .run = stashFn };
        return ss;
    }

    pub fn deinit(ss: *SlotSet) void {
        for (ss.stack.items) |*x| x.backup.free();
        ss.stack.deinit(ss.gpa);
        for (ss.transactions) |*x| x.deinit(ss.gpa);
        ss.gpa.free(ss.transactions);
        for (ss.swapped.items) |*x| x.aside.free();
        ss.swapped.deinit(ss.gpa);
        ss.roles.deinit(ss.gpa);
        ss.gpa.free(ss.states);
        ss.gpa.destroy(ss);
    }

    /// After Runner.bind (and Kv.bind): each stacked role's base and a slot's bytes. The roles stay at their bases.
    pub fn bindBases(ss: *SlotSet, plan: *const buffers.Plan) !void {
        const r = ss.f.runner;
        for (plan.sizes.keys(), plan.sizes.values()) |role, size| {
            if (!(rowmode.isRing(role) or rowmode.isCarry(role)) or !ss.backbone(role)) continue;
            if (size % ss.n != 0) return error.NotBound;
            try ss.roles.put(ss.gpa, role, .{ .base = r.addressOf(role) orelse return error.NotBound, .bytes = size / ss.n });
        }
        const k = ss.f.kv.?;
        const tb: u64 = 4 * @as(u64, k.dev.pts);
        try ss.roles.put(ss.gpa, "s.kv.pt", .{ .base = k.dev.table.ptr, .bytes = tb });
        try ss.roles.put(ss.gpa, "s.kv.ct", .{ .base = k.dev.local.ptr, .bytes = tb });
        // point() binds a slot's view at base + s x bytes through Runner.external (16-byte aligned, Triton's
        // pointer specialization): a bad stride fails here at boot, named, not at the first request on slot 1
        for (ss.roles.keys(), ss.roles.values()) |role, st| {
            if (st.bytes % 16 != 0) {
                std.log.scoped(.dsv41).err("slots: {s}: a slot's {d} B stride is not 16-byte aligned", .{ role, st.bytes });
                return error.NotBound;
            }
        }
    }

    /// After Runner.bind: `roles` (planned ones) become per-slot by swapping, each slot's copy in its own buffer.
    pub fn swap(ss: *SlotSet, plan: *const buffers.Plan, roles: []const []const u8) !void {
        const r = ss.f.runner;
        for (roles) |role| {
            const len = plan.sizes.get(role) orelse continue;
            const addr = r.addressOf(role) orelse return error.NotBound;
            try ss.swapped.ensureUnusedCapacity(ss.gpa, 1);
            ss.swapped.appendAssumeCapacity(.{ .addr = addr, .len = len, .aside = try cuda.DeviceBuffer.alloc(r.d, len * ss.n) });
        }
    }

    /// The swapped roles hold slot s's bytes (the owner's set aside first), on the runner's stream.
    fn swapIn(ss: *SlotSet, s: u32) !void {
        if (ss.swapped.items.len == 0 or ss.owner == s) return;
        const r = ss.f.runner;
        for (ss.swapped.items) |x| {
            if (ss.owner) |o| try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.aside.ptr + o * x.len, x.addr, x.len, r.stream.handle), "cuMemcpyDtoDAsync");
            try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.addr, x.aside.ptr + s * x.len, x.len, r.stream.handle), "cuMemcpyDtoDAsync");
        }
        ss.owner = s;
    }

    /// "s.L<i>.…" of a backbone layer (DSpark's blocks keep their own one-slot rings).
    fn backbone(ss: *const SlotSet, role: []const u8) bool {
        const rest = role["s.L".len..];
        const dot = std.mem.indexOfScalar(u8, rest, '.') orelse return false;
        const L = std.fmt.parseInt(u32, rest[0..dot], 10) catch return false;
        return L < ss.f.cfg.layers;
    }

    /// A stacked role's base (the row programs' view), e.g. "s.L2.carry".
    pub fn base(ss: *const SlotSet, role: []const u8) ?Stacked {
        return ss.roles.get(role);
    }

    pub fn leads(ss: *const SlotSet) bool {
        const f = ss.f;
        return f.link != null and !f.quiet and f.comm.rank() == 0;
    }

    /// Slot s for one-slot operations on every rank (the leader sends op_select first).
    pub fn activate(ss: *SlotSet, s: u32) !void {
        if (s >= ss.n) return error.SlotRange;
        if (ss.leads()) try ss.f.link.?.send(&.{ op_select, s });
        try ss.swapIn(s);
        if (ss.active == s) return;
        ss.store();
        ss.f.slot = ss.states[s];
        ss.f.transaction = ss.transactions[s];
        ss.transactions[s] = .{};
        ss.active = s;
        try ss.point(s);
    }

    /// Slot s's views and state current on this rank alone (no op_select); swapped roles stay with their owner.
    pub fn view(ss: *SlotSet, s: u32) !void {
        if (s >= ss.n) return error.SlotRange;
        if (ss.active == s) return;
        ss.store();
        ss.f.slot = ss.states[s];
        ss.f.transaction = ss.transactions[s];
        ss.transactions[s] = .{};
        ss.active = s;
        try ss.point(s);
    }

    fn viewFn(ctx: *anyopaque, s: u32) anyerror!void {
        const ss: *SlotSet = @ptrCast(@alignCast(ctx));
        return ss.view(s);
    }

    /// Forward.pf_stash: the swapped roles hold the active slot's bytes (the owner's set aside first).
    fn stashFn(ctx: *anyopaque) anyerror!void {
        const ss: *SlotSet = @ptrCast(@alignCast(ctx));
        if (ss.active) |s| try ss.swapIn(s);
    }

    /// The stacked bases again (row mode), the active slot's state kept.
    pub fn deactivate(ss: *SlotSet) !void {
        if (ss.active == null) return;
        ss.store();
        ss.active = null;
        ss.f.slot = .{};
        for (ss.roles.keys(), ss.roles.values()) |role, st| try ss.f.runner.external(role, st.base);
    }

    /// Every slot of a window opens its cache transaction in one pass over the stacked rings: the backup of a role is
    /// one buffer for all slots (slot s at s x stride), so the slots' copies of a role are one memcpy per run of slots
    /// (not one per slot and region: ~130 driver calls a slot, ~1,300 an 8-slot window, 3-4 ms ahead of every
    /// window). Same bytes in the same places as Transaction.begin; keep/rollback use the same regions. Call with no
    /// slot active (after deactivate).
    pub fn beginWindow(ss: *SlotSet, mix: anytype) !void {
        const f = ss.f;
        if (ss.active != null) return error.SlotRange;
        if (ss.stack.items.len == 0) try ss.buildStack();
        var mask: u64 = 0;
        var lo: u32 = ss.n;
        var hi: u32 = 0;
        for (mix) |m| {
            if (m.slot >= ss.n or ss.n > 64) return error.SlotRange;
            const t = &ss.transactions[m.slot];
            if (t.state.pending != null) return error.CacheTransactionActive;
            if (m.rows == 0 or m.rows > 64) return error.BadWindow;
            mask |= @as(u64, 1) << @intCast(m.slot);
            lo = @min(lo, m.slot);
            hi = @max(hi, m.slot);
        }
        if (mask == 0) return;
        // one span lo..hi when no slot inside it keeps a pending window of its own backup (an idle slot's backup is free)
        var span = true;
        var s: u32 = lo;
        while (s <= hi) : (s += 1) {
            if (mask & (@as(u64, 1) << @intCast(s)) == 0 and ss.transactions[s].state.pending != null) span = false;
        }
        const r = f.runner;
        for (ss.stack.items) |x| {
            var a: u32 = lo;
            while (a <= hi) {
                if (!span and mask & (@as(u64, 1) << @intCast(a)) == 0) {
                    a += 1;
                    continue;
                }
                var b: u32 = a;
                while (b + 1 <= hi and (span or mask & (@as(u64, 1) << @intCast(b + 1)) != 0)) b += 1;
                const off = @as(u64, a) * x.stride;
                const len = @as(usize, b - a) * x.stride + x.bytes;
                try r.d.check(r.d.api.cuMemcpyDtoDAsync_v2(x.backup.ptr + off, x.live + off, len, r.stream.handle), "cuMemcpyDtoDAsync");
                a = b + 1;
            }
        }
        for (mix) |m| {
            const t = &ss.transactions[m.slot];
            if (!t.shared) try ss.bindTransaction(m.slot);
            t.state.position = ss.states[m.slot].pos;
            t.state.pending = m.rows;
        }
    }

    fn buildStack(ss: *SlotSet) !void {
        const f = ss.f;
        var nb: [64]u8 = undefined;
        var la = std.heap.ArenaAllocator.init(ss.gpa);
        defer la.deinit();
        for (try f.backbone(la.allocator())) |L| {
            for ([_][]const u8{ "swa.v", "swa.s", "carry" }) |part| {
                const role = try std.fmt.bufPrint(&nb, "s.L{d}.{s}", .{ L, part });
                const st = ss.roles.get(role) orelse continue;
                const carry = std.mem.eql(u8, part, "carry");
                const row: usize = if (carry) 8 * @as(usize, f.cfg.head_dim) else if (std.mem.endsWith(u8, part, ".v")) 576 else 8;
                const rows: u32 = if (carry) 1 else 2 * f.cfg.window;
                if (row * rows > st.bytes) return error.TransactionShapeChanged;
                var buf = try cuda.DeviceBuffer.alloc(f.runner.d, @as(usize, @intCast(st.bytes)) * ss.n);
                errdefer buf.free();
                try ss.stack.append(ss.gpa, .{ .live = st.base, .stride = st.bytes, .bytes = row * rows, .backup = buf, .row_bytes = row, .rows = rows, .carry = carry });
            }
        }
    }

    /// Slot s's transaction regions over the stacked live rings and backups (what Transaction.begin builds per call).
    fn bindTransaction(ss: *SlotSet, s: u32) !void {
        const t = &ss.transactions[s];
        t.runner = ss.f.runner;
        t.regions.clearRetainingCapacity();
        for (ss.stack.items) |x| {
            try t.regions.append(ss.gpa, .{
                .live = x.live + @as(u64, s) * x.stride,
                .backup = x.backup.ptr + @as(u64, s) * x.stride,
                .row_bytes = x.row_bytes,
                .rows = x.rows,
                .kind = if (x.carry) .carry else .ring,
                .encoding = if (x.carry) .f32 else .cuda_fp8_584,
            });
        }
        t.shared = true;
    }

    /// A slot's state: the forward's own while it is active, else the stored one.
    pub fn state(ss: *SlotSet, s: u32) *fwd.Slot {
        return if (ss.active == s) &ss.f.slot else &ss.states[s];
    }

    pub fn transaction(ss: *SlotSet, s: u32) *@import("cuda_transaction.zig").Transaction {
        return if (ss.active == s) &ss.f.transaction else &ss.transactions[s];
    }

    fn store(ss: *SlotSet) void {
        if (ss.active) |a| {
            ss.states[a] = ss.f.slot;
            ss.transactions[a] = ss.f.transaction;
            ss.f.transaction = .{};
        }
    }

    fn point(ss: *SlotSet, s: u32) !void {
        for (ss.roles.keys(), ss.roles.values()) |role, st| try ss.f.runner.external(role, st.base + s * st.bytes);
        ss.f.kv.?.select(s);
    }
};
