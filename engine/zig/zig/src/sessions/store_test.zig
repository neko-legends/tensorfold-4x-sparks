//! The session store on a host pool and a real NVMe tier: chats, evictions, resumes bit for bit, deltas, two ranks.

const std = @import("std");
const testing = std.testing;
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Slot = pool_mod.Slot;
const Family = pool_mod.Family;
const HostPool = @import("pagestore.zig").HostPool;
const disk_mod = @import("disk.zig");
const store_mod = @import("store.zig");
const Store = store_mod.Store;
const Bounded = store_mod.Bounded;

const page = 64;
const fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584, .split = true },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584, .split = true },
};

/// Bounded state in host memory: one blob naming its chat and length.
const HostBounded = struct {
    gpa: std.mem.Allocator,
    bytes: []u8,

    fn make(gpa: std.mem.Allocator, chat: u32, pos: u64) !Bounded {
        const h = try gpa.create(HostBounded);
        h.* = .{ .gpa = gpa, .bytes = try gpa.alloc(u8, 3000) };
        for (h.bytes, 0..) |*b, i| b.* = @truncate(i +% chat *% 17 +% pos);
        return .{ .ptr = h, .vtable = &vt };
    }
    const vt: Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .pieces = pieces, .release = release };
    fn pieces(_: *anyopaque) u32 {
        return 1;
    }
    fn nbytes(p: *anyopaque) u64 {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        return h.bytes.len;
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]disk_mod.Blob {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        const out = try gpa.alloc(disk_mod.Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "rings"), .bytes = try gpa.dupe(u8, h.bytes) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        h.gpa.free(h.bytes);
        h.gpa.destroy(h);
    }
};

/// One rank: its pool, host pages and store, plus what each chat's slot wrote (the reference rows by logical page).
const Rank = struct {
    pool: Pool,
    host: HostPool,
    disk: *disk_mod.Disk,
    store: Store,

    fn init(r: *Rank, gpa: std.mem.Allocator, root: []const u8, world: u32, rank: u32, policy: store_mod.Policy) !void {
        r.pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 96 * page, world, rank);
        r.host = try HostPool.init(gpa, &r.pool);
        r.disk = try disk_mod.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(7), .rank = rank, .world = world, .chunk = 64 << 10, .staging = 4, .lanes = 2, .min_tokens = 1 });
        r.store = try Store.init(gpa, testing.io, &r.pool, r.host.store(), r.disk, policy);
    }

    fn deinit(r: *Rank) void {
        r.store.deinit();
        r.disk.close();
        r.host.deinit();
        r.pool.deinit();
    }

    /// The rows a chat's slot writes at a logical page: a function of (chat, logical page, family) alone.
    fn write(r: *Rank, s: *const Slot, chat: u32, from_page: u32) void {
        for (s.mapped()[from_page..], from_page..) |pg, k| for (fams, 0..) |f, fi| {
            if (f.split and !r.pool.owns(pg)) continue;
            const v = r.host.pageOf(@intCast(fi), r.pool.familyPage(f, pg));
            for (v, 0..) |*b, i| b.* = @truncate(i *% 13 +% k *% 101 +% chat *% 7 +% fi);
        };
    }

    /// Every stored row of the slot's pages below pos equals what write() put there.
    fn check(r: *Rank, s: *const Slot, chat: u32, pos: u64) !void {
        const n = pool_mod.pagesFor(pos, page);
        for (s.mapped()[0..n], 0..) |pg, k| for (fams, 0..) |f, fi| {
            if (f.split and !r.pool.owns(pg)) continue;
            const v = r.host.pageOf(@intCast(fi), r.pool.familyPage(f, pg));
            const rows: usize = if (k + 1 == n and pos % page != 0) (pos % page) / f.ratio * f.row_bytes else v.len;
            for (v[0..rows], 0..) |b, i| try testing.expectEqual(@as(u8, @truncate(i *% 13 +% k *% 101 +% chat *% 7 +% fi)), b);
        };
    }
};

fn chatIds(gpa: std.mem.Allocator, chat: u32, n: usize) ![]i32 {
    const v = try gpa.alloc(i32, n);
    for (v, 0..) |*t, i| t.* = @intCast(chat * 100_000 + i);
    return v;
}

/// Prefill or resume a chat to len(ids) in slot s, save it, free the slot, enforce residency; returns the resume point.
fn turn(gpa: std.mem.Allocator, r: *Rank, s: *Slot, chat: u32, ids: []const i32) ![]const u8 {
    var tier: []const u8 = "fresh";
    var from: u32 = 0;
    if (try r.store.find(1, ids)) |hit| {
        if (hit.ram) {
            const b = try r.store.restoreRam(hit.id, s);
            try testing.expectEqual(@as(u64, 3000), b.vtable.nbytes(b.ptr));
            tier = "ram";
        } else {
            const j = try r.store.beginRestore(hit.id, s);
            var got = (try r.store.settle(j)).?;
            defer got.deinit(gpa);
            try testing.expectEqual(@as(u8, @truncate(chat *% 17 +% hit.pos)), got.blobs[0].bytes[0]);
            // the NVMe restore yields the entry's committed ids (its files hold them), whole file or delta chain
            try testing.expectEqualSlices(i32, ids[0..@intCast(hit.pos)], got.ids);
            tier = "disk";
        }
        try r.check(s, chat, hit.pos);
        from = @intCast(hit.pos / page);
    }
    try s.ensure(ids.len);
    r.write(s, chat, from);
    _ = try r.store.save(s, try HostBounded.make(gpa, chat, ids.len), ids, 1, 0);
    _ = try s.releaseAll();
    _ = try r.store.enforce();
    try r.store.settleAll(); // the plan's settle point (async parks finish here)
    return tier;
}

/// The scenario on one rank; returns a trace of its decisions (stats and page ids) to compare ranks.
fn scenario(gpa: std.mem.Allocator, root: []const u8, world: u32, rank: u32, async_io: bool, trace: *std.ArrayList(u64)) !void {
    var r: Rank = undefined;
    try r.init(gpa, root, world, rank, .{ .chats = 1, .async_io = async_io });
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    const b = try chatIds(gpa, 2, 20 * page);
    defer gpa.free(b);
    const st = &r.store.stats;
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]));
    try testing.expectEqualStrings("ram", try turn(gpa, &r, s, 1, a[0 .. 8 * page + 3]));
    try testing.expectEqual(@as(u64, 0), st.parks);
    // a second idle chat: chat 1 (least recent) parked whole, its older turn dropped (not written)
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 2, b[0 .. 6 * page]));
    try testing.expectEqual(@as(u64, 1), st.parks);
    try testing.expectEqual(@as(u64, 1), st.drops);
    try testing.expectEqual(@as(u32, 1), r.store.resident().chats);
    const whole = r.disk.used;
    // chat 1 resumes from NVMe bit for bit; chat 2 goes
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 1, a[0 .. 12 * page + 40]));
    try testing.expectEqual(@as(u64, 2), st.parks);
    const before = r.disk.used;
    // chat 2 resumes; chat 1 goes again, written as a delta on its first file (pages 8..12 of 13)
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 2, b[0 .. 9 * page]));
    try testing.expectEqual(@as(u64, 3), st.parks);
    try testing.expectEqual(@as(u64, 1), st.delta_parks);
    // the delta holds 5 of the entry's 13 pages: less than chat 1's first whole file of 9 pages
    try testing.expect(r.disk.used - before < whole);
    // chat 1's newest turn restores through its two files, bit for bit
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 1, a[0 .. 15 * page]));
    try testing.expectEqual(@as(u64, 3), st.hits_disk);
    try trace.appendSlice(gpa, &.{ st.parks, st.delta_parks, st.drops, st.covered_drops, st.hits_ram, st.hits_disk, r.pool.freePages(), r.pool.entry_only });
    var id = r.store.oldest;
    while (id != store_mod.none) : (id = r.store.entry(id).extra.newer) for (r.store.entry(id).extra.pages) |pg| try trace.append(gpa, pg);
}

/// After the scenario: a new process on the same tier finds every file and resumes chat 1 through its delta chain.
fn restart(gpa: std.mem.Allocator, root: []const u8, world: u32, rank: u32) !void {
    var r: Rank = undefined;
    try r.init(gpa, root, world, rank, .{ .chats = 1 });
    defer r.deinit();
    try testing.expectEqual(@as(u32, 4), try r.store.reconcile());
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    try testing.expectEqualStrings("disk", try turn(gpa, &r, s, 1, a[0 .. 16 * page]));
    try testing.expectEqual(@as(u64, 1), r.store.stats.hits_disk);
}

fn tmpRoot(gpa: std.mem.Allocator, tmp: *std.testing.TmpDir, sub: []const u8) ![:0]u8 {
    try tmp.dir.createDirPath(testing.io, sub);
    return tmp.dir.realPathFileAlloc(testing.io, sub, gpa);
}

test "chats over turns: residency to NVMe, bit-identical resumes, deltas, older turns dropped (sync and async)" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var t0: std.ArrayList(u64) = .empty;
    defer t0.deinit(gpa);
    var t1: std.ArrayList(u64) = .empty;
    defer t1.deinit(gpa);
    const r0 = try tmpRoot(gpa, &tmp, "sync");
    defer gpa.free(r0);
    try scenario(gpa, r0, 1, 0, false, &t0);
    try restart(gpa, r0, 1, 0);
    const r1 = try tmpRoot(gpa, &tmp, "async");
    defer gpa.free(r1);
    try scenario(gpa, r1, 1, 0, true, &t1);
    try testing.expectEqualSlices(u64, t0.items, t1.items); // the tier thread changes when bytes move, not what is decided
}

test "an evicted (parked) entry's index path is trimmed; its NVMe restore returns the committed ids" {
    // the 2026-10-10 w4b fleet crash: a repeated prompt whose session was RAM-evicted raised NotInRam out of
    // sessions_gpu.restore()'s index.idsOf tail on every rank (a parked entry keeps digests only, by design),
    // and the followers' replay made the request-level error process-fatal. The index behaviour stays; the
    // restore sources the ids from the files.
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmpRoot(gpa, &tmp, "evicted");
    defer gpa.free(root);
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, .{});
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]));
    const hit = (try r.store.find(1, a[0 .. 8 * page])).?;
    try testing.expect(hit.ram);
    // the RAM trim's eviction (a spill's: the same settle path): the entry parked, its index path trimmed
    _ = try r.store.evict(hit.id);
    try r.store.settleAll();
    const parked = (try r.store.find(1, a[0 .. 8 * page])).?;
    try testing.expect(!parked.ram);
    try testing.expectEqual(hit.id, parked.id);
    // the defect: a disk entry's ids are not in the index, so the old restore tail's idsOf raised NotInRam
    const buf = try gpa.alloc(i32, @intCast(parked.pos));
    defer gpa.free(buf);
    try testing.expectError(error.NotInRam, r.store.index.idsOf(parked.id, buf));
    // the fix: the NVMe restore itself returns the entry's committed ids (its files hold them, checksummed)
    const j = try r.store.beginRestore(parked.id, s);
    var got = (try r.store.settle(j)).?;
    defer got.deinit(gpa);
    try testing.expectEqualSlices(i32, a[0..@intCast(parked.pos)], got.ids);
    try r.check(s, 1, parked.pos);
}

fn chmod(path: [:0]const u8, mode: std.c.mode_t) !void {
    if (std.c.fchmodat(std.posix.AT.FDCWD, path, mode, 0) != 0) return error.ChmodFailed;
}

/// The disk's directory of a Rank.init'd rank (compat @splat(7)): <root>/<compat hex>/rank<d>.
fn tierDir(gpa: std.mem.Allocator, root: [:0]const u8, rank: u32) ![:0]u8 {
    return std.fmt.allocPrintSentinel(gpa, "{s}/0707070707070707/rank{d}", .{ root, rank }, 0);
}

test "a spill park that fails on one rank is the fleet-fatal ParkFailed (the review's silent pool divergence)" {
    // the 2026-10-10 review's M1: a follower-local NVMe failure on the admit/spill path (no agreement there)
    // surfaced as the tier's raw error — request-level on a follower (tp/replay.zig) — so the follower logged
    // and served on with its pool still holding the entry's pages the other ranks' parks freed: silent pool
    // divergence, an asymmetric PoolExhausted away from a collective-order desync with every rank alive. A
    // failed park on a multi-rank store is the distinct process-fatal ParkFailed (the fate channel's clean
    // restart, as a replica's OOM); on a one-rank store the raw tier error stands (nothing to diverge from).
    if (std.os.linux.geteuid() == 0) return; // root writes through permission bits: the failure cannot be injected
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmpRoot(gpa, &tmp, "parkfail");
    defer gpa.free(root);
    // a follower's store in a two-rank fleet, a chat saved and idle in RAM
    var r: Rank = undefined;
    try r.init(gpa, root, 2, 1, .{});
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 20 * page);
    defer gpa.free(a);
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 1, a[0 .. 5 * page + 10]));
    const hit = (try r.store.find(1, a[0 .. 8 * page])).?;
    try testing.expect(hit.ram);
    const free_before = r.pool.freePages();
    // this rank's tier stops taking writes (one Spark's full or faulted disk): its rank dir read-only
    const tier = try tierDir(gpa, root, 1);
    defer gpa.free(tier);
    try chmod(tier, 0o555);
    defer chmod(tier, 0o755) catch {};
    // the spill path itself (applyAdmit's evict, its sync park settle): not the tier's raw request-level
    // error but the process-fatal ParkFailed, and the failed park changed nothing on this rank — the entry
    // is still in RAM holding its pages, the failed file's .tmp is gone
    try testing.expectError(error.ParkFailed, r.store.evict(hit.id));
    try testing.expect(r.store.entry(hit.id).ram);
    try testing.expectEqual(free_before, r.pool.freePages());
    var d = try std.Io.Dir.cwd().openDir(testing.io, tier, .{ .iterate = true });
    defer d.close(testing.io);
    var it = d.iterate();
    while (try it.next(testing.io)) |ent| try testing.expect(!std.mem.endsWith(u8, ent.name, ".tmp"));
    // the async settle point (applyAdmit's settleAll) maps the same way
    var r2: Rank = undefined;
    try r2.init(gpa, root, 2, 1, .{ .async_io = true });
    defer r2.deinit();
    const s2 = try r2.pool.newSlot(64 * page);
    try testing.expectEqualStrings("fresh", try turn(gpa, &r2, s2, 1, a[0 .. 5 * page + 10]));
    const hit2 = (try r2.store.find(1, a[0 .. 8 * page])).?;
    _ = try r2.store.evict(hit2.id); // the park job starts; its failure surfaces at the plan's settle
    try testing.expectError(error.ParkFailed, r2.store.settleAll());
    try testing.expect(r2.store.entry(hit2.id).ram);
    // one rank alone has nothing to diverge from: the tier's own error, the entry still in RAM, serving on
    const root1 = try tmpRoot(gpa, &tmp, "parkfail1");
    defer gpa.free(root1);
    var r1: Rank = undefined;
    try r1.init(gpa, root1, 1, 0, .{});
    defer r1.deinit();
    const s1 = try r1.pool.newSlot(64 * page);
    try testing.expectEqualStrings("fresh", try turn(gpa, &r1, s1, 1, a[0 .. 5 * page + 10]));
    const hit1 = (try r1.store.find(1, a[0 .. 8 * page])).?;
    const tier1 = try tierDir(gpa, root1, 0);
    defer gpa.free(tier1);
    try chmod(tier1, 0o555);
    defer chmod(tier1, 0o755) catch {};
    if (r1.store.evict(hit1.id)) |_| return error.TestUnexpectedResult else |err| try testing.expect(err != error.ParkFailed);
    try testing.expect(r1.store.entry(hit1.id).ram);
}

test "two split ranks make the same decisions on the same calls and each restores its own rows" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    var t0: std.ArrayList(u64) = .empty;
    defer t0.deinit(gpa);
    var t1: std.ArrayList(u64) = .empty;
    defer t1.deinit(gpa);
    const root = try tmpRoot(gpa, &tmp, "tier");
    defer gpa.free(root);
    try scenario(gpa, root, 2, 0, false, &t0);
    try scenario(gpa, root, 2, 1, true, &t1);
    try restart(gpa, root, 2, 0);
    try restart(gpa, root, 2, 1);
    try testing.expectEqualSlices(u64, t0.items, t1.items);
}

test "historyOf: a RAM entry's ids come from the index, an NVMe entry's only from the prompt that names it" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmpRoot(gpa, &tmp, "history");
    defer gpa.free(root);
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, .{ .chats = 1 });
    defer r.deinit();
    const s = try r.pool.newSlot(64 * page);
    const a = try chatIds(gpa, 1, 8 * page);
    defer gpa.free(a);
    const b = try chatIds(gpa, 2, 8 * page);
    defer gpa.free(b);
    const n = 5 * page + 10;
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 1, a[0..n]));
    const hit = (try r.store.find(1, a[0 .. n + 1])).?;
    try testing.expect(hit.ram and hit.pos == n);
    var out: [n]i32 = undefined;
    try r.store.historyOf(hit.id, null, &out);
    try testing.expectEqualSlices(i32, a[0..n], &out);
    try testing.expectError(error.BadLength, r.store.historyOf(hit.id, null, out[0 .. n - 1]));
    // a second idle chat parks chat 1 on NVMe: its ids leave the index with its RAM
    try testing.expectEqualStrings("fresh", try turn(gpa, &r, s, 2, b[0 .. 6 * page]));
    const parked = (try r.store.find(1, a[0 .. n + 1])).?;
    try testing.expect(!parked.ram and parked.pos == n);
    try testing.expectError(error.NotInRam, r.store.historyOf(parked.id, null, &out));
    @memset(&out, 0);
    try r.store.historyOf(parked.id, a, &out); // a longer prompt: its first pos ids, checked against the entry's key
    try testing.expectEqualSlices(i32, a[0..n], &out);
    // another chat's ids, or these with the partial page's last id changed, are not this entry
    try testing.expectError(error.NotTheEntry, r.store.historyOf(parked.id, b[0..n], &out));
    const edited = try gpa.dupe(i32, a[0..n]);
    defer gpa.free(edited);
    edited[n - 1] += 1;
    try testing.expectError(error.NotTheEntry, r.store.historyOf(parked.id, edited, &out));
    try testing.expectError(error.BadLength, r.store.historyOf(parked.id, a[0 .. n - 1], &out));
}

test "zero chat or byte budgets evict idle residency while null remains unlimited" {
    const gpa = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmpRoot(gpa, &tmp, "zero-budget");
    defer gpa.free(root);
    var r: Rank = undefined;
    try r.init(gpa, root, 1, 0, .{});
    defer r.deinit();
    const slot = try r.pool.newSlot(64 * page);
    const ids = try chatIds(gpa, 1, 5 * page);
    defer gpa.free(ids);
    _ = try turn(gpa, &r, slot, 1, ids);
    try testing.expectEqual(@as(u32, 1), r.store.resident().chats);
    r.store.policy.chats = 0;
    try testing.expectEqual(@as(u32, 1), try r.store.enforce());
    try testing.expectEqual(@as(u32, 0), r.store.resident().chats);
    r.store.policy.chats = null;
    _ = try turn(gpa, &r, slot, 1, ids);
    try testing.expect(r.store.resident().bytes > 0);
    r.store.policy.resident_bytes = 0;
    try testing.expectEqual(@as(u32, 1), try r.store.enforce());
    try testing.expectEqual(@as(u64, 0), r.store.resident().bytes);
}
