//! The session store: RAM entries holding pool pages and bounded state, a residency budget, and the NVMe tier.

const std = @import("std");
const Io = std.Io;
const pool_mod = @import("pool.zig");
const Pool = pool_mod.Pool;
const Slot = pool_mod.Slot;
const PageStore = @import("pagestore.zig").PageStore;
const prefix = @import("prefix.zig");
const Digest = prefix.Digest;
const disk_mod = @import("disk.zig");
const Disk = disk_mod.Disk;
const format = @import("format.zig");
const up = @import("lanes.zig").up;
const jobs = @import("jobs.zig");
const Job = jobs.Job;

pub const none = prefix.none;

/// A slot's bounded state (rings, carries, lookback, taps) as the family snapshots it: opaque to the store.
pub const Bounded = struct {
    ptr: *anyopaque,
    vtable: *const VTable,

    pub const VTable = struct {
        nbytes: *const fn (ptr: *anyopaque) u64,
        /// host copies of its pieces for a park (owned by the caller: name and bytes allocated with gpa)
        blobs: *const fn (ptr: *anyopaque, gpa: std.mem.Allocator) anyerror![]disk_mod.Blob,
        /// how many pieces blobs() returns (a park writes one segment a piece): the tier's price needs the count without the copies
        pieces: *const fn (ptr: *anyopaque) u32,
        release: *const fn (ptr: *anyopaque) void,
    };
};

pub const Policy = struct {
    /// bounded state the RAM tier keeps (least recently used entries go past it)
    ram_bytes: u64 = 512 << 20,
    /// Idle chats and entry-only pool bytes: null is unlimited; zero retains no idle residency.
    chats: ?u32 = null,
    resident_bytes: ?u64 = null,
    /// park an evicted chat's older turns too (default: drop them, the leaf's file covers them)
    park_prefixes: bool = false,
    /// drop, never write, an entry a newer entry of its chain covers
    skip_covered: bool = true,
    /// write a newer turn as a delta on the file its leading pages came from
    delta: bool = true,
    /// parks and restores on the tier thread (settled by the plan)
    async_io: bool = false,
};

pub const Extra = struct {
    /// RAM: the pool pages it holds, logical [0, npages)
    pages: []u32 = &.{},
    bounded: ?Bounded = null,
    /// a file with this key is registered on the NVMe tier
    file: bool = false,
    used: u64 = 0,
    origin: ?pool_mod.Origin = null,
    /// a newer entry of its chain was parked from RAM: writing this one again is redundant
    covered_by: bool = false,
    kind: u32 = 0,
    /// an unsettled park of it
    job: ?*Job = null,
    older: u32 = none,
    newer: u32 = none,
};

const Index = prefix.Index(Extra);

pub const Hit = struct { id: u32, ram: bool, pos: u64 };

pub const Stats = struct {
    saves: u64 = 0,
    dups: u64 = 0,
    hits_ram: u64 = 0,
    hits_disk: u64 = 0,
    parks: u64 = 0,
    delta_parks: u64 = 0,
    drops: u64 = 0,
    covered_drops: u64 = 0,
    chat_evictions: u64 = 0,
    park_ns: u64 = 0,
    restore_ns: u64 = 0,
};

pub const Store = struct {
    gpa: std.mem.Allocator,
    io: Io,
    pool: *Pool,
    pages: PageStore,
    index: Index,
    disk: ?*Disk,
    /// heap-held: the tier thread keeps its address
    runner: ?*jobs.Runner = null,
    policy: Policy,
    clock: u64 = 0,
    bounded_total: u64 = 0,
    /// RAM entries, least recently used first
    oldest: u32 = none,
    newest: u32 = none,
    ram_count: u32 = 0,
    pending: std.ArrayList(*Job) = .empty,
    scratch: std.ArrayList(Digest) = .empty,
    stats: Stats = .{},

    pub fn init(gpa: std.mem.Allocator, io: Io, pool: *Pool, pages: PageStore, d: ?*Disk, policy: Policy) !Store {
        var s: Store = .{ .gpa = gpa, .io = io, .pool = pool, .pages = pages, .index = Index.init(gpa, pool.page), .disk = d, .policy = policy };
        if (d) |dd| {
            const r = try gpa.create(jobs.Runner);
            r.* = jobs.Runner.init(gpa, io, dd);
            s.runner = r;
            if (policy.async_io) r.startThread() catch |e| {
                gpa.destroy(r);
                return e;
            };
        }
        return s;
    }

    pub fn deinit(s: *Store) void {
        if (s.runner) |r| {
            for (s.pending.items) |j| r.wait(j);
            r.deinit();
            s.gpa.destroy(r);
        }
        for (s.pending.items) |j| j.deinit(s.gpa);
        s.pending.deinit(s.gpa);
        for (s.index.entries.items) |*e| if (e.live) {
            if (e.extra.bounded) |b| b.vtable.release(b.ptr);
            s.pool.drop(e.extra.pages) catch {};
            s.gpa.free(e.extra.pages);
        };
        s.index.deinit();
        s.scratch.deinit(s.gpa);
        s.* = undefined;
    }

    pub fn digests(s: *Store, tag: u32, ids: []const i32) ![]const Digest {
        const n = ids.len / s.pool.page;
        try s.scratch.resize(s.gpa, n);
        prefix.chain(tag, ids, s.pool.page, s.scratch.items);
        return s.scratch.items;
    }

    pub fn entry(s: *Store, id: u32) *Index.Entry {
        return s.index.get(id);
    }

    /// The longest stored entry (RAM or NVMe) whose ids strictly prefix prompt, with the same tag.
    pub fn find(s: *Store, tag: u32, prompt: []const i32) !?Hit {
        const d = try s.digests(tag, prompt);
        const id = s.index.find(tag, prompt, d, .any) orelse return null;
        const e = s.index.get(id);
        return .{ .id = id, .ram = e.ram, .pos = e.pos };
    }

    pub fn link(s: *Store, id: u32) void {
        const x = &s.index.get(id).extra;
        x.older = s.newest;
        x.newer = none;
        if (s.newest != none) s.index.get(s.newest).extra.newer = id else s.oldest = id;
        s.newest = id;
    }

    fn unlink(s: *Store, id: u32) void {
        const x = &s.index.get(id).extra;
        if (x.older != none) s.index.get(x.older).extra.newer = x.newer else s.oldest = x.newer;
        if (x.newer != none) s.index.get(x.newer).extra.older = x.older else s.newest = x.older;
    }

    pub fn touch(s: *Store, id: u32) void {
        s.clock += 1;
        const e = s.index.get(id);
        e.extra.used = s.clock;
        if (e.ram) {
            s.unlink(id);
            s.link(id);
        }
    }

    // RAM ownership transfers live in a separate module; helpers below also serve disk policy work.
    pub const save = @import("store_ram.zig").save;
    pub const restoreRam = @import("store_ram.zig").restoreRam;

    /// A restored entry's ids (pos long): RAM from the index, else `given`, the prompt's prefix, checked.
    pub fn historyOf(s: *Store, id: u32, given: ?[]const i32, out: []i32) !void {
        const e = s.index.get(id);
        if (out.len != e.pos) return error.BadLength;
        if (e.ram) return s.index.idsOf(id, out);
        // an NVMe entry's tokens left the index with its RAM (prefix.dropTokens): only the prompt still holds them
        const ids = given orelse return error.NotInRam;
        if (ids.len < e.pos) return error.BadLength;
        const pos: usize = @intCast(e.pos);
        const full = pos / s.pool.page;
        const d = try s.digests(e.tag, ids[0..pos]);
        const node = if (full == 0) prefix.root(e.tag) else d[full - 1];
        if (!std.mem.eql(u8, &prefix.entryKey(node, e.pos, ids[full * s.pool.page .. pos]), &e.key)) return error.NotTheEntry;
        @memcpy(out, ids[0..pos]);
    }

    /// Starts an NVMe resume into an empty slot (pages mapped now, filled by the job); settle it before the slot runs.
    pub fn beginRestore(s: *Store, id: u32, slot: *Slot) !*Job {
        const d = s.disk orelse return error.NoDisk;
        const e = s.index.get(id);
        if (!e.extra.file) return error.NotOnDisk;
        if (slot.len != 0) return error.SlotNotEmpty;
        try slot.ensure(e.pos);
        const j = try s.gpa.create(Job);
        j.* = .{ .kind = .restore, .entry = id, .slot = slot.id };
        errdefer j.deinit(s.gpa);
        var ext: std.ArrayList(disk_mod.Extent) = .empty;
        errdefer ext.deinit(s.gpa);
        try d.extents(e.key, &ext);
        j.extents = try ext.toOwnedSlice(s.gpa);
        // the entry's committed ids [0, e.pos): the chain's files each hold their [first_page x page, pos) of
        // them (an entry out of RAM keeps digests only in the index); the windows must tile with no gap
        if (j.extents.len == 0 or j.extents[0].first_page != 0) return error.BadChain;
        var covered: u64 = 0;
        for (j.extents) |x| {
            if (@as(u64, x.first_page) * s.pool.page > covered) return error.BadChain;
            covered = @max(covered, x.pos);
        }
        if (covered != e.pos) return error.BadChain;
        j.ids = try s.gpa.alloc(i32, e.pos);
        const np = pool_mod.pagesFor(e.pos, s.pool.page);
        j.lists = try s.familyLists(slot.mapped()[0..np], 0, false);
        j.sink = .{ .store = s.pages, .families = s.pool.layout.families, .page = s.pool.page, .world = s.pool.world, .rank = s.pool.rank, .pages = j.lists };
        try s.pending.append(s.gpa, j);
        try s.runner.?.submit(j);
        return j;
    }

    /// Per family: family-tensor pages of logical [first, len(pages)) (owned_only: this rank's of split families).
    fn familyLists(s: *Store, pages: []const u32, first: u32, owned_only: bool) ![][]u32 {
        const fams = s.pool.layout.families;
        const out = try s.gpa.alloc([]u32, fams.len);
        var made: usize = 0;
        errdefer {
            for (out[0..made]) |l| s.gpa.free(l);
            s.gpa.free(out);
        }
        for (fams, 0..) |f, fi| {
            const w = s.pool.world;
            const sharded = f.split and w > 1 and owned_only;
            var n: usize = pages.len - first;
            if (sharded) {
                var by: [pool_mod.max_world]u32 = undefined;
                pool_mod.span(first, @intCast(pages.len), w, &by);
                n = by[s.pool.rank];
            }
            const l = try s.gpa.alloc(u32, n);
            var i: usize = 0;
            var k: usize = first;
            if (sharded) k = first + (s.pool.rank + w - first % w) % w;
            while (k < pages.len) : (k += if (sharded) w else 1) {
                l[i] = s.pool.familyPage(f, pages[k]);
                i += 1;
            }
            out[fi] = l;
            made += 1;
        }
        return out;
    }

    /// Starts a park of a RAM entry (null: nothing to write, the entry is already settled out of RAM).
    pub fn beginPark(s: *Store, id: u32) !?*Job {
        const d = s.disk orelse return error.NoDisk;
        const e = s.index.get(id);
        if (!e.ram or e.extra.job != null) return error.NotParkable;
        if (e.extra.file and d.get(e.key) != null) {
            try s.leaveRam(id);
            return null;
        }
        const page = s.pool.page;
        const npages = pool_mod.pagesFor(e.pos, page);
        var base: ?Digest = null;
        var first: u32 = 0;
        if (s.policy.delta) if (e.extra.origin) |o| if (o.pages > 0) {
            var r = d.get(o.key);
            while (r) |rec| : (r = if (rec.base == disk_mod.none) null else &d.recs.items[rec.base]) {
                if (rec.first_page < o.pages) break;
            }
            if (r) |rec| if (rec.depth < d.opts.max_chain) {
                base = rec.key;
                first = o.pages;
            };
        };
        const j = try s.gpa.create(Job);
        j.* = .{ .kind = .park, .entry = id };
        errdefer j.deinit(s.gpa);
        j.ids = try s.gpa.alloc(i32, e.pos);
        try s.index.idsOf(id, j.ids);
        j.lists = try s.familyLists(e.extra.pages, first, true);
        j.blobs = try e.extra.bounded.?.vtable.blobs(e.extra.bounded.?.ptr, s.gpa);
        j.src = .{ .key = e.key, .base = base, .tag = e.tag, .kind = e.extra.kind, .page = page, .pos = e.pos, .first_page = first, .npages = npages, .ids = j.ids[first * page ..], .store = s.pages, .fence = try s.pages.fence(), .families = s.pool.layout.families, .pages = j.lists, .blobs = j.blobs };
        if (base) |b| d.get(b).?.refs += 1; // held while the job runs: a trim cannot take the base
        e.extra.job = j;
        try s.pending.append(s.gpa, j);
        try s.runner.?.submit(j);
        return j;
    }

    /// Settles a finished job at the plan's point: a park registers its file, a restore returns the bounded state.
    pub fn settle(s: *Store, j: *Job) !?disk_mod.Restored {
        s.runner.?.wait(j);
        defer {
            for (s.pending.items, 0..) |p, i| if (p == j) {
                _ = s.pending.orderedRemove(i);
                break;
            };
            j.deinit(s.gpa);
        }
        const d = s.disk.?;
        switch (j.kind) {
            .park => {
                if (j.src.base) |b| d.get(b).?.refs -= 1;
                const e = s.index.get(j.entry);
                e.extra.job = null;
                if (j.err) |err| {
                    // a park that failed on this rank only diverges the fleet's pools by construction (the
                    // entry's pages freed where the write landed, held here) and no agreement repairs it: in
                    // a fleet it is the process-fatal ParkFailed (tp/replay.zig's classes — the follower's
                    // exit fires the fate channel, a clean loud restart, as a replica's OOM), not the tier's
                    // raw request-level error a follower would survive into a silent divergence. One rank
                    // alone has nothing to diverge from: the raw error, the engine serves on.
                    if (s.pool.world > 1) {
                        std.log.warn("sessions: the park of entry {d} failed on this rank ({t}); the ranks' pools would diverge, this rank ends so the fleet restarts consistently", .{ j.entry, err });
                        return error.ParkFailed;
                    }
                    return err; // the entry stays in RAM; the .tmp is gone
                }
                _ = try d.register(&j.src, j.size);
                e.extra.file = true;
                s.stats.parks += 1;
                if (j.src.base != null) s.stats.delta_parks += 1;
                s.stats.park_ns += j.elapsed_ns;
                if (s.policy.skip_covered) {
                    var m: std.ArrayList(u32) = .empty;
                    defer m.deinit(s.gpa);
                    _ = try s.index.members(j.entry, &m, usedOf);
                    for (m.items) |o| s.index.get(o).extra.covered_by = true;
                }
                try s.leaveRam(j.entry);
                try s.trimDisk();
                return null;
            },
            .restore => {
                if (j.err) |err| {
                    // the slot as before the restore: empty, its reservation kept
                    _ = try s.pool.slots.items[j.slot].truncate(0);
                    return err;
                }
                const e = s.index.get(j.entry);
                d.touch(e.key);
                s.touch(j.entry);
                s.pool.slots.items[j.slot].origin = .{ .key = e.key, .pages = @intCast(e.pos / s.pool.page) };
                s.stats.hits_disk += 1;
                s.stats.restore_ns += j.elapsed_ns;
                const r = j.restored;
                j.restored = null;
                return r;
            },
        }
    }

    /// Settles every pending job in submission order (one rank, or the plan's "settle all").
    pub fn settleAll(s: *Store) !void {
        while (s.pending.items.len > 0) {
            if (try s.settle(s.pending.items[0])) |r| {
                var rr = r;
                rr.deinit(s.gpa);
            }
        }
    }

    fn usedOf(x: *const Extra) u64 {
        return x.used;
    }

    /// A RAM entry lets go of its pages and bounded state; it stays indexed if a file holds it.
    pub fn leaveRam(s: *Store, id: u32) !void {
        const e = s.index.get(id);
        if (!e.ram) return;
        try s.pool.drop(e.extra.pages);
        s.gpa.free(e.extra.pages);
        e.extra.pages = &.{};
        if (e.extra.bounded) |b| {
            s.bounded_total -= b.vtable.nbytes(b.ptr);
            b.vtable.release(b.ptr);
        }
        e.extra.bounded = null;
        s.unlink(id);
        s.ram_count -= 1;
        try s.index.setRam(id, false, null);
        if (!e.extra.file) s.index.remove(id);
    }

    /// An entry whose file could not be read back: the file deleted, the entry gone unless RAM holds it (every rank).
    pub fn forget(s: *Store, id: u32) !void {
        const e = s.index.get(id);
        if (e.extra.job != null) return error.Busy;
        if (e.extra.file) if (s.disk) |d| {
            d.stats.bad += 1;
            // a delta's base stays while a newer file stacks on it (an opt-in delta mode)
            d.remove(e.key) catch |err| return if (err == error.HasDependents) {} else err;
            e.extra.file = false;
        };
        if (!e.ram) s.index.remove(id);
    }

    /// Drops an entry from RAM without writing it.
    pub fn drop(s: *Store, id: u32) !void {
        try s.leaveRam(id);
        s.stats.drops += 1;
    }

    /// Lets go of a RAM entry: dropped when covered, parked if the tier takes it, else dropped; returns a started park.
    pub fn evict(s: *Store, id: u32) !?*Job {
        const e = s.index.get(id);
        if (e.extra.job != null) return null;
        if (s.policy.skip_covered and (e.extra.covered_by or s.index.covered(id, .any))) {
            try s.drop(id);
            s.stats.covered_drops += 1;
            return null;
        }
        if (s.disk) |d| if (d.accepts(e.pos)) {
            const j = try s.beginPark(id) orelse return null;
            if (!s.policy.async_io) _ = try s.settle(j);
            return if (s.policy.async_io) j else null;
        };
        try s.drop(id);
        return null;
    }

    /// Evicts least recently used RAM entries past the bounded-state budget.
    pub fn trimRam(s: *Store, keep: u32) !void {
        var id = s.oldest;
        while (s.bounded_total > s.policy.ram_bytes and s.ram_count > 1 and id != none) {
            const next = s.index.get(id).extra.newer;
            if (id != keep and s.index.get(id).extra.job == null) _ = try s.evict(id);
            id = next;
        }
    }

    /// Files the NVMe budget deleted leave the index (a RAM entry with that key stays, unwritten again).
    fn trimDisk(s: *Store) !void {
        const d = s.disk orelse return;
        var gone: std.ArrayList(Digest) = .empty;
        defer gone.deinit(s.gpa);
        try d.trim(&gone);
        for (gone.items) |k| if (s.index.lookup(k)) |id| {
            s.index.get(id).extra.file = false;
            if (!s.index.get(id).ram) s.index.remove(id);
        };
    }

    const Chat = struct { leaf: u32, recency: u64 };

    fn chatLess(_: void, a: Chat, b: Chat) bool {
        return a.recency < b.recency or (a.recency == b.recency and a.leaf < b.leaf);
    }

    /// A chat is live while a slot maps its leaf's first page.
    fn idleLeaf(s: *Store, id: u32) bool {
        const e = s.index.get(id);
        return e.extra.job == null and s.index.isLeaf(id) and (e.extra.pages.len == 0 or s.pool.slot_refs[e.extra.pages[0]] == 0);
    }

    /// Evicts LRU idle chats past the residency budget (leaf parked, unshared older turns dropped); returns how many.
    pub fn enforce(s: *Store) !u32 {
        const pol = s.policy;
        if (pol.chats == null and pol.resident_bytes == null) return 0;
        var chats: std.ArrayList(Chat) = .empty;
        defer chats.deinit(s.gpa);
        var members: std.ArrayList(u32) = .empty;
        defer members.deinit(s.gpa);
        var id = s.oldest;
        while (id != none) : (id = s.index.get(id).extra.newer) {
            if (!s.idleLeaf(id)) continue;
            members.clearRetainingCapacity();
            try chats.append(s.gpa, .{ .leaf = id, .recency = try s.index.members(id, &members, usedOf) });
        }
        std.mem.sort(Chat, chats.items, {}, chatLess);
        var idle: u32 = @intCast(chats.items.len);
        var gone: u32 = 0;
        for (chats.items) |c| {
            const over_n = if (pol.chats) |limit| idle > limit else false;
            const over_b = if (pol.resident_bytes) |limit| s.pool.entryOnlyBytes() > limit else false;
            if (!over_n and !over_b) break;
            if (!s.index.get(c.leaf).ram or !s.idleLeaf(c.leaf)) continue;
            members.clearRetainingCapacity();
            _ = try s.index.members(c.leaf, &members, usedOf);
            _ = try s.evict(c.leaf);
            // deepest first: a member more RAM entries extend than this chat's leaf is a branch point, as is all above
            const own: u32 = if (s.index.get(c.leaf).ram) 1 else 0;
            for (members.items) |m| {
                if (s.index.get(m).extra.job != null) continue;
                if (s.index.extensions(m) > own) break;
                if (pol.park_prefixes) _ = try s.evict(m) else try s.drop(m);
            }
            idle -= 1;
            gone += 1;
            s.stats.chat_evictions += 1;
        }
        return gone;
    }

    /// RAM entries to let go of (coldest first, never keep) so need pages are free: [] if they are, null if impossible.
    pub fn planSpill(s: *Store, need: u32, keep: []const u32, out: *std.ArrayList(u32)) !bool {
        out.clearRetainingCapacity();
        var avail = s.pool.available();
        if (avail >= need) return true;
        var left: std.AutoHashMapUnmanaged(u32, u16) = .empty;
        defer left.deinit(s.gpa);
        var id = s.oldest;
        while (id != none and avail < need) : (id = s.index.get(id).extra.newer) {
            if (std.mem.indexOfScalar(u32, keep, id) != null or s.index.get(id).extra.job != null) continue;
            try out.append(s.gpa, id);
            for (s.index.get(id).extra.pages) |pg| {
                const g = try left.getOrPut(s.gpa, pg);
                if (!g.found_existing) g.value_ptr.* = s.pool.refs[pg];
                g.value_ptr.* -= 1;
                if (g.value_ptr.* == 0) avail += 1; // split: a page of the scarcer residue may count for less; the check below decides
            }
        }
        return avail >= need;
    }

    /// A segment's file span as format.zig's Layout.build places it: the whole chunks at their 4 KiB
    /// stride, the last at its own padded tail; its chunks join the header's digest slots.
    fn segSpan(nbytes: u64, chunk: u64, ndigest: *u64) u64 {
        if (nbytes == 0) return 0;
        const n = (nbytes + chunk - 1) / chunk;
        ndigest.* += n;
        return (n - 1) * up(chunk) + up(nbytes - (n - 1) * chunk);
    }

    /// The file-price formula over an entry's numbers: the file as Disk.plan and format.zig's
    /// Layout.build place it — the padded header (a Head, a Seg a segment, a digest slot a chunk),
    /// the ids segment, this rank's pages of every family (split families sharded as beginPark's
    /// familyLists, their chunk a whole number of pages as Disk.plan's) and one segment a bounded
    /// piece. A whole-entry count (a delta's base pages ride along as slack), the pieces' own sizes
    /// known only in sum (each priced a full tail pad: Σceil(piece/chunk) ≤ ceil(Σ/chunk) + pieces − 1).
    fn parkPrice(s: *Store, d: *const Disk, pos: u64, bounded_nb: u64, bounded_pieces: u64) u64 {
        const w = s.pool.world;
        const cap: u64 = d.opts.chunk;
        const page = s.pool.page;
        const npages = pool_mod.pagesFor(pos, page);
        var ndigest: u64 = 0;
        var nseg: u64 = 1;
        var total = segSpan(4 * pos, cap, &ndigest);
        for (s.pool.layout.families) |f| {
            var k: u64 = npages;
            if (f.split and w > 1) {
                var by: [pool_mod.max_world]u32 = undefined;
                pool_mod.span(0, npages, w, &by);
                k = by[s.pool.rank];
            }
            const pb = f.pageBytes(page);
            total += segSpan(k * pb, @max(1, cap / pb) * pb, &ndigest);
            nseg += 1;
        }
        if (bounded_pieces > 0) {
            const chunks: u64 = if (bounded_nb == 0) 0 else (bounded_nb + cap - 1) / cap + bounded_pieces -| 1;
            total += bounded_nb + (up(1) - 1) * chunks;
            ndigest += chunks;
            nseg += bounded_pieces;
        }
        return up(@sizeOf(format.Head) + nseg * @sizeOf(format.Seg) + ndigest * @sizeOf(format.Digest32)) + total;
    }

    /// The bytes a park of `id` asks of the tier (parkPrice over the entry): an overestimate only ever
    /// refuses earlier, and the boundary the gate pins (free == price) fits the write — a payload-only
    /// sum let free == payload through to the write's ENOSPC.
    pub fn parkBytes(s: *Store, id: u32) u64 {
        const d = s.disk orelse return 0;
        const e = s.index.get(id);
        var nb: u64 = 0;
        var pieces: u64 = 0;
        if (e.extra.bounded) |b| {
            nb = b.vtable.nbytes(b.ptr);
            pieces = b.vtable.pieces(b.ptr);
        }
        return s.parkPrice(d, e.pos, nb, pieces);
    }

    /// Would an evict of `id` write a file (true), or let go of the entry without one (a covered or
    /// too-short drop, an entry the tier already holds, a running job)? planSaveParks prices the writes.
    fn wouldPark(s: *Store, id: u32) bool {
        const d = s.disk orelse return false;
        const e = s.index.get(id);
        if (e.extra.job != null) return false;
        if (s.policy.skip_covered and (e.extra.covered_by or s.index.covered(id, .any))) return false;
        if (!d.accepts(e.pos)) return false;
        if (e.extra.file and d.get(e.key) != null) return false;
        return true;
    }

    /// The entries a save of `new_pos` tokens carrying `new_bounded` bytes of bounded state (in
    /// `new_pieces` pieces) would park, in the order the save's own trims evict them: trimRam's walk
    /// with the entry already counted (oldest first, the same would-park filter as evict — a drop asks
    /// nothing of the tier), then enforce's residency trims (budgets off in serving: none). `incoming`
    /// joins the price with the incoming entry's own file: the real walk's last eviction can be that
    /// entry itself (it is linked newest) when every older entry's park is still in flight and the
    /// budget stays crossed. Rank 0 prices these before broadcasting the save (kv/sess.zig's
    /// saveAdmits); the followers' replayed saves run the same trims over the same store, so the price
    /// is what every rank's parks ask of its tier. Slack, never less: an entry trimRam's walk and
    /// enforce's list both name is priced twice (the real enforce never sees it — it left RAM), a
    /// resident-bytes budget is measured against the pool's current entry bytes (the evictions' are
    /// not subtracted), and park_prefixes' members are priced past the branch-point break. The walk
    /// mirrors the synchronous settle (async_io off in serving): an async park leaves the entry in RAM
    /// and the real walk evicts on, parking more than priced — the park's own error stays the guard.
    pub fn planSaveParks(s: *Store, new_pos: u64, new_bounded: u64, new_pieces: u32, out: *std.ArrayList(u32), incoming: *u64) !void {
        out.clearRetainingCapacity();
        incoming.* = 0;
        // trimRam (Store.save's first trim), the entry's bytes already in: the walk's own condition
        var total = s.bounded_total + new_bounded;
        var count = s.ram_count + 1;
        var id = s.oldest;
        while (total > s.policy.ram_bytes and count > 1 and id != none) {
            const next = s.index.get(id).extra.newer;
            const e = s.index.get(id);
            if (e.extra.job == null) {
                if (s.wouldPark(id)) try out.append(s.gpa, id);
                total -= e.extra.bounded.?.vtable.nbytes(e.extra.bounded.?.ptr);
                count -= 1;
            }
            id = next;
        }
        // the walk's last eviction can be the incoming entry itself (the newest): only when every
        // older entry is held by a park still in flight and the budget stays crossed
        if (total > s.policy.ram_bytes and count > 1) if (s.disk) |d| {
            if (d.accepts(new_pos)) incoming.* = s.parkPrice(d, new_pos, new_bounded, new_pieces);
        };
        // enforce (the residency budgets): the idle chats coldest first, the leaf's park priced; the
        // members go with it (dropped by default — park_prefixes prices their parks too)
        const pol = s.policy;
        if (pol.chats == 0 and pol.resident_bytes == 0) return;
        var chats: std.ArrayList(Chat) = .empty;
        defer chats.deinit(s.gpa);
        var members: std.ArrayList(u32) = .empty;
        defer members.deinit(s.gpa);
        id = s.oldest;
        while (id != none) : (id = s.index.get(id).extra.newer) {
            if (!s.idleLeaf(id)) continue;
            members.clearRetainingCapacity();
            try chats.append(s.gpa, .{ .leaf = id, .recency = try s.index.members(id, &members, usedOf) });
        }
        std.mem.sort(Chat, chats.items, {}, chatLess);
        var idle: u32 = @intCast(chats.items.len);
        for (chats.items) |c| {
            const over_n = if (pol.chats) |limit| idle > limit else false;
            const over_b = if (pol.resident_bytes) |limit| s.pool.entryOnlyBytes() > limit else false;
            if (!over_n and !over_b) break;
            if (s.wouldPark(c.leaf)) try out.append(s.gpa, c.leaf);
            if (pol.park_prefixes) {
                members.clearRetainingCapacity();
                _ = try s.index.members(c.leaf, &members, usedOf);
                for (members.items) |m| if (s.wouldPark(m)) try out.append(s.gpa, m);
            }
            idle -= 1;
        }
    }

    /// The priced parks must fit the tier's real free space: a park that meets a full tier on a
    /// follower is the process-fatal ParkFailed (settle's arm: the pools diverged by construction), and
    /// a tier full on every rank turns one admission or save into a fleet crash loop — the restart's
    /// empty start frees nothing (boots never scan the tier), so the next park meets the same wall.
    /// Rank 0 calls this before anything is broadcast and refuses instead (kv/sess.zig's admit for the
    /// admission's spills, its saveAdmits for a save's trims): every rank's store untouched, the
    /// failure request-level. `extra` rides along already priced (the save's incoming entry's own
    /// file). A probe failure refuses nothing: the park's own error stays the guard.
    pub fn tierAdmits(s: *Store, spills: []const u32, extra: u64) !void {
        const d = s.disk orelse return;
        if (spills.len == 0 and extra == 0) return;
        const free = d.freeBytes() catch return;
        var need: u64 = extra;
        for (spills) |id| need += s.parkBytes(id);
        if (free >= need) return;
        std.log.warn("sessions: the tier's {d} free bytes cannot hold this admission's {d} spill file(s) ({d} bytes); the request is refused, not parked into a fleet crash loop", .{ free, spills.len, need });
        return error.TierFull;
    }

    /// Rebuilds the NVMe entries after a restart (an empty store): every valid file indexed, deltas on their bases' paths; returns how many.
    /// Two ranks then intersect what they hold (the engine's plan link) and drop the rest, so a plan never names an entry one rank lacks.
    pub fn reconcile(s: *Store) !u32 {
        const d = s.disk orelse return 0;
        var found: std.ArrayList(Disk.Found) = .empty;
        defer {
            for (found.items) |f| s.gpa.free(f.ids);
            found.deinit(s.gpa);
        }
        try d.scan(&found);
        for (found.items) |f| {
            var anchor: u32 = none;
            if (f.head.hasBase()) {
                const b = s.index.lookup(f.head.base) orelse return error.NoBase;
                anchor = s.index.nodeAt(b, f.head.first_page) orelse return error.BadChain;
            }
            const r = try s.index.insertFrom(f.head.tag, anchor, f.ids, .{ .file = true, .kind = f.head.kind });
            s.clock += 1;
            s.index.get(r.id).extra.used = s.clock;
        }
        return @intCast(found.items.len);
    }

    /// Idle resident chats and the pool bytes held by entries alone.
    pub fn resident(s: *Store) struct { chats: u32, bytes: u64 } {
        var n: u32 = 0;
        var id = s.oldest;
        while (id != none) : (id = s.index.get(id).extra.newer) if (s.idleLeaf(id)) {
            n += 1;
        };
        return .{ .chats = n, .bytes = s.pool.entryOnlyBytes() };
    }
};
