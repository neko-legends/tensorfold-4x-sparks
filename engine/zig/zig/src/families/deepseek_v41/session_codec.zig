//! Validate a complete bounded-state image before uploading or publishing any metadata.
const std = @import("std");
const sg = @import("sessions_gpu.zig");
const ced = @import("ced.zig");
const eh = @import("engram_host.zig");
const Blob = @import("kv").sessions.disk.Blob;
const quiet = @import("builtin").is_test;
pub const Metadata = struct { pos: u64, tail_len: usize, tail: [eh.max_ngram]u32, stash: ced.Stash, ds: u64 };

/// A rejected image names the check it failed before the error is returned: `(BadBlob)` alone said an entry did not
/// read back but never which rule it broke, and the resume path logs only the error name.
fn bad(comptime fmt: []const u8, args: anytype) error{BadBlob} {
    if (!quiet) std.log.scoped(.dsv41).warn("sessions: BadBlob: " ++ fmt, args);
    return error.BadBlob;
}

pub fn validate(s: *const sg.Sessions, bs: []const Blob) !Metadata {
    if (bs.len != s.pieces.len + 1) return bad("{d} blobs for {d} pieces + slot", .{ bs.len, s.pieces.len });
    var slot: ?[]const u8 = null;
    for (bs, 0..) |b, i| {
        for (bs[0..i]) |prior| if (std.mem.eql(u8, b.name, prior.name)) return bad("duplicate blob name {s}", .{b.name});
        if (std.mem.eql(u8, b.name, "slot")) {
            slot = b.bytes;
            continue;
        }
        const pc = for (s.pieces) |p| {
            if (std.mem.eql(u8, p.role, b.name)) break p;
        } else return bad("blob {s} names no piece", .{b.name});
        if (pc.len != b.bytes.len) return bad("blob {s} is {d} B, its piece {d} B", .{ b.name, b.bytes.len, pc.len });
    }
    const bytes = slot orelse return bad("no slot blob among {d} blobs", .{bs.len});
    if (bytes.len != sg.Sessions.slot_bytes) return bad("slot blob is {d} B, not {d} B", .{ bytes.len, sg.Sessions.slot_bytes });
    const pos = std.mem.readInt(u64, bytes[0..8], .little);
    const tail_len = std.mem.readInt(u64, bytes[8..16], .little);
    if (pos > s.kx.opts.capacity or tail_len >= eh.max_ngram or tail_len > pos) return bad("pos {d} (cap {d}) or tail_len {d} (ngram {d}) out of range", .{ pos, s.kx.opts.capacity, tail_len, eh.max_ngram });
    var m: Metadata = .{ .pos = pos, .tail_len = @intCast(tail_len), .tail = @splat(0), .stash = .{}, .ds = std.mem.readInt(u64, bytes[sg.Sessions.ds_at..][0..8], .little) };
    for (&m.tail, 0..) |*t, i| {
        t.* = std.mem.readInt(u32, bytes[16 + 4 * i ..][0..4], .little);
        if (i < tail_len and t.* >= s.f.cfg.vocab) return bad("tail[{d}] = {d} >= vocab {d}", .{ i, t.*, s.f.cfg.vocab });
    }
    const c = bytes[sg.Sessions.ced_at..];
    // The CED stash is replay mode's ring. Full mode plans no such ring (and no such pieces), so a slot blob's
    // positions there are stale rather than wrong: an entry written before the full-mode stash stopped being
    // tracked still resumes on its pages, with the stash read as empty.
    if (s.measure.replay) {
        m.stash.from = std.mem.readInt(u64, c[0..8], .little);
        m.stash.hi = std.mem.readInt(u64, c[8..16], .little);
        if (m.stash.from > m.stash.hi or m.stash.hi > pos or m.stash.hi - m.stash.from > ced.ring) return bad("stash {d}..{d} at pos {d}, ring {d} (replay)", .{ m.stash.from, m.stash.hi, pos, ced.ring });
        for (&m.stash.ids, 0..) |*t, i| t.* = std.mem.readInt(u32, c[16 + 4 * i ..][0..4], .little);
        var at = m.stash.from;
        while (at < m.stash.hi) : (at += 1) if (m.stash.ids[@intCast(at % ced.ring)] >= s.f.cfg.vocab) return bad("stash id at {d} = {d} >= vocab {d} (replay)", .{ at, m.stash.ids[@intCast(at % ced.ring)], s.f.cfg.vocab });
    } else {
        const from = std.mem.readInt(u64, c[0..8], .little);
        const hi = std.mem.readInt(u64, c[8..16], .little);
        if ((from != 0 or hi != 0) and !quiet) std.log.scoped(.dsv41).info("sessions: a {d}-token entry carries a replay-mode stash in a full-mode engine; its pages resume and the stash is dropped", .{pos});
    }
    if (m.ds > pos or m.ds > s.measure.window or (!s.ds_rings and m.ds != 0)) return bad("drafter ds {d} at pos {d}, window {d}, rings {}", .{ m.ds, pos, s.measure.window, s.ds_rings });
    return m;
}
