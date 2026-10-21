//! Process-wide custody admission includes active, quarantined and callback-disposal owners.
const custody_sync = @import("copy_source_sync.zig");
const std = @import("std");
const builtin = @import("builtin");
pub const Limits = struct { bytes: usize = 1 << 30, nodes: usize = 4096 };
var limits: Limits = .{};
// TF_DSV41_COPY_SOURCE_GIB: the bound on pinned copy-source bytes held at once (default 1 GiB). The bound is a
// ceiling, not an allocation; a world-1 weight image (the prefill pipeline's stage copies) can be larger than
// 1 GiB, so the serviced engine's pipe boot raises it instead of failing the load with CopySourceLimit.
var limits_set = false;
fn ensureLimits() void {
    if (limits_set) return;
    limits_set = true;
    const raw = std.c.getenv("TF_DSV41_COPY_SOURCE_GIB") orelse return;
    const text = std.mem.trim(u8, std.mem.span(raw), " ");
    if (text.len == 0) return;
    const gib = std.fmt.parseFloat(f64, text) catch return;
    if (!(gib > 0)) return;
    limits.bytes = @intFromFloat(gib * (1 << 30));
}
var bytes: usize = 0;
var nodes: usize = 0;
var mutex: custody_sync.Mutex = .{};
pub fn acquire(size: usize, count: usize) error{CopySourceLimit}!void {
    mutex.lock();
    defer mutex.unlock();
    ensureLimits();
    if (size > limits.bytes - bytes or count > limits.nodes - nodes) return error.CopySourceLimit;
    bytes += size;
    nodes += count;
}
/// Custody that already exists (a buffer a failed wait left in use) counts even past the limit.
pub fn charge(size: usize) void {
    mutex.lock();
    defer mutex.unlock();
    bytes += size;
}
pub fn release(size: usize, count: usize) void {
    mutex.lock();
    defer mutex.unlock();
    std.debug.assert(size <= bytes and count <= nodes);
    bytes -= size;
    nodes -= count;
}
pub fn usage() struct { bytes: usize, nodes: usize } {
    mutex.lock();
    defer mutex.unlock();
    ensureLimits();
    return .{ .bytes = bytes, .nodes = nodes };
}
pub fn setForTest(value: Limits) void {
    if (!builtin.is_test) @compileError("test only");
    mutex.lock();
    defer mutex.unlock();
    std.debug.assert(bytes == 0 and nodes == 0);
    limits_set = true;
    limits = value;
}
