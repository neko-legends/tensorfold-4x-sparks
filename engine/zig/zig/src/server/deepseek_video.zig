//! Video input for DeepSeek-V4.1's native vision (TF_DSV41_IMAGES=native): each ``video_url`` part holding a base64
//! data URL becomes its sampled frames, in order: a text part naming the frame's time, then an ordinary PNG
//! ``image_url`` part. The image path (marker, cache, virtual ids) then sees only text and images. Remote video URLs
//! are refused. The frames count against the request's image budget (TF_DSV41_VISION_MAX_IMAGES).
//!
//! The frames come from a one-shot ``python3 -I`` child with PyAV and Pillow, behind ``Decoder`` (tests plug a stub).
//! Every limit is checked here on whatever the decoder returns, and the child enforces its own wall-clock limit:
//!
//! - TF_VIDEO_MAX_FRAMES (8; 1-64): frames a clip yields; a longer clip's frames spread evenly over it
//! - TF_VIDEO_FPS (2; 0.1-30): frames a second sampled from a shorter clip
//! - TF_VIDEO_MAX_DURATION (300 s; 1-3600): longer clips are refused
//! - TF_VIDEO_MAX_SIDE (768; 32-2048): a frame's longer side in pixels
//! - TF_VIDEO_MAX_OUTPUT_BYTES (16 MiB): the decoder's whole reply
//! - TF_VIDEO_TIMEOUT (30 s; 1-600): the child's run time (SIGALRM in the child)
//! - TF_VIDEO_MAX_INPUT_BYTES (32 MiB): a clip's decoded size
//! - TF_VIDEO_PYTHON (python3): the interpreter
const std = @import("std");
const json = @import("json");
const errors = @import("errors.zig");
const Value = json.Value;
const Cx = errors.Cx;
const Allocator = std.mem.Allocator;

pub const Config = struct {
    max_frames: u32 = 8,
    fps: f64 = 2,
    max_duration: f64 = 300,
    max_side: u32 = 768,
    max_output_bytes: usize = 16 << 20,
    timeout_s: u32 = 30,
    max_input_bytes: usize = 32 << 20,
    python: []const u8 = "python3",

    /// The TF_VIDEO_* knobs; ``problem`` says which one is wrong. Values are clamped to the ranges above.
    pub fn fromEnv(env: ?*const std.process.Environ.Map, problem: *[]const u8) error{Invalid}!Config {
        var c: Config = .{};
        const m = env orelse return c;
        c.max_frames = try int(m, "TF_VIDEO_MAX_FRAMES", u32, c.max_frames, 1, 64, problem);
        c.fps = try float(m, "TF_VIDEO_FPS", c.fps, 0.1, 30, problem);
        c.max_duration = try float(m, "TF_VIDEO_MAX_DURATION", c.max_duration, 1, 3600, problem);
        c.max_side = try int(m, "TF_VIDEO_MAX_SIDE", u32, c.max_side, 32, 2048, problem);
        c.max_output_bytes = try int(m, "TF_VIDEO_MAX_OUTPUT_BYTES", usize, c.max_output_bytes, 1 << 16, 256 << 20, problem);
        c.timeout_s = try int(m, "TF_VIDEO_TIMEOUT", u32, c.timeout_s, 1, 600, problem);
        c.max_input_bytes = try int(m, "TF_VIDEO_MAX_INPUT_BYTES", usize, c.max_input_bytes, 1 << 10, 1 << 30, problem);
        if (m.get("TF_VIDEO_PYTHON")) |v| if (std.mem.trim(u8, v, " ").len > 0) {
            c.python = std.mem.trim(u8, v, " ");
        };
        return c;
    }

    fn int(m: *const std.process.Environ.Map, name: []const u8, comptime T: type, default: T, lo: T, hi: T, problem: *[]const u8) error{Invalid}!T {
        const v = m.get(name) orelse return default;
        const t = std.mem.trim(u8, v, " ");
        if (t.len == 0) return default;
        const n = std.fmt.parseInt(T, t, 10) catch {
            problem.* = std.fmt.allocPrint(std.heap.page_allocator, "{s}: expected an integer {d}-{d}", .{ name, lo, hi }) catch name;
            return error.Invalid;
        };
        return std.math.clamp(n, lo, hi);
    }

    fn float(m: *const std.process.Environ.Map, name: []const u8, default: f64, lo: f64, hi: f64, problem: *[]const u8) error{Invalid}!f64 {
        const v = m.get(name) orelse return default;
        const t = std.mem.trim(u8, v, " ");
        if (t.len == 0) return default;
        const n = std.fmt.parseFloat(f64, t) catch {
            problem.* = std.fmt.allocPrint(std.heap.page_allocator, "{s}: expected a number {d}-{d}", .{ name, lo, hi }) catch name;
            return error.Invalid;
        };
        if (!std.math.isFinite(n)) {
            problem.* = name;
            return error.Invalid;
        }
        return std.math.clamp(n, lo, hi);
    }
};

pub const Frame = struct { t: f64, png: []const u8 };

/// Whatever turns a clip's bytes into at most ``cfg.max_frames`` PNG frames: refuses (through ``cx``) with the reason.
pub const Decoder = struct {
    ctx: ?*anyopaque = null,
    decode: *const fn (ctx: ?*anyopaque, cx: *Cx, io: std.Io, data: []const u8, cfg: Config) errors.Refused![]Frame,
};

/// The PyAV child.
pub const python: Decoder = .{ .decode = pyDecode };

/// Ahead of the decoder script: the child's own wall-clock limit (SIGALRM's default action ends it, in C code too).
const prelude =
    \\import io, os, signal, struct, sys
    \\fps, cap, side, dur_max, tmo, out_max = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
    \\signal.alarm(tmo)
    \\
;

const script =
    \\def fail(msg):
    \\    sys.stdout.buffer.write(b"E" + msg.encode()[:400]); sys.stdout.buffer.flush(); os._exit(0)
    \\try:
    \\    import av
    \\    from PIL import Image
    \\except Exception as e:
    \\    fail("video input needs PyAV and Pillow in the server's Python: %s" % e)
    \\data = sys.stdin.buffer.read()
    \\try:
    \\    c = av.open(io.BytesIO(data))
    \\    s = c.streams.video[0]
    \\    dur = float(s.duration * s.time_base) if s.duration else (c.duration / 1e6 if c.duration else None)
    \\    if dur is not None and dur > dur_max:
    \\        fail("video input: the clip is %.0f s long; at most %.0f s (TF_VIDEO_MAX_DURATION)" % (dur, dur_max))
    \\    if dur and dur * fps > cap:
    \\        want = [i * dur / cap for i in range(cap)]
    \\    elif dur:
    \\        want = [i / fps for i in range(max(1, min(cap, int(dur * fps + 0.999))))]
    \\    else:
    \\        want = [i / fps for i in range(cap)]
    \\    out, k, total = [], 0, 0
    \\    for f in c.decode(s):
    \\        t = float(f.pts * f.time_base) if f.pts is not None else 0.0
    \\        if t > dur_max: break
    \\        if k < len(want) and t + 1e-6 >= want[k]:
    \\            im = f.to_image().convert("RGB")
    \\            im.thumbnail((side, side))
    \\            b = io.BytesIO(); im.save(b, "PNG")
    \\            out.append((max(t, 0.0), b.getvalue())); total += len(out[-1][1])
    \\            if total > out_max: fail("video input: the frames exceed %d bytes (TF_VIDEO_MAX_OUTPUT_BYTES)" % out_max)
    \\            while k < len(want) and t + 1e-6 >= want[k]: k += 1
    \\        if k >= len(want): break
    \\    if not out: raise ValueError("the video has no decodable frames")
    \\except Exception as e:
    \\    fail("video input: %s" % e)
    \\w = sys.stdout.buffer
    \\w.write(b"F" + struct.pack("<Q", len(out)))
    \\for t, png in out: w.write(struct.pack("<dQ", t, len(png)) + png)
    \\w.flush()
;

fn pyDecode(_: ?*anyopaque, cx: *Cx, io: std.Io, data: []const u8, cfg: Config) errors.Refused![]Frame {
    return runChild(cx, io, script, data, cfg);
}

/// ``body`` (the decoder script) behind the prelude, run on ``data``; its reply is "E" and a message, or "F", a count
/// and per frame a time (f64) and a length (u64) then the PNG, all little-endian.
fn runChild(cx: *Cx, io: std.Io, body: []const u8, data: []const u8, cfg: Config) errors.Refused![]Frame {
    const a = cx.a;
    const code = try std.mem.concat(a, u8, &.{ prelude, body });
    const argv = [_][]const u8{
        cfg.python,                                       "-I",                                          "-c", code,
        try std.fmt.allocPrint(a, "{d}", .{cfg.fps}),     try std.fmt.allocPrint(a, "{d}", .{cfg.max_frames}),
        try std.fmt.allocPrint(a, "{d}", .{cfg.max_side}), try std.fmt.allocPrint(a, "{d}", .{cfg.max_duration}),
        try std.fmt.allocPrint(a, "{d}", .{cfg.timeout_s}), try std.fmt.allocPrint(a, "{d}", .{cfg.max_output_bytes}),
    };
    var child = std.process.spawn(io, .{ .argv = &argv, .stdin = .pipe, .stdout = .pipe, .stderr = .inherit }) catch
        return cx.refuse("video input needs python3 in the server's environment (TF_VIDEO_PYTHON)");
    // the bytes go in on a thread: the child streams frames back while it reads
    const Feeder = struct {
        fn run(f: std.Io.File, i: std.Io, d: []const u8) void {
            f.writeStreamingAll(i, d) catch {};
            f.close(i);
        }
    };
    const in = child.stdin.?;
    child.stdin = null;
    const th = std.Thread.spawn(.{}, Feeder.run, .{ in, io, data }) catch {
        in.close(io);
        child.kill(io);
        return cx.refuse("video input: could not start the decoder");
    };
    var waited = false;
    defer {
        if (!waited) child.kill(io); // an early return: end the child before waiting for the feeder
        th.join();
    }
    var buf: [64 << 10]u8 = undefined;
    var r = child.stdout.?.readerStreaming(io, &buf);
    const all = r.interface.allocRemaining(a, .limited(cfg.max_output_bytes + 1)) catch |e| return switch (e) {
        error.StreamTooLong => cx.fail(.request, "video input: the decoder's reply exceeds {d} bytes (TF_VIDEO_MAX_OUTPUT_BYTES)", .{cfg.max_output_bytes}),
        error.OutOfMemory => error.OutOfMemory,
        else => cx.refuse("video input: the decoder's reply could not be read"),
    };
    const term = child.wait(io) catch null;
    waited = true;
    if (all.len == 0) {
        if (term) |t| if (t == .signal and t.signal == .ALRM) return cx.fail(.request, "video input: decoding took longer than {d} s (TF_VIDEO_TIMEOUT)", .{cfg.timeout_s});
        return cx.refuse("video input: the decoder gave no reply");
    }
    if (all[0] == 'E') return cx.refuse(try a.dupe(u8, all[1..]));
    if (all[0] != 'F' or all.len < 9) return cx.refuse("video input: the decoder's reply is malformed");
    const n = std.mem.readInt(u64, all[1..9], .little);
    if (n > cfg.max_frames) return cx.fail(.request, "video input: the decoder returned {d} frames; at most {d} (TF_VIDEO_MAX_FRAMES)", .{ n, cfg.max_frames });
    const out = try a.alloc(Frame, @intCast(n));
    var at: usize = 9;
    for (out) |*f| {
        if (at + 16 > all.len) return cx.refuse("video input: the decoder's reply is short");
        f.t = @bitCast(std.mem.readInt(u64, all[at..][0..8], .little));
        const len = std.mem.readInt(u64, all[at + 8 ..][0..8], .little);
        at += 16;
        if (len > all.len - at) return cx.refuse("video input: the decoder's reply is short");
        f.png = all[at..][0..@intCast(len)];
        at += @intCast(len);
    }
    return out;
}

const png_magic = "\x89PNG\r\n\x1a\n";

/// The frames a decoder gave, held to the limits whoever decoded them.
fn check(cx: *Cx, frames: []const Frame, cfg: Config) errors.Refused!void {
    if (frames.len == 0) return cx.refuse("video input: the video has no decodable frames");
    if (frames.len > cfg.max_frames) return cx.fail(.request, "video input: {d} frames; at most {d} (TF_VIDEO_MAX_FRAMES)", .{ frames.len, cfg.max_frames });
    var total: usize = 0;
    var last: f64 = 0;
    for (frames) |f| {
        total += f.png.len;
        if (total > cfg.max_output_bytes) return cx.fail(.request, "video input: the frames exceed {d} bytes (TF_VIDEO_MAX_OUTPUT_BYTES)", .{cfg.max_output_bytes});
        if (!std.math.isFinite(f.t) or f.t < 0 or f.t > cfg.max_duration or f.t < last) return cx.refuse("video input: the decoder gave a frame time out of range");
        last = f.t;
        if (f.png.len < 24 or !std.mem.eql(u8, f.png[0..8], png_magic) or !std.mem.eql(u8, f.png[12..16], "IHDR")) return cx.refuse("video input: the decoder gave a frame that is not a PNG");
        const w = std.mem.readInt(u32, f.png[16..20], .big);
        const h = std.mem.readInt(u32, f.png[20..24], .big);
        if (w == 0 or h == 0 or w > cfg.max_side or h > cfg.max_side) return cx.fail(.request, "video input: a frame is {d}x{d}; at most {d} a side (TF_VIDEO_MAX_SIDE)", .{ w, h, cfg.max_side });
    }
}

fn isVideoPart(part: Value) bool {
    return part == .object and part.typeIs("video_url");
}

fn isImagePart(part: Value) bool {
    return part == .object and (part.typeIs("image_url") or part.typeIs("image") or part.typeIs("input_image"));
}

/// Whether any message's content list holds a video part.
pub fn hasVideos(messages: Value) bool {
    if (messages != .array) return false;
    for (messages.array) |m| {
        const content = (if (m == .object) m.get("content") else null) orelse continue;
        if (content != .array) continue;
        for (content.array) |p| if (isVideoPart(p)) return true;
    }
    return false;
}

fn imageCount(messages: Value) usize {
    var n: usize = 0;
    for (messages.array) |m| {
        const content = (if (m == .object) m.get("content") else null) orelse continue;
        if (content != .array) continue;
        for (content.array) |p| {
            if (isImagePart(p)) n += 1;
        }
    }
    return n;
}

/// A ``video_url`` part's bytes: its URL (the part's string, or its object's ``url``) a base64 data URL.
fn videoBytes(cx: *Cx, part: Value, cfg: Config) errors.Refused![]u8 {
    const url: []const u8 = blk: {
        const r = part.get("video_url") orelse break :blk "";
        if (r == .string) break :blk r.string;
        if (r == .object) if (r.get("url")) |u| if (u == .string) break :blk u.string;
        break :blk "";
    };
    if (url.len == 0) return cx.refuse("video input: a video_url part needs a url");
    if (!std.ascii.startsWithIgnoreCase(url, "data:")) return cx.refuse("video input: video_url must be a base64 data: URL (remote video URLs are not fetched)");
    const comma = std.mem.indexOfScalar(u8, url, ',') orelse return cx.refuse("video input: a data: URL without a comma");
    if (std.mem.indexOf(u8, try std.ascii.allocLowerString(cx.a, url[0..comma]), ";base64") == null) return cx.refuse("video input: video data URLs must be base64");
    const b64 = std.mem.trim(u8, url[comma + 1 ..], " \r\n\t");
    const dec = std.base64.standard.Decoder;
    const size = dec.calcSizeForSlice(b64) catch return cx.refuse("video input: the base64 data is malformed");
    if (size == 0) return cx.refuse("video input: the video is empty");
    if (size > cfg.max_input_bytes) return cx.fail(.request, "video input: the clip is larger than {d} bytes (TF_VIDEO_MAX_INPUT_BYTES)", .{cfg.max_input_bytes});
    const out = try cx.a.alloc(u8, size);
    dec.decode(out, b64) catch return cx.refuse("video input: the base64 data is malformed");
    return out;
}

fn textPart(a: Allocator, s: []const u8) Allocator.Error!Value {
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "text" });
    try o.put(a, "text", .{ .string = s });
    return .{ .object = o };
}

fn imagePart(a: Allocator, png: []const u8) Allocator.Error!Value {
    const enc = std.base64.standard.Encoder;
    const prefix = "data:image/png;base64,";
    const url = try a.alloc(u8, prefix.len + enc.calcSize(png.len));
    @memcpy(url[0..prefix.len], prefix);
    _ = enc.encode(url[prefix.len..], png);
    const inner = try json.newObject(a);
    try inner.put(a, "url", .{ .string = url });
    const o = try json.newObject(a);
    try o.put(a, "type", .{ .string = "image_url" });
    try o.put(a, "image_url", .{ .object = inner });
    return .{ .object = o };
}

/// ``messages`` with every video part replaced by its frames (a "Video frame at <t> s:" text part, then the picture,
/// in time order); the order of every other part is kept. ``max_images`` is the request's image budget, which the
/// frames share with its image parts. One decoder run a video part; none when there is no video part.
pub fn expand(cx: *Cx, io: std.Io, dec: Decoder, cfg: Config, max_images: usize, messages: Value) errors.Refused!Value {
    if (!hasVideos(messages)) return messages;
    var used = imageCount(messages);
    const out = try cx.a.alloc(Value, messages.array.len);
    for (messages.array, out) |m, *slot| {
        slot.* = m;
        const content = (if (m == .object) m.get("content") else null) orelse continue;
        if (content != .array) continue;
        var parts: std.ArrayList(Value) = .empty;
        var any = false;
        for (content.array) |p| {
            if (!isVideoPart(p)) {
                try parts.append(cx.a, p);
                continue;
            }
            any = true;
            const bytes = try videoBytes(cx, p, cfg);
            if (used >= max_images) return cx.fail(.request, "video input: the request's {d} image(s) leave no room for frames; at most {d} images and frames (TF_DSV41_VISION_MAX_IMAGES)", .{ used, max_images });
            const frames = try dec.decode(dec.ctx, cx, io, bytes, cfg);
            try check(cx, frames, cfg);
            used += frames.len;
            if (used > max_images) return cx.fail(.request, "video input: {d} images and video frames in one request: at most {d} (TF_DSV41_VISION_MAX_IMAGES; TF_VIDEO_MAX_FRAMES frames a clip)", .{ used, max_images });
            for (frames) |f| {
                try parts.append(cx.a, try textPart(cx.a, try std.fmt.allocPrint(cx.a, "Video frame at {d:.1} s:", .{f.t})));
                try parts.append(cx.a, try imagePart(cx.a, f.png));
            }
        }
        if (!any) continue;
        const copy = try json.copyObject(cx.a, m.object);
        try copy.put(cx.a, "content", .{ .array = parts.items });
        slot.* = .{ .object = copy };
    }
    return .{ .array = out };
}

// -- tests ----------------------------------------------------------------------------------------------------------

/// A PNG's signature and IHDR (no pixel data: ``check`` reads nothing more).
fn fakePng(comptime w: u32, comptime h: u32) *const [24]u8 {
    const bytes = comptime blk: {
        var b: [24]u8 = undefined;
        @memcpy(b[0..8], png_magic);
        std.mem.writeInt(u32, b[8..12], 13, .big);
        @memcpy(b[12..16], "IHDR");
        std.mem.writeInt(u32, b[16..20], w, .big);
        std.mem.writeInt(u32, b[20..24], h, .big);
        break :blk b;
    };
    return &bytes;
}

/// A decoder that counts its runs and answers with fixed frames (or a refusal), for this file's and DeepSeek's tests.
pub const Stub = struct {
    calls: usize = 0,
    frames: []const Frame = &.{},
    fail: ?[]const u8 = null,
    seen: usize = 0,

    pub fn decoder(s: *Stub) Decoder {
        return .{ .ctx = s, .decode = run };
    }
    fn run(ctx: ?*anyopaque, cx: *Cx, _: std.Io, data: []const u8, _: Config) errors.Refused![]Frame {
        const s: *Stub = @ptrCast(@alignCast(ctx.?));
        s.calls += 1;
        s.seen = data.len;
        if (s.fail) |m| return cx.refuse(m);
        return @constCast(s.frames);
    }
};

fn parse(cx: *Cx, comptime text: []const u8) !Value {
    return (try json.parseText(cx.a, text)).ok;
}

const good_frames = [_]Frame{ .{ .t = 0, .png = fakePng(64, 48) }, .{ .t = 0.5, .png = fakePng(64, 48) }, .{ .t = 1.5, .png = fakePng(32, 32) } };
const clip = "data:video/mp4;base64,AAAAIGZ0eXBpc29t"; // 12 bytes

test "a video part becomes timestamp text and PNG image parts, in place among the other parts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var stub: Stub = .{ .frames = &good_frames };
    const m = try parse(&cx, "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"before\"},{\"type\":\"video_url\",\"video_url\":{\"url\":\"" ++ clip ++ "\"}},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}},{\"type\":\"text\",\"text\":\"after\"}]},{\"role\":\"assistant\",\"content\":\"hi\"}]");
    const got = try expand(&cx, std.testing.io, stub.decoder(), .{}, 8, m);
    try std.testing.expectEqual(@as(usize, 1), stub.calls);
    try std.testing.expectEqual(@as(usize, 12), stub.seen);
    const parts = got.array[0].get("content").?.array;
    try std.testing.expectEqual(@as(usize, 1 + 6 + 1 + 1), parts.len);
    try std.testing.expectEqualStrings("before", parts[0].strField("text").?);
    try std.testing.expectEqualStrings("Video frame at 0.0 s:", parts[1].strField("text").?);
    try std.testing.expect(parts[2].typeIs("image_url"));
    try std.testing.expectEqualStrings("Video frame at 0.5 s:", parts[3].strField("text").?);
    try std.testing.expectEqualStrings("Video frame at 1.5 s:", parts[5].strField("text").?);
    const url = parts[6].get("image_url").?.strField("url").?;
    try std.testing.expect(std.mem.startsWith(u8, url, "data:image/png;base64,iVBORw0KGgo"));
    try std.testing.expectEqualStrings("data:image/png;base64,AA==", parts[7].get("image_url").?.strField("url").?); // the client's own image, after the frames
    try std.testing.expectEqualStrings("after", parts[8].strField("text").?);
    try std.testing.expectEqualStrings("hi", got.array[1].strField("content").?); // untouched
}

test "a string video_url works; no video part: the messages come back as they were, no decoder run" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var stub: Stub = .{ .frames = &good_frames };
    const m = try parse(&cx, "[{\"role\":\"user\",\"content\":[{\"type\":\"video_url\",\"video_url\":\"" ++ clip ++ "\"}]}]");
    _ = try expand(&cx, std.testing.io, stub.decoder(), .{}, 8, m);
    try std.testing.expectEqual(@as(usize, 1), stub.calls);
    const plain = try parse(&cx, "[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"x\"}]}]");
    const same = try expand(&cx, std.testing.io, stub.decoder(), .{}, 8, plain);
    try std.testing.expectEqual(plain.array.ptr, same.array.ptr);
    try std.testing.expectEqual(@as(usize, 1), stub.calls);
}

fn expectRefused(comptime url: []const u8, stub: *Stub, cfg: Config, max_images: usize, needle: []const u8) !void {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    const m = try parse(&cx, "[{\"role\":\"user\",\"content\":[{\"type\":\"video_url\",\"video_url\":{\"url\":\"" ++ url ++ "\"}}]}]");
    try std.testing.expectError(error.Refused, expand(&cx, std.testing.io, stub.decoder(), cfg, max_images, m));
    if (std.mem.indexOf(u8, cx.message, needle) == null) {
        std.debug.print("refusal {s} lacks {s}\n", .{ cx.message, needle });
        return error.WrongRefusal;
    }
}

test "remote URLs, non-base64 and malformed or empty data are refused before any decoder run" {
    var stub: Stub = .{ .frames = &good_frames };
    try expectRefused("https://example.com/a.mp4", &stub, .{}, 8, "remote video URLs");
    try expectRefused("data:video/mp4,AAAA", &stub, .{}, 8, "must be base64");
    try expectRefused("data:video/mp4;base64,A", &stub, .{}, 8, "malformed");
    try expectRefused("data:video/mp4;base64,@@@@", &stub, .{}, 8, "malformed");
    try expectRefused("data:video/mp4;base64,", &stub, .{}, 8, "empty");
    try expectRefused("", &stub, .{}, 8, "needs a url");
    try expectRefused(clip, &stub, .{ .max_input_bytes = 4 }, 8, "TF_VIDEO_MAX_INPUT_BYTES");
    try std.testing.expectEqual(@as(usize, 0), stub.calls);
}

test "a decoder's refusal (corrupt video) and an empty frame list are refused" {
    var stub: Stub = .{ .fail = "video input: invalid data found when processing input" };
    try expectRefused(clip, &stub, .{}, 8, "invalid data");
    var none: Stub = .{};
    try expectRefused(clip, &none, .{}, 8, "no decodable frames");
}

test "frame count, time, side and output size are held to the limits whatever the decoder does" {
    var many: Stub = .{ .frames = &good_frames };
    try expectRefused(clip, &many, .{ .max_frames = 2 }, 8, "TF_VIDEO_MAX_FRAMES");
    try expectRefused(clip, &many, .{ .max_side = 48 }, 8, "TF_VIDEO_MAX_SIDE");
    try expectRefused(clip, &many, .{ .max_output_bytes = 40 }, 8, "TF_VIDEO_MAX_OUTPUT_BYTES");
    try expectRefused(clip, &many, .{ .max_duration = 1 }, 8, "out of range"); // a frame at 1.5 s of a 1 s limit
    const backwards = [_]Frame{ .{ .t = 1, .png = fakePng(8, 8) }, .{ .t = 0.5, .png = fakePng(8, 8) } };
    var b: Stub = .{ .frames = &backwards };
    try expectRefused(clip, &b, .{}, 8, "out of range");
    const nan = [_]Frame{.{ .t = std.math.nan(f64), .png = fakePng(8, 8) }};
    var n: Stub = .{ .frames = &nan };
    try expectRefused(clip, &n, .{}, 8, "out of range");
    const bad = [_]Frame{.{ .t = 0, .png = "not a png at all, but long enough" }};
    var p: Stub = .{ .frames = &bad };
    try expectRefused(clip, &p, .{}, 8, "not a PNG");
}

test "frames share the request's image budget with its image parts" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var stub: Stub = .{ .frames = &good_frames };
    const m = try parse(&cx, "[{\"role\":\"user\",\"content\":[{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,AA==\"}},{\"type\":\"video_url\",\"video_url\":\"" ++ clip ++ "\"}]}]");
    _ = try expand(&cx, std.testing.io, stub.decoder(), .{}, 4, m); // 1 + 3 fits
    try std.testing.expectError(error.Refused, expand(&cx, std.testing.io, stub.decoder(), .{}, 3, m)); // 1 + 3 > 3
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "TF_DSV41_VISION_MAX_IMAGES") != null);
    const calls = stub.calls;
    try std.testing.expectError(error.Refused, expand(&cx, std.testing.io, stub.decoder(), .{}, 1, m)); // no room at all: no decoder run
    try std.testing.expectEqual(calls, stub.calls);
}

test "TF_VIDEO_* knobs are read and clamped" {
    var env = std.process.Environ.Map.init(std.testing.allocator);
    defer env.deinit();
    var problem: []const u8 = "";
    const d = try Config.fromEnv(&env, &problem);
    try std.testing.expectEqual(@as(u32, 8), d.max_frames);
    try env.put("TF_VIDEO_MAX_FRAMES", "1000");
    try env.put("TF_VIDEO_FPS", "0.001");
    try env.put("TF_VIDEO_MAX_SIDE", "9999");
    try env.put("TF_VIDEO_TIMEOUT", "5");
    const c = try Config.fromEnv(&env, &problem);
    try std.testing.expectEqual(@as(u32, 64), c.max_frames);
    try std.testing.expectEqual(@as(f64, 0.1), c.fps);
    try std.testing.expectEqual(@as(u32, 2048), c.max_side);
    try std.testing.expectEqual(@as(u32, 5), c.timeout_s);
    try env.put("TF_VIDEO_MAX_DURATION", "soon");
    try std.testing.expectError(error.Invalid, Config.fromEnv(&env, &problem));
}

test "the child's timeout ends a decoder that hangs; a child that cannot start is a refusal" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var probe = std.process.spawn(std.testing.io, .{ .argv = &.{ "python3", "-c", "pass" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore }) catch return error.SkipZigTest;
    _ = probe.wait(std.testing.io) catch {};
    const t0 = std.Io.Clock.awake.now(std.testing.io);
    try std.testing.expectError(error.Refused, runChild(&cx, std.testing.io, "import time\ntime.sleep(60)\n", "x", .{ .timeout_s = 1 }));
    const took = t0.durationTo(std.Io.Clock.awake.now(std.testing.io)).nanoseconds;
    try std.testing.expect(took < 20 * std.time.ns_per_s);
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "TF_VIDEO_TIMEOUT") != null);
    try std.testing.expectError(error.Refused, runChild(&cx, std.testing.io, "pass\n", "x", .{ .python = "/nonexistent/python" }));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "needs python3") != null);
}

test "the child's reply is parsed; an oversized or short reply is refused" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    var cx: Cx = .{ .a = arena.allocator() };
    var probe = std.process.spawn(std.testing.io, .{ .argv = &.{ "python3", "-c", "pass" }, .stdin = .ignore, .stdout = .ignore, .stderr = .ignore }) catch return error.SkipZigTest;
    _ = probe.wait(std.testing.io) catch {};
    const ok = "w = sys.stdout.buffer\nw.write(b'F' + struct.pack('<Q', 1) + struct.pack('<dQ', 1.5, 3) + b'abc')\n";
    const got = try runChild(&cx, std.testing.io, ok, "x", .{});
    try std.testing.expectEqual(@as(usize, 1), got.len);
    try std.testing.expectEqual(@as(f64, 1.5), got[0].t);
    try std.testing.expectEqualStrings("abc", got[0].png);
    const big = "sys.stdout.buffer.write(b'F' + b'x' * 200000)\n";
    try std.testing.expectError(error.Refused, runChild(&cx, std.testing.io, big, "x", .{ .max_output_bytes = 1 << 16 }));
    try std.testing.expect(std.mem.indexOf(u8, cx.message, "TF_VIDEO_MAX_OUTPUT_BYTES") != null);
    const short = "sys.stdout.buffer.write(b'F' + struct.pack('<Q', 1) + b'zz')\n";
    try std.testing.expectError(error.Refused, runChild(&cx, std.testing.io, short, "x", .{}));
    const err = "sys.stdout.buffer.write(b'Ebroken file')\n";
    try std.testing.expectError(error.Refused, runChild(&cx, std.testing.io, err, "x", .{}));
    try std.testing.expectEqualStrings("broken file", cx.message);
}
