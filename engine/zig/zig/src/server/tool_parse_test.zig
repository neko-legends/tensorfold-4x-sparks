//! The tool_parse tests, as their own test binary: whole-reply parses and the close-call helpers.
const std = @import("std");
const json = @import("json");
const tool_parse = @import("tool_parse.zig");
const tool_closers = @import("tool_closers.zig");
const parse = tool_parse.parse;
const closeCall = tool_parse.closeCall;

test "families" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][2][]const u8{
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}</tool_call>", "{\"q\":\"x\"}" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n</tool_call>", "{\"n\":5}" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value></tool_call>", "{\"n\":7}" },
        .{ "<|tool_call>call:lookup{q:<|\"|>hi, there<|\"|>,n:3}<tool_call|>", "{\"q\":\"hi, there\",\"n\":3}" },
        .{ "{\"tool\":\"lookup\",\"query\":\"ping\"}", "{\"query\":\"ping\"}" },
    };
    for (cases) |c| {
        const r = try parse(a, c[0], tools, null);
        try std.testing.expectEqualStrings(c[1], r.calls.?[0].get("function").?.get("arguments").?.string);
    }
    const kept = try parse(a, "hi <tool_call>{\"name\":\"other tool\"}</tool_call>", tools, null);
    try std.testing.expect(kept.calls == null);
    try std.testing.expectEqualStrings("hi <tool_call>{\"name\":\"other tool\"}</tool_call>", kept.content);
}

test "a call the end token left open closes when it parses whole" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const cases = [_][4][]const u8{ // the reply, what closes it, the call's arguments, the content left
        .{ "Looking.<tool_call>\n{\"name\":\"lookup\",\"arguments\":{\"q\":\"x\"}}\n", "</tool_call>", "{\"q\":\"x\"}", "Looking." },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":[\"x\"", "]}}</tool_call>", "{\"q\":[\"x\"]}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n", "</function></tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>\n<function=lookup>\n", "</function></tool_call>", "{}", "" },
        .{ "<tool_call>\n<function=lookup>\n<parameter=n>\n5\n</parameter>\n</function>\n", "</tool_call>", "{\"n\":5}", "" },
        .{ "<tool_call>lookup<arg_key>n</arg_key><arg_value>7</arg_value>", "</tool_call>", "{\"n\":7}", "" },
        .{ "<tool_call>{\"name\":\"lookup\"}</tool_call>\n<tool_call>{\"name\":\"lookup\",\"arguments\":{\"n\":2}", "}</tool_call>", "{\"n\":2}", "" },
        .{ "<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"<tool_call>\"}", "}</tool_call>", "{\"q\":\"<tool_call>\"}", "" },
    };
    for (cases) |c| {
        const close = try closeCall(a, c[0], tools);
        try std.testing.expectEqualStrings(c[1], close);
        const r = try parse(a, try std.mem.concat(a, u8, &.{ c[0], close }), tools, null);
        try std.testing.expectEqualStrings(c[2], r.calls.?[r.calls.?.len - 1].get("function").?.get("arguments").?.string);
        try std.testing.expectEqualStrings(c[3], r.content);
    }
}

test "a call left open that would not parse whole stays the reply's text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\",\"parameters\":{\"properties\":{\"n\":{\"type\":\"integer\"}}}}}]")).ok.array;
    const replies = [_][]const u8{
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":\"fast ca", // inside a string
        "Trying.<tool_call>{\"name\":\"lookup\",\"arguments\":{\"q\":", // a key without its value
        "Trying.<tool_call>\n<function=lookup>\n<parameter=q>\nfast ca", // inside a parameter
        "Trying.<tool_call>\n<function=lookup>\n<parameter=n>\n5\n", // before its </parameter>
        "Trying.<tool_call>{\"name\":\"launch rocket!\",\"arguments\":{}}", // no function name
        "Trying.<tool_call>lookup<arg_key>n</arg_key><arg_value>7", // a GLM value left open
        "Trying.<tool_call>", // nothing written
        "Trying <tool_call> tags.<tool_call>{\"name\":\"lookup\"}", // an earlier opener would take the closer
    };
    for (replies) |text| {
        try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
        const r = try parse(a, text, tools, null);
        try std.testing.expect(r.calls == null);
        try std.testing.expectEqualStrings(text, r.content);
    }
}

test "replies that end outside a call close nothing" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const replies = [_][]const u8{
        "Done.",
        "<tool_call>{\"name\":\"lookup\"}</tool_call>",
        "<tool_call>\n<function=lookup>\n</function>\n</tool_call> Done.",
    };
    for (replies) |text| try std.testing.expectEqualStrings("", try closeCall(a, text, tools));
}

test "a call to a tool the request did not declare stays text" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    const whole = [_][]const u8{
        "<tool_call><function=launch><parameter=when>now</parameter></function></tool_call>",
        "<tool_call>{\"name\": \"launch rocket!\", \"arguments\": {\"when\": \"now\"}}</tool_call>",
        "<tool_call>{\"name\": \"launch\", \"arguments\": {\"when\": NaN}}</tool_call>",
    };
    for (whole) |text| {
        const parallel = try parse(a, text, tools, null);
        try std.testing.expect(parallel.calls == null);
        try std.testing.expectEqualStrings(text, parallel.content);
        const single = try parse(a, text, tools, 1);
        try std.testing.expect(single.calls == null);
        try std.testing.expectEqualStrings("", single.content);
    }
    // an unclosed envelope is not a call in either mode, so the reply keeps it
    const open = "<tool_call>{\"name\":\"launch\",\"arguments\":{\"when\":\"now\"}";
    for ([_]?usize{ null, 1 }) |max_calls| {
        const k = try parse(a, open, tools, max_calls);
        try std.testing.expect(k.calls == null);
        try std.testing.expectEqualStrings(open, k.content);
    }
    const prose = "Launching. <tool_call><function=launch><parameter=when>now</function></tool_call>";
    try std.testing.expectEqualStrings("Launching.", (try parse(a, prose, tools, 1)).content);
    // an offered tool keeps its offered spelling; bare JSON naming one not offered stays content
    const offered = try parse(a, "<tool_call>{\"name\":\"LOOKUP\"}</tool_call>", tools, null);
    try std.testing.expectEqualStrings("lookup", offered.calls.?[0].get("function").?.get("name").?.string);
    try std.testing.expect((try parse(a, "{\"name\":\"launch\",\"arguments\":{}}", tools, null)).calls == null);
}
test "namespaced openers without closers read in linear time" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const tools = (try json.parse(a, "[{\"type\":\"function\",\"function\":{\"name\":\"lookup\"}}]")).ok.array;
    // one namespace repeated, then a new namespace per opener
    for ([_]bool{ false, true }) |distinct| {
        var ns: [2]u64 = undefined;
        for ([_]usize{ 1000, 8000 }, &ns) |n, *best| {
            var reply: std.ArrayList(u8) = .empty;
            for (0..n) |i| try reply.print(a, "<ns{d}:tool_call>", .{if (distinct) i else 0});
            best.* = std.math.maxInt(u64);
            for (0..3) |_| {
                var run: std.heap.ArenaAllocator = .init(std.testing.allocator);
                defer run.deinit();
                const start = std.Io.Clock.awake.now(std.testing.io).nanoseconds;
                const r = try parse(run.allocator(), reply.items, tools, null);
                best.* = @min(best.*, @as(u64, @intCast(std.Io.Clock.awake.now(std.testing.io).nanoseconds - start)));
                try std.testing.expect(r.calls == null);
            }
        }
        // 8x the openers: near 8x the time, not 64x
        try std.testing.expect(ns[1] <= 20 * std.time.ns_per_ms or ns[1] <= 24 * ns[0]);
    }
}

test "a closer found by name is the one a search for it finds" {
    var prng: std.Random.DefaultPrng = .init(0x7f4a7c15);
    const r = prng.random();
    const pieces = [_][]const u8{ "</a:tool_call>", "</A:TOOL_CALL>", "</ab:tool_call>", "</b.c:tool_call>", "</a:tool_cal>", "</", "<", "a", ":tool_call>", "x", "</x\u{e9}:tool_call>", "</x\u{c9}:tool_call>" };
    const names = [_][]const u8{ "a", "A", "ab", "b.c", "B.C", "x\u{e9}", "x\u{c9}", "zz" };
    for (0..5_000) |_| {
        var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena.deinit();
        const a = arena.allocator();
        var text: std.ArrayList(u8) = .empty;
        for (0..r.uintLessThan(usize, 16)) |_| try text.appendSlice(a, pieces[r.uintLessThan(usize, pieces.len)]);
        var closers: tool_closers.Closers = .{ .text = text.items };
        // every start in order, every name at each, as the parser asks
        for (0..text.items.len + 1) |from| for (names) |name| {
            const closer = try std.mem.concat(a, u8, &.{ "</", name, ":tool_call>" });
            try std.testing.expectEqual(std.ascii.findIgnoreCasePos(text.items, from, closer), try closers.next(a, name, from));
        };
    }
}

