//! ``tensorfold-native``: ``capabilities --json`` for the Python switch, and ``serve MODEL [flags]`` with no Python.
const std = @import("std");
const json = @import("json");
const cli = @import("cli.zig");
const log = @import("log.zig");
const hub = @import("hub.zig");
const engines = @import("engines.zig");
const serve = @import("serve.zig");
const hf_text = @import("hf_text.zig");
const deepseek = @import("deepseek.zig");
const model_text = @import("model_text.zig");
const family = @import("family.zig");
const startup = @import("startup.zig");
const slide_cli = @import("slide_cli.zig");
const slide_key = @import("slide_key.zig");
const api = @import("engine_api");
const recall_open = @import("recall_open.zig");
const benchmark = @import("benchmark.zig");
const checkpoint_cli = @import("checkpoint_cli");

const usage_line = "usage: tensorfold serve [-h] [--host HOST] [--port PORT] [--name NAME] [--chat-template FILE]" ++
    " [--alias ALIAS] [--allowed-host NAME] [--allowed-origin ORIGIN] [--api-key API_KEY] [--api-key-file API_KEY_FILE] [--metrics-open] [--dashboard]" ++
    " [--context CONTEXT] [--speed-up SETTINGS] [--prompt-cache-gib PROMPT_CACHE_GIB]" ++
    " [--prompt-cache-over-cap] [--learn] [--learn-dir LEARN_DIR] [--learn-gib LEARN_GIB]" ++
    " [--learn-conversations] [--learn-daily-gib LEARN_DAILY_GIB] [--recall] [--recall-dir RECALL_DIR]" ++
    " [--recall-gib RECALL_GIB] [--recall-read-only] [--max-tokens MAX_TOKENS]" ++
    " [--temperature TEMPERATURE] [--top-p TOP_P] [--top-k TOP_K] [--min-p MIN_P] [--prompt-seed]" ++
    " [--thinking | --no-thinking] [--reasoning-effort {low,medium,high,xhigh}]" ++
    " [--thinking-budget THINKING_BUDGET] [--loop-guard] [--no-drafts] [--drafter DRAFTER]" ++
    " [--drafter-bits {0,4}] [--record RECORD] [--keep-warm SECONDS] [--compact-at COMPACT_AT]" ++
    " [--compact-keep COMPACT_KEEP]" ++
    " [--compact-memory COMPACT_MEMORY] [--slide] [--slide-graph SLIDE_GRAPH] [--parallel PARALLEL]" ++
    " [--cortheon] [--no-update-check] [--backend {auto,mlx,cuda}] [--device DEVICE]" ++
    " [--segments SEGMENTS] model\n";
const not_lanes = "--recall drafts from a record in the lane engine's rounds, " ++
    "which this model's engine does not run: serve it without --recall, --recall-dir, " ++
    "--recall-gib or --recall-read-only";

/// The flags past the common ones this binary's backend serves; cli.zig reads them from the root.
pub const serves = engines.serves;

pub const std_options = engines.std_options;

pub fn main(init: std.process.Init) !u8 {
    const gpa = init.gpa;
    const io = init.io;
    const a = init.arena.allocator();
    const argv = try init.minimal.args.toSlice(a);
    // These commands must work without a model, driver, GPU or checkout.
    if (argv.len == 2 and std.mem.eql(u8, argv[1], "--version")) {
        try std.Io.File.stdout().writeStreamingAll(io, "tensorfold-native " ++ @import("build_options").version ++ "\n");
        return 0;
    }
    if (argv.len == 2 and (std.mem.eql(u8, argv[1], "--help") or std.mem.eql(u8, argv[1], "-h"))) {
        try std.Io.File.stdout().writeStreamingAll(io, "usage: tensorfold-native --version | capabilities --json | models | info MODEL | pull REPO[@REVISION] | slide rollback FOLDER | serve MODEL [flags] | benchmark MODEL [flags]\n" ++ usage_line);
        return 0;
    }
    if (argv.len >= 3 and std.mem.eql(u8, argv[1], "serve")) {
        for (argv[2..]) |arg| if (std.mem.eql(u8, arg, "--help") or std.mem.eql(u8, arg, "-h")) {
            try std.Io.File.stdout().writeStreamingAll(io, usage_line);
            return 0;
        };
    }
    // the checkpoint commands need no model, GPU or driver either
    if (argv.len >= 2 and checkpoint_cli.wants(@ptrCast(argv[1..]))) return checkpoint_cli.main(init, argv[1..]);
    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "slide")) return slide_cli.main(init, argv[2..]);
    log.init(io, false);
    if (argv.len == 3 and std.mem.eql(u8, argv[1], "capabilities") and std.mem.eql(u8, argv[2], "--json")) {
        var out: std.Io.Writer.Allocating = .init(a);
        var caps = engines.capabilities(a);
        caps.optional_libraries = try @import("sys").optional_libraries.probe(a, io, @import("sys"));
        try cli.capabilities(&out.writer, caps);
        try std.Io.File.stdout().writeStreamingAll(io, out.written());
        return 0;
    }
    if (argv.len >= 2 and std.mem.eql(u8, argv[1], "benchmark")) {
        // the benchmark runs its serve child under POSIX signals and waits; Windows refuses it by name
        if (comptime @import("builtin").os.tag == .windows) {
            std.debug.print("tensorfold benchmark is not supported on Windows yet: run tensorfold serve and a client against it\n", .{});
            return 2;
        } else return benchmark.main(init, argv[2..], engines.capabilities(a));
    }
    if (argv.len < 2 or !std.mem.eql(u8, argv[1], "serve")) {
        std.debug.print("usage: tensorfold-native capabilities --json | models | info MODEL | pull REPO[@REVISION] | slide rollback FOLDER | serve MODEL [flags] | benchmark MODEL [flags]\n", .{});
        return 2;
    }
    const started = std.Io.Clock.awake.now(io).toNanoseconds();
    var u: cli.Usage = .{};
    const args = cli.parse(a, argv[2..], &u) catch |e| switch (e) {
        error.Usage => {
            std.debug.print("{s}tensorfold serve: error: {s}\n", .{ usage_line, u.message });
            return 2;
        },
        else => |x| return x,
    };
    if (try slide_key.slideKeyless(a, args, init.environ_map.get("TENSORFOLD_API_KEY") orelse "")) |why| {
        std.debug.print("tensorfold: {s}\n", .{why});
        return 2;
    }
    var problem: []const u8 = "";
    const dir = try hub.resolve(a, io, init.environ_map, args.model, &problem) orelse return fail(problem);
    const model_type = modelType(a, io, dir, &problem) catch |e| {
        if (e == error.OutOfMemory) return e;
        return fail(problem);
    };
    // DeepSeek-V4.1 has its own tokenizer, encoding and reply rules; every other family reads its chat template
    var text: model_text.Text = undefined;
    var fam: ?family.Family = null;
    var hf: ?*hf_text.HfText = null;
    var ds: ?*deepseek.DeepSeek = null;
    if (std.mem.eql(u8, model_type, deepseek.model_type)) {
        const options = deepseek.Options.fromEnv(init.environ_map, &problem) catch return fail(problem);
        ds = deepseek.DeepSeek.load(gpa, io, dir, options, &problem) catch return fail(problem);
        text = ds.?.text();
        fam = ds.?.family_();
    } else {
        hf = hf_text.HfText.load(gpa, io, dir, args.chat_template, a, &problem) catch |e| return fail(if (problem.len > 0) problem else @errorName(e));
        text = hf.?.text();
    }
    defer if (hf) |t| t.deinit();
    defer if (ds) |d| d.deinit(true);
    const opened = try engines.open(a, gpa, io, dir, model_type, args, &problem) orelse return fail(problem);
    var closer: Closer = .{ .opened = opened };
    defer closer.closeOnce();
    const served = hub.servedName(args.name, args.model, dir);
    var record: ?*@import("engine_api").recall_store.Store = null;
    defer if (record) |r| r.close(); // after the engine below has stopped reading it
    if (hf) |t| {
        record = recall_open.open(gpa, io, a, args, t, served, &problem) catch |e| {
            return if (problem.len > 0) fail(problem) else e;
        };
    } else if (args.recall) return fail(not_lanes);
    if (record) |r| if (!opened.engine.attachRecall(r)) return fail(not_lanes);
    return serve.run(gpa, io, args, .{
        .stop = .{ .ctx = &closer, .halt = if (opened.halt != null) Closer.halt else null, .close = Closer.close },
        .engine = opened.engine,
        .text = text,
        .family = fam,
        .served = served,
        .sampling = try sampling(a, io, dir, args, fam != null),
        .recall = record,
        // DeepSeek-V4.1's Python app is the Spark server's: the same HTTP surface (TENSORFOLD_WIRE overrides)
        .wire = wire(init.environ_map, fam != null) orelse return fail("TENSORFOLD_WIRE: expected spark or tensorfold"),
        .model_dir = dir,
        .environ = init.environ_map,
        .started = started,
    });
}

/// Closes once: serve.run closes the engine before freeing the server, main's defer covers paths that end before it.
const Closer = struct {
    opened: engines.Opened,
    closed: bool = false,

    fn closeOnce(c: *Closer) void {
        if (c.closed) return;
        c.closed = true;
        c.opened.close(c.opened.ctx);
    }

    fn close(ctx: *anyopaque) void {
        closeOnce(@ptrCast(@alignCast(ctx)));
    }

    fn halt(ctx: *anyopaque, reason: []const u8) void {
        const c: *Closer = @ptrCast(@alignCast(ctx));
        if (c.opened.halt) |h| h(c.opened.ctx, reason);
    }
};

fn wire(env: ?*const std.process.Environ.Map, family_default: bool) ?@import("spark.zig").Wire {
    const raw = if (env) |m| m.get("TENSORFOLD_WIRE") else null;
    const t = std.mem.trim(u8, raw orelse "", " \t");
    if (t.len == 0) return if (family_default) .spark else .tensorfold;
    return std.meta.stringToEnum(@import("spark.zig").Wire, t);
}

/// TF_SLIDE_FORGET_FAIL in a -Dslide-test-hooks build: the fact whose check a delete fails, so its put-back is seen.
fn forgetFail(env: *const std.process.Environ.Map) u32 {
    const v = env.get("TF_SLIDE_FORGET_FAIL") orelse return 0;
    return std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch 0;
}

/// HfText.load with every failure named in `problem`, so main tells the text's failure from the engine's.
fn loadText(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, template: ?[]const u8, pa: std.mem.Allocator, problem: *[]const u8) !*hf_text.HfText {
    return hf_text.HfText.load(gpa, io, dir, template, pa, problem) catch |e| {
        if (problem.len == 0) problem.* = @errorName(e);
        return e;
    };
}

fn fail(message: []const u8) u8 {
    std.debug.print("tensorfold: {s}\n", .{message});
    return 1;
}

fn modelType(a: std.mem.Allocator, io: std.Io, dir: []const u8, problem: *[]const u8) ![]const u8 {
    const bytes = (try @import("checkpoint_config").read(a, io, dir, problem)) orelse return "unknown";
    const doc = std.json.parseFromSliceLeaky(std.json.Value, a, bytes, .{}) catch return "unknown";
    if (doc != .object) return "unknown";
    const t = doc.object.get("model_type") orelse return "unknown";
    return if (t == .string) t.string else "unknown";
}

/// generation_config.json's sampling (``do_sample`` false is greedy), then the serve flags over it.
fn sampling(a: std.mem.Allocator, io: std.Io, dir: []const u8, args: cli.Args, python_defaults: bool) !?json.Value {
    const out = try json.newObject(a);
    if (python_defaults) { // the Spark server's own defaults under the checkpoint's and the flags (DeepSeek-V4.1)
        try out.put(a, "temperature", .{ .float = 1.0 });
        try out.put(a, "top_k", .{ .int = "20" });
        try out.put(a, "top_p", .{ .float = 0.95 });
    }
    const path = try std.fs.path.join(a, &.{ dir, "generation_config.json" });
    if (std.Io.Dir.cwd().readFileAlloc(io, path, a, .limited(1 << 20))) |bytes| {
        if ((try json.parse(a, bytes)) == .ok) {
            const cfg = (try json.parse(a, bytes)).ok;
            for ([_][]const u8{ "temperature", "top_k", "top_p", "min_p" }) |k| if (cfg.field(k)) |v| try out.put(a, k, v);
            if (cfg.get("do_sample")) |d| if (d == .bool) {
                if (!d.bool) try out.put(a, "temperature", .{ .float = 0 }) else if (out.get("temperature") == null) try out.put(a, "temperature", .{ .float = 1 });
            };
        }
    } else |_| {}
    if (args.temperature) |t| try out.put(a, "temperature", .{ .float = t });
    if (args.top_p) |t| try out.put(a, "top_p", .{ .float = t });
    if (args.top_k) |t| try out.put(a, "top_k", try json.intValue(a, t));
    if (args.min_p) |t| try out.put(a, "min_p", .{ .float = t });
    return if (out.count() > 0) json.Value{ .object = out } else null;
}

