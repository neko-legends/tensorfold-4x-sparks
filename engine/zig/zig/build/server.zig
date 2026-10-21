//! The HTTP server's and DeepSeek-V4.1 serving side's host builds: unit tests on any host, the tokenizer benchmark,
//! and the engine-less server binary. Goldens beyond the committed ones: TF_DSV41_MODEL (the release tokenizer.json's
//! directory) and TF_DSV41_GOLDEN (zig/tests/server/dsv41/gen_*.py's output directory).
const std = @import("std");
const vision_build = @import("vision.zig");

/// DeepSeek-V4.1's serving module (tokenizer, encoding, DSML tools, sampling) over the modules the server shares.
pub fn serveModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, json: *std.Build.Module, tokenizer: *std.Build.Module, lanes: *std.Build.Module) *std.Build.Module {
    const fixtures = b.createModule(.{ .root_source_file = b.path("zig/src/families/deepseek_v41/fixtures/serve/fixtures.zig"), .target = target, .optimize = optimize });
    return b.createModule(.{
        .root_source_file = b.path("zig/src/families/deepseek_v41/serve/root.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "json", .module = json },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "dsv41_serve_fixtures", .module = fixtures },
            .{ .name = "dsv41_vision", .module = vision_build.module(b, target, optimize, json) },
        },
    });
}

/// A builder that makes its own server root (the HIP, Metal and XPU natives) gets the tree's serving imports:
/// `json` and DeepSeek-V4.1's `dsv41_serve` module (server/deepseek.zig), over its tokenizer and lanes modules.
pub fn attachServing(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, tokenizer: *std.Build.Module, lanes: *std.Build.Module) void {
    if (m.import_table.get("dsv41_serve") != null) return;
    const j = m.import_table.get("json") orelse blk: {
        const nj = jsonModule(b, target, optimize);
        m.addImport("json", nj);
        break :blk nj;
    };
    m.addImport("dsv41_serve", serveModule(b, target, optimize, j, tokenizer, lanes));
}

/// The server's JSON values as a module (zig/src/server/json.zig), so a family's serving code shares them.
pub fn jsonModule(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    return b.createModule(.{
        .root_source_file = b.path("zig/src/server/json.zig"),
        .target = target,
        .optimize = optimize,
    });
}

const Server = struct { json: *std.Build.Module, serve: *std.Build.Module, server: *std.Build.Module, api: *std.Build.Module };

/// The server module (zig/src/server/root.zig) and everything it imports, for ``target``.
fn server(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, root: []const u8, natives: ?*std.Build.Module, build_options: *std.Build.Step.Options) Server {
    const json = b.createModule(.{ .root_source_file = b.path("zig/src/server/json.zig"), .target = target, .optimize = optimize });
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const template = b.createModule(.{ .root_source_file = b.path("zig/src/core/template/template.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const serve = serveModule(b, target, optimize, json, tokenizer, lanes);
    // the server's Living Weights families (slide_model): the engines' own module when there is one, so their types are shared
    const living = if (natives) |n| n.import_table.get("living") orelse b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }) else b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const m = b.createModule(.{
        .root_source_file = b.path(root),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = api },
            .{ .name = "tokenizer", .module = tokenizer },
            .{ .name = "template", .module = template },
            .{ .name = "json", .module = json },
            .{ .name = "dsv41_serve", .module = serve },
            .{ .name = "living", .module = living },
        },
    });
    m.addOptions("build_options", build_options);
    if (natives) |n| {
        n.addImport("engine_api", api);
        n.addImport("lanes", lanes); // an engine on the lane core shares the server's (engine_api's) lanes types
        n.addImport("dsv41_serve", serve); // and its serving module (DeepSeek-V4.1's exact sampler)
        n.addImport("tokenizer", tokenizer); // TF_DSV41_CALIB=measure's calibration text (calib_gpu.zig)
        m.addImport("native_engines", n);
        // models / info / pull: the checkpoint commands over the same engines
        const cli_checkpoint = b.createModule(.{ .root_source_file = b.path("zig/src/cli/cli.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "native_engines", .module = n }} });
        { // EXLR8 node pulls need the exlr8 module graph; this build carries none
            const opts = b.addOptions();
            opts.addOption(bool, "exlr8", false);
            opts.addOption(bool, "lora", false);
            cli_checkpoint.addOptions("build_options", opts);
        }
        if (serve.import_table.get("core")) |serve_core| cli_checkpoint.addImport("core", serve_core);
        if (cli_checkpoint.import_table.get("core")) |cc| cli_checkpoint.addImport("exlr8", @import("exlr8.zig").coreAlias(b, target, optimize, cc));
        m.addImport("checkpoint_cli", cli_checkpoint);
    }
    return .{ .json = json, .serve = serve, .server = m, .api = api };
}

/// tensorfold-native's root module with `natives` as its built-in engines (zig/build/cuda.zig: tensorfold-dsv41).
pub fn withEngine(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, natives: *std.Build.Module, build_options: *std.Build.Step.Options) *std.Build.Module {
    return server(b, target, optimize, "zig/src/server/main.zig", natives, build_options).server;
}

/// `test-dsv41-serve`: the serving module's goldens; `test-server`: the HTTP server's tests (DeepSeek over HTTP
/// included); both part of `test`. `tf-dsv41-tokbench`: throughput. `server-none`: tensorfold-native without an engine.
pub fn tests(b: *std.Build, build_options: *std.Build.Step.Options, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const s = server(b, host, .Debug, "zig/src/server/root.zig", null, build_options);
    const serve_step = b.step("test-dsv41-serve", "DeepSeek-V4.1's serving side: tokenizer, encoding, DSML tools, sampling (goldens: TF_DSV41_MODEL, TF_DSV41_GOLDEN)");
    serve_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.serve })).step);
    test_step.dependOn(serve_step);
    if (host.result.os.tag != .macos) { // macOS runs the server's tests in nativeServer, with its Metal engines
        const server_step = b.step("test-server", "The native HTTP server's tests, DeepSeek-V4.1 over HTTP included (TF_DSV41_MODEL: the release tokenizer too)");
        // 1.1.0's required-tools witness runs a lane fake: the server's tests import the lanes module and the
        // 27B's mask policy fake (a no-op gate on this fork: see tool_required_test's structures note)
        s.server.addImport("lanes", s.api.import_table.get("lanes").?);
        const q27 = @import("qwen27_required.zig").modules(b, s.api.import_table.get("lanes").?);
        s.server.addImport("qwen27_mask_fake", q27.fake);
        s.server.addImport("qwen27_masks", q27.masks);
        server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.server })).step);
        server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.json })).step);
        server_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = s.api })).step);
        test_step.dependOn(server_step);
        // the goldens (cases + security cases) are wired by 1.1.1's zig/build/golden.zig from the CUDA build's
        // host tests, which own `zig build test-golden` on Linux; macOS runs them from nativeServer
        const none = b.createModule(.{ .root_source_file = b.path("zig/src/native/none.zig"), .target = host, .optimize = .ReleaseSafe, .link_libc = true });
        const exe = b.addExecutable(.{ .name = "tensorfold-native", .root_module = server(b, host, .ReleaseSafe, "zig/src/server/main.zig", none, build_options).server });
        b.step("server-none", "tensorfold-native without an engine (the server's own checks) into zig-out/server").dependOn(&b.addInstallArtifact(exe, .{ .dest_dir = .{ .override = .{ .custom = "server" } } }).step);
    }
    const fast = server(b, host, .ReleaseFast, "zig/src/server/root.zig", null, build_options);
    const bench = b.addExecutable(.{ .name = "tf-dsv41-tokbench", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/tests/server/dsv41/tokbench.zig"),
        .target = host,
        .optimize = .ReleaseFast,
        .link_libc = true,
        .imports = &.{.{ .name = "dsv41_serve", .module = fast.serve }},
    }) });
    b.step("tf-dsv41-tokbench", "The DeepSeek-V4.1 tokenizer's throughput over gen_tokenizer.py's corpus").dependOn(&b.addInstallArtifact(bench, .{}).step);
}

/// Adds the shared `json` import (and `lanes` through `engine_api` when present); returns the json module.
/// Attach a server's shared JSON module (tests run apart), and Living Weights' families where the build gave none.
pub fn addTo(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const j = m.import_table.get("json") orelse blk: {
        const nj = jsonModule(b, target, optimize);
        m.addImport("json", nj);
        break :blk nj;
    };
    if (m.import_table.get("engine_api")) |api| if (api.import_table.get("lanes")) |lanes| m.addImport("lanes", lanes);
    // a build whose engines learn hands the server their own living module; the rest read only its families
    if (m.import_table.get("living") == null) m.addImport("living", b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true }));
    return j;
}
