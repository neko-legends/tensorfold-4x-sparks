//! Grammar-constrained decoding: xgrammar 0.2.8's C++ core (zig/third_party/xgrammar, Apache-2.0, the sdist's own
//! sources) compiled by Zig's C++ toolchain with the C ABI in zig/src/core/grammar/xgr_c.cc, and the `grammar`
//! module over it. `test-grammar`: the masks, cuts and compiles against xgrammar's Python package on recorded goldens.

const std = @import("std");

const xgr_sources = [_][]const u8{
    "compiled_grammar.cc",        "config.cc",                "earley_parser.cc",            "fsm.cc",
    "fsm_builder.cc",             "grammar.cc",               "grammar_builder.cc",          "grammar_compiler.cc",
    "grammar_functor.cc",         "grammar_matcher.cc",       "grammar_parser.cc",           "grammar_printer.cc",
    "json_schema_converter.cc",   "lark_converter.cc",        "regex_converter.cc",          "structural_tag.cc",
    "suffix_automata.cc",         "testing.cc",               "tokenizer_info.cc",           "support/logging.cc",
    "support/recursion_guard.cc", "converter_ext/cohere.cc",  "converter_ext/deepseek.cc",   "converter_ext/glm.cc",
    "converter_ext/kimi_k3.cc",   "converter_ext/minimax.cc", "converter_ext/minimax_m3.cc", "converter_ext/qwen.cc",
};

/// xgrammar's static library with the C ABI. Always optimized (the masks are the same bits at any level; a debug
/// build of the Earley parser is too slow for the tests' fills).
/// Dist overlay: the fork links xgrammar statically from zig/third_party (no libtfgrammar to bundle), so
/// `zig build dist -Dxgrammar=DIR` has no overlay; the step refuses instead of shipping without it.
pub fn overlay(b: *std.Build, target: std.Build.ResolvedTarget, dir: []const u8) ?std.Build.LazyPath {
    _ = b;
    _ = target;
    _ = dir;
    return null;
}

/// The xgrammar release the vendored tree tracks; the clone hint `zig build dist` prints without -Dxgrammar.
pub const pinned = "v0.2.8";
pub const clone = "git clone --branch " ++ pinned ++ " --recurse-submodules https://github.com/mlc-ai/xgrammar";

pub fn library(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    const root = "zig/third_party/xgrammar/";
    const m = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true, .link_libcpp = true });
    m.addIncludePath(b.path(root ++ "include"));
    m.addIncludePath(b.path(root ++ "cpp"));
    m.addSystemIncludePath(b.path(root ++ "3rdparty/picojson"));
    m.addSystemIncludePath(b.path(root ++ "3rdparty/dlpack/include"));
    m.addIncludePath(b.path("zig/src/core/grammar"));
    const flags = [_][]const u8{ "-std=c++17", "-DXGRAMMAR_ENABLE_CPPTRACE=0", "-fexceptions", "-Wno-everything" };
    m.addCSourceFiles(.{ .root = b.path(root ++ "cpp"), .files = &xgr_sources, .flags = &flags });
    m.addCSourceFile(.{ .file = b.path("zig/src/core/grammar/xgr_c.cc"), .flags = &flags });
    return b.addLibrary(.{ .name = "tf_xgrammar", .root_module = m, .linkage = .static });
}

/// The `grammar` module (zig/src/core/grammar/root.zig) linked with xgrammar.
pub fn module(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) *std.Build.Module {
    const m = b.createModule(.{ .root_source_file = b.path("zig/src/core/grammar/root.zig"), .target = target, .optimize = optimize, .link_libc = true, .link_libcpp = true });
    m.addIncludePath(b.path("zig/src/core/grammar"));
    m.linkLibrary(library(b, target));
    return m;
}

pub fn add(b: *std.Build, test_step: *std.Build.Step) void {
    const host = b.graph.host;
    const t = b.addTest(.{ .root_module = module(b, host, .debug) });
    const run = b.addRunArtifact(t);
    run.setCwd(b.path("."));
    b.step("test-grammar", "Grammar masks / cuts / compiles vs xgrammar 0.2.8's Python package on recorded goldens (no GPU)").dependOn(&run.step);
    test_step.dependOn(&run.step);
}


/// libtfgrammar.so: `zig/src/grammar/tf_grammar.cc` (the C ABI the native server loads at its first structured
/// request) over the same vendored xgrammar the mask path links. The fork does not take an upstream release
/// archive, so the build installs the library beside the binary (packaging/RUNTIME.md) instead of expecting it.
pub fn shared(b: *std.Build, target: std.Build.ResolvedTarget) *std.Build.Step.Compile {
    const root = "zig/third_party/xgrammar/";
    const m = b.createModule(.{ .target = target, .optimize = .ReleaseFast, .link_libc = true, .link_libcpp = true });
    m.addIncludePath(b.path(root ++ "include"));
    m.addIncludePath(b.path(root ++ "cpp"));
    m.addSystemIncludePath(b.path(root ++ "3rdparty/picojson"));
    m.addSystemIncludePath(b.path(root ++ "3rdparty/dlpack/include"));
    m.addCSourceFile(.{ .file = b.path("zig/src/grammar/tf_grammar.cc"), .flags = &.{ "-std=c++17", "-fexceptions", "-Wno-everything" } });
    m.linkLibrary(library(b, target));
    return b.addLibrary(.{ .name = "tfgrammar", .root_module = m, .linkage = .dynamic });
}

/// The mask kernel (zig/src/core/grammar/mask_kernel.zig) as PTX text, for `@embedFile` through an anonymous import
/// named `grammar_mask_ptx` (sm_80 PTX: the driver JIT-compiles it for the GPU it runs on).
pub fn maskPtx(b: *std.Build) std.Build.LazyPath {
    const nvptx = b.resolveTargetQuery(.{ .cpu_arch = .nvptx64, .os_tag = .cuda, .cpu_model = .{ .explicit = &std.Target.nvptx.cpu.sm_80 } });
    const obj = b.addObject(.{ .name = "tf_grammar_mask", .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/core/grammar/mask_kernel.zig"), .target = nvptx, .optimize = .ReleaseFast }) });
    return obj.getEmittedAsm();
}

/// Adds `grammar` (and the mask kernel's PTX as `grammar_mask_ptx`) to a module that serves grammars.
pub fn addTo(b: *std.Build, m: *std.Build.Module, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode) void {
    m.addImport("grammar", module(b, target, optimize));
    m.addAnonymousImport("grammar_mask_ptx", .{ .root_source_file = maskPtx(b) });
}
