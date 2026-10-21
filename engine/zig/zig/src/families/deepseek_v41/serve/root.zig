//! DeepSeek-V4.1's host-only serving side: tokenizer, chat encoding, DSML tool calls, exact sampler (zig/src/server).
pub const tokenizer = @import("tokenizer.zig");
pub const template = @import("template.zig");
pub const tools = @import("tools.zig");
pub const sampling = @import("sampling.zig");
/// The image front end (TF_DSV41_IMAGES=native): parts, decoding, the processor, virtual ids.
pub const vision = @import("dsv41_vision");
/// The committed goldens and the mini tokenizer the server tests read.
pub const fixtures = @import("dsv41_serve_fixtures");

test {
    @import("std").testing.refAllDecls(@This());
    _ = @import("golden_test.zig");
}
