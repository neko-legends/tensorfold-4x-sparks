//! The backend-neutral checkpoint reader alone, as `core` for a module that also lives beside the server's tokenizer
//! (tensorfold-dsv41's engine: DeepSeek-V4.1's pack reads safetensors headers through `core.safetensors`).

pub const safetensors = @import("safetensors.zig");
/// The EXL3 pack geometry (groups, TP parts, expert layouts) DeepSeek-V4.1's plan/names/pack read through `core`.
pub const exl3_layout = @import("exl3_layout.zig");
/// The codebook the pack markers pick (exl3_layout.Codebook lives here).
pub const exl3_load = @import("exl3_load.zig");
/// The per-file SHA-256 record cache (descriptor identity), as `core.digest_cache` beside the server's tokenizer.
pub const digest_cache = @import("digest_cache.zig");
