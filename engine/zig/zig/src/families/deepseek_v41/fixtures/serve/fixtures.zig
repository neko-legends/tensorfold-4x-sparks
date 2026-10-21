//! The serving goldens committed with the code (recipe/server_tests/gen_*.py wrote them from the Python engine).
pub const template = @embedFile("template.jsonl");
pub const dsml = @embedFile("dsml.jsonl");
pub const sampling = @embedFile("sampling.jsonl");
/// HTTP exchanges with prod's Python server (tools/zig/dsv41_ops/http_golden.py; server/wire_test.zig)
pub const wire = @embedFile("wire.jsonl");
pub const mini_tokenizer = @embedFile("mini_tokenizer.json");
