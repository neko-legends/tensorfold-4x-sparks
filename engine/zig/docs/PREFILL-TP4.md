# TP=4 prefill: why upstream refuses every segment size on our pack, and what our branch does

Written 2026-10-22 on `tf112-rebase-20261022` while chasing the PR-code device receipts (an
upstream 1.1.2 + PR-patch image could not boot at world 4: `a 4096-row segment is not on the
prefill path at any position`, `a 2048-row segment …`, `a 1024-row segment …` — every size).

Short answer: **1.1.2 does not fix it.** Between our 1.1.1 upstream base and v1.1.2 nothing under
`zig/src/families/deepseek_v41` changed:

    git -C <upstream> log --oneline d1018b31..4ff78441 -- zig/src/families/deepseek_v41   # 0 commits
    git -C <upstream> diff --stat v1.1.1 v1.1.2 -- zig/src/families/deepseek_v41        # empty

`PF_4K` is not an escape either: it demands a 4,096-row chunk (`TF_DSV41_PREFILL_CHUNK=4096`,
`PREFILL_ROWS=4096`) plus `TF_DSV41_PF_4K_MEMORY_OK=1`, so it cannot serve a 2,048-row round and it
is not what the forward's own segments use (`forward_prefill.k4Options` != `f.opts`). The forward
plans its segments with `f.opts` (pf4k off) and fails closed in `forward_prefill.lastStartWith`
when no start emits them.

## The two checks that fail at world 4, and are absent upstream

Both live in our `block_prefill.zig` as deltas over upstream; upstream still has the unclamped forms
in its split modules (`block_prefill_attn.zig`, `block_prefill_moe.zig`, currently not imported by
any call path in our tree).

1. **The `_fused` attention head tile** — `block_prefill.zig:37` `attnBmq(H) = @min(bmq, H)`,
   used at `:899` and `:994/:1015`. Upstream checks `@mod(H, 32) != 0` in
   `block_prefill_attn.zig:221`. At TP=4 a rank has H = 16 query heads, so that check can never pass
   and **every** segment (1,024 / 2,048 / 4,096) is refused before MoE is even reached. The kernel is
   unchanged; the head tile becomes 16 (its own AOT variant), as the PR patch `patches/0003`
   (`attn_pf.config`) clamps it.

2. **The x3gm expert scratch bound** — `block_prefill.zig:1150-1164`:
   `whole_block = e.o.pf4k or e.o.world > 2;` and the bound is only enforced when
   `!whole_block`. Upstream `block_prefill_moe.zig:103` enforces it unconditionally below PF_4K:
   `P * D > zb or P * I > rows * (topk + 1) * D` with `rows = expert_rows` (1,024) and
   `zb = rows * (topk + 1) * max(8 I, D)`.
   At TP=2 (I = 1,152, so 8 I > D) a 2,048-row block fits; at TP=4 (I = 640 / 512, 8 I <= D) it does
   not — the very failure recorded in `prefill_tp4_test.zig`'s header. Our rule sizes the scratch from
   the segment instead: the block is the whole segment (`s.ex.z` / `s.ex.xg` in the buffer plan,
   ~+100 MB a rank at 2,048 rows).

## What upstream would need

Not the `_fused` head-tile clamp alone — that only moves the refusal from 1,024 rows to 2,048. A
correct upstream fix is one of:

- take the same `world > 2 ⇒ block is the segment` rule, with the buffer plan sizing `s.ex.z` /
  `s.ex.xg` from the views (what we run), or
- keep streaming and size the expert scratch from `min(expert_rows, n)` rows of the *segment* rather
  than always from the decode-sized `expert_rows` (Python halves the block past the scratch; a
  half-block at TP=4 still needs `P/2 * D > zb` to be false, which at I=640, 8 I <= D it is not), or
- make `PF_4K` independent of chunk/rows so a 4,096-row workspace can serve 1,024/2,048-row
  segments — it currently refuses any explicit chunk other than 4,096
  (`pf4k_test.zig`, `TF_DSV41_PREFILL_CHUNK` / `_ROWS` must both be 4,096).

Our branch keeps (1) and (2); `zig build test-dsv41-forward`'s `prefill_tp4_test` covers them at
rank 0..3 with the real 640 / 512 expert widths.
