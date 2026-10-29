#!/usr/bin/env bash
# sweep-knob.sh [KNOB=VALUE ...] — rewrite the perf-sweep override block at the end of
# ~/tf4/zig-1017-uncens.env on ALL FOUR nodes. No args = remove the block (revert to stock).
# v2 2026-10-19: the remote append MUST NOT use ssh -n (it blocks the stdin pipe — v1 never
# appended on ranks 1-3, which the TP settings check caught via TF_TP_MAILBOX_MAX_KB).
set -uo pipefail
MARK_A="# >>> perf-sweep-20261019 (do not edit by hand)"
MARK_B="# <<< perf-sweep-20261019"
ENV=~/tf4/zig-1017-uncens.env
nodes=(anvil ember flame)
esc_a=${MARK_A//\//\\/}
esc_b=${MARK_B//\//\\/}
sed -i "/^${esc_a}\$/,/^${esc_b}\$/d" "$ENV"
if [[ $# -gt 0 ]]; then
  { echo "$MARK_A"; for kv in "$@"; do echo "$kv"; done; echo "$MARK_B"; } >> "$ENV"
fi
for n in "${nodes[@]}"; do
  ssh "$n" "sed -i \"/^${esc_a}\$/,/^${esc_b}\$/d\" $ENV" </dev/null
  if [[ $# -gt 0 ]]; then
    { echo "$MARK_A"; for kv in "$@"; do echo "$kv"; done; echo "$MARK_B"; } | ssh "$n" "cat >> $ENV"
  fi
done
echo "env block now: ${*:-<removed>}"
for n in "${nodes[@]}"; do
  echo "$n tail: $(ssh -n "$n" "tail -2 $ENV" | tr '\n' ' ')"
done
echo "forge tail: $(tail -2 "$ENV" | tr '\n' ' ')"
