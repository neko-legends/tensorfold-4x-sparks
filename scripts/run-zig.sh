#!/usr/bin/env bash
# Run one local rank. Invoke separately on each node with the same serving env.
# TP_WORLD in the config (default 2) sets the ranks: run-zig.sh 0..TP_WORLD-1 (config/tp4-zig.env.example: 4).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
rank=${1:?usage: run-zig.sh <rank> [--dry-run]}
config=${CONFIG:-config/prod-zig.env}
[[ -f "$config" ]] || { echo "missing $config; copy config/prod-zig.env.example first" >&2; exit 2; }
set -a
source "$config"
set +a
world=${TP_WORLD:-2}
[[ "$world" =~ ^[0-9]+$ && "$world" -ge 1 ]] || { echo "TP_WORLD must be a positive integer" >&2; exit 2; }
[[ "$rank" =~ ^[0-9]+$ && "$rank" -lt "$world" ]] || { echo "rank must be 0..$((world - 1))" >&2; exit 2; }
name=${ZIG_NAME:-tensorfold-zig}
for key in IMAGE TF_TP_MASTER NCCL_SOCKET_IFNAME NCCL_IB_HCA MODEL_DIR ENGRAM_DIR ASSETS_DIR PREPARED_DIR STATE_DIR CACHE_DIR; do
    [[ -n "${!key:-}" && "${!key}" != *'<'* ]] || { echo "set $key in $config" >&2; exit 2; }
done
args=(docker run --rm --init --name "$name-r$rank" --gpus all --ipc=host --network host
      --device /dev/infiniband --ulimit memlock=-1 --cap-add IPC_LOCK
      -v "$MODEL_DIR:/model:ro" -v "$ENGRAM_DIR:/engram:ro" -v "$ASSETS_DIR:/assets:ro"
      -v "$PREPARED_DIR:/prepared" -v "$STATE_DIR:/state" -v "$STATE_DIR/sessions:/sessions"
      -v "$CACHE_DIR:/cache")
# Pass values by environment name, preserving spaces in values. TF_TP_* except the ones set below (world, rank,
# device, master, port, backend, kernel image): e.g. TF_TP_ROCE_FALLBACK.
while IFS='=' read -r key value; do
    case "$key" in
        TF_TP_WORLD|TF_TP_RANK|TF_TP_DEVICE|TF_TP_MASTER|TF_TP_PORT|TF_COMM_BACKEND|TF_TP_KERNEL_IMAGE) ;;
        TF_DSV41_*|TF_VIDEO_*|GLM53_TF_*|NCCL_*|TF_TP_*|MALLOC_ARENA_MAX) args+=(-e "$key");;
    esac
done < <(env)
args+=(-e "TF_TP_WORLD=$world" -e "TF_TP_RANK=$rank" -e TF_TP_DEVICE=0
       -e "TF_TP_MASTER=$TF_TP_MASTER" -e "TF_TP_PORT=${TF_TP_PORT:-29571}"
       -e "TF_COMM_BACKEND=${TF_COMM_BACKEND:-roce}"
       -e TF_TP_KERNEL_IMAGE=/opt/tensorfold/share/tp_mailbox.fatbin "$IMAGE")
if [[ "$rank" == 0 ]]; then
    args+=(tensorfold-dsv41 serve /model --host "${HOST:-localhost}" --port "${PORT:-8000}"
           --parallel "$TF_DSV41_SLOTS" --context "$TF_DSV41_CONTEXT"
           --name "${SERVED_NAME:-DeepSeek-V4.1-Flash-TF}" --max-tokens "${MAX_TOKENS:-32768}")
else
    args+=(tf-dsv41-m1 follow /model /assets)
fi
if [[ "${2:-}" == --dry-run ]]; then printf '%q ' "${args[@]}"; printf '\n'; exit; fi
for dir in "$MODEL_DIR" "$ENGRAM_DIR" "$ASSETS_DIR" "$CACHE_DIR"; do
    [[ -d "$dir" ]] || { echo "missing directory $dir" >&2; exit 2; }
done
for asset in rope.bin engram-host.bin aot; do
    [[ -e "$ASSETS_DIR/$asset" ]] || { echo "missing asset $asset" >&2; exit 2; }
done
for l in 1 14; do
    shard="$ENGRAM_DIR/engram-l$l-r${rank}of$world.bin"
    [[ -e "$shard" ]] || { echo "missing Engram shard $shard (pack it for world $world)" >&2; exit 2; }
done
mkdir -p "$PREPARED_DIR" "$STATE_DIR/sessions"
exec "${args[@]}"
