#!/usr/bin/env bash
# WORLD ranks of tf-tp-test on one host with WORLD GPUs (rank r on GPU r; two by default; up to sixteen), as the
# single-GPU gate runs them.
#   [WORLD=2] [SUITES="mailbox nccl"] [FAILFAST="nccl mailbox"] run_pair.sh BIN OUTDIR
# RoCE's GPU path on one host without RDMA: SUITES=roce FAILFAST=roce TF_TP_ROCE_WIRE=shm (the proxy copies through
# shared memory instead of RDMA writes; the kernel, doorbell, flags and proxy loop are the real ones), any WORLD.
# The mailbox is two ranks only: above two it falls back to NCCL (the suite then measures NCCL twice).
# suite: every collective bit-exact on NCCL and the mailbox route, graph replays, latency sweeps.
# failfast: each backend x (the victim's own failure | SIGKILL) x victim rank; every rank must exit 70, the
# survivors about TF_DSV41_FAILFAST_GRACE_S after the injection. OUTDIR/failfast.txt has the delays.
set -u
BIN=${1:?bin}; OUT=${2:?outdir}
WORLD=${WORLD:-2}
mkdir -p "$OUT"
nvidia-smi > "$OUT/smi.txt" 2>&1
nvidia-smi topo -m > "$OUT/topo.txt" 2>&1
ldconfig -p | grep -E "libnccl|libcuda" > "$OUT/libs.txt" 2>&1
PORT=29500
pair() { # NAME BACKEND ARGS...: every rank, logs OUT/NAME.rR.log ending with an EXIT line
    local name=$1 backend=$2 r; shift 2
    PORT=$((PORT + 10))
    for ((r = 0; r < WORLD; r++)); do
        ( TF_TP_WORLD=$WORLD TF_TP_RANK=$r TF_TP_DEVICE=$r TF_TP_PORT=$PORT TF_COMM_BACKEND=$backend NCCL_DEBUG=${NCCL_DEBUG:-WARN} \
            timeout 600 "$BIN" "$@"; echo "EXIT rank $r rc $? at $(date +%s.%N)" ) > "$OUT/$name.r$r.log" 2>&1 &
    done
    wait
}
# ffsum OUT NAME VICTIM: a fail-fast run's verdict line (exit codes, exit times after the injection), any WORLD
ffsum() {
    [ -f "$1/$2.r0.log" ] || { echo "ffsum: $1/$2.r0.log missing" >&2; return 1; }
    python3 - "$1" "$2" "$3" "$WORLD" <<'EOF'
import re, sys
out, name, victim, world = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
logs = [open(f"{out}/{name}.r{r}.log").read() for r in range(world)]
inj = re.search(r"INJECT \w+ rank \d+ at ([\d.]+)", logs[victim])
ex = [re.search(r"EXIT rank \d+ rc (\d+) at ([\d.]+)", l) for l in logs]
if not inj or not all(ex):
    print(f"{name}: FAIL (no inject / exit line)"); sys.exit()
t0 = float(inj.group(1))
rcs = [int(e.group(1)) for e in ex]
dt = [float(e.group(2)) - t0 for e in ex]
want = [137 if (r == victim and "kill" in name) else 70 for r in range(world)]
ok = rcs == want and all(dt[r] < 3.0 for r in range(world) if r != victim)
print(f"{name}: {'PASS' if ok else 'FAIL'} rc {rcs} (want {want}), exit after injection: " + ", ".join(f"rank{r} {dt[r]:.3f} s" for r in range(world)))
EOF
}
for backend in ${SUITES:-mailbox nccl}; do
    pair "suite-$backend" "$backend" suite
done
: > "$OUT/failfast.txt"
for backend in ${FAILFAST:-nccl mailbox}; do
    for mode in fatal kill; do
        for ((victim = 0; victim < WORLD; victim++)); do
            name=ff-$backend-$mode-$victim
            pair "$name" "$backend" failfast "$victim" "$mode"
            ffsum "$OUT" "$name" "$victim" >> "$OUT/failfast.txt"
        done
    done
done
grep -h "^SUMMARY\|^FAIL" "$OUT"/suite*.log
cat "$OUT/failfast.txt"
