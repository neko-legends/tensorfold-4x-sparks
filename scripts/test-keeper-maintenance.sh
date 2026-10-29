#!/usr/bin/env bash
# Table-driven tests for the TensorFold maintenance-window decision (2026-10-21).
#   bash scripts/test-keeper-maintenance.sh
# Pure shell, no framework: each row asserts the exit code (0 stand down / 1 armed)
# and a keyword in the reason line. systemctl is stubbed through PATH.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
helper="$here/keeper-maintenance.sh"
keeper="$here/keeper4.sh"
tmp="$(mktemp -d)"; trap 'rm -rf "$tmp"' EXIT
stub="$tmp/bin"; mkdir -p "$stub"
cat > "$stub/systemctl" <<'STUB'
#!/usr/bin/env bash
# `systemctl --user is-active <unit>` only: report $STUB_ACTIVE (default inactive).
[[ "${1:-}" == --user && "${2:-}" == is-active ]] || { echo "stub: unsupported $*" >&2; exit 1; }
echo "${STUB_ACTIVE:-inactive}"
[[ "${STUB_ACTIVE:-inactive}" == active ]]
STUB
chmod +x "$stub/systemctl"
export PATH="$stub:$PATH"
now=1700000000
fail=0; passed=0
flag="$tmp/maintenance"

check() { # <name> <expected-rc> <expected-substring> <age-seconds|absent> <content> [active]
  local name="$1" want_rc="$2" want_word="$3" age="$4" content="$5" active="${6:-inactive}"
  local out rc
  rm -f "$flag"
  if [[ "$age" != absent ]]; then
    printf '%s\n' "$content" > "$flag"
    touch -d "@$(( now - age ))" "$flag"
  fi
  out=$(KEEPER_NOW=$now STUB_ACTIVE=$active "$helper" "$flag" 2>&1); rc=$?
  if [[ "$rc" != "$want_rc" ]]; then
    printf 'FAIL %-48s rc=%s want %s | %s\n' "$name" "$rc" "$want_rc" "$out"; fail=$((fail+1)); return
  fi
  if [[ "$out" != *"$want_word"* ]]; then
    printf 'FAIL %-48s reason=%s (missing %s)\n' "$name" "$out" "$want_word"; fail=$((fail+1)); return
  fi
  passed=$((passed+1)); printf 'ok   %-48s %s\n' "$name" "$out"
}

# --- Bug B table -----------------------------------------------------------
check 'no flag'                              1 'armed: no maintenance flag'        absent ''                 inactive
check 'legacy empty flag, fresh (5 min)'     0 'legacy'                            300    ''                 inactive
check 'legacy empty flag, expired (61 min)'  1 'armed'                             3660   ''                 inactive
check 'legacy whitespace-only flag, fresh'   0 'legacy'                            120    '   '              inactive
check 'owned + owner active, long expired'   0 'is active'                          7200   'tf111-boot.service 3600' active
check 'owned + owner active, fresh'          0 'is active'                          60     'tf111-boot.service 3600' active
check 'owned + owner finished, ttl fresh'    0 'ttl 3600s not elapsed'             300    'tf111-boot.service 3600' inactive
check 'owned + owner finished, ttl elapsed'  1 'ttl 3600s elapsed'                 3660   'tf111-boot.service 3600' inactive
check 'owned + owner unknown, ttl elapsed'   1 'elapsed'                           3660   'tf111-boot.service 1800' unknown
check 'malformed line, fresh -> legacy'      0 'treated as legacy'                 300    'just-a-word'        active
check 'malformed line, expired -> armed'     1 'treated as legacy'                 3660   'just-a-word'        active
check 'ttl zero -> legacy',                  0 'treated as legacy'                 300    'unit.service 0'     active
check 'future mtime counts as fresh'         0 'legacy'                            -600   ''                   inactive

# --- keeper4.sh wiring: a fresh flag stands down before touching docker/hermes --
export STATE_DIR="$tmp/state"; mkdir -p "$STATE_DIR"
printf '\n' > "$STATE_DIR/maintenance"
out=$(STATE_DIR="$STATE_DIR" bash "$keeper" 2>&1); rc=$?
if [[ "$rc" == 0 && -s "$STATE_DIR/keeper.log" ]] && grep -q 'maintenance: stand down' "$STATE_DIR/keeper.log"; then
  passed=$((passed+1)); printf 'ok   %-48s %s\n' 'keeper exits early on a fresh flag' "$(tail -n1 "$STATE_DIR/keeper.log")"
else
  printf 'FAIL keeper4.sh did not stand down on a fresh flag (rc=%s)\n' "$rc"; fail=$((fail+1))
fi
# The reason is logged once per change, not on every 2-minute tick: a later tick
# has a larger age in the reason but the same decision, and must not re-log.
before=$(wc -l < "$STATE_DIR/keeper.log")
KEEPER_NOW=$(( $(date +%s) + 120 )) STATE_DIR="$STATE_DIR" bash "$keeper" >/dev/null 2>&1
after=$(wc -l < "$STATE_DIR/keeper.log")
if [[ "$before" == "$after" ]]; then
  passed=$((passed+1)); printf 'ok   %-48s lines %s -> %s\n' 'repeat tick does not duplicate the reason' "$before" "$after"
else
  printf 'FAIL repeat tick logged the reason again (%s -> %s)\n' "$before" "$after"; fail=$((fail+1))
fi

printf '\n%d passed, %d failed\n' "$passed" "$fail"
(( fail == 0 ))
