#!/usr/bin/env bash
# TensorFold maintenance-window decision, shared by scripts/keeper4.sh and the
# deployed ~/tf4/keeper.sh.
#
#   keeper-maintenance.sh <flag-path>
#
# Exit 0 = STAND DOWN (an experiment owns the GPUs), 1 = ARMED (the keeper may act).
# The reason is printed on stdout, one line, for the keeper's log.
#
# Flag format — first line of the flag file:
#
#   <owner-unit> <ttl-seconds>          e.g. "tf111-boot-20261021.service 7200"
#
#   * The window holds while <owner-unit> is active (systemctl --user is-active)
#     or until <ttl-seconds> have passed since the flag was written, whichever
#     lasts longer. A long boot cannot outlive its window.
#   * A bare/empty flag keeps the legacy behaviour: holds while younger than
#     KEEPER_LEGACY_TTL (default 3600 s).
#   * A malformed owner line is treated as legacy (fail safe: holds while fresh).
#
# The keeper never touches the flag — the experiment owns and removes it.
# 2026-10-21: a bare 60-minute mtime window expired under a running boot and the
# keeper restarted the live engine mid-experiment ("rank 0 closed the fate
# channel").
set -uo pipefail
flag="${1:?maintenance flag path}"
legacy_ttl="${KEEPER_LEGACY_TTL:-3600}"
now="${KEEPER_NOW:-$(date +%s)}"
[[ "$legacy_ttl" =~ ^[0-9]+$ ]] || legacy_ttl=3600
[[ "$now" =~ ^[0-9]+$ ]] || now=$(date +%s)

legacy_reason() {   # $1 = age, $2 = optional qualifier
  if (( $1 < legacy_ttl )); then
    echo "stand down: ${2}maintenance flag (age ${1}s, legacy ${legacy_ttl}s window)"
    exit 0
  fi
  echo "armed: ${2}maintenance flag (age ${1}s, legacy ${legacy_ttl}s window)"
  exit 1
}

if [[ ! -e "$flag" ]]; then
  echo "armed: no maintenance flag"
  exit 1
fi
mtime=$(stat -c %Y "$flag" 2>/dev/null || echo 0)
[[ "$mtime" =~ ^[0-9]+$ ]] || mtime=0
age=$(( now - mtime ))
(( age < 0 )) && age=0        # future mtime counts as just-written: fail safe

head=$(head -n 1 "$flag" 2>/dev/null | tr -d '\r' | sed -e 's/[[:space:]]*$//')
if [[ -z "${head//[[:space:]]/}" ]]; then
  legacy_reason "$age" "legacy "
fi
read -r owner ttl _rest <<<"$head"
if [[ -z "$owner" || ! "$ttl" =~ ^[0-9]+$ || "${ttl:-0}" -eq 0 ]]; then
  legacy_reason "$age" "unparsable ('$head'), treated as legacy "
fi
state=$(systemctl --user is-active "$owner" 2>/dev/null | head -n 1)
if [[ "$state" == active ]]; then
  echo "stand down: maintenance owner $owner is active (age ${age}s, ttl ${ttl}s)"
  exit 0
fi
if (( age < ttl )); then
  echo "stand down: maintenance owner $owner is ${state:-unknown}, ttl ${ttl}s not elapsed (age ${age}s)"
  exit 0
fi
echo "armed: maintenance owner $owner is ${state:-unknown}, ttl ${ttl}s elapsed (age ${age}s)"
exit 1
