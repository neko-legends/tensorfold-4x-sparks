#!/usr/bin/env bash
# Optional four-Spark keeper (cron on the head): start after a reboot, restart after 3 failed /health checks in a row.
#   @reboot sleep 240 && bash <repo>/scripts/keeper4.sh boot
#   */2 * * * * bash <repo>/scripts/keeper4.sh
# Stands down while a maintenance flag is fresh/owned (see scripts/keeper-maintenance.sh), while another start
# runs, and while a container named in KEEPER_CONFLICTS runs (another model on the same GPUs).
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STATE_DIR="${STATE_DIR:-${XDG_STATE_HOME:-$HOME/.local/state}/dsv41-tf4}"; mkdir -p "$STATE_DIR"
PORT="${PORT:-$(grep -hE '^PORT=' config/prod.env config/tp4.env 2>/dev/null | tail -1 | cut -d= -f2)}"; PORT="${PORT:-8000}"
LOG=$STATE_DIR/keeper.log; F=$STATE_DIR/keeper.fails
log() { echo "$(date -Is) $*" >> "$LOG"; }
# Maintenance window (2026-10-21): the flag may carry `<owner-unit> <ttl-seconds>`
# on its first line and the window holds while that unit is active or until the
# TTL elapses. A bare/legacy flag keeps the 60-minute mtime window. The reason is
# logged once per change (not every 2-minute tick) so the next mystery has an
# answer in keeper.log. The keeper never refreshes the flag; the experiment owns it.
maint_helper="$(dirname "${BASH_SOURCE[0]}")/keeper-maintenance.sh"
maint_state="$STATE_DIR/keeper.maintenance-reason"
if [[ -e $STATE_DIR/maintenance ]]; then
    if [[ -x $maint_helper ]]; then
        maint_reason=$("$maint_helper" "$STATE_DIR/maintenance"); maint_rc=$?
    elif [[ $(( $(date +%s) - $(stat -c %Y "$STATE_DIR/maintenance") )) -lt 3600 ]]; then
        maint_rc=0; maint_reason='stand down: legacy inline maintenance check (helper missing)'
    else
        maint_rc=1; maint_reason='armed: legacy inline maintenance check (helper missing)'
    fi
    # Dedupe on the stable part of the reason: the age inside the trailing
    # "(age Ns…)" changes every tick and must not re-log the same decision.
    maint_key="${maint_reason% (age *}"
    if [[ "$(cat "$maint_state" 2>/dev/null)" != "$maint_key" ]]; then
        log "maintenance: $maint_reason"
        printf '%s\n' "$maint_key" > "$maint_state"
    fi
    (( maint_rc == 0 )) && exit 0
else
    if [[ -s $maint_state ]]; then log 'maintenance: armed (flag removed)'; : > "$maint_state"; fi
fi
exec 9>"$STATE_DIR/keeper.lock"; flock -n 9 || exit 0
for c in ${KEEPER_CONFLICTS:-}; do
    [[ "$(docker inspect -f '{{.State.Running}}' "$c" 2>/dev/null)" == true ]] && { log "$c running: standing down"; exit 0; }
done
if curl -sf -m 15 "localhost:$PORT/health" >/dev/null; then echo 0 > "$F"; exit 0; fi
n=$(( $(cat "$F" 2>/dev/null || echo 0) + 1 )); echo $n > "$F"
[[ "${1:-}" == boot ]] && n=3
log "health check failed ($n)"
(( n >= 3 )) || exit 0
log "restarting"; echo 0 > "$F"
bash scripts/serve4.sh stop >> "$LOG" 2>&1; bash scripts/serve4.sh start >> "$LOG" 2>&1 && log "started" || log "start FAILED"
