#!/usr/bin/env bash
# sweep-restart.sh — touch maintenance, restart the 1017 engine, wait for health + canary.
# Prints READY or FAILED; on failure prints the last boot log lines.
set -uo pipefail
D=~/tf4
touch "$D/maintenance"
"$D/serve4-zig-1017-uncens.sh" stop >/dev/null 2>&1
sleep 5
if ! "$D/serve4-zig-1017-uncens.sh" start >> "$D/logs/sweep-restart.log" 2>&1; then
  echo "FAILED"
  tail -20 "$D/logs/zig-1017-uncens-boot-r0.log" 2>/dev/null
  for r in 1 2 3; do echo "-- rank $r:"; ssh -n $(echo anvil ember flame | cut -d\  -f$r) "tail -8 $D/logs/zig-1017-uncens-boot-r$r.log" 2>/dev/null; done
  exit 1
fi
for i in $(seq 1 40); do
  curl -sf -m 5 http://127.0.0.1:8000/health >/dev/null && break
  sleep 3
done
ans=$(curl -s -m 60 http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"deepseek-v4.1-flash","max_tokens":2000,"temperature":0,"reasoning_effort":"low","messages":[{"role":"user","content":"What is 17 * 23? Answer with the number only."}]}' | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"][:60])' 2>/dev/null)
echo "READY canary=[$ans]"
