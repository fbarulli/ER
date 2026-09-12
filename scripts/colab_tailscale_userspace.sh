#!/usr/bin/env bash
set -Eeuo pipefail
: "${TAILSCALE_AUTH_KEY:?TAILSCALE_AUTH_KEY is required}"
: "${TAILSCALE_HOSTNAME:=euromonitor-hpo-cpu-test}"
: "${TAILSCALE_SOCKET:=/tmp/euromonitor-tailscaled.sock}"
: "${TAILSCALE_STATE:=/tmp/euromonitor-tailscale.state}"
: "${TAILSCALE_LOG:=/tmp/euromonitor-tailscaled.log}"
redact() { sed "s/${TAILSCALE_AUTH_KEY//\//\\/}/<redacted>/g"; }
show_daemon_log() { [[ -f "$TAILSCALE_LOG" ]] && { echo "[tailscale] daemon log (tail):" >&2; tail -200 "$TAILSCALE_LOG" | redact >&2; }; }
on_error() { local status=$?; echo "[tailscale] failed (rc=$status)" >&2; show_daemon_log; exit "$status"; }
trap on_error ERR
missing=()
command -v socat >/dev/null || missing+=(socat)
command -v curl >/dev/null || missing+=(curl)
command -v tailscaled >/dev/null || missing+=(tailscale)
if [[ ${#missing[@]} -gt 0 ]]; then apt-get update -qq; apt-get install -y -qq socat curl; command -v tailscaled >/dev/null || curl -fsSL https://tailscale.com/install.sh | sh; fi
if ! tailscale --socket="$TAILSCALE_SOCKET" status >/dev/null 2>&1; then
  pkill -f "tailscaled --socket=$TAILSCALE_SOCKET" || true
  rm -f "$TAILSCALE_SOCKET"
  tailscaled --socket="$TAILSCALE_SOCKET" --state="$TAILSCALE_STATE" --tun=userspace-networking --socks5-server=127.0.0.1:1055 >>"$TAILSCALE_LOG" 2>&1 &
  for _ in {1..30}; do [[ -S "$TAILSCALE_SOCKET" ]] && break; sleep 1; done
fi
[[ -S "$TAILSCALE_SOCKET" ]] || { echo "[tailscale] daemon socket was not created" >&2; exit 1; }
key_file=$(mktemp); chmod 600 "$key_file"; printf '%s' "$TAILSCALE_AUTH_KEY" > "$key_file"
output_file=$(mktemp); up_status=0
tailscale --socket="$TAILSCALE_SOCKET" up --auth-key="file:$key_file" --hostname="$TAILSCALE_HOSTNAME" >"$output_file" 2>&1 || up_status=$?
rm -f "$key_file"
if [[ $up_status -ne 0 ]]; then redact < "$output_file" >&2; rm -f "$output_file"; exit "$up_status"; fi
rm -f "$output_file"
tailscale --socket="$TAILSCALE_SOCKET" status; tailscale --socket="$TAILSCALE_SOCKET" ip -4
if [[ -n "${TAILSCALE_TARGET_HOST:-}" && -n "${TAILSCALE_TARGET_PORT:-}" ]]; then
  : "${TAILSCALE_LOCAL_PORT:=$TAILSCALE_TARGET_PORT}"
  bridge_pid_file="/tmp/euromonitor-tailscale-bridge-${TAILSCALE_LOCAL_PORT}.pid"
  if [[ -s "$bridge_pid_file" ]]; then
    old_pid=$(cat "$bridge_pid_file")
    kill -0 "$old_pid" 2>/dev/null && kill "$old_pid" || true
    rm -f "$bridge_pid_file"
  fi
  socat --experimental "TCP-LISTEN:${TAILSCALE_LOCAL_PORT},fork,reuseaddr" "SOCKS5-CONNECT:127.0.0.1:1055:${TAILSCALE_TARGET_HOST}:${TAILSCALE_TARGET_PORT}" >>/tmp/euromonitor-tailscale-socat.log 2>&1 &
  bridge_pid=$!
  printf '%s\n' "$bridge_pid" > "$bridge_pid_file"
  for _ in {1..10}; do kill -0 "$bridge_pid" 2>/dev/null || { echo "[tailscale] bridge process died" >&2; exit 1; }; (exec 3<>"/dev/tcp/127.0.0.1/${TAILSCALE_LOCAL_PORT}") 2>/dev/null && { exec 3<&-; break; }; sleep 1; done
  echo "[tailscale] bridge ready: 127.0.0.1:${TAILSCALE_LOCAL_PORT} -> ${TAILSCALE_TARGET_HOST}:${TAILSCALE_TARGET_PORT} (pid $bridge_pid)"
fi
