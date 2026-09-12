#!/usr/bin/env bash
set -Eeuo pipefail
: "${RAW_TCP_PORT:?RAW_TCP_PORT is required}"
: "${RAW_TCP_BIND:=100.91.130.10}"
: "${RAW_TCP_OUTPUT:=/tmp/euromonitor-raw-tcp-${RAW_TCP_PORT}.out}"
echo "[raw-tcp] listening on ${RAW_TCP_BIND}:${RAW_TCP_PORT}" >&2
exec socat "TCP-LISTEN:${RAW_TCP_PORT},bind=${RAW_TCP_BIND},reuseaddr" "OPEN:${RAW_TCP_OUTPUT},creat,append"
