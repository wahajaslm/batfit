#!/usr/bin/env bash
set -euo pipefail
prompt=${1:-"Middle-order, prefers 2.9 lb, mid-high SS, round handle, quick pickup."}
ollama run batfit "$prompt" | tr -d '\r'
