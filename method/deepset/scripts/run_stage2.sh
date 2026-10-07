#!/usr/bin/env bash
# Method Stage2 launcher (formal verification).
# Default: L1 action-bound verification of the frozen Stage1 DeepSet.
# Usage: run_stage2.sh [timeout_s] [samples] [bucket_csv]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
exec bash "$ROOT/scripts/run_stage2_l1.sh" "$@"
