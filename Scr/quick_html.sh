#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REPORTER="$ROOT/Scr/05_generate_report.py"
RESULTS_DIR="$ROOT/results"

RUN_DIR="${1:-}"
if [[ -z "$RUN_DIR" ]]; then
  if compgen -G "$RESULTS_DIR/*_run*/run_config.yaml" > /dev/null; then
    RUN_DIR="$(ls -dt "$RESULTS_DIR"/*_run*/ | head -n 1)"
  else
    echo "No runs found under: $RESULTS_DIR"
    echo "   Usage: $0 <path-to-run-dir>"
    exit 1
  fi
fi

RUN_DIR="${RUN_DIR%/}"
CONFIG="$RUN_DIR/run_config.yaml"
REPORT="$RUN_DIR/report.html"

[[ -f "$CONFIG" ]]   || { echo "Config not found: $CONFIG"; exit 1; }
[[ -f "$REPORTER" ]] || { echo "Reporter not found: $REPORTER"; exit 1; }

echo "📄 Config : $CONFIG"
echo "📂 Outdir : $RUN_DIR"

PYBIN="${PYBIN:-}"
if [[ -z "$PYBIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then PYBIN=python3
  elif command -v python  >/dev/null 2>&1; then PYBIN=python
  else
    echo "Python not found. I need to bloody reinstall Python 3 (also pip {pandas, numpy, pyyaml})."
    exit 1
  fi
fi

"$PYBIN" "$REPORTER" --config "$CONFIG" --outdir "$RUN_DIR"

if command -v open >/dev/null 2>&1; then
  open "$REPORT"
elif command -v xdg-open >/dev/null 2>&1; then
  xdg-open "$REPORT"
else
  echo "Report generated: $REPORT"
fi
