#!/usr/bin/env bash
# Addendum A overnight LLM queue (S12) - strict order per NEXT_STEPS.md §1.
# One Ollama job at a time; a failed job does not stop the chain (jobs 3-6
# are sensitivity rows, not gates). Each runner pre/postflights Ollama and
# exits non-zero if the LLM is unreachable, so a dead-LLM arm cannot leave
# complete-looking CSVs.
#
# Launch DETACHED into its own hidden console (survives closing the
# launching terminal; plain `nohup ... &` dies with a conpty tab such as
# Windows Terminal / VS Code):
#   powershell -Command "Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-c \"cd /c/College/Capstone/CapstoneProject && bash validation/round2/run_addendum_a_llm_queue.sh > validation/logs/decomp_queue.log 2>&1\"'"
# (single pre-quoted -ArgumentList string: PowerShell 5.1 does not quote
# array elements containing spaces, so the '-c','...' array form silently
# runs `bash -c cd` and exits - verified during the S12 launch)
cd "$(dirname "$0")/../.." || exit 1
PY=venv/Scripts/python.exe
mkdir -p validation/logs

echo "[queue] $(date) HEAD=$(git rev-parse --short HEAD)"
echo "[queue] models available:"
ollama list 2>&1 || echo "[queue] ollama CLI not on PATH (runners preflight anyway)"

run() {
  echo "[queue] $(date) START $1 -> validation/logs/$3"
  "$PY" -u "validation/round2/$2" ${4:-} > "validation/logs/$3" 2>&1
  rc=$?
  echo "[queue] $(date) EXIT=$rc $1"
}

run "D-b tier bound"     a3_decomp.py decomp_db.log db
run "D-a no modifier"    a3_decomp.py decomp_da.log da
run "D-c mean-revert v2" a3_decomp.py decomp_dc.log dc
run "D-d qwen v2"        a3_decomp.py decomp_dd.log dd
run "L-1 qwen legacy"    a3_decomp.py decomp_l1.log l1
run "RS-2x seeds 21-40"  a3_rs_ext.py decomp_rs_ext.log

echo "[queue] $(date) ALL JOBS DONE"
