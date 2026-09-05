#!/usr/bin/env bash
# Addendum A non-LLM companions (S12) - runs in parallel with the LLM queue
# (Ollama is the bottleneck; these are CPU-only).
#
# Launch DETACHED into its own hidden console (survives closing the
# launching terminal; plain `nohup ... &` dies with a conpty tab such as
# Windows Terminal / VS Code):
#   powershell -Command "Start-Process -WindowStyle Hidden -FilePath 'C:\Program Files\Git\bin\bash.exe' -ArgumentList '-c','cd /c/College/Capstone/CapstoneProject && bash validation/round2/run_addendum_a_nonllm.sh > validation/logs/nonllm_queue.log 2>&1'"
cd "$(dirname "$0")/../.." || exit 1
PY=venv/Scripts/python.exe
mkdir -p validation/logs

echo "[nonllm] $(date) START A2 v2phys+mr (50 seeds)"
"$PY" -u validation/round2/a2_policy_baselines_v2phys_mr.py \
  > validation/logs/a2_v2phys_mr.log 2>&1
rc=$?
echo "[nonllm] $(date) EXIT=$rc A2 v2phys+mr"

echo "[nonllm] $(date) START E-battery mr (legacy + v2)"
"$PY" -u validation/round2/e_battery_mr.py \
  > validation/logs/e_battery_mr.log 2>&1
rc=$?
echo "[nonllm] $(date) EXIT=$rc E-battery mr"

echo "[nonllm] $(date) ALL DONE"
