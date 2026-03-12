#!/usr/bin/env bash
# One-command optimization of vector rejection kernel on H100.
#
# Usage (from the repo root):
#   # With Claude Opus 4.6:
#   bash examples/run_vec_reject.sh opus
#
#   # With GPT-5.4:
#   bash examples/run_vec_reject.sh gpt
#
#   # Custom rounds:
#   bash examples/run_vec_reject.sh opus 10
#
# Prerequisites:
#   1. uv venv + pip install -e .
#   2. Set ANTHROPIC_API_KEY (for opus) or OPENAI_API_KEY (for gpt) in .env or env
#   3. Running on a machine with an H100 + ncu (nsight compute) available

set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-opus}"
MAX_ROUNDS="${2:-5}"

KERNEL_DIR="examples/optimize_04_vec_reject"

case "$MODEL" in
    opus|claude)
        CONFIG="examples/configs/vec_reject_opus.yaml"
        echo "=== Using Claude Opus 4.6 ==="
        ;;
    gpt|gpt54)
        CONFIG="examples/configs/vec_reject_gpt54.yaml"
        echo "=== Using GPT-5.4 ==="
        ;;
    *)
        echo "Unknown model: $MODEL (use 'opus' or 'gpt')"
        exit 1
        ;;
esac

echo "Config:     $CONFIG"
echo "Kernel dir: $KERNEL_DIR"
echo "Max rounds: $MAX_ROUNDS"
echo ""

# The run_opt_manager.py uses --strategy to pick a config, but we want to
# pass a specific config file. Use the OptimizationManager directly.
python -c "
from pathlib import Path
from triton_kernel_agent.opt_manager import OptimizationManager

kernel_dir = Path('$KERNEL_DIR')
config_path = '$CONFIG'
max_rounds = $MAX_ROUNDS
log_dir = kernel_dir / 'opt_logs'

kernel_code = (kernel_dir / 'input.py').read_text()
test_code = (kernel_dir / 'test.py').read_text()
problem_file = kernel_dir / 'problem.py'

print('Initializing OptimizationManager...')
manager = OptimizationManager(
    config=config_path,
    log_dir=log_dir,
    database_path=log_dir / 'program_db.json',
)

print('Starting optimization...')
result = manager.run_optimization(
    initial_kernel=kernel_code,
    problem_file=problem_file,
    test_code=test_code,
    max_rounds=max_rounds,
)

print()
if result['success']:
    print('=' * 60)
    print('OPTIMIZATION SUCCESSFUL')
    print('=' * 60)
    print(f\"Best time: {result['best_time_ms']:.4f} ms\")
    print(f\"Total rounds: {result['total_rounds']}\")
    if result.get('top_kernels'):
        print(f\"Top kernels: {len(result['top_kernels'])}\")
        for i, k in enumerate(result['top_kernels'][:3], 1):
            print(f\"  {i}. {k['time_ms']:.4f}ms (gen {k['generation']})\")
    if result.get('kernel_code'):
        out = kernel_dir / 'optimized_kernel.py'
        out.write_text(result['kernel_code'])
        print(f\"Saved best kernel to: {out}\")
else:
    print('OPTIMIZATION FAILED -- check logs in', log_dir)
"
