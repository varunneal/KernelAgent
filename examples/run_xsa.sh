#!/usr/bin/env bash
# Optimize XSA fwd and/or bwd kernels on H100.
#
# Usage:
#   bash examples/run_xsa.sh fwd         # optimize forward only
#   bash examples/run_xsa.sh bwd         # optimize backward only
#   bash examples/run_xsa.sh both        # both sequentially
#   bash examples/run_xsa.sh fwd 10      # 10 rounds

set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-both}"
MAX_ROUNDS="${2:-5}"

run_opt() {
    local name="$1"
    local kernel_dir="$2"
    local config="$3"

    echo ""
    echo "================================================================"
    echo "  Optimizing: $name"
    echo "  Kernel dir: $kernel_dir"
    echo "  Config:     $config"
    echo "  Max rounds: $MAX_ROUNDS"
    echo "================================================================"

    python -c "
from pathlib import Path
from triton_kernel_agent.opt_manager import OptimizationManager

kernel_dir = Path('$kernel_dir').resolve()
config_path = Path('$config').resolve()
log_dir = kernel_dir / 'opt_logs'

kernel_code = (kernel_dir / 'input.py').read_text()
test_code = (kernel_dir / 'test.py').read_text()
problem_file = kernel_dir / 'problem.py'

manager = OptimizationManager(
    config=str(config_path),
    log_dir=log_dir,
    database_path=log_dir / 'program_db.json',
)

result = manager.run_optimization(
    initial_kernel=kernel_code,
    problem_file=problem_file,
    test_code=test_code,
    max_rounds=$MAX_ROUNDS,
)

print()
if result['success']:
    print('OPTIMIZATION SUCCESSFUL: $name')
    print(f\"Best time: {result['best_time_ms']:.4f} ms\")
    if result.get('kernel_code'):
        out = kernel_dir / 'optimized_kernel.py'
        out.write_text(result['kernel_code'])
        print(f\"Saved to: {out}\")
else:
    print('OPTIMIZATION FAILED: $name -- check logs in', log_dir)
"
}

case "$TARGET" in
    fwd)
        run_opt "XSA Forward" "examples/optimize_05_xsa_fwd" "examples/configs/xsa_fwd_opus.yaml"
        ;;
    bwd)
        run_opt "XSA Backward" "examples/optimize_06_xsa_bwd" "examples/configs/xsa_bwd_opus.yaml"
        ;;
    both)
        run_opt "XSA Forward" "examples/optimize_05_xsa_fwd" "examples/configs/xsa_fwd_opus.yaml"
        run_opt "XSA Backward" "examples/optimize_06_xsa_bwd" "examples/configs/xsa_bwd_opus.yaml"
        ;;
    *)
        echo "Usage: $0 {fwd|bwd|both} [max_rounds]"
        exit 1
        ;;
esac
