#!/bin/bash
# Run Ablation Study in Background (survives SSH disconnect)
# Usage: ./run_ablation_background.sh

echo "🚀 Starting Ablation Study in tmux session..."
echo ""
echo "Commands:"
echo "  - Detach (keep running): Ctrl+B, then D"
echo "  - Reattach later: tmux attach -t ablation"
echo "  - Kill session: tmux kill-session -t ablation"
echo ""
echo "Press ENTER to start..."
read

# Create tmux session and run ablation study
tmux new-session -s ablation \
  "source .venv/bin/activate && \
   python run_ablation_study.py --all && \
   echo '' && \
   echo '✅ All experiments completed!' && \
   echo 'Generating visualizations...' && \
   python visualize_ablation_results.py && \
   echo '' && \
   echo '🎉 ABLATION STUDY COMPLETE!' && \
   echo 'Results saved in: ablation_results/' && \
   echo 'Press Ctrl+B then D to detach, or Ctrl+C to exit' && \
   bash"
