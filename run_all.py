\
from pathlib import Path
from datetime import datetime
import json

from q_learning import sweep as q_sweep
from dqn import train as dqn_train, DQNConfig

REPORT = Path("report")
OUT = Path("out")

def write_report_intro(md):
    md.write("# CSCN8020 – Assignment 2: Taxi-v3 Q-Learning & DQN\n\n")
    md.write(f"_Auto-generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_\n\n")
    md.write("## Environment\n")
    md.write("- **Env:** Taxi-v3 (500 states, 6 actions)\n")
    md.write("- **Rewards:** -1 per step; +20 for successful dropoff; -10 for illegal pickup/dropoff\n\n")
    md.write("## Algorithms\n- Tabular Q-Learning\n- Deep Q-Network (DQN) with one-hot state encoding\n\n")

def embed_metrics(md, path: Path, title: str):
    md.write(f"### {title}\n\n")
    # Summary
    summary_path = path/"summary.txt"
    if summary_path.exists():
        md.write("**Summary**\n\n```\n")
        md.write(summary_path.read_text())
        md.write("\n```\n\n")
    # Curves
    for img in ["returns.png", "steps.png"]:
        p = path/img
        if p.exists():
            md.write(f"![{img}]({p.as_posix()})\n\n")

def main():
    REPORT.mkdir(exist_ok=True)
    OUT.mkdir(exist_ok=True)

    # 1) Q-Learning sweep
    q_sweep()

    # 2) One DQN baseline
    dqn_dir = dqn_train(DQNConfig())

    # 3) Build Markdown report
    md_path = REPORT/"Assignment2_Report.md"
    with open(md_path, "w") as md:
        write_report_intro(md)
        md.write("## Results\n\n")
        # Append Q-Learning results
        md.write("### Q-Learning Hyperparameter Sweep\n\n")
        q_base = Path("out/q_sweep")
        for combo in sorted(q_base.glob("*")):
            embed_metrics(md, combo, f"Q-Learning — {combo.name}")
        # Append DQN
        md.write("\n### DQN Baseline\n\n")
        embed_metrics(md, dqn_dir, f"DQN — {dqn_dir.name}")

        # 4) Best config placeholder (user to interpret or automate later)
        md.write("\n## Best Configuration (Q-Learning)\n\n")
        md.write("- _Fill in based on highest avg_return and lowest avg_steps; re-run `q_learning.py` with your chosen α and ε._\n")

        md.write("\n## Discussion\n\n- Compare learning curves.\n- Explain how α and ε affected convergence speed and stability.\n- Comment on DQN vs tabular Q-learning.\n")

    print(f"Report written: {md_path}")

if __name__ == "__main__":
    main()
