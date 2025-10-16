# CSCN8020 – Assignment 2 (Full, Step-by-Step)

This package gives you **ready-to-run** scripts to complete the assignment on Taxi-v3 using **Q-Learning** and **DQN**, plus an auto-generated Markdown report with figures.

## 1) Setup

```bash
# create & activate a virtualenv (recommended)
python -m venv .venv
source .venv/bin/activate    # Windows: .venv\Scripts\activate

# install dependencies
pip install -r requirements.txt
```

> If `gymnasium` complains about classic control renderers, you **do not** need GUI to train. For simulations with rendering, use `render_mode="human"` when making the env and run locally (not on headless servers).

## 2) Files

- `q_learning.py` – Tabular Q-learning with a **hyperparameter sweep** for α ∈ [0.1, 0.01, 0.001, 0.2] and **ϵ ∈ [0.1, 0.2, 0.3]** (interpreting the assignment's typo about “exploration factor γ” as ϵ).
- `dqn.py` – Simple DQN with replay + target network using **one-hot state** input.
- `metrics.py` – Logging, summaries, plots.
- `run_all.py` – Runs the full Q-learning sweep **and** one DQN baseline, then builds `report/Assignment2_Report.md` with embedded plots.
- `requirements.txt` – Python dependencies.

## 3) Run

### Option A: One baseline Q-Learning run
```bash
python q_learning.py
```

### Option B: Sweep (recommended for the report)
```bash
python q_learning.py -m sweep   # or edit main() to call sweep()
```
If your shell doesn't pass `-m`, simply **edit the bottom** of `q_learning.py` to call `sweep()` and run it.

### Option C: DQN
```bash
python dqn.py
```

### Option D: Everything + Auto Report
```bash
python run_all.py
```

This will create:
- `out/q_sweep/*/returns.png` and `steps.png` for each α–ϵ combo
- `out/dqn/*/returns.png` and `steps.png`
- `report/Assignment2_Report.md` – Ready to convert to PDF

## 4) Convert the Markdown Report to PDF

Use VS Code Markdown PDF extension, or:
```bash
pip install markdown pdfkit
python - <<'PY'
import markdown, pdfkit, pathlib
src = pathlib.Path("report/Assignment2_Report.md")
html = markdown.markdown(src.read_text(), extensions=["tables"])
(pathlib.Path("report")/"Assignment2_Report.html").write_text("<meta charset='utf-8'>"+html)
pdfkit.from_file("report/Assignment2_Report.html", "report/Assignment2_Report.pdf")
print("Saved: report/Assignment2_Report.pdf")
PY
```

## 5) What to Submit

- **Code** folder (these scripts).
- **PDF report** (`Assignment2_Report.pdf`) with:
  - Total episodes, steps per episode, average return per episode (with plots).
  - Clear commentary on how **α** and **ϵ** influenced learning.
  - Your **chosen best** α–ϵ combo, re-run results, and comparison.
  - A short section comparing Q-Learning vs DQN on Taxi-v3.

Good luck — this setup is designed to help you score **full marks**.
