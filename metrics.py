import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

@dataclass
class EpisodeLog:
    episode: int
    steps: int
    reward: float

@dataclass
class RunResult:
    algo: str
    env_id: str
    seed: int
    episodes: int
    lr: float
    gamma: float
    eps_start: float
    eps_end: float
    eps_decay: float
    notes: str
    episode_logs: List[EpisodeLog]

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["episode_logs"] = [asdict(x) for x in self.episode_logs]
        return d

def summarize(episode_logs: List[EpisodeLog]) -> Dict[str, float]:
    steps = np.array([e.steps for e in episode_logs], dtype=float)
    rets  = np.array([e.reward for e in episode_logs], dtype=float)
    return {
        "episodes": len(episode_logs),
        "avg_steps": float(steps.mean()),
        "std_steps": float(steps.std()),
        "avg_return": float(rets.mean()),
        "std_return": float(rets.std()),
        "min_return": float(rets.min()),
        "max_return": float(rets.max()),
    }

def save_run(run: RunResult, outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)
    # JSON
    (outdir/"metrics.json").write_text(json.dumps(run.to_dict(), indent=2))
    # CSV (episodes)
    with open(outdir/"episodes.csv", "w") as f:
        f.write("episode,steps,return\n")
        for e in run.episode_logs:
            f.write(f"{e.episode},{e.steps},{e.reward}\n")

def plot_learning_curves(run: RunResult, outdir: Path, window: int = 50):
    import numpy as np
    outdir.mkdir(parents=True, exist_ok=True)
    episodes = np.array([e.episode for e in run.episode_logs])
    returns  = np.array([e.reward for e in run.episode_logs], dtype=float)
    steps    = np.array([e.steps for e in run.episode_logs], dtype=float)

    def moving_avg(x, w):
        if len(x) < w:
            w = max(1, len(x)//5 or 1)
        return np.convolve(x, np.ones(w)/w, mode="valid")

    plt.figure()
    plt.plot(episodes, returns, alpha=0.3, label="Return (per-episode)")
    ma = moving_avg(returns, window)
    plt.plot(np.arange(len(ma))+1, ma, label=f"Moving Avg Return (w={max(1, len(returns)//5 or window)})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title(f"{run.algo} on {run.env_id} — Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"returns.png", dpi=160)
    plt.close()

    plt.figure()
    plt.plot(episodes, steps, alpha=0.3, label="Steps (per-episode)")
    ma_s = moving_avg(steps, window)
    plt.plot(np.arange(len(ma_s))+1, ma_s, label=f"Moving Avg Steps (w={max(1, len(steps)//5 or window)})")
    plt.xlabel("Episode")
    plt.ylabel("Steps")
    plt.title(f"{run.algo} on {run.env_id} — Steps per Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir/"steps.png", dpi=160)
    plt.close()
