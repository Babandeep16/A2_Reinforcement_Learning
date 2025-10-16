\
import os, random
import numpy as np
from dataclasses import dataclass
from typing import List
from pathlib import Path
from tqdm import trange

import gymnasium as gym

from metrics import EpisodeLog, RunResult, save_run, plot_learning_curves, summarize

@dataclass
class QConfig:
    env_id: str = "Taxi-v3"
    episodes: int = 5000
    max_steps: int = 200
    alpha: float = 0.1
    gamma: float = 0.9
    eps_start: float = 0.1
    eps_end: float = 0.1
    seed: int = 42
    render_mode: str = None
    outdir: str = "out/q_learning"

class QLearningAgent:
    def __init__(self, n_states: int, n_actions: int, alpha: float, gamma: float, eps: float, seed: int):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros((n_states, n_actions), dtype=np.float32)

    def select_action(self, state: int) -> int:
        if self.rng.random() < self.eps:
            return self.rng.integers(0, self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        best_next = np.max(self.Q[s_next])
        td_target = r + (0 if done else self.gamma * best_next)
        td_error  = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

def train(cfg: QConfig):
    env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
    env.reset(seed=cfg.seed)
    n_states = env.observation_space.n
    n_actions = env.action_space.n

    agent = QLearningAgent(n_states, n_actions, cfg.alpha, cfg.gamma, cfg.eps_start, cfg.seed)

    logs: List[EpisodeLog] = []
    for ep in trange(cfg.episodes, desc="Q-Learning"):
        state, _ = env.reset(seed=cfg.seed + ep)
        total_r = 0.0
        steps = 0
        for t in range(cfg.max_steps):
            a = agent.select_action(state)
            s_next, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            agent.update(state, a, r, s_next, done)
            total_r += r
            steps += 1
            state = s_next
            if done:
                break
        logs.append(EpisodeLog(ep+1, steps, total_r))

    env.close()

    run = RunResult(
        algo="Q-Learning",
        env_id=cfg.env_id,
        seed=cfg.seed,
        episodes=cfg.episodes,
        lr=cfg.alpha,
        gamma=cfg.gamma,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay=1.0,
        notes="Basic tabular Q-learning with epsilon-greedy.",
        episode_logs=logs
    )

    outdir = Path(cfg.outdir) / f"alpha{cfg.alpha}_eps{cfg.eps_start}"
    save_run(run, outdir)
    plot_learning_curves(run, outdir)
    summary = summarize(logs)
    with open(outdir/"summary.txt", "w") as f:
        for k, v in summary.items():
            f.write(f"{k}: {v}\n")
    return outdir

def sweep():
    # As per assignment: vary alpha and (interpreting doc typo) epsilon, keeping gamma fixed
    alphas = [0.1, 0.01, 0.001, 0.2]
    epsilons = [0.1, 0.2, 0.3]
    jobs = []
    for a in alphas:
        for e in epsilons:
            cfg = QConfig(alpha=a, eps_start=e, eps_end=e, episodes=5000, outdir="out/q_sweep")
            outdir = train(cfg)
            jobs.append(str(outdir))
    print("Completed runs:")
    for j in jobs:
        print(j)

if __name__ == "__main__":
    # Default: one baseline run; to run full sweep, execute sweep()
    train(QConfig())
