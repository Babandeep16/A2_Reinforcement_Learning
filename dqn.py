\
import random, math
from collections import deque
from dataclasses import dataclass
from typing import List
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import trange
import gymnasium as gym

from metrics import EpisodeLog, RunResult, save_run, plot_learning_curves, summarize

@dataclass
class DQNConfig:
    env_id: str = "Taxi-v3"
    episodes: int = 4000
    max_steps: int = 200
    gamma: float = 0.99
    lr: float = 1e-3
    batch_size: int = 64
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay: int = 2000  # decay episodes
    target_update: int = 100  # steps
    memory_size: int = 50000
    seed: int = 7
    render_mode: str = None
    outdir: str = "out/dqn"

class QNet(nn.Module):
    def __init__(self, n_states, n_actions, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_states, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity, seed=0):
        self.buf = deque(maxlen=capacity)
        random.seed(seed)

    def push(self, *transition):
        self.buf.append(transition)

    def sample(self, batch_size):
        batch = random.sample(self.buf, batch_size)
        return map(np.array, zip(*batch))

    def __len__(self):
        return len(self.buf)

def one_hot_state(state_scalar: int, n_states: int):
    x = np.zeros(n_states, dtype=np.float32)
    x[state_scalar] = 1.0
    return x

def epsilon_by_episode(ep, cfg: DQNConfig):
    # exponential decay over episodes
    ratio = min(1.0, ep / cfg.eps_decay)
    eps = cfg.eps_end + (cfg.eps_start - cfg.eps_end) * math.exp(-3.0 * ratio)
    return eps

def train(cfg: DQNConfig):
    env = gym.make(cfg.env_id, render_mode=cfg.render_mode)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    rng = np.random.default_rng(cfg.seed)

    policy = QNet(n_states, n_actions)
    target = QNet(n_states, n_actions)
    target.load_state_dict(policy.state_dict())

    opt = optim.Adam(policy.parameters(), lr=cfg.lr)
    crit = nn.SmoothL1Loss()
    memory = ReplayBuffer(cfg.memory_size, seed=cfg.seed)

    logs: List[EpisodeLog] = []
    global_step = 0

    for ep in trange(cfg.episodes, desc="DQN"):
        state, _ = env.reset(seed=cfg.seed + ep)
        total_r, steps = 0.0, 0
        done = False
        while not done and steps < cfg.max_steps:
            eps = epsilon_by_episode(ep, cfg)
            if rng.random() < eps:
                action = rng.integers(0, n_actions)
            else:
                with torch.no_grad():
                    qvals = policy(torch.tensor(one_hot_state(state, n_states)).unsqueeze(0))
                    action = int(torch.argmax(qvals, dim=1).item())

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            memory.push(state, action, reward, next_state, done)
            state = next_state
            total_r += reward
            steps += 1
            global_step += 1

            # Learn
            if len(memory) >= cfg.batch_size:
                s, a, r, s2, d = memory.sample(cfg.batch_size)
                s  = torch.tensor(np.stack([one_hot_state(si, n_states) for si in s]), dtype=torch.float32)
                a  = torch.tensor(a, dtype=torch.int64).unsqueeze(1)
                r  = torch.tensor(r, dtype=torch.float32).unsqueeze(1)
                s2 = torch.tensor(np.stack([one_hot_state(si, n_states) for si in s2]), dtype=torch.float32)
                d  = torch.tensor(d, dtype=torch.float32).unsqueeze(1)

                q = policy(s).gather(1, a)
                with torch.no_grad():
                    q_next = target(s2).max(1, keepdim=True)[0]
                    target_q = r + (1 - d) * cfg.gamma * q_next

                loss = crit(q, target_q)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

            # target network update
            if global_step % cfg.target_update == 0:
                target.load_state_dict(policy.state_dict())

        logs.append(EpisodeLog(ep+1, steps, total_r))

    env.close()

    run = RunResult(
        algo="DQN",
        env_id=cfg.env_id,
        seed=cfg.seed,
        episodes=cfg.episodes,
        lr=cfg.lr,
        gamma=cfg.gamma,
        eps_start=cfg.eps_start,
        eps_end=cfg.eps_end,
        eps_decay=float(cfg.eps_decay),
        notes="One-hot state encoding, simple MLP, target network, replay.",
        episode_logs=logs
    )
    outdir = Path(cfg.outdir) / f"lr{cfg.lr}_gamma{cfg.gamma}"
    save_run(run, outdir)
    plot_learning_curves(run, outdir)
    with open(outdir/"summary.txt", "w") as f:
        for k, v in summarize(logs).items():
            f.write(f"{k}: {v}\n")
    return outdir

if __name__ == "__main__":
    train(DQNConfig())
