"""A shared Q network scores candidates, independent of transient frontier IDs."""

from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

from fronter_exploration.RL.env import FEATURE_SCALING
import numpy as np
import torch
from torch import nn


@dataclass
class DQNConfig:
    """Small CPU network and initial learning settings."""

    learning_rate: float = .001
    gamma: float = .99
    epsilon_start: float = 1.
    epsilon_min: float = .05
    epsilon_decay: float = .995
    replay_capacity: int = 10000
    batch_size: int = 64
    target_update: int = 100
    seed: int = 0


class QNetwork(nn.Module):
    """Map [scaled frontier cells, scaled planned path metres] to scalar Q."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, features):
        """Score a batch of individual candidates."""
        return self.layers(features).squeeze(-1)


class DQNAgent:
    """Replay stores chosen features, never a permanent output index or ID."""

    def __init__(self, config=None):
        self.config = config or DQNConfig()
        c = self.config
        if not (0 <= c.gamma <= 1 and 0 <= c.epsilon_min <= c.epsilon_start <= 1
                and 0 < c.epsilon_decay <= 1 and c.learning_rate > 0
                and 0 < c.batch_size <= c.replay_capacity and c.target_update > 0):
            raise ValueError('Invalid DQN configuration')
        torch.manual_seed(c.seed)
        torch.set_num_threads(1)
        self.random = random.Random(c.seed)
        self.policy = QNetwork()
        self.target = QNetwork()
        self.target.load_state_dict(self.policy.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=c.learning_rate)
        self.loss = nn.SmoothL1Loss()
        self.replay = deque(maxlen=c.replay_capacity)
        self.epsilon = c.epsilon_start
        self.steps = 0
        self.updates = 0
        self.episodes = 0
        self.feature_scaling = dict(FEATURE_SCALING)

    def select_action(self, state, explore=True):
        """Return a key from this state's currently available candidates."""
        if not state:
            return None
        keys = list(state)
        if explore and self.random.random() < self.epsilon:
            return self.random.choice(keys)
        with torch.no_grad():
            features = torch.as_tensor(np.stack([state[key] for key in keys]),
                                       dtype=torch.float32)
            return keys[int(self.policy(features).argmax().item())]

    def remember(self, transition):
        """Freeze action features and next candidates for future replay."""
        state, action, reward, next_state, done = transition
        chosen = np.asarray(state[action], dtype=np.float32).copy()
        candidates = (np.stack(list(next_state.values())).astype(np.float32).copy()
                      if next_state else np.empty((0, 2), dtype=np.float32))
        self.replay.append((chosen, float(reward), candidates,
                            bool(done or not next_state)))
        self.steps += 1
        self.epsilon = max(self.config.epsilon_min,
                           self.epsilon * self.config.epsilon_decay)

    def bellman_targets(self, samples):
        """Bootstrap over each next candidate set, with zero bootstrap at terminal."""
        with torch.no_grad():
            targets = []
            for _, reward, candidates, done in samples:
                future = 0.
                if not done and len(candidates):
                    future = self.target(torch.as_tensor(candidates)).max().item()
                targets.append(reward + self.config.gamma * future)
            return torch.tensor(targets, dtype=torch.float32)

    def optimize(self):
        """Run one replay minibatch and periodically synchronize target weights."""
        if len(self.replay) < self.config.batch_size:
            return None
        samples = self.random.sample(list(self.replay), self.config.batch_size)
        features = torch.as_tensor(np.stack([sample[0] for sample in samples]))
        loss = self.loss(self.policy(features), self.bellman_targets(samples))
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=10.)
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_update == 0:
            self.target.load_state_dict(self.policy.state_dict())
        return float(loss.item())

    def save(self, path, allow_untrained=False):
        """Atomically save networks, optimizer and schedule counters on CPU."""
        path = Path(path).expanduser().resolve()
        if self.updates == 0:
            if not allow_untrained:
                return False
            if path.exists():
                existing = torch.load(path, map_location='cpu', weights_only=True)
                if existing.get('updates', 0) > 0:
                    raise ValueError('Untrained weights cannot overwrite a trained checkpoint')
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + '.tmp')
        torch.save({'format_version': 2, 'feature_scaling': self.feature_scaling,
                    'policy': self.policy.state_dict(),
                    'target': self.target.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'epsilon': self.epsilon, 'steps': self.steps,
                    'updates': self.updates, 'episodes': self.episodes}, temporary)
        temporary.replace(path)
        return True

    def load(self, path, training=False):
        """Load tensor-only checkpoint; restore optimizer only for training."""
        checkpoint = torch.load(Path(path).expanduser().resolve(),
                                map_location='cpu', weights_only=True)
        if checkpoint.get('format_version') not in (1, 2):
            raise ValueError('Unsupported DQN checkpoint format')
        self.feature_scaling = checkpoint.get('feature_scaling')
        self.policy.load_state_dict(checkpoint['policy'])
        self.target.load_state_dict(checkpoint['target'])
        self.steps = checkpoint['steps']
        self.updates = checkpoint['updates']
        self.episodes = checkpoint['episodes']
        if training:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            # The explicitly configured learning rate takes precedence on resume.
            for group in self.optimizer.param_groups:
                group['lr'] = self.config.learning_rate
            self.epsilon = max(self.config.epsilon_min, checkpoint['epsilon'])
        else:
            self.epsilon = 0.
            self.policy.eval()
