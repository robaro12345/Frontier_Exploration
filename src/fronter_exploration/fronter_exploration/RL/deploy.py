"""Inference-only frontier selection through the existing navigation layer."""

from fronter_exploration.RL.dqn import DQNAgent
from fronter_exploration.RL.env import FEATURE_SCALING


class Deployment:
    """Load a required checkpoint and select without epsilon-greedy noise."""

    def __init__(self, checkpoint, allow_untrained=False):
        if not checkpoint:
            raise ValueError('RL deployment requires -p rl.checkpoint:=/absolute/model.pt')
        self.agent = DQNAgent()
        self.agent.load(checkpoint, training=False)
        if self.agent.feature_scaling != FEATURE_SCALING:
            raise ValueError('Checkpoint uses incompatible/legacy feature scaling; '
                             'retrain before deployment')
        if self.agent.updates == 0 and not allow_untrained:
            raise ValueError('Checkpoint has zero optimizer updates; testing only: '
                             '-p rl.allow_untrained_checkpoint:=true')

    def select_action(self, state):
        """Choose the greatest Q among the current frontier IDs, or None if empty."""
        return self.agent.select_action(state, explore=False)
