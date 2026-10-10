"""Callback-driven training with all updates and checkpoint I/O in one worker."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fronter_exploration.RL.dqn import DQNAgent, DQNConfig
from fronter_exploration.RL.env import FEATURE_SCALING
from fronter_exploration.RL.reporting import (
    EpisodeReport, episode_text, plot_records, summary_text)


PARAMETERS = {
    'learning_rate': .001, 'gamma': .99,
    'epsilon_start': 1., 'epsilon_min': .05, 'epsilon_decay': .995,
    'replay_capacity': 10000, 'batch_size': 64, 'target_update': 100, 'seed': 0,
    'reward_gain': .7, 'reward_distance': .2, 'reward_stagnation': .1,
    'reward_collision': 0., 'stagnation_threshold': .05, 'collision_enabled': False,
    'allow_untrained_checkpoint': False, 'map_max_age': 5.,
    'map_settle_time': .2, 'map_wait_timeout': 30.,
    'episode_limit': 1, 'checkpoint_every': 50,
    'checkpoint_dir': './checkpoints', 'checkpoint': '',
    'navigation_timeout': 300., 'planner_timeout': 30.,
    'frontier_goal_min_distance': .5,
    'unreachable_timeout': 60.,
    'frontier_goal_attempts': 4,
}


class Training:
    """Process each completed transition, then choose the following action."""

    def __init__(self, settings, logger):
        """Restore or initialize the learner and create an independent run report."""
        self.agent = DQNAgent(DQNConfig(**{
            key: settings[key] for key in DQNConfig.__dataclass_fields__}))
        if settings['checkpoint']:
            self.agent.load(settings['checkpoint'], training=True)
            if self.agent.feature_scaling != FEATURE_SCALING:
                raise ValueError('Checkpoint uses incompatible/legacy features; retrain '
                                 'with fixed feature scaling in a new checkpoint directory')
        self.directory = Path(settings['checkpoint_dir']).expanduser().resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoint_every = settings['checkpoint_every']
        self.episode_limit = settings['episode_limit']
        if self.checkpoint_every < 1 or self.episode_limit < 1:
            raise ValueError('Checkpoint interval and episode limit must be positive')
        self.logger = logger
        self.report = EpisodeReport(self.directory)
        self.logger.info(f'RL report directory: {self.report.directory}')
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='dqn')
        self.completed_episodes = 0
        self.episode_reward = 0.
        self.episode_steps = 0
        self.successes = 0
        self.explored_area = 0.
        self.coverage = None
        self.episode_context = {'world': '', 'pose_index': '', 'spawn_pose': ''}
        self._episode_active = False
        self._closed = False
        self._exported_count = -1

    @property
    def finished(self):
        """Whether the requested number of episodes in this process has ended."""
        return self.completed_episodes >= self.episode_limit

    def start_episode(self, coverage=None):
        """Clear metrics after an explicit simulation/SLAM episode reset."""
        if self.finished:
            raise RuntimeError('Training episode limit reached')
        self.episode_reward = 0.
        self.episode_steps = 0
        self.successes = 0
        self.explored_area = 0.
        self.coverage = coverage
        self._episode_active = True

    def prepare_episode(self, world, pose_index, pose):
        """Mark a scheduled attempt before launch so startup failures are recorded."""
        self.start_episode()
        self.episode_context = {'world': world, 'pose_index': pose_index,
                                'spawn_pose': list(pose)}

    def decide(self, state, transition=None, info=None, terminal=False):
        """Return a Future so the ROS executor can keep dispatching callbacks."""
        return self.worker.submit(self._process, state, transition, info or {}, terminal)

    def _process(self, state, transition, info, terminal):
        if state or transition is not None:
            self._episode_active = True
        if info.get('known_fraction') is not None:
            self.coverage = float(info['known_fraction'])
        if transition is not None:
            self.agent.remember(transition)
            loss = self.agent.optimize()
            self.episode_reward += transition[2]
            self.episode_steps += 1
            self.successes += int(info.get('navigation_success', False))
            self.explored_area += info.get('explored_area', 0.)
            self.logger.info(
                f'RL step={self.agent.steps} reward={transition[2]:.3f} '
                f'success={info.get("navigation_success")} '
                f'known={info.get("known_fraction", 0.):.3f} '
                f'epsilon={self.agent.epsilon:.3f} loss={loss}')
            if self.agent.steps % self.checkpoint_every == 0:
                self._save(f'step_{self.agent.steps}.pt')
                self._save('latest.pt')
        if terminal:
            self.completed_episodes += 1
            self.agent.episodes += 1
            self._record_episode(
                True, 'completed', info.get('reason', 'No frontier clusters remain'))
            saved = self._save('latest.pt')
            self.logger.info(
                f'RL episode={self.agent.episodes} steps={self.episode_steps} '
                f'successes={self.successes} reward={self.episode_reward:.3f} '
                f'explored_area={self.explored_area:.3f}m^2 '
                f'checkpoint={saved or "not saved (zero optimizer updates)"}')
            if self.finished:
                self._export_results()
            return None
        return self.agent.select_action(state, explore=True)

    def _record_episode(self, success, outcome, reason):
        """Record measured metrics without changing the learning or completion counters."""
        row = self.report.append(
            cumulative_reward=self.episode_reward, steps=self.episode_steps,
            coverage=self.coverage, success=success, epsilon=self.agent.epsilon,
            navigation_successes=self.successes, optimizer_updates=self.agent.updates,
            outcome=outcome, reason=reason, **self.episode_context)
        self._episode_active = False
        self.logger.info('\n' + episode_text(row))
        return row

    def abort_episode(self, reason, coverage=None, transition=None, info=None):
        """Queue failure reporting after any already submitted optimizer job."""
        return self.worker.submit(self._record_abort, reason, coverage,
                                  transition=transition, info=info)

    def _record_abort(self, reason, coverage=None, outcome='aborted',
                      transition=None, info=None):
        if self._episode_active:
            if transition is not None:
                self._process({}, transition, info or {}, terminal=False)
            if coverage is not None:
                self.coverage = coverage
            self._record_episode(False, outcome, reason)
            self._save('latest.pt')

    def _save(self, name):
        """Never publish an untrained model as the latest trained checkpoint."""
        path = self.directory / name
        if self.agent.save(path):
            return path
        self.logger.warn('Checkpoint skipped: zero optimizer updates; existing files preserved')
        return None

    def close(self, coverage=None, reason='Training stopped before episode completion'):
        """Save trained weights, measured results and graphs outside ROS callbacks."""
        if self._closed:
            return
        self.worker.shutdown(wait=True)
        self._record_abort(reason, coverage, outcome='interrupted')
        self._save('latest.pt')
        self._export_results()
        self._closed = True

    def _export_results(self):
        """Export once per finalized record count, on the worker or at shutdown."""
        count = len(self.report.records)
        if count == self._exported_count:
            return
        summary = self.report.write_summary(self.agent.epsilon)
        self.logger.info('\n' + summary_text(summary))
        try:
            paths = plot_records(self.report.records, self.report.directory)
            if paths:
                self.logger.info(f'RL graphs saved: {self.report.directory}')
        except Exception as error:
            self.logger.warn(f'Graph export failed: {error}; episode CSV/text report preserved')
        else:
            self._exported_count = count
