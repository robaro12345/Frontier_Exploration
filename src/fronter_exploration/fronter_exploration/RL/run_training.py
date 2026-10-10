"""Train across the worlds and spawn poses in the workspace's maps.json."""

import argparse
import json
import math
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Episode:
    """One world and one actual spawn pose from the metadata."""

    world: str
    pose_index: int
    pose: tuple


def episode_schedule(metadata, worlds_directory, maps=None, rounds=1, limit=None,
                     pose_index=None, start_episode=1):
    """Validate metadata and enumerate every requested world/spawn combination."""
    if rounds < 1 or (limit is not None and limit < 1):
        raise ValueError('Rounds and episode limit must be positive')
    names = list(metadata) if maps is None else maps
    if not names:
        raise ValueError('No maps selected')
    episodes = []
    for name in names:
        if name not in metadata:
            raise ValueError(f'Map {name!r} is missing from maps.json')
        if Path(name).name != name:
            raise ValueError('Map keys must be world names, not paths')
        world = name if name.endswith('.sdf') else name + '.sdf'
        if not (Path(worlds_directory) / world).is_file():
            raise ValueError(f'World file is missing: {world}')
        poses = metadata[name].get('spawn_poses_xyyaw', [])
        if not poses:
            raise ValueError(f'No spawn_poses_xyyaw for {name}')
        indices = range(len(poses)) if pose_index is None else [pose_index]
        for index in indices:
            if not 0 <= index < len(poses):
                raise ValueError(f'Pose index {index} is invalid for {name}')
            pose = poses[index]
            if (not isinstance(pose, list) or len(pose) != 3 or
                    not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                            and math.isfinite(v) for v in pose)):
                raise ValueError(f'Invalid [x, y, yaw] pose {index} for {name}')
            episodes.append(Episode(world, index, tuple(pose)))
    episodes *= rounds
    if not 1 <= start_episode <= len(episodes):
        raise ValueError(f'Start episode must be between 1 and {len(episodes)}')
    episodes = episodes[start_episode - 1:]
    return episodes if limit is None else episodes[:limit]


def launch_command(episode, headless=True):
    """Reuse the existing simulation launch and its inspected pose arguments."""
    x, y, yaw = episode.pose
    return ['ros2', 'launch', 'fronter_exploration', 'sim_launch.py',
            'world:=' + episode.world, f'x_pose:={x}', f'y_pose:={y}',
            f'yaw:={yaw}', 'use_rviz:=true',
            'headless:=' + str(headless).lower()]


def stop_simulation(process):
    """Stop only the process group created for this runner's simulation launch."""
    for sig, timeout in [(signal.SIGINT, 10.), (signal.SIGTERM, 5.),
                         (signal.SIGKILL, 5.)]:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            continue
        # Launch may exit while one of its children still runs.
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
    process.wait(timeout=5.)


class EpisodeFailure(RuntimeError):
    """A simulator or exploration failure that can be recorded and skipped."""


class EpisodeTimeout(TimeoutError):
    """A bounded startup or exploration attempt exceeded its wall-clock limit."""


def _wait_for_shutdown(node, spin_once, context_ok, timeout=60.):
    """Drain callbacks and require old simulation endpoints to disappear."""
    started = time.monotonic()
    next_log = started + 10.
    while context_ok():
        spin_once(node, timeout_sec=.05)
        maps = node.count_publishers('map')
        navigation_ready = node.nav.client.server_is_ready()
        planner_ready = node.planner.client.server_is_ready()
        if maps == 0 and not navigation_ready and not planner_ready:
            return
        now = time.monotonic()
        status = (f'map publishers={maps}, navigation ready={navigation_ready}, '
                  f'planner ready={planner_ready}')
        if now - started > timeout:
            raise RuntimeError(
                f'Old map/Nav2 publishers still present after shutdown ({status}); '
                f'waited {timeout:g}s')
        if now > next_log:
            node.get_logger().warn(f'Waiting for ROS discovery after shutdown: {status}')
            next_log = now + 10.
    raise RuntimeError('ROS context stopped during simulator shutdown')


def run_episodes(node, episodes, spin_once, context_ok, headless=True,
                 startup_timeout=120., episode_timeout=1800.,
                 startup_retries=0, continue_on_failure=False, shutdown_timeout=60.):
    """Restart the simulator per episode while retaining node/agent/replay."""
    from std_srvs.srv import Trigger

    if not all(math.isfinite(value) and value > 0
               for value in (startup_timeout, episode_timeout, shutdown_timeout)):
        raise ValueError('Startup, episode and shutdown timeouts must be finite and positive')
    if startup_retries < 0:
        raise ValueError('Startup retries must be nonnegative')
    directory = node.trainer.directory / 'episodes'
    directory.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ, GZ_PARTITION=f'fronter-training-{os.getpid()}')
    attempts = 0
    completed_before = node.trainer.completed_episodes
    for number, episode in enumerate(episodes, start=1):
        retry = 0
        while True:
            if not context_ok():
                raise RuntimeError('ROS context stopped before all episodes completed')
            if attempts:
                _wait_for_shutdown(node, spin_once, context_ok, shutdown_timeout)
                if getattr(node, 'episode_aborted', False):
                    response = node._reset_episode(
                        Trigger.Request(), Trigger.Response(), simulation_stopped=True)
                else:
                    response = node._reset_episode(Trigger.Request(), Trigger.Response())
                if not response.success:
                    raise RuntimeError(response.message)
            attempts += 1
            node.latest_map = None
            node.trainer.prepare_episode(episode.world, episode.pose_index, episode.pose)
            log_path = directory / (
                f'{number:04d}_{Path(episode.world).stem}_{episode.pose_index}.log')
            if log_path.exists():
                log_path = log_path.with_name(f'{log_path.stem}_{time.time_ns()}.log')
            node.get_logger().info(
                f'Training environment {number}/{len(episodes)}: '
                f'{episode.world} spawn[{episode.pose_index}]={episode.pose}; '
                f'startup retry={retry}/{startup_retries}; log={log_path}')
            process = None
            failure = None
            was_started = False
            completed_at_start = node.trainer.completed_episodes
            started = time.monotonic()
            try:
                with log_path.open('w') as log:
                    process = subprocess.Popen(launch_command(episode, headless),
                                               env=environment, stdout=log,
                                               stderr=subprocess.STDOUT,
                                               start_new_session=True)
                    while context_ok():
                        spin_once(node, timeout_sec=.05)
                        if node.exploration_complete and node._rl_future is None:
                            if getattr(node, 'episode_aborted', False):
                                raise EpisodeFailure('Exploration aborted; see episode report')
                            if node.trainer.completed_episodes != completed_at_start + 1:
                                raise RuntimeError('RL stopped without completing the episode')
                            break
                        if process.poll() is not None:
                            raise EpisodeFailure(
                                f'Simulation launch exited: {process.returncode}; see {log_path}')
                        elapsed = time.monotonic() - started
                        if not node._episode_started and elapsed > startup_timeout:
                            raise EpisodeTimeout(f'No exploration started; see {log_path}')
                        if elapsed > episode_timeout:
                            raise EpisodeTimeout(
                                f'Episode exceeded wall-time limit; see {log_path}')
                    else:
                        raise RuntimeError('ROS context stopped during exploration')
            except (EpisodeFailure, EpisodeTimeout) as error:
                failure = error
                was_started = node._episode_started
                node._abort_rl(str(error))
            except Exception as error:
                node._abort_rl(str(error))
                raise
            finally:
                if process is not None:
                    stop_simulation(process)
            if failure is None:
                break
            # Preserve each failed attempt before resetting metrics or restarting ROS stacks.
            if node._rl_future is not None:
                node._rl_future.result(timeout=30.)
                node._poll_rl()
            if not was_started and retry < startup_retries:
                retry += 1
                node.get_logger().warn(f'Startup failed; retrying the same spawn: {failure}')
                continue
            if not continue_on_failure:
                raise failure
            node.get_logger().warn(f'Failed attempt recorded; advancing schedule: {failure}')
            break
    completed = node.trainer.completed_episodes - completed_before
    node.get_logger().info(
        f'Processed {len(episodes)} scheduled environments: '
        f'{completed} completed, {attempts - completed} failed attempts')


def main(args=None):
    """Launch the automatic loop; remaining arguments are normal ROS parameters."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps-file', type=Path, default=Path('maps.json'))
    parser.add_argument('--maps', nargs='+', help='Map keys to select; default: all')
    parser.add_argument('--rounds', type=int, default=1)
    parser.add_argument('--episodes', type=int, help='Cap the scheduled episode count')
    parser.add_argument('--start-episode', type=int, default=1,
                        help='Resume at this one-based position after map/pose filters '
                             'and rounds, before applying --episodes')
    parser.add_argument('--pose-index', type=int, help='Use one pose per selected map')
    parser.add_argument('--checkpoint-dir', type=Path, default=Path('checkpoints'))
    parser.add_argument('--domain-id', type=int, default=186,
                        help='Isolated ROS domain for the runner and simulation')
    parser.add_argument('--startup-timeout', type=float, default=120.)
    parser.add_argument('--episode-timeout', type=float, default=1800.)
    parser.add_argument('--shutdown-timeout', type=float, default=60.,
                        help='Seconds to wait for old ROS endpoints after stopping simulation')
    parser.add_argument('--startup-retries', type=int, default=0,
                        help='Additional simulator launches per spawn after startup failure')
    parser.add_argument('--continue-on-failure', action='store_true',
                        help='Record failed attempts and advance to the next scheduled spawn')
    parser.add_argument('--gui', action='store_true', help='Show Gazebo GUI')
    parser.add_argument('--dry-run', action='store_true', help='Print schedule without launching')
    options, ros_arguments = parser.parse_known_args(args)
    from ament_index_python.packages import get_package_share_directory
    share = Path(get_package_share_directory('fronter_exploration'))
    episodes = episode_schedule(json.loads(options.maps_file.read_text()), share / 'worlds',
                                options.maps, options.rounds, options.episodes,
                                options.pose_index, options.start_episode)
    if options.dry_run:
        print(f'{len(episodes)} episodes from {options.maps_file.resolve()}')
        for number, episode in enumerate(episodes, start=1):
            print(number, ' '.join(launch_command(episode, not options.gui)))
        return
    if not all(math.isfinite(value) and value > 0 for value in (
            options.startup_timeout, options.episode_timeout, options.shutdown_timeout)):
        parser.error('Timeouts must be finite and positive')
    if options.startup_retries < 0:
        parser.error('Startup retries must be nonnegative')
    if not 0 <= options.domain_id <= 232:
        parser.error('Domain ID must be between 0 and 232')
    # The simulation children and exploration node must share the isolated domain.
    os.environ['ROS_DOMAIN_ID'] = str(options.domain_id)
    import rclpy
    from fronter_exploration.main_node import main_node
    if '--ros-args' not in ros_arguments:
        ros_arguments.append('--ros-args')
    ros_arguments += ['-p', 'rl:=true', '-p', 'training:=true',
                      '-p', 'use_sim_time:=true',
                      '-p', f'rl.episode_limit:={len(episodes)}',
                      '-p', 'rl.checkpoint_dir:=' + str(options.checkpoint_dir.resolve())]
    rclpy.init(args=ros_arguments, domain_id=options.domain_id)
    node = None
    try:
        node = main_node()
        run_episodes(node, episodes, rclpy.spin_once, rclpy.ok, not options.gui,
                     options.startup_timeout, options.episode_timeout,
                     options.startup_retries, options.continue_on_failure,
                     options.shutdown_timeout)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.trainer.close(coverage=node._report_info().get('known_fraction'))
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
