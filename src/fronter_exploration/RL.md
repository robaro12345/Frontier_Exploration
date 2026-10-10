# Optional DQN frontier selection

RL changes only the selection stage of the existing detector → cluster → planner
→ estimator → navigation pipeline. `rl:=false` keeps heuristic gain/path scoring
and does not import PyTorch. `training:=true` is ignored unless RL is enabled.

`rl:=true training:=true` trains; `rl:=true training:=false` requires a checkpoint
and performs deterministic greedy inference. Missing/incompatible checkpoints fail
at startup rather than falling back to an untrained policy.

## Build and start the simulation

From the workspace root (the commands below use this checkout's absolute path):

```bash
cd /home/omkar/Downloads/Fronter
source /opt/ros/jazzy/setup.bash
# Already created and installed during implementation; needed on another checkout:
python3 -m venv --system-site-packages .venv-rl
.venv-rl/bin/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-rl/bin/python -m colcon build --packages-select fronter_exploration --symlink-install
source install/setup.bash
ros2 launch fronter_exploration sim_launch.py world:=map01_open_obstacles.sdf
```

Wait for SLAM `/map` and Nav2 `compute_path_to_pose` and `navigate_to_pose` action
servers. The existing simulation launch does not start the exploration node.
Only run one exploration node at a time. This implementation was tested on
Python 3.12, ROS 2 Jazzy, and CPU PyTorch 2.14.1. A bounded Gazebo startup check
used the first metadata spawn pose, but the local renderer failed because OpenGL 3.3 was unavailable; physical exploration was
not verified.

## Train

In another terminal:

```bash
cd /home/omkar/Downloads/Fronter
source /opt/ros/jazzy/setup.bash
source install/setup.bash
.venv-rl/bin/python -m fronter_exploration.main_node --ros-args \
  --params-file /home/omkar/Downloads/Fronter/install/fronter_exploration/share/fronter_exploration/config/rl.yaml \
  -p use_sim_time:=true -p rl:=true -p training:=true \
  -p rl.checkpoint_dir:=/home/omkar/Downloads/Fronter/checkpoints
```

The Python module invocation ensures the interpreter has PyTorch even if a prior
`ros2 run` entry point was built with system Python. Plain heuristic use remains:

```bash
ros2 run fronter_exploration main_node --ros-args -p use_sim_time:=true -p rl:=false
```

Training logs reward, success/failure, known fraction of the current grid,
epsilon and loss on each completed transition. Loss is `None` until replay has
at least the configured batch size. At episode completion it logs total steps,
successful navigations, reward, explored area and the saved checkpoint path.
Completed failed/rejected actions use the same four-term reward and are counted
separately from successful navigation; failure itself is not an extra reward term.

The default episode limit is **one**. Completion stops new frontier selection,
saves `latest.pt` once optimizer updates exist, and leaves the node alive until
Ctrl+C. Periodic checkpoints are `step_N.pt` and `latest.pt`; clean shutdown also
saves only when optimizer updates exist. Zero-update saves are skipped with a
warning and preserve any existing trained checkpoint. With the default batch
size, at least 64 completed transitions are needed for the first optimizer update.
An interrupted navigation is not added to replay as a completed transition.

### Automatic training across worlds and restart poses

The workspace `maps.json` defines `spawn_poses_xyyaw` for every SDF world: 10
worlds × 12 poses = **120 episodes per pass**. The add-on runner uses the existing
`sim_launch.py` arguments `world`, `x_pose`, `y_pose`, and `yaw`. It keeps one
exploration node, DQN agent, optimizer, epsilon schedule and replay buffer alive,
and **restarts Gazebo, Nav2 and SLAM for every world/spawn episode**. No manual
simulation launch or episode-reset call is needed when using this runner.

```bash
cd /home/omkar/Downloads/Fronter
source /opt/ros/jazzy/setup.bash
source install/setup.bash
# Preview all world/pose arguments without starting ROS or Gazebo:
.venv-rl/bin/python -m fronter_exploration.RL.run_training --dry-run
# Train one pass through all 120 world/spawn combinations:
.venv-rl/bin/python -m fronter_exploration.RL.run_training \
  --maps-file /home/omkar/Downloads/Fronter/maps.json \
  --checkpoint-dir /home/omkar/Downloads/Fronter/checkpoints
```

The runner enumerates worlds in metadata order, then each world's spawn list in
order. `--rounds 2` repeats that schedule (240 episodes). `--episodes 5` caps the
schedule for a short run. `--maps map01_open_obstacles map02_corridor_loop` selects
worlds; `--pose-index 0` uses the first metadata pose in each selected world.
Neither world files nor metadata start poses are modified. The default metadata
path is `maps.json` in the current working directory.

```bash
# Two worlds, one supplied spawn per world; smaller batch for a short smoke run:
.venv-rl/bin/python -m fronter_exploration.RL.run_training \
  --maps map01_open_obstacles map02_corridor_loop --pose-index 0 \
  --ros-args -p rl.batch_size:=8 -p rl.reward_distance:=0.2
```

It runs Gazebo without its GUI by default using `-s --headless-rendering`. Use
`--gui` to show Gazebo; RViz stays disabled in automatic runs. The launch file's
manual default remains graphical. Headless rendering still needs compatible
OpenGL/EGL drivers for the robot's GPU lidar/camera; a renderer crash aborts the
automatic launch rather than counting an exploration episode.

Simulation and training use isolated ROS domain **186** by default and a private
Gazebo transport partition. Override `--domain-id` if needed. For ROS inspection
in a second terminal use `export ROS_DOMAIN_ID=186`. Do not manually launch an
additional stack in that domain. The runner stops only the process group it
created, waits for the old map/Nav2 endpoints to disappear, then resets the
episode and starts the next world. The simulation clock may restart from zero;
training and ROS timers remain in the same process.

`--startup-timeout 120` and `--episode-timeout 1800` are wall-clock seconds,
including simulation startup. Stalled SLAM, launch failures or unfinished
episodes abort the run and save the latest weights by default; they are not reported as
completed episodes and do not fabricate a transition. Use `--startup-retries 1`
to restart the same world/spawn once if exploration has not started, and
`--continue-on-failure` to record exhausted/active-episode failures and move on.
Every failed launch is a separate recorded attempt. Recovery requires stopping
the owned simulation and waiting for its map/Nav2 endpoints to disappear before
resetting subscriptions and invalidating old action callbacks. The agent,
optimizer, epsilon and replay remain in the same process. Unknown exceptions
and failure to remove old endpoints still stop the run.

`rl.unreachable_timeout` defaults to 60 wall-clock seconds from the first planning
cycle with no reachable candidates; another failed cycle does not restart the
deadline. A reachable candidate clears it. Expiration aborts the attempt rather
than claiming exploration success. If navigation completed and its fresh outcome
map was captured, that final measured transition is learned as terminal before
failure reporting; unfinished actions are discarded on the simulator reset.
Graphs are exported when the run ends, not between recovered attempts.

RL planning now tries at most `rl.frontier_goal_attempts` targets per cluster
(default 4), including the original point and any near-goal adjustment. Ordinary
Nav2 path failures can try spatially separated cells in that same frontier
cluster before pruning it. Rejected requests, cancelled goals and transport
timeouts do not trigger extra trials. The accepted target and its actual planned
path cost remain paired for navigation and DQN features. Heuristic mode still
uses its original one-point planning behavior. Four trials do not prove that
every possible cluster cell is unreachable; exhausted trials remain failures.

`--shutdown-timeout` defaults to 60 wall-clock seconds instead of the earlier
fixed 10-second wait. It reports map publisher counts and both action-server
readiness flags during discovery cleanup. It still stops if any old endpoint
remains after that deadline; it does not bypass the isolation check.

Reward map comparison retains exact integer-cell shifts. For fractional shifts
at the same resolution/orientation/frame, a current known cell earns gain only
if its whole footprint lies inside the previous grid and every old cell it
overlaps was unknown. This conservative comparison prevents shifted known-space
boundaries earning fake gain. It may undercount partially explored footprints;
cells outside the previous snapshot remain excluded. Resolution, orientation,
frame changes and non-overlapping snapshots still skip gain/stagnation with a
warning. `map_alignment` in the environment info identifies exact versus
conservative comparisons. Reward weights, network features and checkpoint
architecture are unchanged; historical rewards are not recomputed.

Logs for each launch are
`checkpoints/episodes/NNNN_WORLD_POSE.log`. Existing logs are preserved;
repeated filenames receive a timestamp suffix. Periodic model checkpoints
continue to use the existing training settings. The runner sets `rl`, `training`,
`use_sim_time`, and `rl.episode_limit` from its actual schedule; remaining reward
and optimizer parameters are configurable with `--ros-args`.

To retry the latest failed default-schedule entry 36 (`map03_four_rooms`, spawn 11):

```bash
.venv-rl/bin/python -m fronter_exploration.RL.run_training \
  --maps-file /home/omkar/Downloads/Fronter/maps.json \
  --checkpoint-dir /home/omkar/Downloads/Fronter/checkpoints \
  --start-episode 36 --gui --startup-retries 1 --continue-on-failure \
  --ros-args -p rl.checkpoint:=/home/omkar/Downloads/Fronter/checkpoints/latest.pt
```

After a run ends, rebuild a single report including every saved run, startup
failure and interrupted attempt. Earlier combined outputs are excluded to avoid
double counting. Repeat this command after future runs finish:

```bash
.venv-rl/bin/python -m fronter_exploration.RL.reporting \
  --all-runs /home/omkar/Downloads/Fronter/checkpoints/reports \
  --output-dir /home/omkar/Downloads/Fronter/checkpoints/reports/all_runs
```

The output contains `episodes.csv`, `training_report.txt`, `summary.json`,
`sources.json`, and the usual four PNG/PDF figure pairs. Attempt numbering is
continuous across runs, including failures and retries; it differs from the
world/spawn schedule position used by `--start-episode`. Success rate includes
all recorded attempts, including earlier startup diagnostics. Source CSVs are
preserved. Recovered source files carry `source_run.json` with their original
run timestamp so they appear at the correct position in the history.

To resume model training across environments, add this ROS argument to the
runner command:

```bash
--ros-args -p rl.checkpoint:=/home/omkar/Downloads/Fronter/checkpoints/latest.pt
```

Use `--start-episode N` to begin at position N in the selected world/spawn schedule
(one-based, after map/pose filters and rounds). `--episodes` limits the number of
remaining entries. For example, after 21 completed episodes in the default
120-entry schedule, `--start-episode 22` retries the failed entry and schedules
99 remaining entries. Supply `rl.checkpoint` as above to restore weights,
optimizer state, epsilon and learning counters. Replay is not saved, so a resumed
run refills a batch before optimizing. Reports and their episode numbers cover
the new process; world/pose fields identify the original schedule entries.

Resume restores policy/target/optimizer, epsilon, and lifetime counters; replay
starts empty after a process restart. In a single automatic run, replay survives
all world changes. Without `--start-episode`, the environment schedule starts
from the beginning on resume; select maps/poses, a start entry, or cap episodes
as needed. Configured learning parameters apply to the resumed run.

For the manual single-world training command above, the `reset_episode` service
is still available after external Gazebo/SLAM resets when using an episode limit
greater than one:

```bash
ros2 service call /main_node/reset_episode std_srvs/srv/Trigger '{}'
```

## Episode report and graphs

Training now prints each episode in this format, using measured values:

```text
Episode: ...
Cumulative Reward: ...
Steps: ...
Coverage: ...%
Success: True/False
Epsilon: ...
```

It also prints navigation successes, the outcome and reason, and the actual
world/start-pose index when using the automatic runner. At training completion
or shutdown it prints Total Episodes, Mean Episode Reward, Success Rate,
Mean Steps per Episode, and Final Epsilon.

The log prints the absolute report directory. Each process writes a separate
`CHECKPOINT_DIR/reports/UTC_TIMESTAMP_UNIQUE_ID/` directory containing:

- `episodes.csv`: full-precision metrics, world/spawn context, termination
  reason, navigation successes and optimizer-update counts, saved after each episode.
- `training_report.txt`: per-episode blocks and the running summary for your report.
- `summary.json`: machine-readable summary.
- `reward_vs_episode.png` / `.pdf`: unsmoothed cumulative reward for each episode.
- `success_rate_vs_episode.png` / `.pdf`: cumulative episode success percentage.
- `steps_per_episode.png` / `.pdf`: completed frontier actions per episode.
- `training_curves.png` / `.pdf`: all three graphs together.

Graphs are exported when the configured episode limit is reached or on orderly
shutdown, once for the final results. Intermediate episodes only update the
metrics and text report; no graph files or plot windows appear between episodes.
They run in the training worker or during cleanup, not inside map/Nav2
callbacks. Matplotlib is required for plotting (`python3-matplotlib` is declared
in the ROS package; it is already available in this checkout's virtual environment).
If plotting fails, CSV and text measurements remain saved and the error is logged.
Graphs are not generated for an empty run. No sample training values are invented.

Metric definitions:

- **Cumulative Reward:** sum of rewards actually submitted to replay during that episode.
- **Steps:** completed frontier-action transitions; a pending unfinished action is excluded.
- **Coverage:** known cells (free or occupied) / all cells in the latest received
  occupancy grid at episode end, shown as a percentage. It is not ground-truth
  world coverage; SLAM map expansion can change its denominator. If no map was
  received, coverage is `N/A`, not zero.
- **Success:** reaching the existing detector completion condition (no eligible
  frontiers) without aborting. This does not prove full physical exploration and
  is distinct from individual Nav2 action success.
- **Success Rate:** successful completed episodes / all recorded attempts.
  Aborted/interrupted attempts count as failures. They do not increment the
  existing completed-episode or checkpoint episode counters.
- **Epsilon:** the agent's actual exploration probability at the end of the episode.

A startup failure in the automatic runner is a failed attempt with zero steps;
an interruption records accumulated rewards/steps without fabricating a terminal
transition. Reports cover only the current process. Resuming a checkpoint makes
a new report, preserves earlier files, and does not invent history that the
checkpoint never stored. A hard kill or power loss can leave the current attempt
unfinished; previously finalized episodes remain in the CSV.

To regenerate/export the graphs from an actual report's CSV:

```bash
# Use the actual episodes.csv path printed in the training log:
.venv-rl/bin/python -m fronter_exploration.RL.reporting \
  /absolute/path/to/episodes.csv --output-dir /absolute/path/to/report_figures
```

After the resumed run ends, combine its CSV with earlier run CSVs, oldest first:

```bash
.venv-rl/bin/python -m fronter_exploration.RL.reporting \
  /absolute/path/to/original/episodes.csv \
  /absolute/path/to/resumed/episodes.csv \
  --output-dir /absolute/path/to/combined_report
```

This writes one `episodes.csv`, `training_report.txt`, `summary.json`, the three
individual graph pairs and the combined graph pair. Source files are preserved,
and `sources.json` records their order. Attempts are renumbered consecutively;
failed/interrupted attempts remain in the summary and graph denominator. A retry
of failed attempt 22 is therefore combined attempt 23, with world/pose fields
identifying the same scheduled environment. Add further run CSVs to the command
after additional resumes. An output directory containing a source report is
rejected, as is listing the same CSV twice. Training still exports graphs only
at run completion or orderly shutdown; combining is a separate command run afterward.

## Deploy

Stop training, restart the simulation/SLAM with a fresh map, then:

```bash
cd /home/omkar/Downloads/Fronter
source /opt/ros/jazzy/setup.bash
source install/setup.bash
.venv-rl/bin/python -m fronter_exploration.main_node --ros-args \
  -p use_sim_time:=true -p rl:=true -p training:=false \
  -p rl.checkpoint:=/home/omkar/Downloads/Fronter/checkpoints/latest.pt
```

Inference selects the highest Q among the current candidates with epsilon zero.
It never collects replay or optimizes. Once no candidates remain, it sends the
existing return-to-origin goal once. Heuristic termination retains the existing
behavior of issuing that goal and immediately exiting.

## Settings and reward

Defaults are in `config/rl.yaml` and declared as ordinary ROS parameters only
when RL is enabled. Override with `-p rl.NAME:=VALUE` or edit a copy of that YAML.
`checkpoint_dir` is created automatically and relative paths resolve against the
process's working directory; absolute paths are recommended.

| Parameter (`rl.` prefix) | Default | Meaning |
| --- | ---: | --- |
| `learning_rate` | 0.001 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `epsilon_start` / `epsilon_min` | 1.0 / 0.05 | Training exploration probabilities |
| `epsilon_decay` | 0.995 | Multiplier after each completed transition |
| `replay_capacity` | 10000 | Stored transitions |
| `batch_size` | 64 | Replay samples per optimizer step |
| `target_update` | 100 | Optimizer updates between target synchronizations |
| `seed` | 0 | Initial PyTorch and action/replay random seed |
| `reward_gain` | 0.7 | Weight of newly known square metres |
| `reward_distance` | 0.2 | Weight of planned path metres |
| `reward_stagnation` | 0.1 | Weight of insufficient-gain indicator |
| `reward_collision` | 0.0 | Weight of genuine contact-event count |
| `stagnation_threshold` | 0.05 | New known area (m²) below which an action stagnates |
| `collision_enabled` | false | Enable the external contact hook only after connecting a real sensor |
| `allow_untrained_checkpoint` | false | Explicit deployment-only testing override for zero-update weights |
| `map_max_age` | 5.0 | Maximum pre/post-map acquisition age in seconds on the ROS clock |
| `map_settle_time` | 0.2 | Minimum map acquisition delay after navigation completes (ROS seconds) |
| `map_wait_timeout` | 30.0 | Wall seconds to await a suitable post-navigation map before aborting |
| `episode_limit` | 1 | Completed episodes in this process |
| `checkpoint_every` | 50 | Completed transitions between saves |
| `checkpoint_dir` | `./checkpoints` | Training output directory |
| `checkpoint` | empty | Optional training resume, required for deployment |
| `navigation_timeout` | 300.0 | Wall seconds before RL navigation cancellation |
| `planner_timeout` | 30.0 | Seconds on the node clock before advancing past a stalled path request |
| `frontier_goal_min_distance` | 0.5 | Minimum RL goal and planned endpoint separation from the planned start, in metres; keep above Nav2's XY goal tolerance |
| `frontier_goal_attempts` | 4 | Maximum requested target points per frontier cluster in RL mode |
| `unreachable_timeout` | 60.0 | Wall seconds of repeated planning failure before recording an aborted attempt |

Reward is:

```text
reward_gain * (unknown_to_known_cells * resolution²)
- reward_distance * selected_planned_path_metres
- reward_stagnation * (1 if newly_known_area < stagnation_threshold, otherwise 0)
- reward_collision * actual_reported_contacts
```

New known cells include free **and occupied** cells, and must previously have
been `-1`. Known-to-known and known-to-unknown changes receive no gain. Using
square metres avoids tying reward magnitude to map resolution. The defaults
sum to **1.0**; all four weights must be finite, nonnegative, and sum to one
within absolute tolerance `1e-6`, or startup fails. Each term is applied once.
There is no reward clipping or normalization. Stagnation is evaluated once from
the settled action outcome, not from map callbacks during motion.
Distance is the planner estimate, not measured robot
odometry; failed actions also incur that configured cost.

When a SLAM grid grows/shifts, only overlapping physical footprints are compared.
Integer shifts retain direct cell comparison. Fractional shifts require every
previous cell overlapping a new known cell to have been unknown. New cells
outside the old grid lack a previous measurement and receive no reward in that
step. Incompatible resolution/rotation, changed frame, nonoverlapping extents,
or non-axis-aligned rotation skip gain and stagnation with a warning rather than
crediting unrelated cells or inventing stagnation.
Distance and genuine contacts remain applicable. Known fraction is a diagnostic, not a
success criterion.

The supplied SDF has physical collision geometry and contact/friction properties,
but **no contact sensor or bridged collision-event topic**. `Env.record_collision()`
is the explicit hook for a future real sensor callback. The hook is disabled by
default and the default collision weight is zero. To use a genuine sensor, connect
its events to that hook, enable `rl.collision_enabled`, and choose four weights
that still sum to one. Without that hookup, leave the contribution disabled.
Aborted/rejected/timed-out navigation is a failure,
never a fabricated collision. No new collision topic or simulation sensor is
introduced by this add-on.

## Transition and Q semantics

`Env.step(frontier_id, state)` validates membership and map age, copies the latest
received acquisition immediately before dispatch (including maps received during
planning), freezes candidate features, and returns its cluster immediately. `main_node` sends
the goal through `Nav2Client`. Nav2 callbacks record real result status and put
the node back in IDLE. Map callbacks keep executing during motion. Duplicate and
out-of-order timestamps are rejected; a post-map must be acquired strictly after
both the baseline and completion plus `map_settle_time`, and pass `map_max_age`.
That outcome is frozen before detection/planning. Only after estimating reachable candidates
is `(state, action_id, reward, next_state, done)` completed. Training never waits
inside a ROS callback for navigation or gradient updates.

The Q network is `2 → 64 → 64 → 1`, shared by all frontiers. Each replay item
freezes the chosen two-feature vector and the complete next candidate feature
set. IDs may be regenerated/reused without invalidating replay. The Bellman
target is `reward + gamma * max(Q_target(next_candidates))`; terminal or empty
next sets use reward alone. Adam, Smooth L1 loss, gradient clipping, a target
network and bounded random replay are implemented. Checkpoints are CPU tensors
loaded with `weights_only=True`. Features are `[frontier_cells / 100,
planned_path_metres / 10]` without clipping, identically in training/deployment.
A single 2 m route is encoded as 0.2 and a 20 m route as 2.0.

Format-2 checkpoints contain `format_version`, `feature_scaling`, `policy`,
`target`, `optimizer`, `epsilon`, `steps`, `updates`, and `episodes`. Replay,
RNG state and full run configuration are not saved. Both networks retain the
same architecture. Format-1 tensor payloads remain readable by `DQNAgent.load`,
but normal training/deployment reject them because their candidate-relative
features are incompatible with the corrected scaling; retrain into a new
directory. Deployment rejects zero optimizer updates unless the explicit testing
override is set, and logs a warning when that override is used. The saver also
rejects explicitly requested untrained saves over existing trained checkpoints.

Episodes complete when the detector finds no eligible clusters (including the
existing minimum size threshold). This criterion does not prove every physical
cell is explored. Planning failures prune individual candidates; if all plans
fail, the node retries on a fresh map, preserves the pending transition and does
not increment completed-episode counters. Any viable candidate can continue
the episode. A genuine terminal transition uses reward alone in the Bellman target.
The two features give a limited observation, so policy quality still requires
measurement in simulation.

In RL mode, a goal or planned endpoint within `frontier_goal_min_distance` of
the planned start is replaced once by a farther cell in the same cluster,
closest to its centroid, and replanned. An unusable replacement is pruned for
that map cycle; it does not declare exploration complete. This prevents a
centroid-near frontier cell inside Nav2's goal tolerance from causing repeated
instant successes without movement. Heuristic target selection is unchanged.

Action rejections, exceptions and timeouts advance the state machine. Late
callbacks after timeouts are ignored and accepted late handles are cancelled.
RL navigation timeouts request cancellation, then wait up to ten wall seconds
for a terminal action result. A cancel acknowledgment alone does not finalize
the action. Unconfirmed cancellation aborts the episode without a fabricated
transition or completion count. Navigation and post-map watchdogs use steady
time; planner timeouts still use the node clock. Missing fresh maps abort after
`map_wait_timeout`. Restart the node and simulation after an aborted episode;
the automatic runner reports the error and shuts down its own simulation.
Transport errors that leave the motion outcome unknown also abort without a
fabricated transition.

An episode reset recreates the map subscription to discard unread samples from
the previous simulator before accepting the next world's maps.

## Verification

```bash
cd /home/omkar/Downloads/Fronter
source /opt/ros/jazzy/setup.bash
source install/setup.bash
.venv-rl/bin/python -m pytest \
  src/fronter_exploration/test/test_rl.py \
  src/fronter_exploration/test/test_rl_fixes.py \
  src/fronter_exploration/test/test_rl_reporting.py \
  src/fronter_exploration/test/test_rl_integration.py \
  src/fronter_exploration/test/test_rl_schedule.py -q
# Actual ROS/DDS action servers on isolated domains 187, 188 and 189:
FRONTER_ROS_TEST=1 .venv-rl/bin/python -m pytest \
  src/fronter_exploration/test/test_rl_ros.py -q
# Original package checks (run from package directory to avoid generated copies):
cd src/fronter_exploration
../../.venv-rl/bin/python -m pytest test -q
```

The synthetic DDS tests drive the real map subscription, detector, clustering,
planner action, navigation action, worker and optimizer for successful and failed goals,
and check that map callbacks execute during navigation. A second test drives the
automatic world loop with two synthetic simulation stacks, clock resets, and
the same agent/replay throughout. It exercises deployment
mode with a saved checkpoint and controlled candidate estimates. It does not
simulate robot motion, contacts, SLAM accuracy, or Gazebo physics.

Focused tests cover current-ID selection, snapshot integrity, reward transitions,
map shifts, empty/terminal Bellman targets, optimization, target update,
checkpoint round-trip, failures, queue progression, late callbacks, episode
reset and heuristic scoring. The ROS test is opt-in; PyTorch-dependent tests skip
on machines without PyTorch. Current results are reported after executing the
commands above; synthetic DDS tests do not establish Gazebo or learned policy
quality. The original package-wide flake8 check also covers existing formatting
issues outside this add-on. No project-wide formatting cleanup is included.
No validated deployment policy is included. Latest executed validation:
99 focused tests passed and all 6 opt-in real ROS/DDS tests passed after the
bounded alternative-goal, conservative map-comparison and shutdown-wait corrections.
The DDS tests include a map/clock-only failed startup followed by a healthy retry
and the next scheduled world. These controlled transports verify recovery, not
the root cause of Nav2 lifecycle service response timeouts in Gazebo.
The environment, runner and planner modules also pass scoped flake8.
A previous DDS run exposed
a stale-map reset failure; recreating the subscription discards that queue.
The ROS package builds successfully. The installed runner dry-run enumerated
120 world/start-pose episodes.

A bounded Gazebo check used the user's trained checkpoint as read-only input,
test batch size 1, and 105-second episode limits. It reached three goals on
`map06_warehouse`, spawn 11, stopped that owned stack, and launched `map07_office`,
spawn 0. Five further completed transitions were learned there; alternative
frontier-point trials were exercised. Eight transitions produced eight optimizer
updates, and all six policy tensors changed in the private test checkpoint.
Both attempts aborted at the deliberate time limits; neither completed
exploration. Logs and weights are in `/tmp/fronter-target-gz.5pPu6R`. The user
checkpoint's SHA256 was unchanged. A read-only sample of actual SLAM snapshots
measured a 198.0787-cell fractional origin shift and 3,063 conservative
unknown-to-known cells. This verifies fractional comparison and a real restart,
not guaranteed warehouse completion or long-run discovery reliability.

A bounded Gazebo run on `map01_open_obstacles`, spawn index 0, used a test batch
size of 1 and a 100-second episode limit. Odometry reported a maximum displacement
of 6.29 m; known map cells increased from 1,385 to 47,131. Two completed navigation
transitions produced two optimizer updates; all six policy parameter tensors
differed from their seeded initialization. The run aborted at its deliberate
wall-time limit, with zero completed episodes. This verifies motion/map growth
and updates, not exploration completion or learned policy quality. One transition
skipped gain/stagnation because its map origins did not align to the cell lattice.
Test logs and weights were kept under `/tmp/fronter-motion-after`, preserving
the user's checkpoint directory. Default training still uses batch size 64.

The earlier full package suite returned 60 passed, 4 skipped and 1 failed: its
broad flake8 check reported 175 existing style findings. That full suite was not
rerun for the motion correction; no project-wide formatting changes were made.

## Files added or modified for the add-on

- `fronter_exploration/RL/env.py`, `dqn.py`, `training.py`, `deploy.py`,
  `run_training.py`, `reporting.py`: environment, candidate network, worker,
  inference, metadata-driven world/spawn training loop and measured report/graph export.
- `launch/sim_launch.py`: optional headless Gazebo launch and prompt shutdown
  when that simulator exits. Manual graphical launch defaults are preserved.
- `fronter_exploration/main_node.py`: optional selection branches, transition
  coordination, episode reset service and shutdown checkpoint.
- `fronter_exploration/nav2_client.py`, `planner_client.py`: asynchronous status,
  rejected requests, exceptions, timeout/cancellation and queue progression.
- `fronter_exploration/cluster.py`: safe removal while iterating and zero-cost
  denominator guard; normal heuristic scores are unchanged.
- `config/rl.yaml`, `RL.md`, `test/test_rl*.py`: configuration, instructions and tests.
- `setup.py`, `package.xml`: install configuration and declare NumPy/std_srvs.
- Workspace `.gitignore` and `README.md`: local environment/checkpoint ignores
  and a link to this guide.

Existing local changes, RL package initialization, simulation assets, detector,
visualization, coordinate utilities, world definitions and metadata are retained.
