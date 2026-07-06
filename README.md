# Fronter Exploration

A **ROS 2** package for autonomous frontier-based robot exploration. The robot continuously detects unexplored boundaries (frontiers) in an occupancy grid map, clusters them, scores each cluster by balancing information gain against travel cost, and navigates to the best candidate — repeating until the entire environment has been mapped.

---

## Table of Contents

- [Overview](#overview)
- [How It Works](#how-it-works)
- [Architecture](#architecture)
- [Modules](#modules)
- [State Machine](#state-machine)
- [ROS 2 Topics & Actions](#ros-2-topics--actions)
- [Dependencies](#dependencies)
- [Building](#building)
- [Running](#running)
- [Project Structure](#project-structure)

---

## Overview

Frontier exploration is a classic autonomous navigation strategy: the robot identifies the boundary between known free space and unknown space (the *frontier*), and drives towards it to expand its knowledge of the environment. This package implements that strategy on top of the **Nav2** navigation stack, and continues until no frontiers remain — at which point the robot returns to its starting position `(0, 0)`.

---

## How It Works

1. **Map Subscription** — The node subscribes to the `/map` topic (`nav_msgs/OccupancyGrid`) published by a SLAM system (e.g., SLAM Toolbox).
2. **Frontier Detection** — Every free cell adjacent to an unknown cell is marked as a frontier cell.
3. **Clustering** — Frontier cells are grouped into connected clusters using BFS with 8-connectivity. Clusters smaller than 20 cells are discarded as noise.
4. **Path Planning** — For each cluster, the Nav2 `ComputePathToPose` action is called to get an actual path cost to the cluster's nearest point.
5. **Scoring** — Each cluster is scored as:

   ```
   score = information_gain / path_cost
   ```

   where `information_gain` is the cluster size (number of frontier cells).

6. **Navigation** — The robot navigates to the highest-scoring cluster using the Nav2 `NavigateToPose` action.
7. **Repeat** — Once the goal is reached, the state resets to `IDLE` and the cycle restarts on the next map update.
8. **Termination** — When no clusters are found, exploration is complete and the robot returns home.

---

## Architecture

```
┌──────────────────────────────────────────────────────────┐
│                        main_node                         │
│                                                          │
│  /map ──► Detector ──► Cluster ──► Planner ──► Nav2      │
│                           │                    Client    │
│                      Visualization                       │
│                  (frontier_map, markers)                 │
└──────────────────────────────────────────────────────────┘
```

The `main_node` orchestrates the entire pipeline. It delegates frontier detection, clustering, planning, visualization, and navigation to dedicated helper classes.

---

## Modules

| File | Class | Responsibility |
|---|---|---|
| `main_node.py` | `main_node` | Central ROS 2 node; orchestrates the full explore–plan–navigate loop |
| `detector.py` | `Detector` | Scans the occupancy grid and marks frontier cells (free cells bordering unknown cells) |
| `cluster.py` | `Cluster`, `FrontierCluster` | BFS-based clustering of frontier cells; computes centroid, nearest point, and final score |
| `planner_client.py` | `Planner` | Asynchronously requests paths to each cluster via the Nav2 `ComputePathToPose` action; calculates Euclidean path cost |
| `nav2_client.py` | `Nav2Client` | Sends navigation goals to Nav2 via the `NavigateToPose` action; resets state on completion |
| `visualization.py` | `Visulatization` | Publishes frontier mask as an `OccupancyGrid` and coloured cluster markers as a `MarkerArray` for RViz |
| `utils.py` | `Utils` | Converts between grid (row, col) and world (x, y) coordinates using map origin and resolution |
| `state.py` | `State` | Enum defining the four node states: `IDLE`, `PLANNING`, `SCORING`, `NAVIGATING` |

---

## State Machine

```
         ┌─────────────────────────────────────────┐
         │                                         │
         ▼                                         │
┌──────────────┐   map received   ┌─────────────┐  │ goal reached
│     IDLE     │ ──────────────► │   PLANNING  │  │
└──────────────┘                 └─────────────┘  │
                                       │           │
                              paths computed       │
                                       │           │
                                       ▼           │
                               ┌─────────────┐     │
                               │  NAVIGATING │ ────┘
                               └─────────────┘
```

- **IDLE** → Waits for a new map message. When received, begins frontier detection.
- **PLANNING** → Requests path costs for all clusters sequentially via a request queue.
- **NAVIGATING** → Robot is en-route to the best-scored frontier cluster.
- On goal completion → Returns to **IDLE** for the next map update.

---

## ROS 2 Topics & Actions

### Subscribed Topics

| Topic | Type | Description |
|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | Occupancy grid from SLAM |

### Published Topics

| Topic | Type | Description |
|---|---|---|
| `/frontier_map` | `nav_msgs/OccupancyGrid` | Binary map showing detected frontier cells |
| `/frontier_markers` | `visualization_msgs/MarkerArray` | Coloured cluster cubes, centroids, and nearest-point spheres for RViz |

### Action Clients

| Action Server | Type | Description |
|---|---|---|
| `navigate_to_pose` | `nav2_msgs/action/NavigateToPose` | Drives the robot to the selected frontier |
| `compute_path_to_pose` | `nav2_msgs/action/ComputePathToPose` | Computes a path to each frontier cluster for cost estimation |

---

## Dependencies

- **ROS 2** (Humble or later recommended)
- **Nav2** (`nav2_msgs`, `nav2_bringup`)
- **SLAM Toolbox** (or any SLAM system publishing `/map`)
- Python packages: `rclpy`, `numpy`

ROS 2 package dependencies (declared in `package.xml`):

```
rclpy
nav_msgs
geometry_msgs
tf2_ros
tf2_geometry_msgs
visualization_msgs
nav2_msgs
action_msgs
```

---

## Building

This is a standard **colcon** workspace. From the workspace root (`Fronter/`):

```bash
# Source your ROS 2 installation
source /opt/ros/<distro>/setup.bash

# Build the package
colcon build --packages-select fronter_exploration

# Source the install overlay
source install/setup.bash
```

---

## Running

Make sure a SLAM system and Nav2 are already running and publishing the `/map` topic and the required action servers.

```bash
# In a sourced terminal
ros2 run fronter_exploration main_node
```

To visualise the exploration in **RViz**, add the following displays:

| Display | Topic/Config |
|---|---|
| Map | `/map` |
| Map (frontiers) | `/frontier_map` |
| MarkerArray | `/frontier_markers` |
| Robot model | TF from your URDF |

---

## Project Structure

```
Fronter/
├── src/
│   └── fronter_exploration/
│       ├── fronter_exploration/
│       │   ├── __init__.py
│       │   ├── main_node.py       # ROS 2 node entry point
│       │   ├── detector.py        # Frontier cell detection
│       │   ├── cluster.py         # BFS clustering & scoring
│       │   ├── planner_client.py  # Nav2 path planning client
│       │   ├── nav2_client.py     # Nav2 navigation client
│       │   ├── visualization.py   # RViz marker publishing
│       │   ├── utils.py           # Coordinate conversion utilities
│       │   └── state.py           # Robot state enum
│       ├── package.xml
│       ├── setup.py
│       └── setup.cfg
├── build/                         # colcon build artifacts (generated)
├── install/                       # colcon install overlay (generated)
├── log/                           # colcon build logs (generated)
└── README.md
```
