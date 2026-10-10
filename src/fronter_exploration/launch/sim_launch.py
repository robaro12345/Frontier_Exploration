import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, TimerAction, DeclareLaunchArgument, ExecuteProcess
from launch.actions import EmitEvent, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.events import Shutdown
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.conditions import IfCondition, UnlessCondition


def generate_launch_description():
    world = LaunchConfiguration('world')
    x_pose = LaunchConfiguration('x_pose')
    y_pose = LaunchConfiguration('y_pose')
    yaw = LaunchConfiguration('yaw')
    use_rviz = LaunchConfiguration('use_rviz')
    headless = LaunchConfiguration('headless')

    gz_pkg = 'ros_gz_sim'
    tb3_pkg = 'nav2_minimal_tb3_sim'
    nav2_pkg = 'nav2_bringup'
    exploration_pkg = 'fronter_exploration'

    tb3_share = get_package_share_directory(tb3_pkg)
    exploration_share = get_package_share_directory(exploration_pkg)

    gz_launch = os.path.join(
        get_package_share_directory(gz_pkg),
        'launch',
        'gz_sim.launch.py'
    )

    spawn_launch = os.path.join(
        tb3_share,
        'launch',
        'spawn_tb3.launch.py'
    )

    nav2_launch = os.path.join(
        get_package_share_directory(nav2_pkg),
        'launch',
        'bringup_launch.py'
    )

    rviz_config = os.path.join(
        exploration_share,
        'rviz',
        'nav2_sim.rviz'
    )

    nav2_params_file = os.path.join(
        get_package_share_directory(nav2_pkg),
        'params',
        'nav2_params.yaml'
    )

    world_file = PathJoinSubstitution([
        exploration_share,
        'worlds',
        world
    ])

    robot_description_path = os.path.join(
        exploration_share,
        'urdf',
        'turtlebot3_waffle.urdf'
    )

    robot_sdf = os.path.join(
        exploration_share,
        'urdf',
        'gz_waffle_rgb.sdf.xacro'
    )

    with open(robot_description_path, 'r', encoding='utf-8') as f:
        robot_description = f.read()

    ld = LaunchDescription()

    ld.add_action(
        DeclareLaunchArgument(
            'world',
            default_value='map01_open_obstacles.sdf',
            description='SDF world file located in the worlds directory'
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            'x_pose',
            default_value='0.0',
            description='Robot initial X position'
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            'y_pose',
            default_value='0.0',
            description='Robot initial Y position'
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            'yaw',
            default_value='0.0',
            description='Robot initial yaw'
        )
    )

    ld.add_action(
        DeclareLaunchArgument(
            'use_rviz',
            default_value='true',
            description='Launch RViz'
        )
    )

    ld.add_action(DeclareLaunchArgument(
        'headless', default_value='false', description='Run Gazebo without its GUI'))

    headless_sim = ExecuteProcess(
        cmd=['gz', 'sim', '-r', '-s', '--headless-rendering', world_file],
        output='screen', condition=IfCondition(headless))
    ld.add_action(headless_sim)
    ld.add_action(RegisterEventHandler(OnProcessExit(
        target_action=headless_sim,
        on_exit=[EmitEvent(event=Shutdown(reason='Headless Gazebo exited'))])))

    gui_sim = ExecuteProcess(
        condition=UnlessCondition(headless),
        cmd=['gz', 'sim', '-r', world_file],
        output='screen')
    ld.add_action(gui_sim)
    ld.add_action(RegisterEventHandler(OnProcessExit(
        target_action=gui_sim,
        on_exit=[EmitEvent(event=Shutdown(reason='GUI Gazebo exited'))])))

    ld.add_action(
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[
                {'use_sim_time': True},
                {'robot_description': robot_description},
            ],
        )
    )

    ld.add_action(
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(spawn_launch),
            launch_arguments={
                'robot_name': 'turtlebot3_waffle',
                'robot_sdf': robot_sdf,
                'x_pose': x_pose,
                'y_pose': y_pose,
                'z_pose': '0.01',
                'roll': '0.00',
                'pitch': '0.00',
                'yaw': yaw,
            }.items(),
        )
    )

    ld.add_action(
        Node(
            package='ros_gz_bridge',
            executable='parameter_bridge',
            name='camera_bridge',
            output='screen',
            arguments=[
                '/camera/image_raw@sensor_msgs/msg/Image@gz.msgs.Image'
            ],
        )
    )

    nav2_args = {
        'slam': 'True',
        'use_localization': 'True',
        'use_sim_time': 'true',
        'autostart': 'true',
        'use_composition': 'False',
        'use_respawn': 'False',
        'params_file': nav2_params_file,
    }

    ld.add_action(
        TimerAction(
            period=20.0,
            actions=[
                IncludeLaunchDescription(
                    PythonLaunchDescriptionSource(nav2_launch),
                    launch_arguments=nav2_args.items(),
                )
            ],
        )
    )

    ld.add_action(
        TimerAction(
            period=5.0,
            actions=[
                Node(
                    package='rviz2',
                    executable='rviz2',
                    output='screen',
                    parameters=[
                        {'use_sim_time': True}
                    ],
                    arguments=[
                        '-d',
                        rviz_config
                    ],
                    condition=IfCondition(use_rviz)
                )
            ],
        )
    )

    return ld
