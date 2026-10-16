import os

from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.actions import SetEnvironmentVariable

from launch.launch_description_sources import PythonLaunchDescriptionSource

from launch_ros.actions import Node


def generate_launch_description():

    pkg_name = 'pcap'

    # ------------------------------------------------------------
    # Package paths
    # ------------------------------------------------------------

    package_share = get_package_share_directory(pkg_name)

    urdf_file = os.path.join(
        package_share,
        'urdf',
        'pcap_model.urdf'
    )

    world_file = os.path.join(
        package_share,
        'worlds',
        'pcap_test.world'
    )

    models_path = os.path.join(
        package_share,
        'models'
    )

    packages_path = os.path.dirname(package_share)

    # ------------------------------------------------------------
    # Read URDF
    # ------------------------------------------------------------

    with open(urdf_file, 'r') as f:
        robot_description = f.read()

    # ------------------------------------------------------------
    # Add our Gazebo models to GAZEBO_MODEL_PATH
    # while preserving any existing model paths.
    # ------------------------------------------------------------

    existing_model_path = os.environ.get(
        'GAZEBO_MODEL_PATH',
        ''
    )

    gazebo_model_paths = [
        models_path,
        packages_path,
    ]

    if existing_model_path:
        gazebo_model_paths.append(existing_model_path)

    gazebo_model_path = os.pathsep.join(gazebo_model_paths)

    gazebo_model_path_env = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=gazebo_model_path
    )

    # ------------------------------------------------------------
    # Robot State Publisher
    # ------------------------------------------------------------

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',

        parameters=[
            {
                'robot_description': robot_description,
                'use_sim_time': True
            }
        ]
    )

    # ------------------------------------------------------------
    # Gazebo
    # ------------------------------------------------------------

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),

        launch_arguments={
            'world': world_file
        }.items()
    )

    # ------------------------------------------------------------
    # Spawn robot
    # ------------------------------------------------------------

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',

        arguments=[
            '-topic',
            'robot_description',

            '-entity',
            'my_bot'
        ],

        output='screen'
    )

    # ------------------------------------------------------------
    # Launch everything
    # ------------------------------------------------------------

    return LaunchDescription([
        gazebo_model_path_env,
        gazebo,
        robot_state_publisher,
        spawn_entity
    ])
