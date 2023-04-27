from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='pointPaintingRos',
            executable='paintLidarNode',
            name='paintLidarNode',
            output='screen',
            emulate_tty=True
        )
    ])
