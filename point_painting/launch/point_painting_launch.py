import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
        point_painting_node = Node(
            package='point_painting',
            executable='pointPaintingNode',
            name='point_painting_node',
            output='screen',
            emulate_tty=True
        )

        return LaunchDescription([point_painting_node])