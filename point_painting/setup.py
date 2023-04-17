from setuptools import setup

package_name = 'point_painting'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name], 
    py_modules=[
        'point_painting.pointPaintingNode',
        'point_painting.KittiCalibration',
        'point_painting.BiseNetv2',
        'point_painting.utils',
        'point_painting.bev_utils',
        'point_painting.pointpainting',
    ], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/point_painting_launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Markus Hillreiner',
    maintainer_email='Markus.hillreiner1@hs-augsburg.de',
    description='ROS2 package for Point Painting',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pointPaintingNode = point_painting.pointPaintingNode:main'
        ],
    },
)
