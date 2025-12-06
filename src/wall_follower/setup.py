import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'wall_follower'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'slam_driver = wall_follower.slam_driver:main',
            'local_planner = wall_follower.local_planner:main',
            'global_planner = wall_follower.global_planner:main',
            'mpc_controller = wall_follower.mpc_controller:main',
            'tf_broadcaster = wall_follower.tf_broadcaster:main',
            'disparity_extender = wall_follower.disparity_extender:main',
            'ftg_driver = wall_follower.ftg_driver:main',
            'rw_follower = wall_follower.rw_follower:main',
            'map_driver = wall_follower.map_driver:main',
            'track_mapper = wall_follower.track_mapper:main',
            'path_racer = wall_follower.path_racer:main',
        ],
    },
)
