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
            'simple_driver = wall_follower.simple_driver:main',
            'mpc_driver = wall_follower.mpc_driver:main',
            'map_follower = wall_follower.map_follower:main',
            'robust_driver = wall_follower.robust_driver:main',
        ],
    },
)
