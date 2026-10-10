import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'fronter_exploration'

launch_files = glob('launch/*.py')

data_files = [
    ('share/ament_index/resource_index/packages',
     ['resource/' + package_name]),
    ('share/' + package_name,
     ['package.xml']),
    (os.path.join('share', package_name, 'rviz'),
     glob('rviz/*.rviz')),
    (os.path.join('share', package_name, 'worlds'),
     glob('worlds/*.sdf')),
    (os.path.join('share', package_name, 'urdf'),
     glob('urdf/*.urdf')),
     (os.path.join('share', package_name, 'urdf'),
     glob('urdf/*.xacro')),
    (os.path.join('share', package_name, 'launch'),
     launch_files),
    (os.path.join('share', package_name, 'config'),
     glob('config/*.yaml')),
]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='omkar',
    maintainer_email='omkar@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'main_node = fronter_exploration.main_node:main',
        ],
    },
)
