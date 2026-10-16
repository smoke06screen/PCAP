from glob import glob
import os

from setuptools import Extension, find_packages, setup

package_name = 'pcap'


# ------------------------------------------------------------
# Collect all files inside models/ recursively
# This preserves the Gazebo model directory structure.
# ------------------------------------------------------------
model_data_files = []

for root, dirs, files in os.walk('models'):
    if files:
        install_dir = os.path.join(
            'share',
            package_name,
            root
        )

        model_data_files.append(
            (
                install_dir,
                [os.path.join(root, f) for f in files]
            )
        )


setup(
    name=package_name,
    version='0.0.0',

    packages=find_packages(exclude=['test']),

    ext_modules=[
        Extension(
            'pcap.libperception',
            sources=['pcap/native/perception.c'],
            extra_compile_args=['-O3', '-fopenmp'],
            extra_link_args=['-fopenmp'],
        ),
    ],

    data_files=[
        # ROS package index
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name]
        ),

        # package.xml
        (
            'share/' + package_name,
            ['package.xml']
        ),

        # Launch files
        (
            os.path.join('share', package_name, 'launch'),
            glob('launch/*.py')
        ),

        # URDF / model files
        (
            os.path.join('share', package_name, 'urdf'),
            glob('urdf/*')
        ),

        # Mesh files
        (
            os.path.join('share', package_name, 'meshes'),
            glob('meshes/*')
        ),

        # World files
        (
            os.path.join('share', package_name, 'worlds'),
            glob('worlds/*.world')
        ),

    ]

    # Add all Gazebo model files
    + model_data_files,

    install_requires=['setuptools'],

    zip_safe=False,

    maintainer='shubh',
    maintainer_email='shubh06kesar@gmail.com',

    description='PCAP parallel computer vision project',
    license='Apache-2.0',

    extras_require={
        'test': [
            'pytest',
        ],
    },

    entry_points={
        'console_scripts': [
            'camera_test = pcap.camera_test:main',
        ],
    },
)
