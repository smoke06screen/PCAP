"""Launch the camera node without user-installed Python packages."""

import os
import site
import sys


def main():
    """Run the node using the ROS-compatible system Python environment."""
    if site.ENABLE_USER_SITE:
        environment = os.environ.copy()
        environment['PYTHONNOUSERSITE'] = '1'
        os.execve(
            sys.executable,
            [
                sys.executable,
                '-s',
                '-m',
                'fcv.camera_test',
                *sys.argv[1:],
            ],
            environment,
        )

    from fcv.camera_test import main as camera_main

    camera_main()
