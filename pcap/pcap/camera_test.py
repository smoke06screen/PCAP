import ctypes
from importlib.machinery import EXTENSION_SUFFIXES
import os

import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraTest(Node):

    def __init__(self):
        super().__init__('camera_test')

        # ---------------------------------------------------------
        # Load the C shared library
        # ---------------------------------------------------------

        library_path = next(
            os.path.join(os.path.dirname(__file__), 'libperception' + suffix)
            for suffix in EXTENSION_SUFFIXES
            if os.path.isfile(
                os.path.join(os.path.dirname(__file__), 'libperception' + suffix)
            )
        )

        self.lib = ctypes.CDLL(library_path)

        # ---------------------------------------------------------
        # Define C function signature
        # ---------------------------------------------------------

        uchar_ptr = ctypes.POINTER(ctypes.c_uint8)

        self.lib.process_rgb8_to_gray.argtypes = [
            uchar_ptr,          # input image
            uchar_ptr,          # output grayscale image
            ctypes.c_int,       # width
            ctypes.c_int        # height
        ]

        self.lib.process_rgb8_to_gray.restype = ctypes.c_int

        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter(
            'show_window'
        ).get_parameter_value().bool_value

        self.get_logger().info(
            f'Loaded C library: {library_path}'
        )

        # ---------------------------------------------------------
        # ROS image subscriber
        # ---------------------------------------------------------

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        self.get_logger().info(
            'Subscribed to /camera/image_raw'
        )

    def image_callback(self, msg):

        # ---------------------------------------------------------
        # ROS Image -> NumPy
        # ---------------------------------------------------------

        if msg.encoding != 'rgb8':
            self.get_logger().error(
                f'Expected rgb8, got {msg.encoding}'
            )
            return

        row_size = msg.width * 3
        buffer_size = msg.step * msg.height

        if msg.step < row_size or len(msg.data) < buffer_size:
            self.get_logger().error('Invalid RGB image buffer dimensions')
            return

        rgb = np.frombuffer(
            msg.data,
            dtype=np.uint8,
            count=buffer_size
        ).reshape(msg.height, msg.step)[:, :row_size]

        rgb = np.ascontiguousarray(
            rgb.reshape(msg.height, msg.width, 3)
        )

        # ---------------------------------------------------------
        # Allocate output image
        # ---------------------------------------------------------

        gray = np.empty(
            (msg.height, msg.width),
            dtype=np.uint8
        )

        # ---------------------------------------------------------
        # NumPy buffer -> C
        # ---------------------------------------------------------

        input_ptr = rgb.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )

        output_ptr = gray.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )

        result = self.lib.process_rgb8_to_gray(
            input_ptr,
            output_ptr,
            msg.width,
            msg.height
        )

        if result != 0:
            self.get_logger().error(
                'C image processing failed'
            )
            return

        # ---------------------------------------------------------
        # Display C/OpenMP result
        # ---------------------------------------------------------

        if self.show_window:
            cv2.imshow('C + OpenMP - Grayscale', gray)
            cv2.waitKey(1)


def main(args=None):

    rclpy.init(args=args)

    node = CameraTest()

    try:
        rclpy.spin(node)

    except KeyboardInterrupt:
        pass

    finally:
        node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()

        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
