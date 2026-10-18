import ctypes
from importlib.machinery import EXTENSION_SUFFIXES
import os

import cv2
import matplotlib.pyplot as plt
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

        self.lib.process_traversability_pipeline.argtypes = [
            uchar_ptr,          # input rgb image
            uchar_ptr,          # grayscale output
            uchar_ptr,          # denoised output
            uchar_ptr,          # edge output
            uchar_ptr,          # road mask
            uchar_ptr,          # hough lines
            uchar_ptr,          # boundary mask
            uchar_ptr,          # traversable mask
            ctypes.c_int,       # width
            ctypes.c_int        # height
        ]

        self.lib.process_traversability_pipeline.restype = ctypes.c_int

        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter(
            'show_window'
        ).get_parameter_value().bool_value

        self.figure = None
        self.axes = None
        self.panels = None

        if self.show_window:
            plt.ion()
            self.figure, self.axes = plt.subplots(2, 3, figsize=(15, 8))
            manager = getattr(self.figure.canvas, 'manager', None)
            if manager is not None and hasattr(manager, 'set_window_title'):
                manager.set_window_title('PCAP Pipeline View')

            titles = [
                'Original RGB',
                'Blurred',
                'Edges',
                'Hough Lines',
                'Boundary Mask',
                'Traversable Mask',
            ]

            self.panels = []
            for panel_index, (axis, title) in enumerate(zip(self.axes.flat, titles)):
                axis.set_title(title)
                axis.axis('off')
                if panel_index == 0:
                    image = axis.imshow(np.zeros((10, 10, 3), dtype=np.uint8))
                else:
                    image = axis.imshow(
                        np.zeros((10, 10), dtype=np.uint8),
                        cmap='gray',
                        vmin=0,
                        vmax=255
                    )
                self.panels.append(image)

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

        gray = np.empty((msg.height, msg.width), dtype=np.uint8)
        denoised = np.empty((msg.height, msg.width), dtype=np.uint8)
        edges = np.empty((msg.height, msg.width), dtype=np.uint8)
        road_mask = np.empty((msg.height, msg.width), dtype=np.uint8)
        hough_lines = np.empty((msg.height, msg.width), dtype=np.uint8)
        boundary_mask = np.empty((msg.height, msg.width), dtype=np.uint8)
        traversable_mask = np.empty((msg.height, msg.width), dtype=np.uint8)

        # ---------------------------------------------------------
        # NumPy buffer -> C
        # ---------------------------------------------------------

        input_ptr = rgb.ctypes.data_as(
            ctypes.POINTER(ctypes.c_uint8)
        )

        gray_ptr = gray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        denoised_ptr = denoised.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        edges_ptr = edges.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        road_ptr = road_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        hough_ptr = hough_lines.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        boundary_ptr = boundary_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
        traversable_ptr = traversable_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))

        result = self.lib.process_traversability_pipeline(
            input_ptr,
            gray_ptr,
            denoised_ptr,
            edges_ptr,
            road_ptr,
            hough_ptr,
            boundary_ptr,
            traversable_ptr,
            msg.width,
            msg.height
        )

        if result != 0:
            self.get_logger().error(
                'C image processing failed'
            )
            return

        if self.show_window and self.figure is not None:
            updates = [
                rgb,
                denoised,
                edges,
                hough_lines,
                boundary_mask,
                traversable_mask,
            ]

            for panel, data in zip(self.panels, updates):
                panel.set_data(data)

            self.figure.tight_layout()
            self.figure.canvas.draw_idle()
            plt.pause(0.001)


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

        plt.close('all')


if __name__ == '__main__':
    main()
