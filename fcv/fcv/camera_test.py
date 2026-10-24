import cv2
from fcv.perception import process_traversability_pipeline
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image


VALIDITY_LABELS = {
    0: 'No reliable boundary',
    1: 'Left boundary only',
    2: 'Right boundary only',
    3: 'Valid boundary pair',
    4: 'Pair rejected by geometry',
    5: 'Low confidence pair',
}

DISPLAY_TITLES = (
    'Original RGB',
    'Blurred',
    'Edges',
    'Hough Lines',
    'Boundary Mask',
    'Traversable Mask',
)
WINDOW_NAME = 'FCV Pipeline View'
DISPLAY_SCALE = 0.5
PANEL_TITLE_HEIGHT = 26
STATUS_HEIGHT = 32


def rgb8_message_to_array(msg):
    """Return the active RGB8 pixels from a possibly padded ROS image."""
    if msg.encoding != 'rgb8':
        raise ValueError(f'Expected rgb8, got {msg.encoding}')

    row_size = msg.width * 3
    buffer_size = msg.step * msg.height
    if (
        msg.width <= 0
        or msg.height <= 0
        or msg.step < row_size
        or len(msg.data) < buffer_size
    ):
        raise ValueError('Invalid RGB image buffer dimensions')

    rows = np.frombuffer(
        msg.data,
        dtype=np.uint8,
        count=buffer_size,
    ).reshape(msg.height, msg.step)
    return np.ascontiguousarray(
        rows[:, :row_size].reshape(msg.height, msg.width, 3)
    )


def create_display_frame(perception, images):
    """Compose the six diagnostic images into one OpenCV display frame."""
    height, width = images[0].shape[:2]
    panel_size = (
        max(1, int(width * DISPLAY_SCALE)),
        max(1, int(height * DISPLAY_SCALE)),
    )
    panels = []

    for index, (title, image) in enumerate(zip(DISPLAY_TITLES, images)):
        if index == 0:
            panel = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            interpolation = cv2.INTER_AREA
        else:
            panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            interpolation = (
                cv2.INTER_AREA if index == 1 else cv2.INTER_NEAREST
            )

        panel = cv2.resize(
            panel,
            panel_size,
            interpolation=interpolation,
        )
        panel = cv2.copyMakeBorder(
            panel,
            PANEL_TITLE_HEIGHT,
            0,
            0,
            0,
            cv2.BORDER_CONSTANT,
            value=(24, 24, 24),
        )
        cv2.putText(
            panel,
            title,
            (7, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        panels.append(panel)

    grid = np.vstack((
        np.hstack(panels[:3]),
        np.hstack(panels[3:]),
    ))
    status = np.full(
        (STATUS_HEIGHT, grid.shape[1], 3),
        24,
        dtype=np.uint8,
    )
    label = VALIDITY_LABELS.get(perception.validity, 'Unknown')
    cv2.putText(
        status,
        f'{label} | confidence {perception.confidence:.2f}',
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    return np.vstack((status, grid))


class CameraTest(Node):

    def __init__(self):
        super().__init__('camera_test')

        self.last_validity = None

        self.declare_parameter('show_window', True)
        self.show_window = self.get_parameter(
            'show_window'
        ).get_parameter_value().bool_value

        self.pending_display = None
        self.display_timer = None
        self.window_created = False

        if self.show_window:
            self.display_timer = self.create_timer(
                1.0 / 30.0,
                self.update_display,
            )

        # ---------------------------------------------------------
        # ROS image subscriber
        # ---------------------------------------------------------

        camera_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.subscription = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            camera_qos,
        )

        self.get_logger().info(
            'Subscribed to /camera/image_raw'
        )

    def image_callback(self, msg):

        # ---------------------------------------------------------
        # ROS Image -> NumPy
        # ---------------------------------------------------------

        try:
            rgb = rgb8_message_to_array(msg)
        except ValueError as error:
            self.get_logger().error(str(error))
            return

        perception, outputs = process_traversability_pipeline(rgb)
        denoised = outputs.denoised
        edges = outputs.edges
        hough_lines = outputs.hough_lines
        boundary_mask = outputs.boundary_mask
        traversable_mask = outputs.traversable_mask

        if perception.validity != self.last_validity:
            label = VALIDITY_LABELS.get(perception.validity, 'Unknown')
            self.get_logger().info(
                f'{label}; confidence={perception.confidence:.2f}'
            )
            self.last_validity = perception.validity

        if self.show_window:
            self.pending_display = (
                perception,
                (
                    rgb,
                    denoised,
                    edges,
                    hough_lines,
                    boundary_mask,
                    traversable_mask,
                ),
            )

    def update_display(self):
        if self.pending_display is None:
            return

        perception, updates = self.pending_display
        self.pending_display = None

        try:
            display = create_display_frame(perception, updates)
            cv2.imshow(WINDOW_NAME, display)
            self.window_created = True
            cv2.waitKey(1)

            visibility = cv2.getWindowProperty(
                WINDOW_NAME,
                cv2.WND_PROP_VISIBLE,
            )
            # GTK on Wayland returns -1 when this property is unsupported.
            if visibility == 0.0:
                self.show_window = False
                self.display_timer.cancel()
        except cv2.error as error:
            self.get_logger().error(f'Disabling display: {error}')
            self.show_window = False
            self.display_timer.cancel()

    def close_display(self):
        if self.window_created:
            try:
                cv2.destroyWindow(WINDOW_NAME)
                cv2.waitKey(1)
            except cv2.error:
                pass


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

        node.close_display()


if __name__ == '__main__':
    main()
