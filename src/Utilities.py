import numpy as np
import time


def deadband(value, band_radius):
    return max(value - band_radius, 0) + min(value + band_radius, 0)


def clipped_first_order_filter(input, target, max_rate, tau):
    rate = (target - input) / tau
    return np.clip(rate, -max_rate, max_rate)

def start_buffer(t):
    """
    Buffer Mini Pupper activation for user to be ready (hands off, recording, etc.)
    """
    start_buffer = t # seconds to buffer imu data before starting trot
    print(f"Start buffer: {start_buffer} seconds")
    time.sleep(start_buffer)

def initialize_stance(initial_command, config):
    """
    Mini Pupper requires a default stance to be set before any movement. This function initializes the robot to the default stance, which also aligns with the joystick REST posture.
    """
    state_height = initial_command.height
    state_foot_locations = (
    config.default_stance + np.array([0.0, 0.0, initial_command.height])[:, np.newaxis]
    )

    return state_height, state_foot_locations