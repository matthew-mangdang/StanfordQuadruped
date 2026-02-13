import numpy as np
import time
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for SSH
import matplotlib.pyplot as plt


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
    print(f"Initialized to default stance with height: {state_height} m and foot locations:\n{state_foot_locations}")
    return state_height, state_foot_locations


def save_velocity_plot(v_data, filename="velocity_plot.png"):
    """
    Save velocity plot as PNG file (for SSH, no visualization).
    
    Args:
        v_data: List of velocity measurements
        filename: Output PNG filename (default: velocity_plot.png)
    """
    if len(v_data) == 0:
        print("No velocity data to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(v_data, linewidth=2, label='Forward Velocity (m/s)')
    plt.xlabel('Sample')
    plt.ylabel('Velocity (m/s)')
    plt.title('Forward Velocity Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Velocity plot saved to: {filename}")

def save_acceleration_plot(a_data, filename="acceleration_plot.png"):
    """
    Save acceleration plot as PNG file (for SSH, no visualization).
    
    Args:
        a_data: List of acceleration measurements
        filename: Output PNG filename (default: acceleration_plot.png)
    """
    if len(a_data) == 0:
        print("No acceleration data to plot.")
        return
    
    plt.figure(figsize=(12, 6))
    plt.plot(a_data, linewidth=2, label='Forward Acceleration (m/s²)')
    plt.xlabel('Sample')
    plt.ylabel('Acceleration (m/s²)')
    plt.title('Forward Acceleration Over Time')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=100)
    plt.close()
    print(f"Acceleration plot saved to: {filename}")