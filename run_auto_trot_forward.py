import numpy as np
import time
from src.IMU import IMU
from src.Controller import Controller
from src.State import State, BehaviorState
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from MangDang.mini_pupper.display import Display
from src.Command import Command


def main(use_imu=False):
    """Run robot: auto-activate, auto-trot, constant forward motion."""

    # Create config and hardware interfaces
    config = Configuration()
    hardware_interface = HardwareInterface()
    disp = Display()
    disp.show_ip()

    # Optional IMU
    if use_imu:
        imu = IMU(port="/dev/ttyACM0")
        imu.flush_buffer()
    else:
        imu = None

    # Controller and state
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()

    last_loop = time.time()
    initialize_time = last_loop ### For time control

    # Flag so we only send trot toggle once
    trot_enabled = False

    # Choose forward speed (m/s). Keep within config.max_x_velocity.
    forward_speed = min(0.15, config.max_x_velocity)

    print("Auto-trot script started.")
    print(f"Using forward speed: {forward_speed} m/s")

    while True:
        now = time.time()
        if now - last_loop < config.dt:
            continue
        last_loop = now

        #if now - initialize_time > 5.0:
            # Give some time to initialize
            #continue

        # Build a fresh command each cycle
        command = Command()

        # One-time trot toggle from REST to TROT
        if (not trot_enabled) and state.behavior_state == BehaviorState.REST:
            command.trot_event = True
            trot_enabled = True
            print("Sending trot_event -> robot should enter trot gait.")

        # After both toggles, stay in trot and just command velocity
        # IMU orientation (optional)
        if imu is not None:
            quat_orientation = imu.read_orientation()
        else:
            quat_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        state.quat_orientation = quat_orientation

        # Constant forward command, no lateral or turning motion
        command.horizontal_velocity = np.array([forward_speed, 0.0])
        command.yaw_rate = 0.0

        # Run controller and update hardware
        controller.run(state, command, disp)
        hardware_interface.set_actuator_postions(state.joint_angles)


if __name__ == "__main__":
    main()
