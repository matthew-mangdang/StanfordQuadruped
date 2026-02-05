import numpy as np
import time
from src.Controller import Controller
from src.State import State, BehaviorState
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from MangDang.mini_pupper.display import Display
from src.Command import Command
from src.Imu_Esp32 import IMU
from MangDang.mini_pupper.ESP32Interface import ESP32Interface


def main(use_imu=False):
    """Run robot: auto-activate, auto-trot, constant forward motion."""

    # Create config and hardware interfaces
    config = Configuration()
    hardware_interface = HardwareInterface()
    disp = Display()
    disp.show_ip()


    # Controller and state
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()
    last_loop = time.time()
    initialize_time = last_loop  # Track start time for protection
    last_print_time = last_loop

    ### IMU handler
    # Set up esp32 interface
    esp32 = ESP32Interface()
    imu_handler = IMU(esp32)

    ### Movement handler
    # Flag so we only send trot toggle once
    trot_enabled = False

    # Choose forward speed (m/s). Keep within config.max_x_velocity.
    forward_speed = min(0.15, config.max_x_velocity)
    forward_speed = 0.0
    end_time = 10.0 # seconds until auto-stop for safety

    print("Auto-trot script started.")
    print(f"Using forward speed: {forward_speed} m/s")
    print(f"Auto-stop time set to: {end_time} seconds")


    while True:
        now = time.time()
        if now - last_loop < config.dt:
            continue
        last_loop = now

        # Protection: Return to default position after 5 seconds
        if now - initialize_time > end_time:
            print(f"{end_time} seconds elapsed - returning robot to REST position")
            command = Command()
            if state.behavior_state == BehaviorState.TROT:
                command.trot_event = True  # Toggle back to REST
            controller.run(state, command, disp)
            hardware_interface.set_actuator_postions(state.joint_angles)
            time.sleep(0.5)  # Give time to settle
            break


        # Get imu data
        imu_data = imu_handler.get_raw_data()
        if (now - last_print_time)%0.5 > 0:
            #print(f"IMU Raw Data: {imu_data}")
            print(f"IMU Orientation pitch, roll: {imu_handler.get_orientation(imu_data)[0]}, {imu_handler.get_orientation(imu_data)[1]}")
            last_print_time = now

        # Build a fresh command each cycle
        command = Command()

        # One-time trot toggle from REST to TROT
        if (not trot_enabled) and state.behavior_state == BehaviorState.REST:
            command.trot_event = True
            trot_enabled = True
            print("Sending trot_event -> robot should enter trot gait.")

        # After both toggles, stay in trot and just command velocity
        # IMU orientation (optional) Currently manual set to no rotation
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
