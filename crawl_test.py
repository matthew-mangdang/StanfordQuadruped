import numpy as np
import time
import math
from collections import deque
from src.Controller import Controller
from src.State import State, BehaviorState
from src.MovementScheme import MovementScheme
from src.crawl_dance_action_list import MovementLib
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from pupper.Config_mp2 import Configuration
#from MangDang.mini_pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from MangDang.mini_pupper.display import Display
from src.Command import Command
from src.Imu_Esp32 import IMU
from src.Utilities import start_buffer, initialize_stance 
from src.filter import butter_filter_live

from MangDang.mini_pupper.ESP32Interface import ESP32Interface


def main(use_imu=False):
    """Test script to run the crawl gait."""

    start_buffer(0.0) # seconds

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

    # Initialize to default stance (align with joystick REST posture)
    initial_command = Command()
    state.height, state.foot_locations = initialize_stance(initial_command, config)
    #Create movement group scheme instance and set a default True state
    movementCtl = MovementScheme(MovementLib)
    dance_active_state = True
    lib_length = len(MovementLib)

    ### IMU handler
    # Set up esp32 interface
    esp32 = ESP32Interface()
    imu_handler = IMU(esp32)
    #pitch_offset, roll_offset = imu_handler.get_orientation_offset(50, 2.74, 1.51) ### Get offset for pitch and roll at zero
    a_offset = imu_handler.get_a_offset(50, [-0.03, 0.047, -1.053]) #[0.03, 0.04, -1.05]
    lp_filter = butter_filter_live(cutoff_freq=10.0, dt=config.dt, order=1)
    hit_flag = 0.0
    hit_tol = 0.30
    last_hit_time = last_loop

    ### Movement handler - user input  
    crawl_forward_speed = min(0.03, config.max_x_velocity)
    trot_forward_speed = min(0.09, config.max_x_velocity)
    side_speed = min (0.00, config.max_y_velocity)
    run_time = 60.0  # seconds until auto-stop for safety

    # Temporary crawl "mode" parameters
    crawl_duration = 20.0  # seconds to stay in crawl after a hit
    crawl_mode_active = False
    crawl_mode_start_time = None

    print("Script started and start trot}")
    print(f"Trot & crawl forward speed: {trot_forward_speed} m/s & {crawl_forward_speed} m/s")
    print(f"Side speed: {side_speed} m/s")
    print(f"Auto-stop time set to: {run_time} seconds")


    while True:
        now = time.time()
        if now - last_loop < config.dt:
            continue
        last_loop = now

        # Build a fresh command each cycle
        command = Command()

        # Protection: Return to default position after run_time seconds
        if now - initialize_time > run_time:
            print(f"{run_time} seconds elapsed - returning robot to REST position")
            command = Command()
            if state.behavior_state == BehaviorState.TROT:
                command.trot_event = True  # Toggle back to REST
                command.horizontal_velocity = np.array([0.0, 0.0])
            if state.behavior_state == BehaviorState.CRAWL:
                command.crawl_event = True  # Toggle back to REST
                command.horizontal_velocity = np.array([0.0, 0.0])
            controller.run(state, command, disp)
            hardware_interface.set_actuator_postions(state.joint_angles)
            time.sleep(0.5)  # Give time to settle
            break


        crawl_mode_active = True
        crawl_mode_start_time = now
        if state.behavior_state != BehaviorState.CRAWL:
            command.crawl_event = False
            command.pitch = 0.0
            command.horizontal_velocity = np.array([crawl_forward_speed, side_speed])


        # IMU orientation (optional) Currently manual set to no rotation
        quat_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        state.quat_orientation = quat_orientation

        # Step the controller forward by dt
        if dance_active_state == True:
            # Caculate legsLocation, attitudes and speed using custom movement script
            movementCtl.runMovementScheme()
            command.legslocation        = movementCtl.getMovemenLegsLocation()
            command.horizontal_velocity = movementCtl.getMovemenSpeed()
            command.roll                = movementCtl.attitude_now[0]
            command.pitch               = movementCtl.attitude_now[1]
            command.yaw                 = movementCtl.attitude_now[2]
            command.yaw_rate            = movementCtl.getMovemenTurn()
            controller.run(state, command, disp)
        else:
            controller.run(state, command, disp)
        if movementCtl.movement_now_number >= lib_length - 1 and movementCtl.tick >= movementCtl.now_ticks:
            print("exit the process")
            break



        # Run controller and update hardware
        #controller.run(state, command, disp)
        hardware_interface.set_actuator_postions(state.joint_angles)

if __name__ == "__main__":
    main()

