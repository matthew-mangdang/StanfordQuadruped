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


def reset_movement_counters(lib):
    """Reset execution counters in all movements so they can replay."""
    for mov in lib:
        for leg in mov.LegsMovements:
            leg.SequenceExecuteCounter = leg.ExecuteCounterMax
        mov.SpeedMovements.SequenceExecuteCounter = mov.SpeedMovements.ExecuteCounterMax
        mov.AttitudeMovements.SequenceExecuteCounter = mov.AttitudeMovements.ExecuteCounterMax
        mov.TurnMovements.SequenceExecuteCounter = mov.TurnMovements.ExecuteCounterMax


def main(use_imu=False):
    """Use accelerometer-based hit detection to trigger a crawl dance movement."""

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
    
    # Create movement group scheme instance for dance sequence
    movementCtl = MovementScheme(MovementLib)
    lib_length = len(MovementLib)
    dance_active = False

    # Initialize to default stance (align with joystick REST posture)
    initial_command = Command()
    state.height, state.foot_locations = initialize_stance(initial_command, config)

    ### IMU handler
    # Set up esp32 interface
    esp32 = ESP32Interface()
    imu_handler = IMU(esp32)
    a_offset = imu_handler.get_a_offset(50, [-0.03, 0.047, -1.053]) #[0.03, 0.04, -1.05]
    lp_filter = butter_filter_live(cutoff_freq=10.0, dt=config.dt, order=1)
    hit_flag = 0.0
    hit_tol = 0.31
    hit_count_tol = 2.0
    last_hit_time = last_loop

    ### Movement handler - user input  
    trot_forward_speed = min(0.09, config.max_x_velocity)
    side_speed = min (0.00, config.max_y_velocity)
    run_time = 600.0  # seconds until auto-stop for safety

    print("Script started and start trot")
    print(f"Trot forward speed: {trot_forward_speed} m/s")
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
        
        # Signal processing
        a_raw = imu_handler.get_accelerometer()
        a_calibrated = a_raw - a_offset
        a_filtered = lp_filter.apply_filter(a_calibrated[0])
        #print(f"Filtered acceleration: {a_filtered}")
        # Check for hit detection based on filtered accleration
        if a_filtered > hit_tol:  # Threshold for detecting a hit; 0.25 for Cabled Power; 0.15 for Battery
            hit_flag += 1.0
            last_hit_time = now
            print(f"Detected hit with filtered acceleration: {a_filtered}")
        
        if now - last_hit_time > 3.0 and hit_flag > 0.0:  # Reset hit flag after some time without a hit
            hit_flag = 0.0
            print(f"Reset hit flag after 3 seconds without a hit.")
        

        if not dance_active:
            if state.behavior_state != BehaviorState.TROT:
                command.trot_event = True
                print("Currently in trot gait")
            command.horizontal_velocity = np.array([trot_forward_speed, side_speed])

        if hit_flag > hit_count_tol and not dance_active:
            print("Hit detected: starting dance movement.")
            dance_active = True
            hit_flag = 0.0
            reset_movement_counters(MovementLib)
            movementCtl.resetMovementNumber()
            movementCtl.tick = 0  # Reset tick counter to start from beginning
            movementCtl.transition = False  # Ensure we don't skip forward


        if dance_active:
            if state.behavior_state == BehaviorState.TROT:
                command.trot_event = True  # toggle to REST for dance
            command.horizontal_velocity = np.array([0.0, 0.0])

            command.pseudo_dance_event = True
            movementCtl.runMovementScheme()
            command.legslocation = movementCtl.getMovemenLegsLocation()
            command.horizontal_velocity = movementCtl.getMovemenSpeed()
            command.roll = movementCtl.attitude_now[0]
            command.pitch = movementCtl.attitude_now[1]
            command.yaw = movementCtl.attitude_now[2]
            command.yaw_rate = movementCtl.getMovemenTurn()

            if movementCtl.movement_now_number >= lib_length - 1 and movementCtl.tick >= movementCtl.now_ticks:
                print("Dance finished, returning to trot.")

                dance_active = False
                if state.behavior_state != BehaviorState.TROT:
                    command.trot_event = True
                command.horizontal_velocity = np.array([trot_forward_speed, side_speed])


        # IMU orientation (optional) Currently manual set to no rotation
        quat_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        state.quat_orientation = quat_orientation

        # Run controller and update hardware
        controller.run(state, command, disp)
        hardware_interface.set_actuator_postions(state.joint_angles)

if __name__ == "__main__":
    main()

