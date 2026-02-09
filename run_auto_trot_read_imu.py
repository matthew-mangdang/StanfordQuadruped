import numpy as np
import time
import math
from collections import deque
from src.Controller import Controller
from src.State import State, BehaviorState
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from MangDang.mini_pupper.display import Display
from src.Command import Command
from src.Imu_Esp32 import IMU
from src.Imu_Esp32 import IIRLowPassFilter
from src.Imu_Esp32 import first_order_IIRLowPassFilter
from src.Imu_Esp32 import PIDController

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

    # Initialize to default stance (align with joystick REST posture)
    initial_command = Command()
    state.height = initial_command.height
    state.foot_locations = (
        config.default_stance + np.array([0.0, 0.0, initial_command.height])[:, np.newaxis]
    )

    ### IMU handler
    # Set up esp32 interface
    esp32 = ESP32Interface()
    imu_handler = IMU(esp32)

    # Set up filters
    initial_pitch = 3.3  ### Later use self calibration to set this automatically
    initial_roll = -0.55
    
    sample_rate = 1/0.005
    cutoff_freq = 5.0  # Hz
    pitch_filter = IIRLowPassFilter(cutoff_freq, sample_rate, order=1) 
    roll_filter = IIRLowPassFilter(cutoff_freq, sample_rate, order=1) 

    pitch_filter_fo = first_order_IIRLowPassFilter(cutoff_freq, sample_rate)
    roll_filter_fo = first_order_IIRLowPassFilter(cutoff_freq, sample_rate)

    median_window = 5
    pitch_median_buf = deque(maxlen=median_window)
    roll_median_buf = deque(maxlen=median_window)

    alpha = 1.0 # Weight for the pitch & roll

    # Set up PID controller 
    # Parameters can be tuned to be suitable with the need
    kp = 0.6
    ki = 0.05
    kd = 0.01
    pid_pitch = PIDController(kp, ki, kd, 0.0)    ### setpoint = initial_pitch
    pid_roll = PIDController(kp, ki, kd, 0.0)      ### setpoint = initial_roll
    


    ### Movement handler
    # Flag so we only send trot toggle once
    trot_enabled = False

    # Choose forward speed (m/s). Keep within config.max_x_velocity.
    forward_speed = min(0.09, config.max_x_velocity)
    side_speed = min (0.02, config.max_y_velocity)
    #forward_speed = 0.0
    #side_speed = 0.0
    run_time = 10.0 # seconds until auto-stop for safety
    previous_time_pitch = time.time()
    previous_time_roll = time.time()
    print("Auto-trot script started.")
    print(f"Using forward speed: {forward_speed} m/s")
    print(f"Using side speed: {side_speed} m/s")
    print(f"Auto-stop time set to: {run_time} seconds")




    while True:
        now = time.time()
        if now - last_loop < config.dt:
            continue
        last_loop = now

        # Protection: Return to default position after 5 seconds
        if now - initialize_time > run_time:
            print(f"{run_time} seconds elapsed - returning robot to REST position")
            command = Command()
            if state.behavior_state == BehaviorState.TROT:
                command.trot_event = True  # Toggle back to REST
                command.horizontal_velocity = np.array([0.0, 0.0])
            controller.run(state, command, disp)
            hardware_interface.set_actuator_postions(state.joint_angles)
            time.sleep(0.5)  # Give time to settle
            break


        # Get imu data & filtering
        imu_data = imu_handler.get_raw_data()
        pitch = imu_handler.get_orientation(imu_data)[0]
        roll = imu_handler.get_orientation(imu_data)[1]
        imu_data_filtered1 = (pitch_filter.update(pitch), roll_filter.update(roll))
        imu_data_filtered2 = (pitch_filter_fo.update(pitch), roll_filter_fo.update(roll))
        pitch_median_buf.append(imu_data_filtered2[0])
        roll_median_buf.append(imu_data_filtered2[1])
        median_pitch = float(np.median(pitch_median_buf))
        median_roll = float(np.median(roll_median_buf))
        
        if (now - last_print_time) >= 0.15:
            #print(f"IMU Raw Data: {imu_data}")
            #print(f"{'Raw pitch, roll:':35} {pitch:.2f}, {roll:.2f}")
            #print(f"{'Filtered pitch, roll:':35} {imu_data_filtered1[0]:.2f}, {imu_data_filtered1[1]:.2f}")
            #print(f"{'First Order Filtered pitch, roll:':35} {imu_data_filtered2[0]:.2f}, {imu_data_filtered2[1]:.2f}")
            print(f"Pitch&Roll: Raw; Filter; FO Filter: {pitch:.2f},{roll:.2f}; {imu_data_filtered1[0]:.2f},{imu_data_filtered1[1]:.2f}; {imu_data_filtered2[0]:.2f},{imu_data_filtered2[1]:.2f}")
            #print(f"The state pitch and roll are {state.pitch:.2f} and {state.roll:.2f}")
            last_print_time = now

        # Build a fresh command each cycle
        command = Command()

        # One-time trot toggle from REST to TROT
        if (not trot_enabled) and state.behavior_state == BehaviorState.REST:
            command.trot_event = False
            trot_enabled = True
            print(f"Sending trot_event {command.trot_event} -> robot should enter trot gait.")

        # After both toggles, stay in trot and just command velocity
        # IMU orientation (optional) Currently manual set to no rotation
        quat_orientation = np.array([1.0, 0.0, 0.0, 0.0])
        state.quat_orientation = quat_orientation


        # Constant forward command, no lateral or turning motion
        command.horizontal_velocity = np.array([forward_speed, side_speed])
        command.yaw_rate = 0.0

        # Add pitch and roll correction
        corrected_pitch = median_pitch - initial_pitch
        corrected_roll = median_roll - initial_roll
        #state.pitch = math.radians(corrected_pitch)
        #state.roll = math.radians(roll)

        #if corrected_pitch > 1.0 or corrected_pitch < -1.0:
        #command.pitch = 0
        elapsed_pitch = time.time() - previous_time_pitch
        #print(f"elapsed_pitch: {elapsed_pitch:.4f} seconds")
        #Calculate driven angles by PID controller
        error_pitch = pid_pitch.compute(corrected_pitch , 0.015)
        previous_time_pitch = time.time()
        #command.pitch = -math.radians(error_pitch* alpha)
        #print(f"Error pitch: {error_pitch:.2f}, Commanded pitch: {command.pitch:.2f}")
        
        #if corrected_roll > 1.0 or corrected_roll < -1.0:
        #command.roll = 0
        #command.roll = -math.radians(corrected_roll)
        elapsed_roll = time.time() - previous_time_roll
        #print(f"elapsed_roll: {elapsed_roll:.4f} seconds")
        #Calculate driven angles by PID controller
        error_roll = pid_roll.compute(corrected_roll , 0.015)
        previous_time_roll = time.time()
        #command.roll = math.radians(error_roll* alpha)
        #print(f"Error roll: {error_roll:.2f}, Commanded roll: {command.roll:.2f}")
        #print(f"Error pitch: {error_pitch:7.2f}, Commanded pitch: {command.pitch:7.2f}    Error roll: {error_roll:7.2f}, Commanded roll: {command.roll:7.2f}")   

        #print(f"Corrected Pitch: {corrected_pitch:.2f}, Corrected Roll: {corrected_roll:.2f}")
    

        # Run controller and update hardware
        #print(f"The command at run step: Pitch: {math.degrees(command.pitch):.2f}, Roll: {math.degrees(command.roll):.2f}")
        controller.run(state, command, disp)
        hardware_interface.set_actuator_postions(state.joint_angles)


if __name__ == "__main__":
    main()

