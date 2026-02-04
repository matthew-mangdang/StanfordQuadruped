import numpy as np
import time
from src.IMU import IMU
from src.Controller import Controller
from src.JoystickInterface import JoystickInterface
from src.State import BehaviorState, State
from MangDang.mini_pupper.HardwareInterface import HardwareInterface
from MangDang.mini_pupper.Config import Configuration
from pupper.Kinematics import four_legs_inverse_kinematics
from MangDang.mini_pupper.display import Display
from src.MovementScheme import MovementScheme
from src.createDanceActionListSample import MovementLib
from src.Command import Command

from MangDang.mini_pupper.ESP32Interface import ESP32Interface
from ahrs.filters import Madgwick

class LowPassFilter:
    def __init__(self, alpha=0.1):
        self.alpha = alpha
        self.value = None

    def filter(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value






def main(use_imu=False):
    """Main program
    """

    # Create config
    config = Configuration()
    hardware_interface = HardwareInterface()
    disp = Display()
    disp.show_ip()

    # Create imu handle
    if use_imu:
        imu = IMU(port="/dev/ttyACM0")
        imu.flush_buffer()
    esp32 = ESP32Interface()

    # Create controller and user input handles
    controller = Controller(
        config,
        four_legs_inverse_kinematics,
    )
    state = State()

    #Create movement group scheme instance and set a default false state
    movementCtl = MovementScheme(MovementLib)
    dance_active_state = False

    last_loop = time.time()
    # Initialize imu quat
    quat = np.array([1.0, 0.0, 0.0, 0.0])

    print("Summary of gait parameters:")
    print("overlap time: ", config.overlap_time)
    print("swing time: ", config.swing_time)
    print("z clearance: ", config.z_clearance)
    print("x shift: ", config.x_shift)

    # Wait until the activate button has been pressed
    while True:

        while True:
            now = time.time()
            if now - last_loop < config.dt:
                continue
            last_loop = time.time()



            # Read imu data. Orientation will be None if no data was available
            """
            quat_orientation = (
                imu.read_orientation() if use_imu else np.array([1, 0, 0, 0])
            )
            state.quat_orientation = quat_orientation
            """
            imu_data = esp32.imu_get_data()
            # Initialize filter (set sample rate and beta parameter)
            madgwick = Madgwick(sample_rate=100.0)
            # Example usage for IMU channels
            ax_filter, ay_filter, az_filter = LowPassFilter(0.2), LowPassFilter(0.2), LowPassFilter(0.2)
            gx_filter, gy_filter, gz_filter = LowPassFilter(0.2), LowPassFilter(0.2), LowPassFilter(0.2)

            ax = ax_filter.filter(imu_data['ax'])
            ay = ay_filter.filter(imu_data['ay'])
            az = az_filter.filter(imu_data['az'])
            gx = gx_filter.filter(imu_data['gx'])
            gy = gy_filter.filter(imu_data['gy'])
            gz = gz_filter.filter(imu_data['gz'])

            # Convert IMU data into numpy arrays
            acc  = np.array([ax,ay,az])
            gyro = np.array([gx,gy,gz])  # in rad/s
            
            # Update filter with new IMU data
            quat = madgwick.updateIMU(quat, gyro, acc)

            # Store in robot state
            state.quat_orientation = np.array([1, 0, 0, 0])
            #state.quat_orientation = quat
            print(f"quat_orientaiton: {quat}")



            # If "circle" button is clicked, switch dance_active_state between False/True.
            if command.dance_activate_event == True:
                if dance_active_state == False:
                    dance_active_state = True
                else:
                    dance_active_state = False

            # Step the controller forward by dt
            if dance_active_state == True:
            	# Caculate legsLocation, attitudes and speed using custom movement script
                movementCtl.runMovementScheme()
                command.horizontal_velocity = movementCtl.getMovemenSpeed()
                command.legslocation        = movementCtl.getMovemenLegsLocation()
                command.roll                = movementCtl.attitude_now[0]
                command.pitch               = movementCtl.attitude_now[1]
                command.yaw                 = movementCtl.attitude_now[2]
                command.yaw_rate            = movementCtl.getMovemenTurn()
                controller.run(state, command, disp)
            else:
                controller.run(state, command, disp)

            # Update the pwm widths going to the servos
            hardware_interface.set_actuator_postions(state.joint_angles)


main()
