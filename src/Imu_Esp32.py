from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import math
import numpy as np
from threading import Thread, Lock
import queue
import time

class IMU:
    """ 
    IMU handler to handler data processing regarding the IMU connected to ESP32
    """
    def __init__(self, esp32):
    
        self.esp32 = esp32

    def get_raw_data(self):
        self.imu_raw = self.esp32.imu_get_data()
        return self.imu_raw

    def get_orientation(self, data):
        """
        Gets pitch and roll in degrees
        """
        ax = data['ax']
        ay = data['ay']
        az = data['az']
        gx = data['gx']
        gy = data['gy']
        gz = data['gz']  # Assume in  deg/s 

        roll_deg  = math.degrees(math.atan(ax / math.sqrt(ay * ay + az * az)))
        pitch_deg = math.degrees(math.atan(ay / math.sqrt(ax * ax + az * az)))

        return pitch_deg, roll_deg


class IIRLowPassFilter:
    """IIR Low Pass Filter implementation with configurable order."""
    
    def __init__(self, cutoff_freq, sample_rate, order):
        """
        Initialize IIR low pass filter.
        
        Args:
            cutoff_freq: Cutoff frequency in Hz
            sample_rate: Sampling frequency in Hz
            order: Filter order (1 or 2)
        """
        self.cutoff = cutoff_freq
        self.fs = sample_rate
        self.order = order
        
        # Calculate filter coefficients
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        
        if order == 1:
            # First order Butterworth coefficients
            self.b = [normal_cutoff, normal_cutoff]
            self.a = [1, normal_cutoff - 1]
        else:
            # Second order Butterworth coefficients
            sqrt2 = np.sqrt(2)
            self.b = [normal_cutoff**2, 2*normal_cutoff**2, normal_cutoff**2]
            self.a = [1, 2*(normal_cutoff**2 - 1), 1 - sqrt2*normal_cutoff + normal_cutoff**2]
            
        self.x_hist = [0] * (order + 1)
        self.y_hist = [0] * (order + 1)
    
    def update(self, new_value):
        """
        Update the filter with a new measurement.
        
        Returns:
            The filtered value
        """
        # Shift history
        self.x_hist.pop()
        self.x_hist.insert(0, new_value)
        self.y_hist.pop()
        
        # Compute new output
        y = 0
        for i in range(len(self.b)):
            y += self.b[i] * self.x_hist[i]
        for i in range(1, len(self.a)):
            y -= self.a[i] * self.y_hist[i-1]
        
        y /= self.a[0]
        self.y_hist.insert(0, y)
        
        return y


class first_order_IIRLowPassFilter:
    """
    Simple first-order IIR low pass filter.
    """
    def __init__(self, cutoff_freq, sample_rate):
        # RC filter approximation
        dt = 1.0 / sample_rate
        rc = 1.0 / (2 * np.pi * cutoff_freq)
        self.alpha = dt / (rc + dt)
        self.y = None

    def update(self, x):
        if self.y is None:
            self.y = x
        else:
            self.y = self.y + self.alpha * (x - self.y)
        return self.y


class PIDController:
    """Standard PID controller implementation."""
    
    def __init__(self, Kp, Ki, Kd, setpoint):
        """
        Initialize PID controller.
        
        Args:
            Kp: Proportional gain
            Ki: Integral gain
            Kd: Derivative gain
            setpoint: Target value
        """
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.previous_error = 0
        self.integral = 0

    def compute(self, process_variable, dt):
        """
        Compute PID output.
        
        Args:
            process_variable: Current measured value
            dt: Time step since last computation
            
        Returns:
            PID control output
        """
        # Calculate error
        error = self.setpoint - process_variable
        
        # Proportional term
        P_out = self.Kp * error
        
        # Integral term
        self.integral += error * dt
        I_out = self.Ki * self.integral
        
        # Derivative term
        derivative = (error - self.previous_error) / dt
        D_out = self.Kd * derivative
        
        # Compute total output
        output = P_out + I_out + D_out
        
        # Update previous error
        self.previous_error = error
        
        return output



class KalmanFilter:
    def __init__(self, dt=None):
        # State [pitch, roll]
        self.x = np.zeros((2,1))
        self.P = np.eye(2) * 0.1
        self.last_time = time.time()

        # Transition (identity)
        self.F = np.eye(2)
        self.dt = dt
        self.B = None

        # Process noise (gyro drift)
        self.Q = np.eye(2) * 1e-3

        # Measurement model (accelerometer tilt)
        self.H = np.eye(2)
        self.R = np.eye(2) * 0.05  # measurement noise

        # Threaded processing support
        self._lock = Lock()
        self._measurement_queue = queue.Queue()
        self._running = False
        self._filter_thread = None

    def set_dt(self, dt):
        self.dt = dt
        self.B = np.eye(2) * dt

    def predict(self, gyro, current_time=None):

        if current_time is None:
            current_time = time.time()
        self.dt = current_time - self.last_time
        self.last_time = current_time
        self.B = np.eye(2) * self.dt

        u = np.array(gyro).reshape(2,1)  # [gyro_pitch_rate, gyro_roll_rate]
        self.x = self.F @ self.x + self.B @ u
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, accel):
        """
        Updates the State, Prediction and Measurement models of the Kalman Filter based on the new accelerometer measurement.
        """
        # Compute pitch/roll from accelerometer
        pitch_acc = np.arctan2(accel[0], np.sqrt(accel[1]**2 + accel[2]**2))
        roll_acc = np.arctan2(accel[1], np.sqrt(accel[0]**2 + accel[2]**2))
        #roll_acc  = np.arctan2(accel[1], accel[2])
        #print(f"Debug: Pitch & Roll : {math.degrees(pitch_acc):.2f}, {math.degrees(roll_acc):.2f} degrees")
        z = np.array([pitch_acc, roll_acc]).reshape(2,1)

        # Kalman update
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(2) - K @ self.H) @ self.P

    def get_state(self):
        """
        Returns:
            Current estimated pitch and roll (rad) as a flat array [pitch, roll]
        """
        with self._lock:
            return self.x.flatten().copy()

    def add_measurement(self, gyro, accel, dt=None):
        self._measurement_queue.put({"gyro": gyro, "accel": accel, "dt": dt})

    def _filter_loop(self):
        while self._running:
            try:
                measurement = self._measurement_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if measurement["dt"] is not None:
                self.set_dt(measurement["dt"])

            with self._lock:
                self.predict(measurement["gyro"])
                self.update(measurement["accel"])

    def start(self):
        if not self._running:
            self._running = True
            self._filter_thread = Thread(target=self._filter_loop)
            self._filter_thread.start()

    def stop(self):
        self._running = False
        if self._filter_thread is not None:
            self._filter_thread.join()

