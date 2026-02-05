from MangDang.mini_pupper.ESP32Interface import ESP32Interface
import math

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