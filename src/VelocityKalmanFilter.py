import numpy as np


class VelocityKalmanFilter1D_WithBias:
    """
    1D Kalman Filter: velocity + bias learning via ZUPT.
    
    ⚠️ IMPORTANT: Bias learning ONLY works with ZUPT (Zero-Velocity Updates)
    
    State: [v, b]^T
      v: velocity (m/s)
      b: accel bias (m/s²)
    
    Process model:
      v_{k+1} = v_k + (a_raw_k - b_k) * dt
      b_{k+1} = b_k
    
    Measurement model (ZUPT only):
      z = 0  (when foot contact detected, velocity = 0)
    
    How it works:
      1. Continuously integrate: v += (a_raw - b) * dt
      2. When motion stops (stance), call update_zupt(0)
      3. If v_pred ≠ 0, the difference reveals bias error
      4. Filter adjusts both v (→ 0) and b (learns true bias)
    """
    
    def __init__(self, dt, process_noise_v=0.01, process_noise_b=0.0001, measurement_noise=0.01):
        """
        Args:
            dt: Time step (seconds)
            process_noise_v: Process noise for velocity (0.001-0.1)
                - How much accel varies unexpectedly
            process_noise_b: Process noise for bias (0-0.001)
                - Assume bias is nearly constant (lower = more constant)
            measurement_noise: Measurement noise for ZUPT (small, e.g., 0.001-0.1)
                - How confident we are that v=0 during stance
        """
        self.dt = dt
        self.q_v = process_noise_v
        self.q_b = process_noise_b
        self.r_zupt = measurement_noise
        
        # State: [v, b]
        self.v = 0.0      # velocity estimate
        self.b = 0.0      # bias estimate
        
        # Covariance (2x2)
        self.P = np.array([[1.0, 0.0], [0.0, 1.0]])
    
    def predict(self, a_raw):
        """
        Predict step: integrate acceleration (no measurement).
        
        Args:
            a_raw: Raw acceleration measurement (m/s²)
        """
        # Corrected acceleration
        a_true_est = a_raw - self.b
        
        # Velocity prediction
        v_pred = self.v + a_true_est * self.dt
        b_pred = self.b
        
        # State transition matrix
        F = np.array([[1.0, -self.dt],
                      [0.0,   1.0]])
        
        # Process noise
        Q = np.array([[self.q_v,     0.0],
                      [0.0,    self.q_b]])
        
        # Covariance prediction
        P_pred = F @ self.P @ F.T + Q
        
        # Store (don't update state yet)
        self.v_pred = v_pred
        self.b_pred = b_pred
        self.P_pred = P_pred
    
    def update_zupt(self, v_measured=0.0):
        """
        Update step: apply ZUPT (Zero-Velocity Update).
        
        Use this ONLY when:
          - Foot is in contact with ground (swing → stance transition)
          - Or when motion definitely stopped
        
        Args:
            v_measured: Measured velocity (typically 0.0 for stationary)
        
        Returns:
            Updated velocity estimate
        """
        # Innovation: difference between measurement and prediction
        innovation = v_measured - self.v_pred
        
        # Measurement matrix: H = [1, 0]
        # (we measure velocity, which depends on v but not directly on b)
        # But through integration: v_meas = v_pred + bias_error
        # So: measurement sensitivity is [1, 0]
        
        # Innovation covariance: S = H @ P_pred @ H^T + R
        #                         = P_pred[0,0] + r
        S = self.P_pred[0, 0] + self.r_zupt
        
        # Kalman gain: K = P_pred @ H^T / S
        #              = [P_pred[0,0], P_pred[1,0]]^T / S
        K = self.P_pred[:, 0] / S
        
        # State update
        self.v = self.v_pred + K[0] * innovation
        self.b = self.b_pred + K[1] * innovation  # ← Learns bias here!
        
        # Covariance update: P = (I - K @ H) @ P_pred
        # where H = [1, 0]
        H = np.array([[1.0, 0.0]])
        self.P = (np.eye(2) - np.outer(K, H)) @ self.P_pred
        
        return self.v
    
    def step(self, a_raw):
        """Predict only (no measurement update)."""
        self.predict(a_raw)
        return self.v_pred
    
    def get_velocity(self):
        return self.v
    
    def get_bias(self):
        return self.b
    
    def reset_velocity(self, v=0.0):
        """Manual velocity reset (alternative to ZUPT)."""
        self.v = v
        self.P[0, 0] = 0.001


class VelocityKalmanFilter1D_Simple:
    """
    Simple 1D velocity filter: just integrates acceleration.
    
    NO bias learning (no external measurements).
    For short-term use only or when bias is already calibrated out.
    """
    
    def __init__(self, dt, process_noise=0.01, measurement_noise=0.1):
        """
        Args:
            dt: Time step
            process_noise: How much to trust the model
            measurement_noise: Accel noise (not used in simple version)
        """
        self.dt = dt
        self.q = process_noise
        self.v = 0.0
        self.p = 1.0
    
    def step(self, a):
        """Direct integration: v += a * dt."""
        self.v = self.v + a * self.dt
        return self.v
    
    def get_velocity(self):
        return self.v
    
    def reset_velocity(self, v=0.0):
        self.v = v
        self.p = 0.01
