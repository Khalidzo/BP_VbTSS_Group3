import numpy as np

class KalmanSpeedFilter:
    """Kalman filter for smoothing vehicle speed estimation"""
    
    def __init__(self, dt=1.0/30.0):
        self.dt = dt  # Time step (assuming 30 FPS)
        
        # State vector: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(4)
        
        # State transition matrix (constant velocity model)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Observation matrix (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        q = 0.1  # Process noise
        self.Q = np.array([
            [dt**4/4, 0, dt**3/2, 0],
            [0, dt**4/4, 0, dt**3/2],
            [dt**3/2, 0, dt**2, 0],
            [0, dt**3/2, 0, dt**2]
        ]) * q
        
        # Measurement noise covariance
        r = 1.0  # Measurement noise
        self.R = np.array([
            [r, 0],
            [0, r]
        ])
        
        # Error covariance matrix
        self.P = np.eye(4) * 1000  # High initial uncertainty
        
        # Initialization flag
        self.initialized = False
        
    def predict(self):
        """Predict step of Kalman filter"""
        if not self.initialized:
            return
            
        # Predict state
        self.state = self.F @ self.state
        
        # Predict error covariance
        self.P = self.F @ self.P @ self.F.T + self.Q
    
    def update(self, measurement):
        """Update step of Kalman filter"""
        if not self.initialized:
            # Initialize with first measurement
            self.state[0] = measurement[0]  # x position
            self.state[1] = measurement[1]  # y position
            self.state[2] = 0  # initial x velocity
            self.state[3] = 0  # initial y velocity
            self.initialized = True
            return
        
        # Calculate innovation
        z = np.array(measurement)
        y = z - self.H @ self.state
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # Update state
        self.state = self.state + K @ y
        
        # Update error covariance
        I = np.eye(4)
        self.P = (I - K @ self.H) @ self.P
    
    def get_velocity(self):
        """Get current velocity estimate (m/s)"""
        if not self.initialized:
            return 0.0
        vx, vy = self.state[2], self.state[3]
        return np.sqrt(vx**2 + vy**2)
    
    def get_speed_kmh(self):
        """Get current speed estimate in km/h"""
        return self.get_velocity() * 3.6