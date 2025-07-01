import socket
import struct
import numpy as np
from scipy.spatial.transform import Rotation as R
import threading
import time
import math

def parse_imu_data(data):
    if len(data) != 61:
        return None
    unpacked = struct.unpack('<II3f3f3f4fB', data)
    secs = unpacked[0]
    nsecs = unpacked[1]
    acc = np.array(unpacked[2:5])        
    gyro = np.array(unpacked[5:8])      
    mag = np.array(unpacked[8:11])      
    quat = np.array(unpacked[11:15])    
    extra = unpacked[15]
    timestamp = secs + nsecs * 1e-9
    return {
        "timestamp": timestamp,
        "acc": acc,
        "gyro": gyro,
        "mag": mag,
        "quat": quat,
        "extra": extra
    }

class EKFStateEstimator:
    def __init__(self, dt=0.01):
        self.dt = dt
        # [x, y, z, vx, vy, vz, yaw]
        self.x = np.zeros(7)
        self.P = np.eye(7) * 0.5
        self.Q = np.diag([0.01]*3 + [0.1]*3 + [0.01])  
        self.H = np.zeros((4,7))
        self.H[0,0] = 1  # x
        self.H[1,1] = 1  # y
        self.H[2,2] = 1  # z
        self.H[3,6] = 1  # yaw
        self.R_base = np.diag([0.2, 0.2, 0.2, 0.05]) 

    def predict(self, acc_body, gyro_z, dt, roll, pitch):
        # [x, y, z, vx, vy, vz, yaw]
        if dt != self.dt:
            self.dt = dt
        yaw = self.x[6]

        R_world_body = R.from_euler('xyz', [roll, pitch, yaw]).as_matrix()
        acc_world = R_world_body @ acc_body

        px, py, pz, vx, vy, vz, yaw = self.x
        px_new = px + vx*dt + 0.5*acc_world[0]*dt*dt
        py_new = py + vy*dt + 0.5*acc_world[1]*dt*dt
        pz_new = pz + vz*dt + 0.5*acc_world[2]*dt*dt
        vx_new = vx + acc_world[0]*dt
        vy_new = vy + acc_world[1]*dt
        vz_new = vz + acc_world[2]*dt
        yaw_new = self._angle_normalize(yaw + gyro_z*dt)

        self.x = np.array([px_new, py_new, pz_new, vx_new, vy_new, vz_new, yaw_new])

        F = np.eye(7)
        F[0,3] = dt
        F[1,4] = dt
        F[2,5] = dt

        self.P = F @ self.P @ F.T + self.Q

    def update_with_vpr(self, vpr_xyz, vpr_yaw, sim_score):
        z = np.array([vpr_xyz[0], vpr_xyz[1], vpr_xyz[2], vpr_yaw])
        Rm = self.R_base * np.exp(-sim_score)  
        y = z - self.H @ self.x
        y[3] = self._angle_normalize(y[3])
        S = self.H @ self.P @ self.H.T + Rm
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.x[6] = self._angle_normalize(self.x[6])
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def get_position(self):
        return self.x[:3]
    def get_velocity(self):
        return self.x[3:6]
    def get_yaw(self):
        return self.x[6]
    @staticmethod
    def _angle_normalize(a):
        return (a + np.pi) % (2*np.pi) - np.pi

class IMUOdometry:
    def __init__(self, fc=1, bias_calib_samples=200, dt=0.01):
        self.prev_time = None
        self.bias_calib_samples = bias_calib_samples
        self.acc_buffer = []
        self.gyro_buffer = []
        self.mag_buffer = []
        self.acc_bias = np.zeros(3)
        self.gyro_bias = np.zeros(3)
        self.mag_bias = np.zeros(3)
        self.bias_calibrated = False
        self.is_ready = True
        self.r_align = None
        self.r_aligned = None

        self.acc_filt = np.array([0, 0, 9.80])
        self.gyro_filt = np.zeros(3)
        self.mag_filt = np.zeros(3)

        self.ekf = EKFStateEstimator(dt=dt)

        self.tau = 1 / (2 * np.pi * fc)

    def update(self, imu):
        acc = imu["acc"]
        gyro = imu["gyro"]
        mag = imu["mag"]
        timestamp = imu["timestamp"]
        quat = imu["quat"]
        dt = timestamp - self.prev_time if self.prev_time else 0.01

        if not self.bias_calibrated:
            self.acc_buffer.append(acc)
            self.gyro_buffer.append(gyro)
            self.mag_buffer.append(mag)
            if len(self.acc_buffer) >= self.bias_calib_samples:
                self.acc_bias = np.mean(self.acc_buffer, axis=0)
                self.acc_bias = self.acc_bias - np.array([0, 0, 9.80])
                self.gyro_bias = np.mean(self.gyro_buffer, axis=0)
                self.mag_bias = np.mean(self.mag_buffer, axis=0)
                self.bias_calibrated = True
            self.prev_time = timestamp
            return np.zeros(3), np.zeros(3), 0, 0, 0, np.array([0, 0, 9.80]), np.zeros(3)

        acc = acc - self.acc_bias 
        gyro = gyro - self.gyro_bias
        mag = mag - self.mag_bias

        alpha = math.exp(-dt / self.tau)
        self.acc_filt = alpha * self.acc_filt + (1 - alpha) * acc
        self.gyro_filt = alpha * self.gyro_filt + (1 - alpha) * gyro
        self.mag_filt = alpha * self.mag_filt + (1 - alpha) * mag

        if self.r_align is None:    
            self.r_align = R.from_quat([quat[0], quat[1], quat[2], quat[3]]).inv()
        self.r_aligned = self.r_align * R.from_quat([quat[0], quat[1], quat[2], quat[3]])
        roll, pitch, _ = self.r_aligned.as_euler('xyz', degrees=False)

        acc_body = self.acc_filt
        gyro_z = self.gyro_filt[2]

        self.ekf.predict(acc_body, gyro_z, dt, roll, pitch)

        position = self.ekf.get_position()
        velocity = self.ekf.get_velocity()
        yaw_fused = self.ekf.get_yaw()

        self.prev_time = timestamp
        return position, velocity, roll, pitch, yaw_fused, self.acc_filt, self.gyro_filt

    def vpr_update(self, vpr_xyz, vpr_yaw, sim_score):
        self.ekf.update_with_vpr(vpr_xyz, vpr_yaw, sim_score)

def imu_receiver(odometry, data_queue):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', 12345))
    print("Listening for IMU UDP packets...")
    while True:
        data, _ = sock.recvfrom(4096)
        imu = parse_imu_data(data)
        if imu is None:
            print("Wrong length:", len(data))
            continue
        pos, vel, r, p, y, acc_, gyro_ = odometry.update(imu)
        if not odometry.is_ready:
            continue
        data_queue.append((pos, vel, r, p, y))
        if len(data_queue) > 10:
            data_queue.pop(0)
        # —— 你可以随时模拟调用 VPR 融合接口 ——
        # 例子:
        # if vpr_available:
        #     odometry.vpr_update(vpr_xyz, vpr_yaw, sim_score)

def main():
    odometry = IMUOdometry()
    data_queue = []

    t = threading.Thread(target=imu_receiver, args=(odometry, data_queue))
    t.daemon = True
    t.start()

    try:
        while True:
            if data_queue:
                pos, vel, r, p, yaw = data_queue.pop(0)

                print(f"Position: {pos}, Velocity: {vel}, Roll: {r:.2f}, Pitch: {p:.2f}, Yaw(fused): {yaw:.2f}")
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Exiting...")

if __name__ == "__main__":
    main()
