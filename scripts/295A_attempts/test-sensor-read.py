import airsim
import numpy as np
import time

# Connect to the simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

client.takeoffAsync().join()
print("Drone is airborne")

for i in range(5):
    lidar_data = client.getLidarData(lidar_name="LidarSensor1")
    imu_data = client.getImuData(imu_name="Imu")

    if len(lidar_data.point_cloud) > 3:
        points = np.array(lidar_data.point_cloud, dtype=np.float32)
        points = points.reshape(-1, 3)
        print(f"Lidar sample #{i}: {len(points)} points")

    print(f"IMU: Linear Acceleration: {imu_data.linear_acceleration}")
    time.sleep(1)

client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)
print("Test complete.")
