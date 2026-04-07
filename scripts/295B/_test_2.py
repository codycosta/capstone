import heapq  # for future A* extension
import time

import numpy as np

import airsim

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)
client.takeoffAsync().join()
client.moveToPositionAsync(0, 0, -5, 5).join()

GOAL = np.array([50.0, 0.0, -5.0])  # Example goal (50m forward)
OBSTACLE_THRESHOLD = 8.0  # meters


def get_forward_clearance(points, forward_angle=30):
    # Simple reactive: min distance in ±forward_angle cone
    if len(points) == 0:
        return 100.0
    # Filter points roughly ahead (using vector math)
    drone_pos = np.array([0, 0, 0])  # relative
    vectors = points[:, :2]  # 2D
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    forward_mask = np.abs(angles) < forward_angle
    if not np.any(forward_mask):
        return 100.0
    return np.min(np.linalg.norm(vectors[forward_mask], axis=1))


print("Starting reactive avoidance loop (goal: 50m forward). Ctrl+C to land.")

try:
    while True:
        state = client.getMultirotorState()
        pos = np.array(
            [
                state.kinematics_estimated.position.x_val,
                state.kinematics_estimated.position.y_val,
                state.kinematics_estimated.position.z_val,
            ]
        )

        lidar_data = client.getLidarData(lidar_name="Lidar1")
        if len(lidar_data.point_cloud) >= 3:
            points = np.array(lidar_data.point_cloud).reshape(-1, 3)
            clearance = get_forward_clearance(points)

            if clearance < OBSTACLE_THRESHOLD:
                print(f"Obstacle ahead ({clearance:.1f}m)! Turning right...")
                client.moveByVelocityAsync(0, 5, 0, 2)  # yaw right + lateral move
                time.sleep(1)
            else:
                # Move toward goal (simple velocity command)
                direction = GOAL[:2] - pos[:2]
                direction /= np.linalg.norm(direction) + 1e-6
                client.moveByVelocityAsync(direction[0] * 5, direction[1] * 5, 0, 1)

        time.sleep(0.2)
except KeyboardInterrupt:
    client.landAsync().join()
    client.armDisarm(False)
    client.enableApiControl(False)
    print("Landed.")
