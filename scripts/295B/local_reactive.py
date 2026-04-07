import math

import numpy as np

import airsim

# Scenario 1: Local/Reactive Planning in an Unknown Map (Bug-like algorithm).
# Prerequisites: AirSim environment running with a multirotor (Drone1) in a warehouse map.
# Ensure AirSim settings.json has a LiDAR sensor "LidarSensor1" on Drone1 (see above for example config).
# This script uses ground-truth data from AirSim for drone position (no GPS, but direct simulator state).

# Connect to AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name="Drone1")
client.armDisarm(True, vehicle_name="Drone1")

# Takeoff and set a reference flight altitude
takeoff_height = (
    -2.0
)  # fly 2 meters above ground (NED coordinate: negative value is above ground)
client.takeoffAsync(vehicle_name="Drone1").join()
client.moveToZAsync(
    takeoff_height, 1.0, vehicle_name="Drone1"
).join()  # smooth climb to desired altitude

# Goal position in world (NED) coordinates – this should be set to the desired destination.
# Example: goal at 10m North, 5m East from start position, at same altitude as takeoff_height.
start_state = client.getMultirotorState(vehicle_name="Drone1")
start_pos = start_state.kinematics_estimated.position
goal_position = airsim.Vector3r(
    start_pos.x_val + 50.0,  # 10 m North
    start_pos.y_val + 5.0,  # 5 m East
    takeoff_height,
)  # same altitude (2m above ground)


# Helper function: parse LiDAR data into numpy array of points
def get_lidar_points():
    lidar_data = client.getLidarData("LidarSensor1", vehicle_name="Drone1")
    if len(lidar_data.point_cloud) < 3:
        return np.zeros((0, 3))  # no points
    # Convert flat array into Nx3 array of [x, y, z] coordinates
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    return points


# Helper function: check for obstacle directly in front of the drone within a given distance
def is_obstacle_ahead(points, distance_threshold=3.0, fov_deg=30):
    """
    Determines if there's any obstacle within distance_threshold in the drone's forward direction.
    Assumes LiDAR points are in drone's **local NED** frame (SensorLocalFrame or VehicleInertialFrame with known orientation).
    This example uses VehicleInertialFrame, so we will rotate points to body frame based on drone's orientation.
    """
    if points.shape[0] == 0:
        return False, None  # no points, no obstacle
    # Get drone's orientation (to transform inertial to body frame if needed)
    drone_state = client.getMultirotorState(vehicle_name="Drone1")
    # Orientation as quaternion (AirSim uses (w,x,y,z) format)
    orient = drone_state.kinematics_estimated.orientation
    # Convert quaternion to yaw (rotation around z). AirSim NED: yaw 0 = facing North (X-axis).
    # Yaw calculation reference: yaw = atan2(2*(w*z + x*y), 1 - 2*(y*y + z*z)) in AirSim's coordinate convention.
    # Here we use a built-in helper if available, otherwise compute.
    # AirSim provides a utility to convert quaternion to Euler, but we'll implement yaw extraction for completeness.
    qw, qx, qy, qz = orient.w_val, orient.x_val, orient.y_val, orient.z_val
    # Yaw (in radians)
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    # Create rotation matrix for yaw (Z-axis rotation) to transform points to body frame
    cos_yaw = math.cos(-yaw)  # note: to transform inertial->body, use negative yaw
    sin_yaw = math.sin(-yaw)
    # Rotate points from inertial frame to body frame (assuming SensorLocalFrame was set to VehicleInertialFrame)
    # Body frame: x-forward, y-right, z-down
    xi = points[:, 0] - drone_state.kinematics_estimated.position.x_val
    yi = points[:, 1] - drone_state.kinematics_estimated.position.y_val
    # (No need to adjust z for horizontal obstacle detection)
    x_body = cos_yaw * xi - sin_yaw * yi
    y_body = sin_yaw * xi + cos_yaw * yi
    # Filter points roughly in a forward cone (±fov_deg) in front of the drone
    angles = np.degrees(
        np.arctan2(y_body, x_body)
    )  # angle 0 = front, positive = right, negative = left
    in_front = np.where(np.abs(angles) < fov_deg)[0]
    if in_front.size == 0:
        return False, None
    # Among points in front sector, find the nearest distance
    dist_ahead = np.sqrt(x_body[in_front] ** 2 + y_body[in_front] ** 2)
    min_dist = float(np.min(dist_ahead))
    if min_dist < distance_threshold:
        return True, min_dist
    else:
        return False, min_dist


# Helper for choosing avoidance turn direction
def choose_avoidance_turn(points):
    """
    Decide whether to turn left or right to avoid an obstacle, based on which side has more clearance.
    Returns an angle (in degrees) to add to current yaw.
    """
    if points.shape[0] == 0:
        return 0  # no points, no preference
    # Compute angles and distances in body frame (reusing part of logic above)
    drone_state = client.getMultirotorState(vehicle_name="Drone1")
    orient = drone_state.kinematics_estimated.orientation
    qw, qx, qy, qz = orient.w_val, orient.x_val, orient.y_val, orient.z_val
    yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    xi = points[:, 0] - drone_state.kinematics_estimated.position.x_val
    yi = points[:, 1] - drone_state.kinematics_estimated.position.y_val
    x_body = cos_yaw * xi - sin_yaw * yi
    y_body = sin_yaw * xi + cos_yaw * yi
    angles = np.degrees(np.arctan2(y_body, x_body))
    # Define left and right sectors (excluding the front 60 deg to avoid counting the obstacle itself)
    left_sector = np.where((angles > 60) & (angles <= 180))[0]
    right_sector = np.where((angles < -60) & (angles >= -180))[0]
    # Note: Because of angle sign convention (0 front, positive right),
    # left obstacles might appear as negative angles (to the left).
    # Adjust: treat left side as negative angles, right as positive angles.
    left_sector = np.where((angles < -60) & (angles >= -180))[0]
    right_sector = np.where((angles > 60) & (angles <= 180))[0]
    # Compute nearest obstacle distance on each side
    left_min = (
        np.min(np.sqrt(x_body[left_sector] ** 2 + y_body[left_sector] ** 2))
        if left_sector.size > 0
        else float("inf")
    )
    right_min = (
        np.min(np.sqrt(x_body[right_sector] ** 2 + y_body[right_sector] ** 2))
        if right_sector.size > 0
        else float("inf")
    )
    # Choose the side with greater clearance (larger min distance) to turn towards
    if left_min > right_min:
        return -45  # turn left 45 degrees
    else:
        return 45  # turn right 45 degrees


# Main reactive navigation loop
collision_count = 0
last_safe_pos = client.getMultirotorState(
    vehicle_name="Drone1"
).kinematics_estimated.position
max_iterations = 500  # safety break to avoid infinite loops
for step in range(max_iterations):
    # Get current state and distance to goal
    state = client.getMultirotorState(vehicle_name="Drone1")
    cur_pos = state.kinematics_estimated.position
    dx = goal_position.x_val - cur_pos.x_val
    dy = goal_position.y_val - cur_pos.y_val
    dist_to_goal = math.sqrt(dx * dx + dy * dy)
    if dist_to_goal < 1.0:  # goal tolerance < 1m
        print(f"Goal reached (distance {dist_to_goal:.2f} m). Landing...")
        break

    # Sense environment with LiDAR
    points = get_lidar_points()
    # For a mostly flat environment, filter out ground points (assume ground is near z=0 in world frame)
    if points.shape[0] > 0:
        # Remove points close to ground (z > -0.2 in NED frame, i.e., within 20cm of ground level)
        points = points[points[:, 2] < -0.2]

    # Reactive planning: decide movement based on sensor data and goal direction
    obstacle_ahead, min_front = is_obstacle_ahead(
        points, distance_threshold=3.0, fov_deg=30
    )
    if obstacle_ahead:
        # Obstacle is within 3m ahead. Decide a new direction to avoid the obstacle.
        turn_angle = choose_avoidance_turn(points)
        # Rotate the drone by the chosen angle (relative turn)
        new_yaw = math.degrees(yaw) + turn_angle if "yaw" in locals() else turn_angle
        client.rotateToYawAsync(new_yaw, vehicle_name="Drone1").join()
        print(
            f"Obstacle detected at {min_front:.1f}m ahead. Turning {turn_angle} degrees."
        )
        # After turning, we do not move forward this iteration (we will re-evaluate in next loop).
        continue
    else:
        # No close obstacle in front, proceed towards goal (or intermediate step towards goal).
        # Choose a small step towards the goal to allow reactive adjustments.
        step_distance = 2.0  # meters to move in this iteration (tunable)
        move_x = (
            cur_pos.x_val + (dx / dist_to_goal) * step_distance
            if dist_to_goal > step_distance
            else goal_position.x_val
        )
        move_y = (
            cur_pos.y_val + (dy / dist_to_goal) * step_distance
            if dist_to_goal > step_distance
            else goal_position.y_val
        )
        # Maintain constant altitude (takeoff_height)
        move_z = takeoff_height
        # Command the drone to move to the new position
        client.moveToPositionAsync(
            move_x, move_y, move_z, 2.0, vehicle_name="Drone1"
        ).join()
        print(f"Moving towards goal: new position = ({move_x:.1f}, {move_y:.1f})")

    # After movement, check for collision
    collision_info = client.simGetCollisionInfo(vehicle_name="Drone1")
    if collision_info.has_collided:
        collision_count += 1
        # Log collision details
        col_pos = collision_info.position
        print(
            f"Collision {collision_count}: hit {collision_info.object_name} at ({col_pos.x_val:.1f}, {col_pos.y_val:.1f}, {col_pos.z_val:.1f}). Retreating..."
        )
        # Retreat: move back to last safe position (or simply step backward)
        safe = last_safe_pos
        client.moveToPositionAsync(
            safe.x_val, safe.y_val, safe.z_val, 1.0, vehicle_name="Drone1"
        ).join()
        # Optionally, you could also try ascending or stopping here.
        # After retreating, continue the loop (the drone will choose a new direction next).
        continue
    else:
        # Update last safe position since we moved without collision
        last_safe_pos = cur_pos

# Land the drone if goal reached or loop ended
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, vehicle_name="Drone1")
client.enableApiControl(False, vehicle_name="Drone1")
print("Scenario 1 complete. Collisions encountered:", collision_count)
