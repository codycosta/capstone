import math

import numpy as np
from pathfinding.core.diagonal_movement import DiagonalMovement

# Using the 'pathfinding' library for A* pathfinding (install via: pip install pathfinding)
from pathfinding.core.grid import Grid
from pathfinding.finder.a_star import AStarFinder

import airsim

# Scenario 2: Global Planning with A* and Dynamic Replanning
# Prerequisites: AirSim environment running with Drone1 and LiDAR sensor configured (as per settings.json above).
# The drone uses a 2D occupancy grid (top-down map) for path planning. Ground truth position is used for localization.

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name="Drone1")
client.armDisarm(True, vehicle_name="Drone1")

# Takeoff and hover at target altitude
target_altitude = -3.0  # 3 meters above ground
client.takeoffAsync(vehicle_name="Drone1").join()
client.moveToZAsync(target_altitude, 1.0, vehicle_name="Drone1").join()

# Define goal position (in world NED coordinates)
# Example: goal offset 15m North, 10m East from start.
start_state = client.getMultirotorState(vehicle_name="Drone1")
start_pos = start_state.kinematics_estimated.position
goal_position = airsim.Vector3r(
    start_pos.x_val + 50.0,  # 15 m North
    start_pos.y_val + 10.0,  # 10 m East
    target_altitude,
)  # same altitude as flight

# Occupancy grid setup
GRID_SIZE = 200  # grid dimension (200x200 cells covering a large area)
RESOLUTION = 1.0  # 1 cell = 1 meter
# We'll center the grid around the start position
origin_x = start_pos.x_val
origin_y = start_pos.y_val
origin_index = GRID_SIZE // 2  # index in grid that corresponds to the start position
# Initialize grid with all free (0 = free, 1 = obstacle)
occupancy = [[0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]


# Helper: convert world (NED) position to grid indices
def world_to_grid(x, y):
    # Compute offsets relative to origin
    dx = x - origin_x
    dy = y - origin_y
    # In NED, x is North (grid row axis), y is East (grid col axis).
    # We map North -> row (with north increases -> smaller row index if we treat 0,0 at top-left)
    # To avoid negative indices, we offset by origin_index.
    col = int(round(dy / RESOLUTION)) + origin_index  # east offset -> column index
    row = (
        origin_index - int(round(dx / RESOLUTION))
    )  # north offset -> row index (subtract because north increase means decreasing row index)
    return row, col


# Helper: check if a grid cell index is within bounds
def valid_index(row, col):
    return 0 <= row < GRID_SIZE and 0 <= col < GRID_SIZE


# Helper: update occupancy grid with new LiDAR data
def update_occupancy_from_lidar():
    lidar_data = client.getLidarData("LidarSensor1", vehicle_name="Drone1")
    if len(lidar_data.point_cloud) < 3:
        return False  # no points, nothing to update
    points = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1, 3)
    updated = False
    # Filter out ground hits (assume ground at z ~ 0; drone in NED above ground has negative z)
    ground_threshold = -0.3  # points with z > -0.3 are near ground
    points = points[points[:, 2] < ground_threshold]
    for px, py, pz in points:
        # Only consider obstacles roughly at the drone's altitude (within a certain vertical band, if needed)
        # In a mostly flat environment, any object above ground is a vertical obstacle
        # Mark the cell at (px, py) as occupied
        r, c = world_to_grid(px, py)
        if valid_index(r, c) and occupancy[r][c] == 0:
            occupancy[r][c] = 1  # mark obstacle
            updated = True
    return updated


# Helper: plan a path using A* on the occupancy grid
# Returns a list of waypoints in world coordinates (each as airsim.Vector3r)
def plan_path(start_pos, goal_pos, allow_diagonal=False):
    # Convert start and goal positions to grid indices
    start_row, start_col = world_to_grid(start_pos.x_val, start_pos.y_val)
    goal_row, goal_col = world_to_grid(goal_pos.x_val, goal_pos.y_val)
    # If any index is out of bounds or if goal is currently marked occupied, return empty path
    if not valid_index(start_row, start_col) or not valid_index(goal_row, goal_col):
        print("Error: Start or goal is outside the grid bounds.")
        return []
    if occupancy[goal_row][goal_col] == 1:
        print("Goal is in an occupied cell - no path!")
        return []
    # Create pathfinding grid and run A*
    grid = Grid(matrix=occupancy)
    # grid uses (x=col, y=row) internally, so get nodes accordingly
    start_node = grid.node(start_col, start_row)
    goal_node = grid.node(goal_col, goal_row)
    finder = AStarFinder(
        diagonal_movement=DiagonalMovement.never
        if not allow_diagonal
        else DiagonalMovement.always
    )
    path_nodes, _ = finder.find_path(start_node, goal_node, grid)
    if len(path_nodes) == 0:
        return []  # no path found
    # Convert path nodes back to world coordinates (each node is (x_col, y_row) in grid indexing)
    waypoints = []
    for col, row in path_nodes:
        # Convert grid indices back to world
        north_offset = (origin_index - row) * RESOLUTION
        east_offset = (col - origin_index) * RESOLUTION
        wx = origin_x + north_offset  # world X (north)
        wy = origin_y + east_offset  # world Y (east)
        wz = target_altitude  # maintain constant altitude
        waypoints.append(airsim.Vector3r(wx, wy, wz))
    return waypoints


# Main loop: dynamic planning and navigation
current_path = []  # list of waypoints (airsim.Vector3r) for the current path
path_index = 0  # index of next waypoint in current_path
last_safe_pos = client.getMultirotorState(
    vehicle_name="Drone1"
).kinematics_estimated.position
collision_count = 0
MAX_STEPS = 1000

for step in range(MAX_STEPS):
    # Update map with latest sensor data
    new_obs = update_occupancy_from_lidar()
    # Get current position and distance to goal
    state = client.getMultirotorState(vehicle_name="Drone1")
    current_pos = state.kinematics_estimated.position
    dx = goal_position.x_val - current_pos.x_val
    dy = goal_position.y_val - current_pos.y_val
    dist_to_goal = math.sqrt(dx * dx + dy * dy)
    if dist_to_goal < 1.0:
        print(f"Goal reached (distance {dist_to_goal:.2f} m). Landing...")
        break

    # Check if we need to (re)plan:
    if not current_path or path_index >= len(current_path) or new_obs:
        # Plan a new path from current position to goal
        current_path = plan_path(current_pos, goal_position, allow_diagonal=False)
        path_index = 0
        if not current_path:
            print("No path found to goal - stopping navigation.")
            break
        # Skip directly to the next iteration (we'll start following the new path in the next loop iteration)
        # This ensures the sensor data used for obstacle detection corresponds to before moving on the new path.
        # (Alternatively, you could proceed to move to first waypoint immediately.)
        continue

    # Follow the current path: move towards the next waypoint
    waypoint = current_path[path_index]
    client.moveToPositionAsync(
        waypoint.x_val, waypoint.y_val, waypoint.z_val, 2.0, vehicle_name="Drone1"
    ).join()
    print(
        f"Moving to waypoint {path_index + 1}/{len(current_path)}: ({waypoint.x_val:.1f}, {waypoint.y_val:.1f})"
    )
    path_index += 1

    # Collision check after movement
    collision_info = client.simGetCollisionInfo(vehicle_name="Drone1")
    if collision_info.has_collided:
        collision_count += 1
        impact = collision_info.position
        print(
            f"Collision {collision_count}: hit {collision_info.object_name} at ({impact.x_val:.1f}, {impact.y_val:.1f}). Retreating and replanning..."
        )
        # Retreat to last safe position
        safe = last_safe_pos
        client.moveToPositionAsync(
            safe.x_val, safe.y_val, safe.z_val, 1.0, vehicle_name="Drone1"
        ).join()
        # Mark the current path as invalid and trigger replan in next loop
        current_path = []
        continue
    else:
        # Update last safe position if move was successful
        last_safe_pos = current_pos

# Land when done
client.landAsync(vehicle_name="Drone1").join()
client.armDisarm(False, vehicle_name="Drone1")
client.enableApiControl(False, vehicle_name="Drone1")
print("Scenario 2 complete. Collisions encountered:", collision_count)
