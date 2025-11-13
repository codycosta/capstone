"""
airsim_lidar_astar_nav.py

End-to-end LiDAR -> 2D occupancy -> A* -> waypoint follower prototype for AirSim.

How to run:
1. Launch Blocks.exe (AirSim).
2. In a terminal/VSCode run: python airsim_lidar_astar_nav.py

Tweak PARAMETERS below to your environment (grid size, resolution, safety radii).
"""

import airsim
import numpy as np
import math
import time
from heapq import heappush, heappop

# ----------------- PARAMETERS -----------------
LIDAR_NAME = "LidarSensor1"    # name in settings.json
GRID_RES = 0.5                 # meters per cell
GRID_RADIUS_M = 25.0           # half-width of grid (meters); grid covers [-R,R] in x,y
GRID_SIZE = int((GRID_RADIUS_M * 2) / GRID_RES)  # cells per side
DRONE_ALT = -3.0               # NED altitude to keep (negative down)
DRONE_RADIUS = 0.6             # meters (used for obstacle inflation)
OCCUPIED_THRESHOLD = 1         # number of hits to mark occupied
REPLAN_INTERVAL = 1.0          # seconds between forced replans
WAYPOINT_TOL = 0.6             # meters; when to switch to next waypoint
MAX_VEL = 3.0                  # m/s for moveToPosition
LOCAL_SAFETY_DIST = 1.0        # if obstacle closer than this, stop and replan
A_STAR_DIAGONAL_COST = 1.4     # heuristic weight for diagonal steps
TIMEOUT = 300                  # seconds to give up
DEBUG_PLOT = False             # set True to show grid+path visualization (matplotlib)
# ------------------------------------------------

# ----------------- UTILITIES -----------------
def quat_to_rot_mat(q):
    # q is AirSim quaternion object with fields w_val,x_val,y_val,z_val
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    R = np.array([
        [1 - 2*(y*y+z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x+z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),   1 - 2*(x*x+y*y)]
    ])
    return R

def transform_lidar_points(lidar_data, pose):
    """
    Transform lidar points into world frame.
    lidar_data.point_cloud: flat list [x1,y1,z1,x2,y2,z2,...] in sensor-local coordinates (usually)
    pose: multirotor kinematics_estimated.pose (position + orientation)
    Returns Nx3 numpy array in world (NED) coordinates.
    """
    if not hasattr(lidar_data, 'point_cloud') or len(lidar_data.point_cloud) < 3:
        return np.zeros((0,3), dtype=np.float32)

    pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1,3)
    # AirSim Lidar coordinates are usually in vehicle/world frame depending on sensor settings.
    # To be robust, transform from sensor-local to world using pose:
    pos = pose.position
    ori = pose.orientation
    R = quat_to_rot_mat(ori)  # rotation from local to world
    # pts assumed to be in local: apply rotation + translation
    transformed = (R @ pts.T).T + np.array([pos.x_val, pos.y_val, pos.z_val])
    return transformed

def world_xy_to_grid_ix(origin_xy, x, y):
    """
    origin_xy: (x0, y0) center of grid in world coords
    convert world (x,y) to grid indices (i,j)
    """
    dx = x - origin_xy[0]
    dy = y - origin_xy[1]
    ix = int(round((dx + GRID_RADIUS_M) / GRID_RES))
    iy = int(round((dy + GRID_RADIUS_M) / GRID_RES))
    return ix, iy

def grid_ix_to_world_xy(origin_xy, ix, iy):
    x = origin_xy[0] + (ix * GRID_RES) - GRID_RADIUS_M
    y = origin_xy[1] + (iy * GRID_RES) - GRID_RADIUS_M
    return x, y

# ----------------- A* IMPLEMENTATION -----------------
def astar_grid(grid, start, goal):
    """
    grid: 2D numpy array, 0 free, 1 occupied
    start, goal: tuples (ix,iy)
    returns: list of grid indices from start to goal (inclusive) or None
    """
    rows, cols = grid.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < cols): return None
    if not (0 <= goal[0] < rows and 0 <= goal[1] < cols): return None
    if grid[goal[0], goal[1]] != 0:
        return None

    def heuristic(a,b):
        # Euclidean
        return math.hypot(a[0]-b[0], a[1]-b[1])

    open_set = []
    heappush(open_set, (heuristic(start,goal), 0.0, start, None))
    came_from = {}
    gscore = {start: 0.0}
    closed = set()

    while open_set:
        f, g, current, parent = heappop(open_set)
        if current in closed:
            continue
        came_from[current] = parent
        if current == goal:
            # reconstruct
            path = [current]
            cur = current
            while came_from[cur] is not None:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(current)
        # neighbors (8-connected)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx == 0 and dy == 0: continue
                nei = (current[0] + dx, current[1] + dy)
                if not (0 <= nei[0] < rows and 0 <= nei[1] < cols): continue
                if grid[nei[0], nei[1]] != 0: continue
                tentative_g = g + (A_STAR_DIAGONAL_COST if dx!=0 and dy!=0 else 1.0)
                if nei not in gscore or tentative_g < gscore[nei]:
                    gscore[nei] = tentative_g
                    heappush(open_set, (tentative_g + heuristic(nei,goal), tentative_g, nei, current))
    return None

# ----------------- HELPER POSE CLASS -----------------
class Pose:
    """Simple replacement for old kinematics_estimated.pose"""
    def __init__(self, position, orientation):
        self.position = position
        self.orientation = orientation

# ----------------- NAVIGATION CLASS -----------------
class LidarAstarNavigator:
    def __init__(self, client):
        self.client = client
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.origin_xy = None
        self.last_plan = None
        self.last_plan_time = 0

    def build_grid_from_lidar(self):
        state = self.client.getMultirotorState()
        pose = Pose(state.kinematics_estimated.position, state.kinematics_estimated.orientation)
        self.origin_xy = (pose.position.x_val, pose.position.y_val)

        lidar = self.client.getLidarData(lidar_name=LIDAR_NAME)
        pts_world = transform_lidar_points(lidar, pose)

        if pts_world.shape[0] == 0:
            self.grid.fill(0)
            return

        z_min, z_max = DRONE_ALT - 2.0, DRONE_ALT + 2.0
        mask = (pts_world[:,2] >= z_min) & (pts_world[:,2] <= z_max)
        pts2 = pts_world[mask][:,:2]

        self.grid.fill(0)
        for x, y in pts2:
            ix, iy = world_xy_to_grid_ix(self.origin_xy, x, y)
            if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
                self.grid[ix, iy] += 1

        occ = (self.grid >= OCCUPIED_THRESHOLD).astype(np.uint8)
        inflate_cells = int(math.ceil(DRONE_RADIUS / GRID_RES))
        if inflate_cells > 0:
            from scipy.ndimage import binary_dilation
            occ = binary_dilation(occ, iterations=inflate_cells).astype(np.uint8)

        self.grid = occ

    def plan_path(self, goal_xy):
        if self.origin_xy is None:
            return None

        start = (GRID_SIZE//2, GRID_SIZE//2)
        goal_ix, goal_iy = world_xy_to_grid_ix(self.origin_xy, goal_xy[0], goal_xy[1])
        goal_ix = max(0, min(GRID_SIZE-1, goal_ix))
        goal_iy = max(0, min(GRID_SIZE-1, goal_iy))

        if self.grid[goal_ix, goal_iy] != 0:
            print("[plan_path] goal cell occupied - cannot plan")
            return None

        path = astar_grid(self.grid, start, (goal_ix, goal_iy))
        if path is None:
            print("[plan_path] A* failed to find path")
            return None

        world_path = [grid_ix_to_world_xy(self.origin_xy, p[0], p[1]) for p in path]
        world_path = [(float(x), float(y)) for (x, y) in world_path]

        self.last_plan = world_path
        self.last_plan_time = time.time()
        return world_path

    def nearest_obstacle_distance(self):
        occ = np.argwhere(self.grid == 1)
        if occ.size == 0:
            return float('inf')
        occ_xy = np.array([grid_ix_to_world_xy(self.origin_xy, int(i), int(j)) for i,j in occ])
        pos = self.client.getMultirotorState().kinematics_estimated.position
        px, py = pos.x_val, pos.y_val
        dists = np.hypot(occ_xy[:,0]-px, occ_xy[:,1]-py)
        return float(np.min(dists))

    def follow_path(self, path):
        if path is None or len(path) == 0:
            return False

        idx = 0
        while idx < len(path):
            # update current position
            state = self.client.getMultirotorState()
            pos = state.kinematics_estimated.position
            px, py = pos.x_val, pos.y_val

            # print real-time coordinates
            print(f"Drone position -> X: {px:.2f}, Y: {py:.2f}, Z: {pos.z_val:.2f}")

            # safety check
            dist_obs = self.nearest_obstacle_distance()
            if dist_obs < LOCAL_SAFETY_DIST:
                print("[follow_path] Obstacle too close -> hover and replan")
                self.client.moveByVelocityAsync(0,0,0,1).join()
                return False

            wx, wy = path[idx]
            dist = math.hypot(wx - px, wy - py)
            if dist < WAYPOINT_TOL:
                idx += 1
                continue

            # slow down if obstacle nearby
            speed_factor = min(1.0, dist_obs / LOCAL_SAFETY_DIST)
            vx = (wx - px) / (dist + 1e-6) * min(MAX_VEL, dist) * speed_factor
            vy = (wy - py) / (dist + 1e-6) * min(MAX_VEL, dist) * speed_factor

            self.client.moveByVelocityAsync(vx, vy, 0, 0.5).join()

        return True


# ----------------- MAIN MISSION -----------------
def run_mission(goal_xy):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)

    print("[mission] takeoff")
    client.takeoffAsync().join()
    client.moveToZAsync(DRONE_ALT, 1.0).join()

    nav = LidarAstarNavigator(client)
    t_start = time.time()

    try:
        while time.time() - t_start < TIMEOUT:
            nav.build_grid_from_lidar()

            # decide if we need to plan/replan
            need_plan = False
            if nav.last_plan is None: need_plan = True
            elif time.time() - nav.last_plan_time > REPLAN_INTERVAL: need_plan = True
            elif nav.nearest_obstacle_distance() < LOCAL_SAFETY_DIST + 0.2: need_plan = True

            if need_plan:
                print("[mission] planning path to goal", goal_xy)
                path = nav.plan_path(goal_xy)
                if path is None:
                    print("[mission] no path found; hovering and retrying")
                    client.moveByVelocityAsync(0,0,0,1).join()
                    time.sleep(1.0)
                    continue
            else:
                path = nav.last_plan

            # follow the path
            ok = nav.follow_path(path)
            if ok:
                pos = client.getMultirotorState().kinematics_estimated.position
                if math.hypot(pos.x_val - goal_xy[0], pos.y_val - goal_xy[1]) < WAYPOINT_TOL:
                    print("[mission] reached goal!")
                    return True
            else:
                print("[mission] path follow aborted; will replan")
                time.sleep(0.2)

    finally:
        print("[mission] landing")
        client.moveToZAsync(-1, 1).join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)

    return False


# ----------------- ENTRY POINT -----------------
if __name__ == "__main__":
    # example target relative to world (choose something within GRID_RADIUS_M of starting position)
    # set this to desired world coordinates (meters)
    TARGET_X = 50.0
    TARGET_Y = 22.0
    print("Starting mission to:", (TARGET_X, TARGET_Y))
    success = run_mission((TARGET_X, TARGET_Y))
    print("Mission success:", success)
