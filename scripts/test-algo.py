# airsim_nav.py
import airsim
import time
import numpy as np
import math
import heapq

# ---------- PARAMETERS ----------
LIDAR_NAME = "LidarSensor1"   # name from settings.json
GRID_RES = 0.2               # meters per grid cell
GRID_SIZE = 200              # grid will be GRID_SIZE x GRID_SIZE
DRONE_HEIGHT = -2.0          # flight altitude in NED (negative Z)
HEIGHT_TOLERANCE = 1.0       # consider points within ± this meter of DRONE_HEIGHT as obstacles
OCCUPIED_THRESHOLD = 1       # cells with >= this hits are obstacles
REPLAN_DISTANCE = 2.0        # meters ahead to trigger replan
WAYPOINT_TOL = 0.5           # meters to accept waypoint
MAX_VELOCITY = 2.0          # m/s
LOCAL_SAFETY_DIST = 1.0     # m — if obstacle within this, stop and replan
PLANNER_HZ = 1.0
LOCAL_LOOP_HZ = 10.0

# Grid origin is centered at initial position
# ---------- UTILITIES ----------
def euler_from_quat(q):
    # AirSim stores quaternion as (w,x,y,z)
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    # convert to roll, pitch, yaw
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = math.asin(t2)
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(t3, t4)
    return roll, pitch, yaw

def world_to_grid(idx_origin, origin_xyz, x, y):
    # origin_xyz = [x0,y0] of grid center in world coords
    dx = x - origin_xyz[0]
    dy = y - origin_xyz[1]
    ix = int(round(dx / GRID_RES)) + idx_origin
    iy = int(round(dy / GRID_RES)) + idx_origin
    return ix, iy

def grid_to_world(idx_origin, origin_xyz, ix, iy):
    dx = (ix - idx_origin) * GRID_RES
    dy = (iy - idx_origin) * GRID_RES
    return origin_xyz[0] + dx, origin_xyz[1] + dy

# ---------- A* ----------
def astar(grid, start, goal):
    # grid: 2D numpy where 0=free, 1=occupied
    h = lambda a,b: math.hypot(a[0]-b[0], a[1]-b[1])
    rows, cols = grid.shape
    open_set = []
    heapq.heappush(open_set, (0 + h(start,goal), 0, start, None))
    came_from = {}
    gscore = {start: 0}
    visited = set()
    while open_set:
        f, g, current, parent = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)
        came_from[current] = parent
        if current == goal:
            # reconstruct path
            path = []
            cur = current
            while cur is not None:
                path.append(cur)
                cur = came_from[cur]
            path.reverse()
            return path
        # neighbors (8-connected)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0:
                    continue
                nei = (current[0]+dx, current[1]+dy)
                if not (0 <= nei[0] < rows and 0 <= nei[1] < cols):
                    continue
                if grid[nei[0], nei[1]] != 0:
                    continue
                tentative_g = g + math.hypot(dx,dy)
                if nei not in gscore or tentative_g < gscore[nei]:
                    gscore[nei] = tentative_g
                    fscore = tentative_g + h(nei, goal)
                    heapq.heappush(open_set, (fscore, tentative_g, nei, current))
    return None

# ---------- MAIN AUTONOMY CLASS ----------
class SimpleNav:
    def __init__(self, client, grid_res=GRID_RES, grid_size=GRID_SIZE):
        self.client = client
        self.grid_res = grid_res
        self.grid_size = grid_size
        self.idx_origin = grid_size // 2
        self.grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
        self.origin_world = None  # will be set to initial xy
        self.last_plan = []
        self.last_plan_time = 0.0

    def update_pose(self):
        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        pose_xyz = (pos.x_val, pos.y_val, pos.z_val)
        ori = state.kinematics_estimated.orientation
        _,_,yaw = euler_from_quat(ori)
        return pose_xyz, yaw

    def clear_grid(self):
        self.grid.fill(0)

    def build_occupancy_from_lidar(self):
        # reset
        self.grid.fill(0)
        lidar_data = self.client.getLidarData(lidar_name=LIDAR_NAME)
        if not hasattr(lidar_data, 'point_cloud') or len(lidar_data.point_cloud) == 0:
            return
        pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1,3)
        # get world origin if not set
        pose, _ = self.update_pose()
        if self.origin_world is None:
            self.origin_world = (pose[0], pose[1])
        # Transform lidar points (are in vehicle local or world? AirSim Lidar returns sensor-local)
        # AirSim LIDAR points are in world coords when return in this function; if not, you'd transform using pose.
        # We'll assume world coords for simplicity; check your Lidar settings
        for (x,y,z) in pts:
            # height filter relative to DRONE_HEIGHT
            if abs(z - DRONE_HEIGHT) <= HEIGHT_TOLERANCE:
                ix, iy = world_to_grid(self.idx_origin, self.origin_world, x, y)
                if 0 <= ix < self.grid_size and 0 <= iy < self.grid_size:
                    self.grid[ix, iy] += 1
        # threshold to binary occupancy
        self.grid = (self.grid >= OCCUPIED_THRESHOLD).astype(np.uint8)

    def plan_to_goal(self, goal_world):
        # goal_world = (x,y) in world coords
        if self.origin_world is None:
            print("Origin not set - run lidar update once")
            return None
        # start
        pose, _ = self.update_pose()
        start_ix, start_iy = world_to_grid(self.idx_origin, self.origin_world, pose[0], pose[1])
        goal_ix, goal_iy = world_to_grid(self.idx_origin, self.origin_world, goal_world[0], goal_world[1])
        # bounds check
        rows,cols = self.grid.shape
        start = (max(0,min(rows-1,start_ix)), max(0,min(cols-1,start_iy)))
        goal = (max(0,min(rows-1,goal_ix)), max(0,min(cols-1,goal_iy)))
        path = astar(self.grid, start, goal)
        if path is None:
            print("No path found")
            return None
        # convert to world
        world_path = [grid_to_world(self.idx_origin, self.origin_world, p[0], p[1]) for p in path]
        self.last_plan = world_path
        self.last_plan_time = time.time()
        return world_path

    def nearest_obstacle_dist(self):
        # quick safety check: compute min distance from current pose to any occupied grid cell
        pose, _ = self.update_pose()
        px, py = pose[0], pose[1]
        # compute coordinates of all occupied cells
        occ = np.argwhere(self.grid==1)
        if occ.size==0:
            return float('inf')
        dists = np.hypot((occ[:,0]-self.idx_origin)*self.grid_res - (px - self.origin_world[0]),
                         (occ[:,1]-self.idx_origin)*self.grid_res - (py - self.origin_world[1]))
        return float(np.min(dists))

    def follow_path(self, path):
        # path is list of (x,y) world coords
        if path is None or len(path)==0:
            return False
        # simple waypoint follower
        idx = 0
        rate = 1.0 / LOCAL_LOOP_HZ
        while idx < len(path):
            pose, yaw = self.update_pose()
            px, py, pz = pose
            wx, wy = path[idx]
            dx = wx - px; dy = wy - py
            dist = math.hypot(dx, dy)
            # safety check
            if self.nearest_obstacle_dist() < LOCAL_SAFETY_DIST:
                print("Obstacle too close - abort and replan")
                self.client.moveByVelocityAsync(0,0,0,1).join()
                return False
            if dist < WAYPOINT_TOL:
                idx += 1
                continue
            # velocity command toward waypoint
            vx = (dx/dist) * min(MAX_VELOCITY, dist)
            vy = (dy/dist) * min(MAX_VELOCITY, dist)
            vz = 0  # maintain altitude (we assume DRONE_HEIGHT constant)
            # send body-frame velocity or world-frame: moveByVelocityAsync uses world frame by default
            self.client.moveByVelocityAsync(vx, vy, 0, duration=1.0/LOCAL_LOOP_HZ).join()
            time.sleep(rate)
        return True

# ---------- DEMO RUN ----------
def demo_run(goal_xy):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    client.takeoffAsync().join()
    # climb to DRONE_HEIGHT (negative in NED)
    client.moveToZAsync(DRONE_HEIGHT, 1.0).join()

    nav = SimpleNav(client)
    t0 = time.time()
    try:
        # initial occupancy build
        nav.build_occupancy_from_lidar()
        # plan
        path = nav.plan_to_goal(goal_xy)
        if path is None:
            print("Can't find path to goal")
            return
        print("Planned %d waypoints" % len(path))
        # follow with replanning on failure or periodically
        success = False
        while time.time() - t0 < 300:  # 5 minute timeout
            # if no path or need replan
            if nav.last_plan is None or (time.time() - nav.last_plan_time) > (1.0/PLANNER_HZ):
                nav.build_occupancy_from_lidar()
                path = nav.plan_to_goal(goal_xy)
            # follow current path
            ok = nav.follow_path(path)
            if ok:
                # check final dist to goal
                pose, _ = nav.update_pose()
                dgoal = math.hypot(pose[0]-goal_xy[0], pose[1]-goal_xy[1])
                if dgoal <= WAYPOINT_TOL:
                    print("Reached goal!")
                    success = True
                    break
            else:
                # replan after short pause
                time.sleep(0.5)
                nav.build_occupancy_from_lidar()
                path = nav.plan_to_goal(goal_xy)
        if not success:
            print("Failed to reach goal within timeout")
    finally:
        client.moveToZAsync(-1, 1).join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)

if __name__ == "__main__":
    # set your target in world coords (x,y). For example 10, 0 meters ahead.
    target = (10.0, 0.0)
    demo_run(target)
