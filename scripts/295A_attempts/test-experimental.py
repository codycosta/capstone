"""
airsim_astar_stable.py

Stable A* navigation for AirSim Multirotor:
- LiDAR -> 2D occupancy grid
- A* path planning
- PID altitude hold
- Adjustable forward speed
- Smooth landing
- Live 2D visualization
"""

import airsim
import numpy as np
import math
import time
from heapq import heappush, heappop
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt

# ---------------- PARAMETERS ----------------
LIDAR_NAME = "LidarSensor1"
GRID_RES = 0.5
GRID_RADIUS_M = 25.0
GRID_SIZE = int((GRID_RADIUS_M*2)/GRID_RES)
DRONE_ALT = -3.0
DRONE_RADIUS = 0.6
OCCUPIED_THRESHOLD = 1
REPLAN_INTERVAL = 1.0
WAYPOINT_TOL = 1.0
FORWARD_SPEED = 2.0
LOCAL_SAFETY_DIST = 1.0
A_STAR_DIAGONAL_COST = 1.4
TIMEOUT = 300
ALT_KP, ALT_KI, ALT_KD = 0.9, 0.05, 0.4

# ---------------- Utilities ----------------
def quat_to_rot_mat(q):
    w,x,y,z = q.w_val,q.x_val,q.y_val,q.z_val
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1-2*(x*x+z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1-2*(x*x+y*y)]
    ])

def transform_lidar_points(lidar_data, pose):
    if not hasattr(lidar_data,'point_cloud') or len(lidar_data.point_cloud)<3:
        return np.zeros((0,3), dtype=np.float32)
    pts = np.array(lidar_data.point_cloud, dtype=np.float32).reshape(-1,3)
    pos = pose.position
    ori = pose.orientation
    R = quat_to_rot_mat(ori)
    return (R @ pts.T).T + np.array([pos.x_val, pos.y_val, pos.z_val])

def world_xy_to_grid_ix(origin_xy, x, y):
    dx, dy = x-origin_xy[0], y-origin_xy[1]
    ix = int(round((dx + GRID_RADIUS_M)/GRID_RES))
    iy = int(round((dy + GRID_RADIUS_M)/GRID_RES))
    return ix, iy

def grid_ix_to_world_xy(origin_xy, ix, iy):
    x = origin_xy[0] + (ix*GRID_RES) - GRID_RADIUS_M
    y = origin_xy[1] + (iy*GRID_RES) - GRID_RADIUS_M
    return x, y

def astar_grid(grid, start, goal):
    rows, cols = grid.shape
    if not (0<=start[0]<rows and 0<=start[1]<cols): return None
    if not (0<=goal[0]<rows and 0<=goal[1]<cols): return None
    if grid[goal[0],goal[1]] !=0: return None
    def heuristic(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])
    open_set = []
    heappush(open_set, (heuristic(start,goal),0.0,start,None))
    came_from = {}
    gscore = {start:0.0}
    closed = set()
    while open_set:
        f,g,current,parent = heappop(open_set)
        if current in closed: continue
        came_from[current]=parent
        if current==goal:
            path=[current]
            cur=current
            while came_from[cur] is not None:
                cur=came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(current)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx==0 and dy==0: continue
                nei=(current[0]+dx, current[1]+dy)
                if not (0<=nei[0]<rows and 0<=nei[1]<cols): continue
                if grid[nei[0],nei[1]]!=0: continue
                tentative_g = g + (A_STAR_DIAGONAL_COST if dx!=0 and dy!=0 else 1.0)
                if nei not in gscore or tentative_g<gscore[nei]:
                    gscore[nei]=tentative_g
                    heappush(open_set,(tentative_g+heuristic(nei,goal),tentative_g,nei,current))
    return None

# ---------------- Navigator ----------------
class AstarNavigator:
    def __init__(self, client, enable_plot=True):
        self.client = client
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.origin_xy = None
        self.last_plan = None
        self.last_plan_time = 0
        self.alt_integral = 0
        self.prev_alt_error = 0
        self.prev_alt_time = time.time()
        self.enable_plot = enable_plot
        if enable_plot:
            plt.ion()
            self.fig,self.ax = plt.subplots(figsize=(6,6))

    def build_grid_from_lidar(self):
        state = self.client.getMultirotorState()
        pose = state.kinematics_estimated
        self.origin_xy = (pose.position.x_val, pose.position.y_val)
        lidar = self.client.getLidarData(lidar_name=LIDAR_NAME)
        pts_world = transform_lidar_points(lidar, pose)
        if pts_world.shape[0]==0:
            self.grid.fill(0)
            return
        mask = (pts_world[:,2]>=(DRONE_ALT-2.0)) & (pts_world[:,2]<=(DRONE_ALT+2.0))
        pts2 = pts_world[mask][:,:2]
        self.grid.fill(0)
        for x,y in pts2:
            ix,iy = world_xy_to_grid_ix(self.origin_xy,x,y)
            if 0<=ix<GRID_SIZE and 0<=iy<GRID_SIZE:
                self.grid[ix,iy]+=1
        occ = (self.grid>=OCCUPIED_THRESHOLD).astype(np.uint8)
        inflate_cells = int(round(DRONE_RADIUS/GRID_RES))
        if inflate_cells>0: occ = binary_dilation(occ,iterations=inflate_cells).astype(np.uint8)
        self.grid = occ

    def plan_path(self,goal_xy):
        if self.origin_xy is None: return None
        start=(GRID_SIZE//2,GRID_SIZE//2)
        goal_ix,goal_iy = world_xy_to_grid_ix(self.origin_xy,goal_xy[0],goal_xy[1])
        goal_ix = max(0,min(GRID_SIZE-1,goal_ix))
        goal_iy = max(0,min(GRID_SIZE-1,goal_iy))
        if self.grid[goal_ix,goal_iy]!=0: return None
        path = astar_grid(self.grid,start,(goal_ix,goal_iy))
        if path is None: return None
        world_path=[(float(grid_ix_to_world_xy(self.origin_xy,p[0],p[1])[0]),
                     float(grid_ix_to_world_xy(self.origin_xy,p[0],p[1])[1])) for p in path]
        self.last_plan=world_path
        self.last_plan_time=time.time()
        return world_path

    def nearest_obstacle_distance(self):
        occ = np.argwhere(self.grid==1)
        if occ.size==0: return float('inf')
        occ_xy=np.array([grid_ix_to_world_xy(self.origin_xy,int(i),int(j)) for i,j in occ])
        pos=self.client.getMultirotorState().kinematics_estimated.position
        px,py = pos.x_val,pos.y_val
        return float(np.min(np.hypot(occ_xy[:,0]-px,occ_xy[:,1]-py)))

    def altitude_hold(self):
        state=self.client.getMultirotorState()
        current_alt=state.kinematics_estimated.position.z_val
        error=DRONE_ALT-current_alt
        now=time.time()
        dt = max(now-self.prev_alt_time,0.01)
        self.alt_integral+=error*dt
        derivative=(error-self.prev_alt_error)/dt
        vz = ALT_KP*error + ALT_KI*self.alt_integral + ALT_KD*derivative
        vz = max(min(vz,2.0),-2.0)
        self.prev_alt_error = error
        self.prev_alt_time = now
        return float(vz)

    def plot_grid_and_path(self,path=None):
        if not self.enable_plot: return
        self.ax.clear()
        self.ax.imshow(self.grid.T, origin='lower', cmap='Greys', extent=(-GRID_RADIUS_M, GRID_RADIUS_M, -GRID_RADIUS_M, GRID_RADIUS_M))
        if path:
            px = [p[0]-self.origin_xy[0] for p in path]
            py = [p[1]-self.origin_xy[1] for p in path]
            self.ax.plot(px, py, 'r.-')
        self.ax.set_title("LiDAR Grid + Path")
        plt.pause(0.001)

    def follow_path(self,path):
        if path is None or len(path)==0: return False
        idx=0
        while idx<len(path):
            if self.nearest_obstacle_distance()<LOCAL_SAFETY_DIST:
                self.client.moveByVelocityAsync(float(0),float(0),float(0),float(1)).join()
                return False

            state=self.client.getMultirotorState()
            px,py,pz = state.kinematics_estimated.position.x_val, state.kinematics_estimated.position.y_val, state.kinematics_estimated.position.z_val

            wx,wy=path[idx]
            dist=math.hypot(wx-px,wy-py)
            if dist<WAYPOINT_TOL:
                idx+=1
                continue

            vx=float((wx-px)/dist*FORWARD_SPEED)
            vy=float((wy-py)/dist*FORWARD_SPEED)
            vz=float(self.altitude_hold())

            self.client.moveByVelocityAsync(vx,vy,vz,0.2).join()
            self.plot_grid_and_path(path)
        return True

# ---------------- Main Mission ----------------
def run_mission(goal_xy):
    client=airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[mission] takeoff")
    client.takeoffAsync().join()
    client.moveToZAsync(float(DRONE_ALT),1.0).join()

    nav=AstarNavigator(client,enable_plot=True)
    t_start=time.time()

    try:
        while time.time()-t_start<TIMEOUT:
            nav.build_grid_from_lidar()
            need_plan=(nav.last_plan is None
                       or time.time()-nav.last_plan_time>REPLAN_INTERVAL
                       or nav.nearest_obstacle_distance()<LOCAL_SAFETY_DIST)
            if need_plan:
                path=nav.plan_path(goal_xy)
                if path is None:
                    client.moveByVelocityAsync(float(0),float(0),float(0),float(1)).join()
                    time.sleep(1.0)
                    continue
            else:
                path=nav.last_plan

            ok=nav.follow_path(path)
            pos=client.getMultirotorState().kinematics_estimated.position
            print(f"[mission] Pos x={pos.x_val:.2f}, y={pos.y_val:.2f}, z={pos.z_val:.2f}")

            if ok and math.hypot(pos.x_val-goal_xy[0],pos.y_val-goal_xy[1])<WAYPOINT_TOL:
                print("[mission] reached goal! Performing smooth landing...")
                client.moveToZAsync(float(DRONE_ALT)+2,1.0).join()
                client.moveToZAsync(float(-1),0.5).join()
                client.landAsync().join()
                client.armDisarm(False)
                client.enableApiControl(False)
                if nav.enable_plot: plt.ioff(); plt.show()
                return True

            if not ok:
                time.sleep(0.2)
                continue

    finally:
        print("[mission] landing (fallback)")
        client.moveToZAsync(float(-1),1.0).join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        if nav.enable_plot: plt.ioff(); plt.show()
    return False

# ---------------- Entry ----------------
if __name__=="__main__":
    TARGET_X=30.0
    TARGET_Y=80.0
    print("Starting mission to:",(TARGET_X,TARGET_Y))
    success=run_mission((TARGET_X,TARGET_Y))
    print("Mission success:",success)
