# airsim_astar_dashboard_smoothed.py
"""
Smoothed A* dashboard for AirSim (classic API)
- 6-panel matplotlib dashboard
- Fixed grid origin
- PID altitude hold
- Smoothed XY motion: PD position -> velocity, accel limiting, EMA
- Float-casted AirSim calls
- Robust plotting guards
- Smooth landing
"""

import airsim
import numpy as np
import math
import time
from heapq import heappush, heappop
from collections import deque
from scipy.ndimage import binary_dilation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import traceback

# ---------------- PARAMETERS ----------------
LIDAR_NAME = "LidarSensor1"
GRID_RES = 0.5
GRID_RADIUS_M = 25.0
GRID_SIZE = int((GRID_RADIUS_M * 2) / GRID_RES)
DRONE_ALT = -3.0
DRONE_RADIUS = 0.6
OCCUPIED_THRESHOLD = 1
REPLAN_INTERVAL = 1.5
WAYPOINT_TOL = 1.0
FORWARD_SPEED = 3.0
LOCAL_SAFETY_DIST = 1.0
A_STAR_DIAGONAL_COST = 1.4
TIMEOUT = 300

# plotting / history
PLOT_ENABLED = True
PLOT_FPS = 8               # limit updates per second (approx)
HISTORY_SEC = 30           # seconds of history for time plots
HIST_LEN = int(HISTORY_SEC * PLOT_FPS)

# altitude PID
ALT_KP, ALT_KI, ALT_KD = 0.9, 0.03, 0.25

# smoothing / XY control parameters
POS_KP = 0.8                # P gain: meters -> m/s (velocity magnitude per meter of error)
POS_KD = 0.01                # D gain for damping (small)
MAX_ACCEL = 2.0             # max change in velocity per second (m/s^2)
VELOCITY_EMA_ALPHA = 0.50   # EMA smoothing factor (0..1)
VELOCITY_CMD_DURATION = 1 # seconds to hold each velocity command

# LiDAR histogram bins
LIDAR_MAX_RANGE = 30.0
LIDAR_BINS = 30

# ------------------------------------------------

# --------------- Utilities ----------------------
def quat_to_rot_mat(q):
    w, x, y, z = q.w_val, q.x_val, q.y_val, q.z_val
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ], dtype=np.float32)
    return R

def transform_lidar_points(lidar_data, kin):
    if not hasattr(lidar_data, 'point_cloud') or lidar_data.point_cloud is None:
        return np.zeros((0,3), dtype=np.float32)
    pc = np.array(lidar_data.point_cloud, dtype=np.float32)
    if pc.size == 0:
        return np.zeros((0,3), dtype=np.float32)
    try:
        pts = pc.reshape(-1, 3)
    except Exception:
        return np.zeros((0,3), dtype=np.float32)
    R = quat_to_rot_mat(kin.orientation)
    t = np.array([kin.position.x_val, kin.position.y_val, kin.position.z_val], dtype=np.float32)
    pts_world = (R @ pts.T).T + t
    return pts_world

def world_xy_to_grid_ix(origin_xy, x, y):
    ix = int(round((x - origin_xy[0] + GRID_RADIUS_M) / GRID_RES))
    iy = int(round((y - origin_xy[1] + GRID_RADIUS_M) / GRID_RES))
    return ix, iy

def grid_ix_to_world_xy(origin_xy, ix, iy):
    x = origin_xy[0] + (ix * GRID_RES) - GRID_RADIUS_M
    y = origin_xy[1] + (iy * GRID_RES) - GRID_RADIUS_M
    return x, y

# --------------- A* --------------------------------
def astar_grid(grid, start, goal):
    rows, cols = grid.shape
    sx, sy = start
    gx, gy = goal
    if not (0 <= sx < rows and 0 <= sy < cols): return None
    if not (0 <= gx < rows and 0 <= gy < cols): return None
    if grid[gx, gy] != 0: return None

    def heur(a,b): return math.hypot(a[0]-b[0], a[1]-b[1])

    open_set = []
    heappush(open_set, (heur(start, goal), 0.0, start, None))
    came_from = {}
    gscore = {start: 0.0}
    closed = set()

    while open_set:
        f, g, current, parent = heappop(open_set)
        if current in closed: continue
        came_from[current] = parent
        if current == goal:
            path = [current]
            cur = current
            while came_from[cur] is not None:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path
        closed.add(current)
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                if dx == 0 and dy == 0: continue
                nei = (current[0]+dx, current[1]+dy)
                if not (0 <= nei[0] < rows and 0 <= nei[1] < cols): continue
                if grid[nei[0], nei[1]] != 0: continue
                tentative_g = g + (A_STAR_DIAGONAL_COST if dx!=0 and dy!=0 else 1.0)
                if nei not in gscore or tentative_g < gscore[nei]:
                    gscore[nei] = tentative_g
                    heappush(open_set, (tentative_g + heur(nei, goal), tentative_g, nei, current))
    return None

# --------------- Navigator ------------------------
class DashboardNavigator:
    def __init__(self, client):
        self.client = client
        self.grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint8)
        self.origin_xy = None       # fixed on first grid build
        self.origin_locked = False
        self.path_world = []
        self.last_plan_time = 0.0

        # altitude PID
        self.alt_integral = 0.0
        self.prev_alt_err = 0.0
        self.prev_alt_time = time.time()

        # smoothing state
        self.prev_vx = 0.0
        self.prev_vy = 0.0
        self.prev_ctrl_time = time.time()
        self.prev_err = (0.0, 0.0)

        # plotting/history buffers
        self.traj_history = deque(maxlen=HIST_LEN)   # list of (x,y)
        self.time_history = deque(maxlen=HIST_LEN)
        self.alt_history = deque(maxlen=HIST_LEN)
        self.alt_cmd_history = deque(maxlen=HIST_LEN)
        self.vx_cmd_history = deque(maxlen=HIST_LEN)
        self.vy_cmd_history = deque(maxlen=HIST_LEN)
        self.vx_act_history = deque(maxlen=HIST_LEN)
        self.vy_act_history = deque(maxlen=HIST_LEN)
        self.lidar_ranges_hist = deque(maxlen=HIST_LEN)
        self.last_plot_time = 0.0

        # plotting setup
        if PLOT_ENABLED:
            self._init_plots()

    # ---------- plotting init ----------
    def _init_plots(self):
        plt.ion()
        self.fig = plt.figure(figsize=(14,8))
        gs = self.fig.add_gridspec(2,3, wspace=0.35, hspace=0.35)

        self.ax_grid = self.fig.add_subplot(gs[0,0])
        self.ax_traj = self.fig.add_subplot(gs[0,1])
        self.ax_heading = self.fig.add_subplot(gs[0,2])
        self.ax_alt = self.fig.add_subplot(gs[1,0])
        self.ax_vel = self.fig.add_subplot(gs[1,1])
        self.ax_hist = self.fig.add_subplot(gs[1,2])

        # initial artists
        self.im = self.ax_grid.imshow(np.zeros((GRID_SIZE, GRID_SIZE)), origin='lower',
                                     extent=[-GRID_RADIUS_M, GRID_RADIUS_M, -GRID_RADIUS_M, GRID_RADIUS_M],
                                     cmap='Greys', vmin=0, vmax=1)
        self.scatter_lidar = self.ax_grid.scatter([], [], s=5, c='cyan', alpha=0.8)
        self.line_path, = self.ax_grid.plot([], [], 'r.-', linewidth=2, markersize=4)
        self.goal_pt, = self.ax_grid.plot([], [], 'g*', markersize=12)
        self.drone_pt, = self.ax_grid.plot([], [], 'ro', markersize=6)
        self.safety_circle = patches.Circle((0,0), LOCAL_SAFETY_DIST, edgecolor='orange', facecolor='none', alpha=0.6)
        self.ax_grid.add_patch(self.safety_circle)
        self.ax_grid.set_title('Occupancy Grid + LiDAR + Path')
        self.ax_grid.set_xlabel('X (m)')
        self.ax_grid.set_ylabel('Y (m)')

        # traj plot
        self.traj_line, = self.ax_traj.plot([], [], '-', linewidth=1)
        self.ax_traj.plot([], [], 'go', label='goal')
        self.cur_pt, = self.ax_traj.plot([], [], 'ro', label='drone')
        self.ax_traj.set_title('Trajectory (Breadcrumbs)')
        self.ax_traj.set_xlabel('X (m)')
        self.ax_traj.set_ylabel('Y (m)')
        self.ax_traj.legend(loc='upper right')

        # heading plot (arrow)
        self.ax_heading.set_xlim(-1,1)
        self.ax_heading.set_ylim(-1,1)
        self.ax_heading.set_title('Heading (blue) and Commanded Velocity (red)')
        self.ax_heading.axis('off')

        # altitude
        self.alt_line, = self.ax_alt.plot([], [], label='actual z')
        self.alt_cmd_line, = self.ax_alt.plot([], [], label='target z')
        self.ax_alt.set_title('Altitude (NED: negative up)')
        self.ax_alt.set_xlabel('Time (s)')
        self.ax_alt.set_ylabel('Z (m)')
        self.ax_alt.legend()

        # velocity vs time (prepare empty lines; will set data later)
        self.vx_cmd_line, = self.ax_vel.plot([], [], label='vx_cmd')
        self.vy_cmd_line, = self.ax_vel.plot([], [], label='vy_cmd')
        self.vx_act_line, = self.ax_vel.plot([], [], label='vx_act', linestyle='--')
        self.vy_act_line, = self.ax_vel.plot([], [], label='vy_act', linestyle='--')
        self.ax_vel.set_title('Commanded vs Actual Velocity')
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_ylabel('m/s')
        self.ax_vel.legend()

        # lidar histogram
        self.hist_bins = np.linspace(0, LIDAR_MAX_RANGE, LIDAR_BINS+1)
        self.ax_hist.set_title('LiDAR Ranges Histogram')
        self.ax_hist.set_xlabel('Range (m)')
        self.ax_hist.set_ylabel('Counts')

        plt.show()

    # ---------- grid build (fixed origin) ----------
    def build_grid_from_lidar(self):
        state = self.client.getMultirotorState()
        kin = state.kinematics_estimated
        lidar = self.client.getLidarData(lidar_name=LIDAR_NAME)
        pts_world = transform_lidar_points(lidar, kin)

        # lock origin on first valid call
        if not self.origin_locked:
            self.origin_xy = (kin.position.x_val, kin.position.y_val)
            self.origin_locked = True

        # reset counts
        counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.uint16)
        if pts_world.shape[0] > 0:
            zmin, zmax = DRONE_ALT - 3.0, DRONE_ALT + 3.0
            mask = (pts_world[:,2] >= zmin) & (pts_world[:,2] <= zmax)
            pts_xy = pts_world[mask][:,:2]
            for x,y in pts_xy:
                ix, iy = world_xy_to_grid_ix(self.origin_xy, x, y)
                if 0 <= ix < GRID_SIZE and 0 <= iy < GRID_SIZE:
                    counts[ix, iy] += 1

        occ = (counts >= OCCUPIED_THRESHOLD).astype(np.uint8)
        inflate_cells = int(math.ceil(DRONE_RADIUS / GRID_RES))
        if inflate_cells > 0:
            occ = binary_dilation(occ, iterations=inflate_cells).astype(np.uint8)
        self.grid = occ

        # store last LiDAR points and ranges for plotting + histogram
        self.last_lidar_pts = pts_world
        horiz = None
        if pts_world.shape[0] > 0:
            horiz = np.hypot(pts_world[:,0] - kin.position.x_val, pts_world[:,1] - kin.position.y_val)
        self.last_lidar_ranges = horiz if horiz is not None else np.array([])

    # ---------- planning ----------
    def plan_path(self, goal_xy):
        if not self.origin_locked: return None
        start = (GRID_SIZE // 2, GRID_SIZE // 2)
        gx, gy = world_xy_to_grid_ix(self.origin_xy, goal_xy[0], goal_xy[1])
        gx = max(0, min(GRID_SIZE - 1, gx))
        gy = max(0, min(GRID_SIZE - 1, gy))
        if self.grid[gx, gy] == 1:
            # try small local search for nearby free cell
            found = False
            for r in range(1, 6):
                for dx in range(-r, r+1):
                    for dy in range(-r, r+1):
                        nx, ny = gx + dx, gy + dy
                        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE and self.grid[nx, ny] == 0:
                            gx, gy = nx, ny
                            found = True
                            break
                    if found: break
                if found: break
            if not found:
                return None

        state = self.client.getMultirotorState()
        pos = state.kinematics_estimated.position
        sx, sy = world_xy_to_grid_ix(self.origin_xy, pos.x_val, pos.y_val)
        idx_path = astar_grid(self.grid, (sx, sy), (gx, gy))
        if idx_path is None: return None

        path_world = []
        for ix, iy in idx_path:
            wx, wy = grid_ix_to_world_xy(self.origin_xy, ix, iy)
            path_world.append((wx, wy))
        self.path_world = path_world
        self.last_plan_time = time.time()
        return path_world

    # ---------- obstacle & nearest check ----------
    def nearest_obstacle_distance(self):
        occ = np.argwhere(self.grid == 1)
        if occ.size == 0:
            return float('inf')
        occ_xy = np.array([grid_ix_to_world_xy(self.origin_xy, int(i), int(j)) for i, j in occ])
        pos = self.client.getMultirotorState().kinematics_estimated.position
        px, py = pos.x_val, pos.y_val
        return float(np.min(np.hypot(occ_xy[:,0] - px, occ_xy[:,1] - py)))

    # ---------- altitude PID ----------
    def altitude_hold(self):
        state = self.client.getMultirotorState()
        current_alt = state.kinematics_estimated.position.z_val
        error = DRONE_ALT - current_alt
        now = time.time()
        dt = max(now - self.prev_alt_time, 1e-3)
        self.alt_integral += error * dt
        derivative = (error - self.prev_alt_err) / dt
        vz = ALT_KP * error + ALT_KI * self.alt_integral + ALT_KD * derivative
        vz = max(min(vz, 2.0), -2.0)
        self.prev_alt_err = error
        self.prev_alt_time = now
        return float(vz)

    # ---------- follow path (single-step + smoothed command) ----------
    def follow_path(self, goal_xy):
        if not self.path_world or len(self.path_world) == 0:
            return False

        # single-step follow of the next waypoint
        state = self.client.getMultirotorState()
        kin = state.kinematics_estimated
        px, py = kin.position.x_val, kin.position.y_val

        # record history
        tnow = time.time()
        self.traj_history.append((px, py))
        self.time_history.append(tnow)
        self.alt_history.append(kin.position.z_val)
        self.alt_cmd_history.append(DRONE_ALT)

        # commanded waypoint
        wx, wy = self.path_world[0]
        err_x = wx - px
        err_y = wy - py
        dist = math.hypot(err_x, err_y)
        if dist < WAYPOINT_TOL:
            # reached this waypoint - pop and continue
            self.path_world.pop(0)
            return True

        # ----------------- SMOOTHING CONTROLLER -----------------
        now = time.time()
        dt = max(now - self.prev_ctrl_time, 1e-3)

        # direction
        if dist > 1e-6:
            dir_x = err_x / dist
            dir_y = err_y / dist
        else:
            dir_x = 0.0; dir_y = 0.0

        # P term: velocity magnitude from position error
        vel_mag_p = min(FORWARD_SPEED, POS_KP * dist)

        # D term (damping) along direction (use change in error)
        derr_x = (err_x - self.prev_err[0]) / dt
        derr_y = (err_y - self.prev_err[1]) / dt
        vel_mag_d = POS_KD * (derr_x * dir_x + derr_y * dir_y)

        desired_vx = vel_mag_p * dir_x + vel_mag_d * dir_x
        desired_vy = vel_mag_p * dir_y + vel_mag_d * dir_y

        # accel limiting
        dvx = desired_vx - self.prev_vx
        dvy = desired_vy - self.prev_vy
        max_dv = MAX_ACCEL * dt
        dv_norm = math.hypot(dvx, dvy)
        if dv_norm > max_dv and max_dv > 0:
            scale = max_dv / (dv_norm + 1e-9)
            dvx *= scale
            dvy *= scale

        smooth_vx = self.prev_vx + dvx
        smooth_vy = self.prev_vy + dvy

        # EMA smoothing
        smooth_vx = VELOCITY_EMA_ALPHA * smooth_vx + (1.0 - VELOCITY_EMA_ALPHA) * self.prev_vx
        smooth_vy = VELOCITY_EMA_ALPHA * smooth_vy + (1.0 - VELOCITY_EMA_ALPHA) * self.prev_vy

        vz_cmd = self.altitude_hold()

        # send velocities (float-casted) with longer hold for smoother motion
        self.client.moveByVelocityAsync(float(smooth_vx), float(smooth_vy), float(vz_cmd), float(VELOCITY_CMD_DURATION)).join()

        # store commanded/actual velocities for plotting (approx actual from kin.)
        vel = getattr(kin, 'linear_velocity', None)
        if vel is not None:
            vx_act = float(vel.x_val); vy_act = float(vel.y_val)
        else:
            if len(self.traj_history) >= 2:
                (px_prev, py_prev) = self.traj_history[-2]
                dt_hist = max(tnow - self.time_history[-2], 1e-3)
                vx_act = float((px - px_prev) / dt_hist)
                vy_act = float((py - py_prev) / dt_hist)
            else:
                vx_act = 0.0; vy_act = 0.0

        vx_cmd = float(smooth_vx); vy_cmd = float(smooth_vy)

        self.vx_cmd_history.append(vx_cmd)
        self.vy_cmd_history.append(vy_cmd)
        self.vx_act_history.append(vx_act)
        self.vy_act_history.append(vy_act)

        # lidar ranges for histogram
        if hasattr(self, 'last_lidar_ranges') and self.last_lidar_ranges is not None:
            self.lidar_ranges_hist.append(self.last_lidar_ranges)
        else:
            self.lidar_ranges_hist.append(np.array([]))

        # update smoothing state
        self.prev_vx = float(smooth_vx)
        self.prev_vy = float(smooth_vy)
        self.prev_ctrl_time = now
        self.prev_err = (err_x, err_y)

        return True

    # ---------- plotting update (robust) ----------
    def update_plots(self, goal_xy):
        if not PLOT_ENABLED:
            return
        now = time.time()
        if now - self.last_plot_time < 1.0 / PLOT_FPS:
            return
        self.last_plot_time = now

        try:
            # 1) Grid + LiDAR + Path
            try:
                self.im.set_data(np.flipud(self.grid.T))
                if hasattr(self, 'last_lidar_pts') and self.last_lidar_pts.shape[0] > 0:
                    pts = self.last_lidar_pts
                    xs = pts[:,0] - self.origin_xy[0]
                    ys = pts[:,1] - self.origin_xy[1]
                    mask = (np.abs(xs) <= GRID_RADIUS_M) & (np.abs(ys) <= GRID_RADIUS_M)
                    xs = xs[mask]; ys = ys[mask]
                    self.scatter_lidar.set_offsets(np.column_stack((xs, ys)))
                else:
                    self.scatter_lidar.set_offsets(np.zeros((0,2)))
                if self.path_world and len(self.path_world) > 0:
                    pxs = [p[0] - self.origin_xy[0] for p in self.path_world]
                    pys = [p[1] - self.origin_xy[1] for p in self.path_world]
                    self.line_path.set_data(pxs, pys)
                    self.goal_pt.set_data([goal_xy[0] - self.origin_xy[0]], [goal_xy[1] - self.origin_xy[1]])
                else:
                    self.line_path.set_data([], [])
                    self.goal_pt.set_data([], [])
            except Exception:
                traceback.print_exc()

            # 2) Trajectory & safety circle
            try:
                if len(self.traj_history) > 0:
                    dx, dy = self.traj_history[-1]
                    self.drone_pt.set_data([dx - self.origin_xy[0]], [dy - self.origin_xy[1]])
                    self.safety_circle.center = (dx - self.origin_xy[0], dy - self.origin_xy[1])
                    xs = [p[0] for p in self.traj_history]
                    ys = [p[1] for p in self.traj_history]
                    xs_local = [x - self.origin_xy[0] for x in xs]
                    ys_local = [y - self.origin_xy[1] for y in ys]
                    if len(xs_local) >= 2:
                        self.traj_line.set_data(xs_local, ys_local)
                        self.ax_traj.set_xlim(min(xs_local) - 2, max(xs_local) + 2)
                        self.ax_traj.set_ylim(min(ys_local) - 2, max(ys_local) + 2)
                        self.cur_pt.set_data([xs_local[-1]], [ys_local[-1]])
                else:
                    self.traj_line.set_data([], [])
                    self.cur_pt.set_data([], [])
            except Exception:
                traceback.print_exc()

            # 3) Heading & commanded velocity arrow
            try:
                self.ax_heading.clear()
                self.ax_heading.set_xlim(-1,1)
                self.ax_heading.set_ylim(-1,1)
                if len(self.vx_act_history) > 0 or len(self.vx_cmd_history) > 0:
                    vx_act = self.vx_act_history[-1] if len(self.vx_act_history)>0 else 0.0
                    vy_act = self.vy_act_history[-1] if len(self.vy_act_history)>0 else 0.0
                    vxc = self.vx_cmd_history[-1] if len(self.vx_cmd_history)>0 else 0.0
                    vyc = self.vy_cmd_history[-1] if len(self.vy_cmd_history)>0 else 0.0
                    scale = 0.5
                    self.ax_heading.arrow(0, 0, vx_act*scale, vy_act*scale, head_width=0.08, color='blue', length_includes_head=True)
                    self.ax_heading.arrow(0, 0, vxc*scale, vyc*scale, head_width=0.08, color='red', length_includes_head=True)
                self.ax_heading.set_title('Heading (blue) and Commanded Velocity (red)')
                self.ax_heading.axis('off')
            except Exception:
                traceback.print_exc()

            # 4) Altitude over time (robust)
            try:
                if len(self.time_history) >= 2 and len(self.alt_history) >= 2:
                    t0 = self.time_history[0]
                    times = np.array(self.time_history) - t0
                    n = min(len(times), len(self.alt_history), len(self.alt_cmd_history))
                    if n >= 2:
                        times_trim = times[:n]
                        alt_trim = list(self.alt_history)[:n]
                        alt_cmd_trim = list(self.alt_cmd_history)[:n]
                        self.alt_line.set_data(times_trim, alt_trim)
                        self.alt_cmd_line.set_data(times_trim, alt_cmd_trim)
                        self.ax_alt.relim(); self.ax_alt.autoscale_view(True,True,True)
            except Exception:
                traceback.print_exc()

            # 5) Commanded vs Actual velocity (robust)
            try:
                n = min(len(self.time_history), len(self.vx_cmd_history), len(self.vy_cmd_history),
                        len(self.vx_act_history), len(self.vy_act_history))
                if n >= 2:
                    t0 = self.time_history[0]
                    times = np.array(self.time_history) - t0
                    t_trim = times[:n]
                    vx_cmd = list(self.vx_cmd_history)[:n]
                    vy_cmd = list(self.vy_cmd_history)[:n]
                    vx_act = list(self.vx_act_history)[:n]
                    vy_act = list(self.vy_act_history)[:n]
                    self.vx_cmd_line.set_data(t_trim, vx_cmd)
                    self.vy_cmd_line.set_data(t_trim, vy_cmd)
                    self.vx_act_line.set_data(t_trim, vx_act)
                    self.vy_act_line.set_data(t_trim, vy_act)
                    self.ax_vel.relim(); self.ax_vel.autoscale_view(True,True,True)
            except Exception:
                traceback.print_exc()

            # 6) LiDAR histogram (robust)
            try:
                all_ranges = np.concatenate([r for r in self.lidar_ranges_hist if r is not None and r.size>0]) if len(self.lidar_ranges_hist)>0 else np.array([])
                self.ax_hist.clear()
                if all_ranges.size>0:
                    self.ax_hist.hist(all_ranges, bins=self.hist_bins)
                else:
                    self.ax_hist.text(0.5, 0.5, 'No LiDAR points', ha='center', va='center')
                self.ax_hist.set_xlim(0, LIDAR_MAX_RANGE)
                self.ax_hist.set_title('LiDAR Ranges Histogram')
            except Exception:
                traceback.print_exc()

            # refresh canvas
            try:
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
            except Exception:
                traceback.print_exc()

        except Exception:
            print("Unexpected plotting error (caught):")
            traceback.print_exc()

# --------------- Mission loop -------------------
def run_mission(goal_xy):
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    client.armDisarm(True)
    print("[mission] takeoff")
    client.takeoffAsync().join()
    client.moveToZAsync(float(DRONE_ALT), 1.0).join()

    nav = DashboardNavigator(client)
    t_start = time.time()

    try:
        while time.time() - t_start < TIMEOUT:
            nav.build_grid_from_lidar()

            need_plan = (nav.path_world is None or len(nav.path_world)==0
                         or time.time() - nav.last_plan_time > REPLAN_INTERVAL
                         or nav.nearest_obstacle_distance() < LOCAL_SAFETY_DIST)
            if need_plan:
                path = nav.plan_path(goal_xy)
                if path is None:
                    client.moveByVelocityAsync(float(0), float(0), float(0), float(1)).join()
                    time.sleep(0.5)
                    continue
            else:
                path = nav.path_world

            ok = nav.follow_path(goal_xy)
            nav.update_plots(goal_xy)

            pos = client.getMultirotorState().kinematics_estimated.position
            dist_to_goal = math.hypot(pos.x_val - goal_xy[0], pos.y_val - goal_xy[1])
            if ok and dist_to_goal < WAYPOINT_TOL:
                print("[mission] reached goal! smooth landing...")
                client.moveToZAsync(float(DRONE_ALT)+2, 1.0).join()
                client.moveToZAsync(float(-1), 0.6).join()
                client.landAsync().join()
                client.armDisarm(False)
                client.enableApiControl(False)
                if PLOT_ENABLED:
                    plt.ioff(); plt.show()
                return True

            time.sleep(0.02)

    finally:
        print("[mission] landing (fallback)")
        try:
            client.moveToZAsync(float(-1), 1.0).join()
            client.landAsync().join()
        except Exception:
            pass
        client.armDisarm(False)
        client.enableApiControl(False)
        if PLOT_ENABLED:
            plt.ioff(); plt.show()
    return False

# ---------------- Entry ----------------
if __name__ == "__main__":
    TARGET_X = 10.0
    TARGET_Y = 8.0
    print("Starting mission to:", (TARGET_X, TARGET_Y))
    ok = run_mission((TARGET_X, TARGET_Y))
    print("Mission finished. Success =", ok)
