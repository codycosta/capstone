# ================================================================
# planner_airsim_px4_full.py
# Fully Integrated:
#   - AirSim LiDAR + IMU
#   - Occupancy Grid
#   - A* Path Planning
#   - MAVSDK Offboard Velocity Control (PX4)
# ================================================================

import asyncio
import math
import numpy as np
import airsim
from mavsdk import System
from mavsdk.offboard import OffboardError, VelocityNedYaw
from collections import deque

# ===========================
# CONFIGURATION
# ===========================
AIRSIM_VEHICLE_NAME = "Drone1"
LIDAR_NAME = "LidarSensor1"
IMU_NAME = "Imu"

# Local grid config
GRID_RES = 1.0         # meters per cell
GRID_WIDTH = 80        # +/-40m x-y
GRID_HEIGHT = 80

OBSTACLE_INFLATION = 2     # inflate Lidar obstacles
LIDAR_MAX_RANGE = 40

# Controller tuning
KP = 0.6
KD = 0.12
MAX_SPEED = 3.0
MAX_ACCEL = 2.0
EMA_ALPHA = 0.25

# ===========================
# HELPER: Build Occupancy Grid
# ===========================
def make_grid(lidar_points):
    grid = np.zeros((GRID_WIDTH, GRID_HEIGHT), dtype=np.uint8)
    half_w = GRID_WIDTH // 2
    half_h = GRID_HEIGHT // 2

    for p in lidar_points:
        x, y, z = p
        r = math.sqrt(x*x + y*y + z*z)
        if r > LIDAR_MAX_RANGE:
            continue

        gx = int(x / GRID_RES) + half_w
        gy = int(y / GRID_RES) + half_h

        if 0 <= gx < GRID_WIDTH and 0 <= gy < GRID_HEIGHT:
            grid[gx, gy] = 1

    # Inflate obstacles
    inflated = grid.copy()
    for i in range(GRID_WIDTH):
        for j in range(GRID_HEIGHT):
            if grid[i, j] == 1:
                for dx in range(-OBSTACLE_INFLATION, OBSTACLE_INFLATION + 1):
                    for dy in range(-OBSTACLE_INFLATION, OBSTACLE_INFLATION + 1):
                        ix = i + dx
                        iy = j + dy
                        if 0 <= ix < GRID_WIDTH and 0 <= iy < GRID_HEIGHT:
                            inflated[ix, iy] = 1
    return inflated

# ===========================
# A* Planner
# ===========================
def astar(grid, start, goal):
    sx, sy = start
    gx, gy = goal

    if grid[gx, gy] == 1:
        return None

    open_set = [(0, (sx, sy))]
    came_from = {}
    g_score = { (sx, sy): 0 }

    def h(a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    while open_set:
        open_set.sort(key=lambda x: x[0])
        _, current = open_set.pop(0)

        if current == (gx, gy):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.reverse()
            return path

        cx, cy = current
        for dx, dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nx = cx + dx
            ny = cy + dy

            if not (0 <= nx < GRID_WIDTH and 0 <= ny < GRID_HEIGHT):
                continue
            if grid[nx, ny] == 1:
                continue

            new_g = g_score[(cx, cy)] + 1
            if (nx, ny) not in g_score or new_g < g_score[(nx, ny)]:
                g_score[(nx, ny)] = new_g
                f = new_g + h((nx, ny), (gx, gy))
                open_set.append((f, (nx, ny)))
                came_from[(nx, ny)] = (cx, cy)

    return None

# ===========================
# MAVSDK Helper
# ===========================
async def start_offboard(drone):
    try:
        await drone.offboard.start()
        print("Offboard started.")
    except OffboardError as e:
        print(f"Offboard start failed: {e}")
        return False
    return True

# ===========================
# MAIN LOGIC
# ===========================
async def main():
    # -----------------------
    # CONNECT: AirSim client
    # -----------------------
    sim = airsim.MultirotorClient()
    sim.confirmConnection()

    print("Connected to AirSim.")

    # -----------------------
    # CONNECT: PX4 (MAVSDK)
    # -----------------------
    drone = System()
    await drone.connect(system_address="udp://:14540")

    print("Waiting for PX4...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 connected.")
            break

    # Arm + takeoff
    await drone.action.arm()
    await drone.action.takeoff()
    await asyncio.sleep(3)

    # Start Offboard (must send at least one setpoint before start)
    await drone.offboard.set_velocity_ned(VelocityNedYaw(0, 0, 0, 0))
    ok = await start_offboard(drone)
    if not ok:
        return

    # -----------------------
    # GOAL in NED (relative)
    # -----------------------
    goal_north = 15
    goal_east = 15

    prev_vx = 0
    prev_vy = 0

    while True:
        # --------------------------------------------------
        # Read LiDAR
        # --------------------------------------------------
        ld = sim.getLidarData(LIDAR_NAME, AIRSIM_VEHICLE_NAME)
        pts = np.array(ld.point_cloud, dtype=np.float32).reshape(-1, 3) if ld.point_cloud else np.zeros((0, 3))

        grid = make_grid(pts)

        # --------------------------------------------------
        # Get current drone pose
        # --------------------------------------------------
        pose = sim.getMultirotorState(AIRSIM_VEHICLE_NAME).kinematics_estimated
        px = pose.position.x_val
        py = pose.position.y_val

        half_w = GRID_WIDTH // 2
        half_h = GRID_HEIGHT // 2

        gx = int(goal_north / GRID_RES) + half_w
        gy = int(goal_east / GRID_RES) + half_h

        sx = int(px / GRID_RES) + half_w
        sy = int(py / GRID_RES) + half_h

        if not (0 <= sx < GRID_WIDTH and 0 <= sy < GRID_HEIGHT):
            print("Drone outside grid!")
            break

        # --------------------------------------------------
        # PLAN
        # --------------------------------------------------
        path = astar(grid, (sx, sy), (gx, gy))
        if not path:
            print("No valid path.")
            break

        if len(path) > 1:
            tx, ty = path[1]
        else:
            tx, ty = path[0]

        tnx = (tx - half_w) * GRID_RES
        tny = (ty - half_h) * GRID_RES

        # --------------------------------------------------
        # CONTROL
        # --------------------------------------------------
        ex = tnx - px
        ey = tny - py
        dist = math.sqrt(ex*ex + ey*ey)

        raw_vx = KP * ex
        raw_vy = KP * ey

        raw_v = math.sqrt(raw_vx**2 + raw_vy**2)
        if raw_v > MAX_SPEED:
            s = MAX_SPEED / (raw_v + 1e-6)
            raw_vx *= s
            raw_vy *= s

        dvx = raw_vx - prev_vx
        dvy = raw_vy - prev_vy
        dv = math.sqrt(dvx*dvx + dvy*dvy)

        if dv > MAX_ACCEL:
            s = MAX_ACCEL / (dv + 1e-6)
            dvx *= s
            dvy *= s

        vx_cmd = prev_vx + dvx
        vy_cmd = prev_vy + dvy

        # EMA smoothing
        vx_cmd = EMA_ALPHA * vx_cmd + (1 - EMA_ALPHA) * prev_vx
        vy_cmd = EMA_ALPHA * vy_cmd + (1 - EMA_ALPHA) * prev_vy

        prev_vx, prev_vy = vx_cmd, vy_cmd

        # --------------------------------------------------
        # SEND PX4 VELOCITY
        # --------------------------------------------------
        await drone.offboard.set_velocity_ned(
            VelocityNedYaw(
                north_m_s=vx_cmd,
                east_m_s=vy_cmd,
                down_m_s=0,
                yaw_deg=0
            )
        )

        # Goal reached?
        if dist < 1.0:
            print("Reached goal!")
            break

        await asyncio.sleep(0.05)

    print("Landing...")
    await drone.action.land()
    await asyncio.sleep(3)

# --------------------------------------------------
# ENTRY POINT
# --------------------------------------------------
if __name__ == "__main__":
    WAYPOINTS = [
        (0, 0, -4),     # takeoff hover
        (10, 0, -4),    # move forward
        (10, 10, -4),   # right
        (0, 10, -4),    # back
        (0, 0, -3),     # return and descend
    ]
    
    asyncio.run(main())
