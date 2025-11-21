# planner_airsim_px4_FIXED.py
import airsim
import asyncio
import numpy as np
from mavsdk import System
from mavsdk.offboard import VelocityNedYaw, OffboardError
from astar import AStarPlanner  # your astar.py is perfect, just needs grid fix


class Planner:
    def __init__(self):
        self.vx_filt = self.vy_filt = self.vz_filt = 0.0
        self.grid_res = 1.0
        self.grid_size = 80
        self.grid_half = self.grid_size // 2
        self.max_speed = 3.0

        self.drone = System()
        self.client = airsim.MultirotorClient()
        self.running = True

    async def connect_all(self):
        self.client.confirmConnection()
        self.client.enableApiControl(True, "Drone1")
        print("AirSim connected")

        await self.drone.connect(system_address="udp://:14540")
        print("Waiting for PX4...")
        async for state in self.drone.core.connection_state():
            if state.is_connected:
                print("PX4 connected")
                break

        # CRITICAL: start offboard heartbeat immediately
        asyncio.create_task(self.offboard_heartbeat())

        async for health in self.drone.telemetry.health():
            if health.is_global_position_ok:
                print("Health OK")
                break

        await self.drone.offboard.start()
        print("Offboard mode started")

        await self.drone.action.arm()
        await self.drone.action.takeoff()
        await asyncio.sleep(6)

    async def offboard_heartbeat(self):
        """ 20 Hz persistent velocity setpoint — this is the real magic """
        while self.running:
            await self.drone.offboard.set_velocity_ned(VelocityNedYaw(
                north_m_s=self.vy_filt,
                east_m_s=self.vx_filt,
                down_m_s=self.vz_filt,
                yaw_deg=0
            ))
            await asyncio.sleep(0.05)

    def get_lidar_grid(self):
        data = self.client.getLidarData(lidar_name="LidarSensor1", vehicle_name="Drone1")
        if len(data.point_cloud) < 3:
            return np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)

        pts = np.array(data.point_cloud, dtype=np.float32).reshape(-1, 3)
        grid = np.zeros((self.grid_size, self.grid_size), dtype=np.uint8)
        offset = self.grid_half

        for x_east, y_north, _ in pts:
            gx = int(round(x_east / self.grid_res)) + offset
            gy = int(round(y_north / self.grid_res)) + offset
            if 0 <= gx < self.grid_size and 0 <= gy < self.grid_size:
                grid[gy, gx] = 1  # FIXED: row = north, col = east

        return grid

    async def fly_to_waypoint(self, wp):
        grid = self.get_lidar_grid()

        pose = self.client.getMultirotorState().kinematics_estimated.position
        start = (
            int(pose.y_val / self.grid_res) + self.grid_half,  # north → row
            int(pose.x_val / self.grid_res) + self.grid_half   # east → col
        )
        goal = (
            int(wp[1] / self.grid_res) + self.grid_half,  # north → row
            int(wp[0] / self.grid_res) + self.grid_half   # east → col
        )

        planner = AStarPlanner(grid)
        path = planner.plan(start, goal)

        if not path:
            print("No path — hovering")
            self.vx_filt = self.vy_filt = self.vz_filt = 0.0
            return

        # Convert back to world
        for r, c in path[1:]:  # skip start
            target_x = (c - self.grid_half) * self.grid_res
            target_y = (r - self.grid_half) * self.grid_res
            await self.goto_point(target_x, target_y, -wp[2])  # negative Z = up

    async def goto_point(self, x_east, y_north, z_up):
        pose = self.client.getMultirotorState().kinematics_estimated.position
        dx = x_east - pose.x_val
        dy = y_north - pose.y_val
        dz = -(z_up + pose.z_val)  # AirSim z is negative down

        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        if dist < 0.5:
            return

        Kp = 0.8
        vx = np.clip(Kp * dx, -self.max_speed, self.max_speed)
        vy = np.clip(Kp * dy, -self.max_speed, self.max_speed)
        vz = np.clip(Kp * dz, -self.max_speed, self.max_speed)

        # EMA smoothing
        alpha = 0.3
        self.vx_filt = alpha * vx + (1 - alpha) * self.vx_filt
        self.vy_filt = alpha * vy + (1 - alpha) * self.vy_filt
        self.vz_filt = alpha * vz + (1 - alpha) * self.vz_filt


async def main():
    p = Planner()
    await p.connect_all()

    WAYPOINTS = [(0,0,-4), (10,0,-4), (10,10,-4), (0,10,-4), (0,0,-3)]
    for wp in WAYPOINTS:
        print(f"Flying to {wp}")
        await p.fly_to_waypoint(wp)
        await asyncio.sleep(2)

    print("Mission complete — landing")
    await p.drone.action.land()
    p.running = False
    await asyncio.sleep(10)


if __name__ == "__main__":
    asyncio.run(main())