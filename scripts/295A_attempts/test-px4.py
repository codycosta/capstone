# test_px4.py  ← 100 % working minimal PX4 + AirSim test (Dec 2025)
import asyncio
from mavsdk import System                 # ← correct import
from mavsdk.offboard import VelocityNedYaw, OffboardError   # ← needed for velocity


async def run():
    drone = System()
    print('connecting to IP address...')
    await drone.connect("udpin://0.0.0.0:14540")

    print("Waiting for PX4 to connect...")
    async for state in drone.core.connection_state():
        if state.is_connected:
            print("PX4 connected!")
            break

    # Persistent 10 Hz offboard heartbeat (required!)
    async def heartbeat():
        while True:
            await drone.offboard.set_velocity_ned(VelocityNedYaw(0.0, 0.0, 0.0, 0.0))
            await asyncio.sleep(0.1)

    asyncio.create_task(heartbeat())

    print("Starting offboard mode...")
    try:
        await drone.offboard.start()
    except OffboardError as error:
        print(f"Offboard failed: {error}")
        return

    print("Arming...")
    await drone.action.arm()

    print("Taking off...")
    await drone.action.takeoff()
    await asyncio.sleep(8)          # give it time to climb

    print("Hovering 10 seconds...")
    await asyncio.sleep(10)

    print("Landing...")
    await drone.action.land()


if __name__ == "__main__":
    asyncio.run(run())