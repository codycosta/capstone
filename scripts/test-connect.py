# file to test api connection to airsim

import airsim

# connect to AirSim
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

print("Taking off...")
client.takeoffAsync().join()

# Move forward 10 m at 3 m/s
client.moveToPositionAsync(10, 0, -3, 3).join()

# Land
client.landAsync().join()
client.armDisarm(False)
client.enableApiControl(False)

print("Mission complete!")
