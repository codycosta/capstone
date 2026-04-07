import airsim

client = airsim.MultirotorClient()
try:
    client.confirmConnection()
    print("Connected to AirSim successfully!")
except Exception as e:
    print("Connection failed:", e)
