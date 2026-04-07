import airsim
import math
import numpy as np

# Scenario 1: Local/Reactive Planning in an Unknown Map (Bug-like algorithm).
# Prerequisites: AirSim environment running with a multirotor(Drone1) in a warehouse map.
# Ensure AirSim settings.json has a LiDAR sensor &quot;LidarSensor1&quot;on Drone1 (see above for example config).
# This script uses ground-truth data from AirSim for droneposition (no GPS, but direct simulator state).
# Connect to AirSim simulator
client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True, vehicle_name='Drone1')
client.armDisarm(True, vehicle_name='Drone1')

# Takeoff and set a reference flight altitude
takeoff_height = -2.0 # fly 2 meters above ground (NED coordinate: negative value is above ground)
client.takeoffAsync(vehicle_name=&quot;Drone1&quot;).join()
client.moveToZAsync(takeoff_height, 1.0,
vehicle_name=&quot;Drone1&quot;).join() # smooth climb to desired altitude
20
21 # Goal position in world (NED) coordinates – this should be
set to the desired destination.
22 # Example: goal at 10m North, 5m East from start position, at
same altitude as takeoff_height.
23 start_state = client.getMultirotorState(vehicle_name=&quot;Drone1&quot;)
24 start_pos = start_state.kinematics_estimated.position
25 goal_position = airsim.Vector3r(start_pos.x_val + 10.0, # 10
m North
26 start_pos.y_val + 5.0, # 5 m
East
27 takeoff_height) #
same altitude (2m above ground)
28
29 # Helper function: parse LiDAR data into numpy array of points
30 def get_lidar_points():
31 lidar_data = client.getLidarData(&quot;LidarSensor1&quot;,
vehicle_name=&quot;Drone1&quot;)
32 if len(lidar_data.point_cloud) &lt; 3:
33 return np.zeros((0,3)) # no points
34 # Convert flat array into Nx3 array of [x, y, z]
coordinates
35 points = np.array(lidar_data.point_cloud,
dtype=np.float32).reshape(-1, 3)
36 return points
37
38 # Helper function: check for obstacle directly in front of the
drone within a given distance
39 def is_obstacle_ahead(points, distance_threshold=3.0,
fov_deg=30):
40 &quot;&quot;&quot;
41 Determines if there&#39;s any obstacle within
distance_threshold in the drone&#39;s forward direction.
42 Assumes LiDAR points are in drone&#39;s **local NED** frame
(SensorLocalFrame or VehicleInertialFrame with known orientation).
43 This example uses VehicleInertialFrame, so we will rotate
points to body frame based on drone&#39;s orientation.
44 &quot;&quot;&quot;

45 if points.shape[0] == 0:
46 return False, None # no points, no obstacle
47 # Get drone&#39;s orientation (to transform inertial to body
frame if needed)
48 drone_state =
client.getMultirotorState(vehicle_name=&quot;Drone1&quot;)
49 # Orientation as quaternion (AirSim uses (w,x,y,z) format)
50 orient = drone_state.kinematics_estimated.orientation
51 # Convert quaternion to yaw (rotation around z). AirSim
NED: yaw 0 = facing North (X-axis).
52 # Yaw calculation reference: yaw = atan2(2*(w*z + x*y), 1
- 2*(y*y + z*z)) in AirSim&#39;s coordinate convention.
53 # Here we use a built-in helper if available, otherwise
compute.
54 # AirSim provides a utility to convert quaternion to
Euler, but we&#39;ll implement yaw extraction for completeness.
55 qw, qx, qy, qz = orient.w_val, orient.x_val, orient.y_val,
orient.z_val
56 # Yaw (in radians)
57 yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
58 # Create rotation matrix for yaw (Z-axis rotation) to
transform points to body frame
59 cos_yaw = math.cos(-yaw) # note: to transform inertial-
&gt;body, use negative yaw
60 sin_yaw = math.sin(-yaw)
61 # Rotate points from inertial frame to body frame
(assuming SensorLocalFrame was set to VehicleInertialFrame)
62 # Body frame: x-forward, y-right, z-down
63 xi = points[:,0] -
drone_state.kinematics_estimated.position.x_val
64 yi = points[:,1] -
drone_state.kinematics_estimated.position.y_val
65 # (No need to adjust z for horizontal obstacle detection)
66 x_body = cos_yaw * xi - sin_yaw * yi
67 y_body = sin_yaw * xi + cos_yaw * yi
68 # Filter points roughly in a forward cone (±fov_deg) in
front of the drone
69 angles = np.degrees(np.arctan2(y_body, x_body)) # angle 0
= front, positive = right, negative = left
70 in_front = np.where(np.abs(angles) &lt; fov_deg)[0]
71 if in_front.size == 0:
72 return False, None

73 # Among points in front sector, find the nearest distance
74 dist_ahead = np.sqrt(x_body[in_front]**2 +
y_body[in_front]**2)
75 min_dist = float(np.min(dist_ahead))
76 if min_dist &lt; distance_threshold:
77 return True, min_dist
78 else:
79 return False, min_dist
80
81 # Helper for choosing avoidance turn direction
82 def choose_avoidance_turn(points):
83 &quot;&quot;&quot;
84 Decide whether to turn left or right to avoid an obstacle,
based on which side has more clearance.
85 Returns an angle (in degrees) to add to current yaw.
86 &quot;&quot;&quot;
87 if points.shape[0] == 0:
88 return 0 # no points, no preference
89 # Compute angles and distances in body frame (reusing part
of logic above)
90 drone_state =
client.getMultirotorState(vehicle_name=&quot;Drone1&quot;)
91 orient = drone_state.kinematics_estimated.orientation
92 qw, qx, qy, qz = orient.w_val, orient.x_val, orient.y_val,
orient.z_val
93 yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))
94 cos_yaw = math.cos(-yaw)
95 sin_yaw = math.sin(-yaw)
96 xi = points[:,0] -
drone_state.kinematics_estimated.position.x_val
97 yi = points[:,1] -
drone_state.kinematics_estimated.position.y_val
98 x_body = cos_yaw * xi - sin_yaw * yi
99 y_body = sin_yaw * xi + cos_yaw * yi
100 angles = np.degrees(np.arctan2(y_body, x_body))
101 # Define left and right sectors (excluding the front 60
deg to avoid counting the obstacle itself)
102 left_sector = np.where((angles &gt; 60) &amp; (angles &lt;=
180))[0]
103 right_sector = np.where((angles &lt; -60) &amp; (angles &gt;= -
180))[0]
104 # Note: Because of angle sign convention (0 front,

positive right),
105 # left obstacles might appear as negative angles (to the
left).
106 # Adjust: treat left side as negative angles, right as
positive angles.
107 left_sector = np.where((angles &lt; -60) &amp; (angles &gt;= -
180))[0]
108 right_sector = np.where((angles &gt; 60) &amp; (angles &lt;=
180))[0]
109 # Compute nearest obstacle distance on each side
110 left_min = np.min(np.sqrt(x_body[left_sector]**2 +
y_body[left_sector]**2)) if left_sector.size &gt; 0 else float(&#39;inf&#39;)
111 right_min = np.min(np.sqrt(x_body[right_sector]**2 +
y_body[right_sector]**2)) if right_sector.size &gt; 0 else
float(&#39;inf&#39;)
112 # Choose the side with greater clearance (larger min
distance) to turn towards
113 if left_min &gt; right_min:
114 return -45 # turn left 45 degrees
115 else:
116 return 45 # turn right 45 degrees
117
118 # Main reactive navigation loop
119 collision_count = 0
120 last_safe_pos =
client.getMultirotorState(vehicle_name=&quot;Drone1&quot;).kinematics_estima
ted.position
121 max_iterations = 500 # safety break to avoid infinite loops
122 for step in range(max_iterations):
123 # Get current state and distance to goal
124 state = client.getMultirotorState(vehicle_name=&quot;Drone1&quot;)
125 cur_pos = state.kinematics_estimated.position
126 dx = goal_position.x_val - cur_pos.x_val
127 dy = goal_position.y_val - cur_pos.y_val
128 dist_to_goal = math.sqrt(dx*dx + dy*dy)
129 if dist_to_goal &lt; 1.0: # goal tolerance &lt; 1m
130 print(f&quot;Goal reached (distance {dist_to_goal:.2f} m).
Landing...&quot;)
131 break
132
133 # Sense environment with LiDAR
134 points = get_lidar_points()

135 # For a mostly flat environment, filter out ground points
(assume ground is near z=0 in world frame)
136 if points.shape[0] &gt; 0:
137 # Remove points close to ground (z &gt; -0.2 in NED
frame, i.e., within 20cm of ground level)
138 points = points[points[:,2] &lt; -0.2]
139
140 # Reactive planning: decide movement based on sensor data
and goal direction
141 obstacle_ahead, min_front = is_obstacle_ahead(points,
distance_threshold=3.0, fov_deg=30)
142 if obstacle_ahead:
143 # Obstacle is within 3m ahead. Decide a new direction
to avoid the obstacle.
144 turn_angle = choose_avoidance_turn(points)
145 # Rotate the drone by the chosen angle (relative
turn)
146 new_yaw = math.degrees(yaw) + turn_angle if &#39;yaw&#39; in
locals() else turn_angle
147 client.rotateToYawAsync(new_yaw,
vehicle_name=&quot;Drone1&quot;).join()
148 print(f&quot;Obstacle detected at {min_front:.1f}m ahead.
Turning {turn_angle} degrees.&quot;)
149 # After turning, we do not move forward this
iteration (we will re-evaluate in next loop).
150 continue
151 else:
152 # No close obstacle in front, proceed towards goal
(or intermediate step towards goal).
153 # Choose a small step towards the goal to allow
reactive adjustments.
154 step_distance = 2.0 # meters to move in this
iteration (tunable)
155 move_x = cur_pos.x_val + (dx/dist_to_goal) *
step_distance if dist_to_goal &gt; step_distance else
goal_position.x_val
156 move_y = cur_pos.y_val + (dy/dist_to_goal) *
step_distance if dist_to_goal &gt; step_distance else
goal_position.y_val
157 # Maintain constant altitude (takeoff_height)
158 move_z = takeoff_height
159 # Command the drone to move to the new position

160 client.moveToPositionAsync(move_x, move_y, move_z,
2.0, vehicle_name=&quot;Drone1&quot;).join()
161 print(f&quot;Moving towards goal: new position =
({move_x:.1f}, {move_y:.1f})&quot;)
162
163 # After movement, check for collision
164 collision_info =
client.simGetCollisionInfo(vehicle_name=&quot;Drone1&quot;)
165 if collision_info.has_collided:
166 collision_count += 1
167 # Log collision details
168 col_pos = collision_info.position
169 print(f&quot;Collision {collision_count}: hit
{collision_info.object_name} at ({col_pos.x_val:.1f},
{col_pos.y_val:.1f}, {col_pos.z_val:.1f}). Retreating...&quot;)
170 # Retreat: move back to last safe position (or simply
step backward)
171 safe = last_safe_pos
172 client.moveToPositionAsync(safe.x_val, safe.y_val,
safe.z_val, 1.0, vehicle_name=&quot;Drone1&quot;).join()
173 # Optionally, you could also try ascending or
stopping here.
174 # After retreating, continue the loop (the drone will
choose a new direction next).
175 continue
176 else:
177 # Update last safe position since we moved without
collision
178 last_safe_pos = cur_pos
179
180 # Land the drone if goal reached or loop ended
181 client.landAsync(vehicle_name=&quot;Drone1&quot;).join()
182 client.armDisarm(False, vehicle_name=&quot;Drone1&quot;)
183 client.enableApiControl(False, vehicle_name=&quot;Drone1&quot;)
184 print(&quot;Scenario 1 complete. Collisions encountered:&quot;,
collision_count)
