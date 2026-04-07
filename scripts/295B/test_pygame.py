import heapq
import math
import sys
import time

import numpy as np
import pygame

# ========================= CONFIG =========================
WIDTH, HEIGHT = 1000, 700
PIXELS_PER_METER = 10.0
DRONE_RADIUS = 10
MAX_LIDAR_RANGE = 40.0
NUM_LIDAR_RAYS = 180
OBSTACLE_THRESHOLD = 12.0
GOAL_REACHED_THRESHOLD = 4.0
NEAR_MISS_DISTANCE = 3.0

# Low-light theme
BG_COLOR = (5, 5, 20)
DRONE_COLOR = (0, 255, 120)
GOAL_COLOR = (255, 60, 60)
OBSTACLE_COLOR = (90, 90, 130)
OBSTACLE_BORDER = (180, 200, 255)
LIDAR_COLOR = (100, 230, 255)
PATH_COLOR = (80, 180, 255)
REACTIVE_COLOR = (255, 140, 60)
WARNING_COLOR = (255, 80, 80)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("LiDAR Drone Navigation - Fixed (Capstone POC)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

# World setup (meters)
drone_pos = np.array([8.0, 35.0])
goal = np.array([92.0, 35.0])

# Obstacles in METERS (x, y, width, height) — clearly visible
obstacles_m = [
    (25.0, 15.0, 10.0, 35.0),  # Tall left
    (50.0, 8.0, 8.0, 45.0),  # Center tall
    (72.0, 40.0, 12.0, 18.0),  # Right lower
    (38.0, 52.0, 18.0, 8.0),  # Bottom
]

# Convert to pygame.Rect (pixels)
obstacles = []
for ox, oy, ow, oh in obstacles_m:
    obstacles.append(
        pygame.Rect(
            int(ox * PIXELS_PER_METER),
            int(oy * PIXELS_PER_METER),
            int(ow * PIXELS_PER_METER),
            int(oh * PIXELS_PER_METER),
        )
    )

GRID_RES = 2.0
GRID_W = int(WIDTH / PIXELS_PER_METER / GRID_RES)
GRID_H = int(HEIGHT / PIXELS_PER_METER / GRID_RES)


def world_to_screen(p):
    return (int(p[0] * PIXELS_PER_METER), int(p[1] * PIXELS_PER_METER))


def point_in_obstacle(p):
    test_rect = pygame.Rect(
        p[0] * PIXELS_PER_METER - 2, p[1] * PIXELS_PER_METER - 2, 4, 4
    )
    for obs in obstacles:
        if obs.colliderect(test_rect.inflate(6, 6)):
            return True
    return False


# ========================= A* =========================
def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def a_star(start, goal_pos):
    start_g = (int(start[0] / GRID_RES), int(start[1] / GRID_RES))
    goal_g = (int(goal_pos[0] / GRID_RES), int(goal_pos[1] / GRID_RES))
    open_set = []
    heapq.heappush(open_set, (0.0, start_g))
    came_from = {}
    g_score = {start_g: 0.0}
    f_score = {start_g: heuristic(start_g, goal_g)}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal_g:
            path = []
            while current in came_from:
                path.append(np.array(current) * GRID_RES)
                current = came_from[current]
            path.append(np.array(start_g) * GRID_RES)
            path.reverse()
            return path
        for dx, dy in [
            (0, 1),
            (1, 0),
            (0, -1),
            (-1, 0),
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        ]:
            neigh = (current[0] + dx, current[1] + dy)
            if not (0 <= neigh[0] < GRID_W and 0 <= neigh[1] < GRID_H):
                continue
            if point_in_obstacle(np.array(neigh) * GRID_RES):
                continue
            tent_g = g_score[current] + (1.414 if dx and dy else 1.0)
            if neigh not in g_score or tent_g < g_score[neigh]:
                came_from[neigh] = current
                g_score[neigh] = tent_g
                f_score[neigh] = tent_g + heuristic(neigh, goal_g)
                heapq.heappush(open_set, (f_score[neigh], neigh))
    return None


# ========================= Simulated LiDAR =========================
def simulate_lidar(pos):
    points = []
    for i in range(NUM_LIDAR_RAYS):
        angle = i * (2 * math.pi / NUM_LIDAR_RAYS)
        direction = np.array([math.cos(angle), math.sin(angle)])
        end = pos + MAX_LIDAR_RANGE * direction

        min_dist = MAX_LIDAR_RANGE
        for obs in obstacles:
            # Sample along the ray for intersection
            for t in np.linspace(0.0, 1.0, 25):
                test_p = pos + t * MAX_LIDAR_RANGE * direction
                test_rect = pygame.Rect(
                    test_p[0] * PIXELS_PER_METER - 2,
                    test_p[1] * PIXELS_PER_METER - 2,
                    4,
                    4,
                )
                if obs.colliderect(test_rect):
                    dist = t * MAX_LIDAR_RANGE
                    if dist < min_dist:
                        min_dist = dist
                    break
        hit_point = pos + (min_dist / MAX_LIDAR_RANGE) * direction
        points.append(hit_point)
    return np.array(points)


# ========================= MAIN LOOP =========================
path = None
target_waypoint_idx = 0
lidar_points = np.zeros((NUM_LIDAR_RAYS, 2))  # safe default to prevent NameError
running = True
start_time = time.time()
path_length = 0.0
near_misses = 0

print("=== LiDAR Drone Simulator (Fixed) ===")
print("Obstacles are now clearly visible as purple blocks with bright borders.")
print("Click anywhere to set new goal • Watch drone avoid obstacles")
print("Red warning = reactive avoidance active\n")

while running:
    dt = clock.tick(40) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            mouse_pos = np.array(pygame.mouse.get_pos()) / PIXELS_PER_METER
            goal = mouse_pos
            path = None
            print(f"→ New goal set: ({goal[0]:.1f}, {goal[1]:.1f})")

    # === Planning & Control ===
    reactive = False
    if path is None or target_waypoint_idx >= len(path) - 1:
        path = a_star(drone_pos, goal)
        target_waypoint_idx = 1 if path and len(path) > 1 else 0
        if path:
            print(f"Replanned A* path ({len(path)} waypoints)")

    if path and target_waypoint_idx < len(path):
        target = path[target_waypoint_idx]
        direction = target - drone_pos
        dist_to_target = np.linalg.norm(direction)

        if dist_to_target < GOAL_REACHED_THRESHOLD:
            target_waypoint_idx += 1
        else:
            # Get fresh LiDAR scan
            lidar_points = simulate_lidar(drone_pos)

            # Check forward cone for obstacles
            rel_points = lidar_points - drone_pos
            angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
            forward_mask = np.abs(angles) < np.deg2rad(50)

            if np.any(forward_mask):
                forward_dists = np.linalg.norm(rel_points[forward_mask], axis=1)
                min_forward = np.min(forward_dists)

                if min_forward < OBSTACLE_THRESHOLD:
                    reactive = True
                    # Reactive avoidance: bias direction right
                    perp = np.array([-direction[1], direction[0]])
                    direction = 0.55 * direction + 0.85 * perp
                    print(
                        f"Obstacle detected ({min_forward:.1f}m) → Reactive avoidance"
                    )

            direction /= np.linalg.norm(direction) + 1e-8
            speed = 18.0
            drone_pos += direction * speed * dt
            path_length += speed * dt

            if "min_forward" in locals() and min_forward < NEAR_MISS_DISTANCE:
                near_misses += 1

    # === Rendering ===
    screen.fill(BG_COLOR)

    # Draw obstacles (very visible now)
    for obs in obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, obs)
        pygame.draw.rect(screen, OBSTACLE_BORDER, obs, width=5)

    # Draw LiDAR points (cyan)
    for p in lidar_points:
        sp = world_to_screen(p)
        pygame.draw.circle(screen, LIDAR_COLOR, sp, 2)

    # Draw planned path
    if path:
        path_color = REACTIVE_COLOR if reactive else PATH_COLOR
        for i in range(len(path) - 1):
            p1 = world_to_screen(path[i])
            p2 = world_to_screen(path[i + 1])
            pygame.draw.line(screen, path_color, p1, p2, 3)

    # Draw goal
    pygame.draw.circle(screen, GOAL_COLOR, world_to_screen(goal), 14)
    pygame.draw.circle(screen, (255, 255, 255), world_to_screen(goal), 14, 3)

    # Draw drone
    ds = world_to_screen(drone_pos)
    pygame.draw.circle(screen, DRONE_COLOR, ds, DRONE_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), ds, DRONE_RADIUS + 2, 2)

    # HUD
    elapsed = time.time() - start_time
    status = (
        f"Time: {elapsed:.1f}s | Path: {path_length:.1f}m | Near-misses: {near_misses}"
    )
    screen.blit(font.render(status, True, (220, 220, 220)), (12, 12))

    if reactive:
        warn = font.render("⚠ OBSTACLE AVOIDANCE ACTIVE ⚠", True, WARNING_COLOR)
        screen.blit(warn, (WIDTH // 2 - 190, 12))

    pygame.display.flip()

print("\n=== Simulation Ended ===")
print(
    f"Total time: {elapsed:.1f}s | Path length: {path_length:.1f}m | Near-misses: {near_misses}"
)
pygame.quit()
sys.exit()
