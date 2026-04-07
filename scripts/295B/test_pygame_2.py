import heapq
import math
import sys
import time

import numpy as np
import pygame

# ========================= v8 CONFIG (Better Path Following) =========================
WIDTH, HEIGHT = 1000, 700
PIXELS_PER_METER = 10.0
DRONE_RADIUS = 11
MAX_LIDAR_RANGE = 80.0
NUM_LIDAR_RAYS = 180

OBSTACLE_THRESHOLD = 16.0
GOAL_REACHED_THRESHOLD = 5.0  # slightly larger
NORMAL_SPEED = 10.0
ESCAPE_SPEED = 7.5

BG_COLOR = (5, 5, 20)
DRONE_COLOR = (0, 255, 120)
ESCAPE_COLOR = (255, 140, 0)
GOAL_COLOR = (255, 60, 60)
OBSTACLE_COLOR = (90, 90, 130)
OBSTACLE_BORDER = (220, 100, 100)
LIDAR_COLOR = (100, 230, 255)  # original small cyan dots
PATH_COLOR = (80, 180, 255)
REACTIVE_COLOR = (255, 200, 80)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption(
    "LiDAR Drone v8 - Improved Path Following + Simple LiDAR Dots"
)
clock = pygame.time.Clock()
font = pygame.font.SysFont("consolas", 18)

drone_pos = np.array([8.0, 35.0])
goal = np.array([92.0, 35.0])

obstacles_m = [
    (25.0, 15.0, 10.0, 35.0),
    (50.0, 8.0, 8.0, 45.0),
    (72.0, 40.0, 12.0, 18.0),
    (38.0, 52.0, 18.0, 8.0),
]

obstacles = [
    pygame.Rect(
        int(ox * PIXELS_PER_METER),
        int(oy * PIXELS_PER_METER),
        int(ow * PIXELS_PER_METER),
        int(oh * PIXELS_PER_METER),
    )
    for ox, oy, ow, oh in obstacles_m
]

GRID_RES = 1.5


def world_to_screen(p):
    return (int(p[0] * PIXELS_PER_METER), int(p[1] * PIXELS_PER_METER))


def is_in_obstacle(pos):
    test_rect = pygame.Rect(
        int(pos[0] * PIXELS_PER_METER - DRONE_RADIUS - 10),
        int(pos[1] * PIXELS_PER_METER - DRONE_RADIUS - 10),
        int(DRONE_RADIUS * 2 + 20),
        int(DRONE_RADIUS * 2 + 20),
    )
    for obs in obstacles:
        if obs.colliderect(test_rect):
            return True
    return False


def heuristic(a, b):
    return np.hypot(a[0] - b[0], a[1] - b[1])


def a_star(start, goal_pos):
    # ... (same as v7 - copy the full function from previous script)
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
            if not (
                0 <= neigh[0] < int(WIDTH / PIXELS_PER_METER / GRID_RES)
                and 0 <= neigh[1] < int(HEIGHT / PIXELS_PER_METER / GRID_RES)
            ):
                continue
            if is_in_obstacle(np.array(neigh) * GRID_RES):
                continue
            tent_g = g_score[current] + (1.414 if dx and dy else 1.0)
            if neigh not in g_score or tent_g < g_score[neigh]:
                came_from[neigh] = current
                g_score[neigh] = tent_g
                f_score[neigh] = tent_g + heuristic(neigh, goal_g)
                heapq.heappush(open_set, (f_score[neigh], neigh))
    return None


def simulate_lidar(pos):
    points = []
    for i in range(NUM_LIDAR_RAYS):
        angle = i * (2 * math.pi / NUM_LIDAR_RAYS)
        direction = np.array([math.cos(angle), math.sin(angle)])
        min_dist = MAX_LIDAR_RANGE
        for obs in obstacles:
            for t in np.linspace(0.0, 1.0, 30):
                test_p = pos + t * MAX_LIDAR_RANGE * direction
                test_rect = pygame.Rect(
                    test_p[0] * PIXELS_PER_METER - 3,
                    test_p[1] * PIXELS_PER_METER - 3,
                    6,
                    6,
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
lidar_points = np.zeros((NUM_LIDAR_RAYS, 2))
running = True
start_time = time.time()
path_length = 0.0
near_misses = 0
escape_timer = 0

print("=== v8: Improved Path Following with Cross-Track Correction ===")
print("Drone should now stay much closer to the blue A* line.\n")

while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT or (
            event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE
        ):
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            goal = np.array(pygame.mouse.get_pos()) / PIXELS_PER_METER
            path = None
            target_waypoint_idx = 0
            escape_timer = 0
            print(f"New goal: ({goal[0]:.1f}, {goal[1]:.1f})")

    if path is None or target_waypoint_idx >= len(path) - 1 or escape_timer > 0:
        path = a_star(drone_pos, goal)
        target_waypoint_idx = 1 if path and len(path) > 1 else 0

    reactive = False
    if path and target_waypoint_idx < len(path):
        # === IMPROVED PATH FOLLOWING ===
        current_target = path[target_waypoint_idx]
        # Find closest point on the current segment to drone (cross-track correction)
        if target_waypoint_idx > 0:
            prev = path[target_waypoint_idx - 1]
            seg_vec = current_target - prev
            seg_len = np.linalg.norm(seg_vec)
            if seg_len > 0.01:
                t = np.clip(
                    np.dot(drone_pos - prev, seg_vec) / (seg_len * seg_len), 0.0, 1.0
                )
                closest_on_path = prev + t * seg_vec
            else:
                closest_on_path = current_target
        else:
            closest_on_path = current_target

        direction = closest_on_path - drone_pos
        dist_to_path = np.linalg.norm(direction)

        if dist_to_path < GOAL_REACHED_THRESHOLD:
            target_waypoint_idx += 1
        else:
            lidar_points = simulate_lidar(drone_pos)
            rel_points = lidar_points - drone_pos
            angles = np.arctan2(rel_points[:, 1], rel_points[:, 0])
            forward_mask = np.abs(angles) < np.deg2rad(55)
            min_forward = MAX_LIDAR_RANGE
            if np.any(forward_mask):
                min_forward = np.min(np.linalg.norm(rel_points[forward_mask], axis=1))

            if min_forward < OBSTACLE_THRESHOLD or escape_timer > 0:
                reactive = True
                perp = np.array([-direction[1], direction[0]])
                strength = (
                    0.75 if escape_timer > 0 else 0.35
                )  # gentler when following path
                direction = (1 - strength) * direction + strength * perp

            direction /= np.linalg.norm(direction) + 1e-8
            speed = ESCAPE_SPEED if escape_timer > 0 else NORMAL_SPEED
            drone_pos += direction * speed * dt
            path_length += speed * dt

            if escape_timer > 0:
                escape_timer -= 1

    # Rendering - original simple LiDAR dots
    screen.fill(BG_COLOR)
    for obs in obstacles:
        pygame.draw.rect(screen, OBSTACLE_COLOR, obs)
        pygame.draw.rect(screen, OBSTACLE_BORDER, obs, width=6)

    for p in lidar_points:
        pygame.draw.circle(screen, LIDAR_COLOR, world_to_screen(p), 2)

    if path:
        col = REACTIVE_COLOR if reactive else PATH_COLOR
        for i in range(len(path) - 1):
            p1 = world_to_screen(path[i])
            p2 = world_to_screen(path[i + 1])
            pygame.draw.line(screen, col, p1, p2, 3)

    pygame.draw.circle(screen, GOAL_COLOR, world_to_screen(goal), 14)
    pygame.draw.circle(screen, (255, 255, 255), world_to_screen(goal), 14, 3)

    ds = world_to_screen(drone_pos)
    drone_col = ESCAPE_COLOR if escape_timer > 0 else DRONE_COLOR
    pygame.draw.circle(screen, drone_col, ds, DRONE_RADIUS)
    pygame.draw.circle(screen, (255, 255, 255), ds, DRONE_RADIUS + 2, 2)

    elapsed = time.time() - start_time
    status = (
        f"Time: {elapsed:.1f}s | Path: {path_length:.1f}m | Near-misses: {near_misses}"
    )
    screen.blit(font.render(status, True, (220, 220, 220)), (12, 12))

    pygame.display.flip()

print("\nSimulation ended.")
print(f"Total time: {elapsed:.1f}s | Path length: {path_length:.1f}m")
pygame.quit()
sys.exit()
