# astar.py — FINAL PRODUCTION VERSION (Nov 2025)
import heapq
import math
import random

class AStarPlanner:
    def __init__(self, occupancy_grid, resolution=1.0):
        self.grid = occupancy_grid
        self.res = resolution
        self.rows, self.cols = occupancy_grid.shape

        # 8-connected moves: (dr, dc, cost)
        self.moves = [
            (1, 0, 1.0), (-1, 0, 1.0), (0, 1, 1.0), (0, -1, 1.0),
            (1, 1, 1.415), (1, -1, 1.415), (-1, 1, 1.415), (-1, -1, 1.415),
        ]

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r, c):
        return self.grid[r, c] == 0

    def heuristic(self, a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def is_diagonal_blocked(self, r, c, dr, dc):
        """ Prevent corner-cutting for diagonal moves """
        if abs(dr) + abs(dc) == 2:  # diagonal
            if self.grid[r + dr][c] or self.grid[r][c + dc]:
                return True
        return False

    def plan(self, start, goal):
        if not (self.is_free(*start) and self.is_free(*goal)):
            return None

        open_set = []
        heapq.heappush(open_set, (0 + self.heuristic(start, goal), 0, random.random(), start))

        came_from = {}
        g_score = {start: 0}
        visited = set()

        while open_set:
            _, g, _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)

            if current == goal:
                return self.reconstruct_path(came_from, current)

            r, c = current
            for dr, dc, cost in self.moves:
                nr, nc = r + dr, c + dc
                if not self.in_bounds(nr, nc) or not self.is_free(nr, nc):
                    continue
                if self.is_diagonal_blocked(r, c, dr, dc):
                    continue

                tentative_g = g_score[current] + cost
                neighbor = (nr, nc)

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f = tentative_g + self.heuristic(neighbor, goal)
                    # Tiny random tie-breaker → smoother, less grid-aligned paths
                    heapq.heappush(open_set, (f, tentative_g, random.random(), neighbor))

        return None

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()

        # ---- Path pruning (remove collinear points) ----
        if len(path) < 3:
            return path

        pruned = [path[0]]
        for i in range(1, len(path) - 1):
            prev = pruned[-1]
            curr = path[i]
            next_p = path[i + 1]
            # Vector cross-product test for collinearity
            if (curr[0] - prev[0]) * (next_p[1] - prev[1]) != (curr[1] - prev[1]) * (next_p[0] - prev[0]):
                pruned.append(curr)
        pruned.append(path[-1])
        return pruned