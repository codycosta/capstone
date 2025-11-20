import heapq
import math


class AStarPlanner:
    def __init__(self, occupancy_grid, resolution=1.0, diag_cost=1.414):
        """
        occupancy_grid: 2D list or numpy array of 0 (free) and 1 (occupied)
        resolution: meters per grid cell
        diag_cost: cost for diagonal moves
        """
        self.grid = occupancy_grid
        self.res = resolution
        self.diag = diag_cost

        self.rows = len(self.grid)
        self.cols = len(self.grid[0])

        # 8-directional moves
        self.moves = [
            (1, 0, 1.0),
            (-1, 0, 1.0),
            (0, 1, 1.0),
            (0, -1, 1.0),
            (1, 1, self.diag),
            (1, -1, self.diag),
            (-1, 1, self.diag),
            (-1, -1, self.diag),
        ]

    def in_bounds(self, r, c):
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_free(self, r, c):
        return self.grid[r][c] == 0

    def heuristic(self, r, c, goal_r, goal_c):
        # Euclidean distance
        return math.sqrt((goal_r - r)**2 + (goal_c - c)**2)

    def plan(self, start, goal):
        """
        start: (row, col)
        goal: (row, col)
        returns: list of (row, col) if path found, else None
        """
        sr, sc = start
        gr, gc = goal

        open_set = []
        heapq.heappush(open_set, (0, sr, sc))

        came_from = {}
        g_score = {(sr, sc): 0}

        while open_set:
            _, r, c = heapq.heappop(open_set)

            if (r, c) == (gr, gc):
                return self.reconstruct_path(came_from, (gr, gc))

            for dr, dc, move_cost in self.moves:
                nr, nc = r + dr, c + dc

                if not self.in_bounds(nr, nc):
                    continue
                if not self.is_free(nr, nc):
                    continue

                tentative_g = g_score[(r, c)] + move_cost

                if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                    g_score[(nr, nc)] = tentative_g
                    f = tentative_g + self.heuristic(nr, nc, gr, gc)
                    heapq.heappush(open_set, (f, nr, nc))
                    came_from[(nr, nc)] = (r, c)

        return None  # no path found

    def reconstruct_path(self, came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
