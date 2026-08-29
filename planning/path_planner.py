import heapq
import numpy as np
from typing import List, Tuple

class AStarPlanner:
    """
    Grid-based A* global path planner.
    """
    def __init__(self, grid_size: float = 1.0):
        self.grid_size = grid_size
        # 8-connected movement
        self.motions = [
            (1, 0, 1.0), (0, 1, 1.0), (-1, 0, 1.0), (0, -1, 1.0),
            (1, 1, 1.414), (1, -1, 1.414), (-1, 1, 1.414), (-1, -1, 1.414)
        ]

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return np.hypot(a[0] - b[0], a[1] - b[1])

    def plan(self, start: Tuple[float, float], goal: Tuple[float, float], obstacles: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """
        Calculates optimal waypoint path from start to goal avoiding obstacles.
        """
        start_node = (int(round(start[0] / self.grid_size)), int(round(start[1] / self.grid_size)))
        goal_node = (int(round(goal[0] / self.grid_size)), int(round(goal[1] / self.grid_size)))
        
        obs_set = set((int(round(x / self.grid_size)), int(round(y / self.grid_size))) for x, y in obstacles)

        open_set = []
        heapq.heappush(open_set, (0 + self._heuristic(start_node, goal_node), 0, start_node, None))
        came_from = {}
        g_score = {start_node: 0.0}

        while open_set:
            f, cost, current, parent = heapq.heappop(open_set)
            came_from[current] = parent

            if current == goal_node:
                # Reconstruct path
                path = []
                curr = current
                while curr:
                    path.append((curr[0] * self.grid_size, curr[1] * self.grid_size))
                    curr = came_from.get(curr)
                return path[::-1]

            for dx, dy, move_cost in self.motions:
                neighbor = (current[0] + dx, current[1] + dy)
                if neighbor in obs_set:
                    continue
                
                tentative_g = g_score[current] + move_cost
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + self._heuristic(neighbor, goal_node)
                    heapq.heappush(open_set, (f_score, tentative_g, neighbor, current))

        # Fallback straight path if goal unreachable
        return [start, goal]
