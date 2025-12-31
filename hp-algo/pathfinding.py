from __future__ import annotations

from dataclasses import dataclass
import heapq
import math
from typing import Callable, List, Optional, Tuple

Coord = Tuple[int, int]


def manhattan(a: Coord, b: Coord) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def euclidean(a: Coord, b: Coord) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass(frozen=True)
class Result:
    path: Optional[List[Coord]]
    cost: float
    expanded: int
    found: bool


def astar_grid(
    grid: List[List[int]],
    start: Coord,
    goal: Coord,
    heuristic: Callable[[Coord, Coord], float] = manhattan,
    weight: float = 1.0,
) -> Result:
    """
    A* (weight=1.0) and Weighted A* (weight>1.0) on a 4-neighbor grid.

    grid: 0 = free, 1 = blocked
    start/goal: (row, col)
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0

    sr, sc = start
    gr, gc = goal

    if not (0 <= sr < rows and 0 <= sc < cols):
        return Result(None, float("inf"), 0, False)
    if not (0 <= gr < rows and 0 <= gc < cols):
        return Result(None, float("inf"), 0, False)
    if grid[sr][sc] or grid[gr][gc]:
        return Result(None, float("inf"), 0, False)
    if start == goal:
        return Result([start], 0.0, 0, True)

    def idx(r: int, c: int) -> int:
        return r * cols + c

    INF = float("inf")
    n = rows * cols
    gscore = [INF] * n
    parent = [-1] * n
    closed = [False] * n

    start_i = idx(sr, sc)
    gscore[start_i] = 0.0

    heap: List[Tuple[float, int, int, int]] = []
    tie = 0
    heapq.heappush(heap, (heuristic(start, goal) * weight, tie, sr, sc))

    expanded = 0

    push = heapq.heappush
    pop = heapq.heappop

    while heap:
        _, _, r, c = pop(heap)
        i = idx(r, c)

        if closed[i]:
            continue

        closed[i] = True
        expanded += 1

        if (r, c) == goal:
            path: List[Coord] = []
            cur = i
            while cur != -1:
                rr = cur // cols
                cc = cur - rr * cols
                path.append((rr, cc))
                cur = parent[cur]
            path.reverse()
            return Result(path, gscore[i], expanded, True)

        ng = gscore[i] + 1.0

        # Up
        if r > 0 and not grid[r - 1][c]:
            ni = idx(r - 1, c)
            if not closed[ni] and ng < gscore[ni]:
                gscore[ni] = ng
                parent[ni] = i
                tie += 1
                push(heap, (ng + heuristic((r - 1, c), goal) * weight, tie, r - 1, c))

        # Down
        if r < rows - 1 and not grid[r + 1][c]:
            ni = idx(r + 1, c)
            if not closed[ni] and ng < gscore[ni]:
                gscore[ni] = ng
                parent[ni] = i
                tie += 1
                push(heap, (ng + heuristic((r + 1, c), goal) * weight, tie, r + 1, c))

        # Left
        if c > 0 and not grid[r][c - 1]:
            ni = idx(r, c - 1)
            if not closed[ni] and ng < gscore[ni]:
                gscore[ni] = ng
                parent[ni] = i
                tie += 1
                push(heap, (ng + heuristic((r, c - 1), goal) * weight, tie, r, c - 1))

        # Right
        if c < cols - 1 and not grid[r][c + 1]:
            ni = idx(r, c + 1)
            if not closed[ni] and ng < gscore[ni]:
                gscore[ni] = ng
                parent[ni] = i
                tie += 1
                push(heap, (ng + heuristic((r, c + 1), goal) * weight, tie, r, c + 1))

    return Result(None, float("inf"), expanded, False)


def make_random_grid(rows: int, cols: int, obstacle_p: float, seed: int) -> List[List[int]]:
    x = seed & 0xFFFFFFFF

    def rand() -> float:
        nonlocal x
        x = (1664525 * x + 1013904223) & 0xFFFFFFFF
        return x / 2**32

    grid = [[0] * cols for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            grid[r][c] = 1 if rand() < obstacle_p else 0
    return grid

def render_grid(
    grid: List[List[int]],
    path: Optional[List[Coord]] = None,
    start: Optional[Coord] = None,
    goal: Optional[Coord] = None,
    max_rows: int = 40,
    max_cols: int = 80,
) -> str:
    """
    ASCII renderer for small grids.

    Legend:
      # = obstacle
      . = free
      * = path
      S = start
      G = goal
    """
    rows = min(len(grid), max_rows)
    cols = min(len(grid[0]) if grid else 0, max_cols)

    path_set = set(path) if path else set()

    out_lines: List[str] = []
    for r in range(rows):
        line_chars: List[str] = []
        for c in range(cols):
            ch = "#" if grid[r][c] else "."
            if (r, c) in path_set:
                ch = "*"
            if start == (r, c):
                ch = "S"
            if goal == (r, c):
                ch = "G"
            line_chars.append(ch)
        out_lines.append("".join(line_chars))
    return "\n".join(out_lines)


def demo_small(seed: int = 7) -> None:
    """
    Quick demo: generates a small grid, runs A*, prints an ASCII map + metrics.
    """
    rows, cols = 20, 40
    obstacle_p = 0.22
    grid = make_random_grid(rows, cols, obstacle_p, seed)

    start = (0, 0)
    goal = (rows - 1, cols - 1)
    grid[start[0]][start[1]] = 0
    grid[goal[0]][goal[1]] = 0

    res = astar_grid(grid, start, goal, heuristic=manhattan, weight=1.0)

    print(f"found={res.found} cost={res.cost} expanded={res.expanded}")
    print(render_grid(grid, res.path, start=start, goal=goal))


if __name__ == "__main__":
    # Optional: run a tiny demo if you execute pathfinding.py directly
    demo_small()
