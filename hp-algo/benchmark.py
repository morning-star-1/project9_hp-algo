import time
from statistics import mean
from pathfinding import astar_grid, make_random_grid, manhattan, euclidean


def run_suite(
    rows: int = 200,
    cols: int = 200,
    obstacle_p: float = 0.25,
    trials: int = 20,
    seed: int = 123,
) -> None:
    configs = [
        ("A* Manhattan", manhattan, 1.0),
        ("A* Euclidean", euclidean, 1.0),
        ("Weighted A* (w=1.5) Manhattan", manhattan, 1.5),
        ("Weighted A* (w=2.0) Manhattan", manhattan, 2.0),
    ]

    results = {name: {"ms": [], "expanded": [], "success": 0, "cost": []} for name, _, _ in configs}

    for t in range(trials):
        grid = make_random_grid(rows, cols, obstacle_p, seed + t)
        start = (0, 0)
        goal = (rows - 1, cols - 1)

        # Ensure endpoints are traversable
        grid[start[0]][start[1]] = 0
        grid[goal[0]][goal[1]] = 0

        for name, h, w in configs:
            t0 = time.perf_counter()
            res = astar_grid(grid, start, goal, heuristic=h, weight=w)
            dt = (time.perf_counter() - t0) * 1000.0

            results[name]["ms"].append(dt)
            results[name]["expanded"].append(res.expanded)
            if res.found:
                results[name]["success"] += 1
                results[name]["cost"].append(res.cost)

    print(f"Grid: {rows}x{cols}, obstacle_p={obstacle_p}, trials={trials}\n")
    print(f"{'Config':38} {'avg ms':>10} {'avg expanded':>14} {'success':>10} {'avg cost(found)':>16}")
    print("-" * 96)

    for name in results:
        ms = results[name]["ms"]
        ex = results[name]["expanded"]
        succ = results[name]["success"]
        cost = results[name]["cost"]

        avg_cost = mean(cost) if cost else float("nan")
        print(f"{name:38} {mean(ms):10.2f} {mean(ex):14.1f} {succ:10d} {avg_cost:16.2f}")


if __name__ == "__main__":
    run_suite()
