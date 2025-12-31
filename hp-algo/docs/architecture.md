# Architecture

## Overview
- A* and Weighted A* implementations
- Benchmark runner generates random grids and compares runs

## Data flow
Grid generator -> pathfinding algorithm -> metrics output

## Key decisions
- Use arrays for performance
- Deterministic random seeds for repeatable benchmarks
