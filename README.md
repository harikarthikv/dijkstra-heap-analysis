# Dijkstra's Algorithm - Heap Implementation Analysis

A comparative analysis of Dijkstra's shortest path algorithm using different heap data structures: Binary Heap, AVL Tree, Binomial Heap, Fibonacci Heap, and Leftist Heap.

## Overview

This project implements and benchmarks Dijkstra's algorithm with five different priority queue implementations to analyze their performance characteristics on graph shortest path problems.

## Project Structure

```
dijkstra-heap-analysis/
├── dijkstra.py           # Shared utilities (Graph, load_graph_from_file, save_results)
├── binary.py             # Binary Heap implementation
├── avl.py                # AVL Tree implementation
├── binomial.py           # Binomial Heap implementation
├── fibonnaci.py          # Fibonacci Heap implementation
├── leftist.py            # Leftist Heap implementation
└── dataset/              # Input and output files
```

## Implementations

### Heap Data Structures

1. **Binary Heap** (`binary.py`)
   - Classic binary min-heap using array representation
   - O(log n) insert and extract-min
   - O(1) amortized time for multiple inserts

2. **AVL Tree** (`avl.py`)
   - Self-balancing binary search tree
   - O(log n) insert, delete, and find-min
   - Maintains strict balance (height difference ≤ 1)

3. **Binomial Heap** (`binomial.py`)
   - Collection of binomial trees with heap property
   - O(log n) extract-min, O(1) insert
   - Efficient merge operation

4. **Fibonacci Heap** (`fibonnaci.py`)
   - Lazy binomial queue with deferred consolidation
   - O(1) amortized insert and decrease-key
   - O(log n) amortized extract-min
   - Theoretical optimal for Dijkstra's algorithm

5. **Leftist Heap** (`leftist.py`)
   - Binary heap with leftist property
   - O(log n) merge, insert, and extract-min
   - Simpler than Fibonacci heaps with good practical performance

## Requirements

- Python 3.8+
- [uv](https://github.com/astral-sh/uv) (recommended for fast execution)

## Installation

```bash
# Using uv (recommended)
pip install uv

# Or install Python directly
python3 --version  # Ensure Python 3.8+
```

## Usage

Run any implementation using uv:

```bash
uv run binary.py
uv run avl.py
uv run binomial.py
uv run fibonnaci.py
uv run leftist.py
```

Or with standard Python:

```bash
python binary.py
python avl.py
python binomial.py
python fibonnaci.py
python leftist.py
```

Run all implementations:

```bash
uv run binary.py && uv run avl.py && uv run binomial.py && uv run leftist.py && uv run fibonnaci.py
```

## Dataset

The `dataset/` directory contains:

- **Input**: `dataset_ip.txt` - Graph in DIMACS format (vertices, edges)
- **Output**: `*_op.txt` - Shortest path distances and execution times for each implementation

Format:
- First line: `num_vertices num_edges`
- Edge lines: `a u v weight` (1-based vertex indices)

## Output Format

Each implementation generates an output file with:
- Vertex indices (0-based)
- Shortest distance from source vertex (0)
- Execution time in milliseconds

Example output:
```
Running Dijkstra with Binary Min-Heap...
Vertex  Distance from Source
0		0
1		803
2		50999
...
Execution time: 125.45 ms
```

## Shared Utilities (`dijkstra.py`)

The refactored codebase uses a shared module for common functionality:

- **Graph class**: Adjacency list representation with undirected edges
- **load_graph_from_file()**: Reads DIMACS format graph files
- **save_results()**: Writes results to output files in consistent format
- **dijkstra_algorithm()**: Generic algorithm supporting multiple heap interfaces
- **run_dijkstra_benchmark()**: Runs algorithm and measures execution time

This design promotes:
- **Code reusability**: Eliminates duplication across heap implementations
- **Maintainability**: Bug fixes in common logic benefit all implementations
- **Consistency**: All implementations use identical input/output handling
- **Testability**: Easy to test heap performance in isolation

## Performance Comparison

For detailed performance analysis and comparison across different graph sizes and densities, refer to `report.pdf`.

### Asymptotic Complexity (Dijkstra's Algorithm)

| Heap Type | Insert | Extract Min | Decrease Key | Total Time |
|-----------|--------|-------------|--------------|------------|
| Binary    | O(log n) | O(log n) | O(log n) | O((V + E) log V) |
| AVL Tree  | O(log n) | O(log n) | O(log n) | O((V + E) log V) |
| Binomial  | O(1) | O(log n) | O(log n) | O((V + E) log V) |
| Fibonacci | O(1) | O(log n) | O(1) | O(E + V log V) |
| Leftist   | O(log n) | O(log n) | O(log n) | O((V + E) log V) |

## License

This project is for educational and research purposes.