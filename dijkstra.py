"""
Shared Dijkstra's Algorithm utilities and Graph class.
This module contains common functionality used across all heap implementations.
"""

import time


class Graph:
    """Undirected weighted graph represented as adjacency list."""
    
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = [[] for _ in range(num_vertices)]

    def add_edge(self, u, v, weight):
        """Add an undirected edge between vertices u and v with given weight."""
        self.edges[u].append((v, weight))
        self.edges[v].append((u, weight))


def load_graph_from_file(file_path):
    """
    Load graph from file in DIMACS format.
    
    Expected format:
    - First line: num_vertices num_edges
    - Subsequent lines: 'a' u v weight (edge format)
    
    Args:
        file_path: Path to the input file
        
    Returns:
        tuple: (Graph object, source vertex index)
    """
    with open(file_path, 'r') as f:
        num_vertices, _ = map(int, f.readline().strip().split())
        src = 0  # Default source vertex
        graph = Graph(num_vertices)
        
        for line in f:
            parts = line.strip().split()
            if parts[0] == 'a':
                u = int(parts[1]) - 1  # Convert to 0-based index
                v = int(parts[2]) - 1
                weight = int(parts[3])
                graph.add_edge(u, v, weight)
    
    return graph, src


def dijkstra_algorithm(heap, graph, src):
    """
    Generic Dijkstra's algorithm implementation.
    
    Args:
        heap: Priority queue object with insert() and extract_min() methods
        graph: Graph object
        src: Source vertex index
        
    Returns:
        list: Shortest distances from source to all vertices
    """
    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0
    heap.insert(0, src)

    # Determine which interface the heap uses
    has_heap_attr = hasattr(heap, 'heap')
    has_is_empty = hasattr(heap, 'is_empty')
    has_root = hasattr(heap, 'root')

    while True:
        # Check if heap is empty based on heap type
        if has_heap_attr:
            if not heap.heap:
                break
        elif has_is_empty:
            if heap.is_empty():
                break
        elif has_root:
            if not heap.root:
                break
        else:
            break
        
        # Extract minimum - handle different return types
        result = heap.extract_min()
        if result is None:
            break
        if isinstance(result, tuple):
            current_distance, u = result
        else:
            current_distance = result.key
            u = result.vertex
        
        if current_distance > distances[u]:
            continue

        for v, weight in graph.edges[u]:
            distance = current_distance + weight
            if distance < distances[v]:
                distances[v] = distance
                heap.insert(distance, v)

    return distances


def save_results(distances, execution_time, output_file, heap_name):
    """
    Save Dijkstra results to output file.
    
    Args:
        distances: List of shortest distances
        execution_time: Execution time in milliseconds
        output_file: Path to output file
        heap_name: Name of heap type for display
    """
    with open(output_file, 'w') as f:
        f.write(f"Running Dijkstra with {heap_name}...\n")
        f.write("Vertex  Distance from Source\n")
        for i, distance in enumerate(distances):
            f.write(f"{i}\t\t{distance}\n")
        f.write(f"Execution time: {execution_time:.2f} ms\n")


def run_dijkstra_benchmark(heap_class, heap_init_args, graph, src, output_file, heap_name):
    """
    Run Dijkstra's algorithm with a specific heap implementation and benchmark it.
    
    Args:
        heap_class: Class of the heap to use
        heap_init_args: Arguments for heap initialization (list or dict)
        graph: Graph object
        src: Source vertex
        output_file: Path to output file
        heap_name: Display name of heap type
        
    Returns:
        tuple: (distances, execution_time_ms)
    """
    # Initialize heap
    if isinstance(heap_init_args, dict):
        heap = heap_class(**heap_init_args)
    else:
        heap = heap_class(*heap_init_args)
    
    # Run and benchmark
    start_time = time.time()
    distances = dijkstra_algorithm(heap, graph, src)
    end_time = time.time()
    
    execution_time = (end_time - start_time) * 1000  # Convert to milliseconds
    
    # Save results
    save_results(distances, execution_time, output_file, heap_name)
    
    return distances, execution_time
