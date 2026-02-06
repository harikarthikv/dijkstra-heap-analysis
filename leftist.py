"""
Leftist Heap implementation of Dijkstra's algorithm.
"""

from dijkstra import load_graph_from_file, run_dijkstra_benchmark


class LeftistHeapNode:
    """Node in Leftist Heap."""
    
    def __init__(self, key, vertex):
        self.key = key
        self.vertex = vertex
        self.left = None
        self.right = None
        self.npl = 0  # Null Path Length


class LeftistHeap:
    """Leftist heap for efficient priority queue operations."""
    
    def __init__(self):
        self.root = None

    def merge(self, h1, h2):
        """Merge two leftist heaps."""
        if not h1:
            return h2
        if not h2:
            return h1
        if h1.key > h2.key:
            h1, h2 = h2, h1
        h1.right = self.merge(h1.right, h2)
        if not h1.left or (h1.left.npl < h1.right.npl):
            h1.left, h1.right = h1.right, h1.left
        h1.npl = 0 if not h1.right else h1.right.npl + 1
        return h1

    def insert(self, key, vertex):
        """Insert key-vertex pair into heap."""
        new_node = LeftistHeapNode(key, vertex)
        self.root = self.merge(self.root, new_node)

    def extract_min(self):
        """Extract and return minimum element."""
        if not self.root:
            return None
        min_node = self.root
        self.root = self.merge(self.root.left, self.root.right)
        return min_node


if __name__ == "__main__":
    graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
    print("Running Dijkstra with Leftist Heap...")
    
    # Custom implementation for Leftist Heap
    import time
    leftist_heap = LeftistHeap()
    leftist_heap.insert(0, src)
    
    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0
    
    start_time = time.time()
    
    while leftist_heap.root:
        min_node = leftist_heap.extract_min()
        current_dist, u = min_node.key, min_node.vertex

        if current_dist > distances[u]:
            continue

        for v, weight in graph.edges[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                leftist_heap.insert(distances[v], v)
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    
    from dijkstra import save_results
    save_results(distances, execution_time, r'dataset/leftist_op.txt', 'Leftist Heap')
    print("Output written to leftist_op.txt")
