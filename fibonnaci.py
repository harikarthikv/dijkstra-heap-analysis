"""
Fibonacci Heap implementation of Dijkstra's algorithm.
"""

from dijkstra import load_graph_from_file, save_results
import time
import math


class FibonacciNode:
    """Node in Fibonacci Heap."""
    
    def __init__(self, key, vertex):
        self.key = key
        self.vertex = vertex
        self.parent = None
        self.child = None
        self.left = None
        self.right = None
        self.degree = 0
        self.mark = False


class FibonacciHeap:
    """Fibonacci heap for efficient priority queue operations."""
    
    def __init__(self):
        self.min_node = None
        self.num_nodes = 0

    def insert(self, key, vertex):
        """Insert key-vertex pair into heap."""
        new_node = FibonacciNode(key, vertex)
        if self.min_node is None:
            self.min_node = new_node
            new_node.left = new_node
            new_node.right = new_node
        else:
            self._insert_into_root_list(new_node)
            if new_node.key < self.min_node.key:
                self.min_node = new_node

        self.num_nodes += 1
        return new_node

    def _insert_into_root_list(self, node):
        """Insert node into root list."""
        if self.min_node is None:
            node.left = node
            node.right = node
            self.min_node = node
        else:
            node.left = self.min_node
            node.right = self.min_node.right
            self.min_node.right.left = node
            self.min_node.right = node

    def is_empty(self):
        """Check if heap is empty."""
        return self.min_node is None

    def extract_min(self):
        """Extract and return minimum element."""
        z = self.min_node
        if z is not None:
            if z.child is not None:
                children = []
                child = z.child
                while True:
                    children.append(child)
                    child.parent = None
                    child = child.right
                    if child == z.child:
                        break

                for child in children:
                    self._insert_into_root_list(child)

            self._remove_from_root_list(z)
            if z == z.right:
                self.min_node = None
            else:
                self.min_node = z.right

            self.num_nodes -= 1

        return z

    def _remove_from_root_list(self, node):
        """Remove node from root list."""
        if node == self.min_node:
            self.min_node = node.right

        node.left.right = node.right
        node.right.left = node.left

    def _consolidate(self):
        """Consolidate heap to merge trees of same degree."""
        A = [None] * (self.num_nodes + 1)
        roots = self._get_root_list()
        for w in roots:
            x = w
            d = x.degree
            while A[d] is not None:
                y = A[d]
                if x.key > y.key:
                    x, y = y, x
                self._heap_link(y, x)
                A[d] = None
                d += 1

            A[d] = x
        self.min_node = None
        for node in A:
            if node is not None:
                if self.min_node is None:
                    self.min_node = node
                else:
                    self._insert_into_root_list(node)
                    if node.key < self.min_node.key:
                        self.min_node = node

    def _heap_link(self, y, x):
        """Link two nodes."""
        self._remove_from_root_list(y)
        y.parent = x
        y.mark = False

        if x.child is None:
            x.child = y
            y.left = y
            y.right = y
        else:
            y.left = x.child
            y.right = x.child.right
            x.child.right.left = y
            x.child.right = y

        x.degree += 1

    def _get_root_list(self):
        """Get list of all roots in heap."""
        if self.min_node is None:
            return []

        roots = []
        current = self.min_node
        while True:
            roots.append(current)
            current = current.right
            if current == self.min_node:
                break

        return roots


if __name__ == "__main__":
    graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
    print("Running Dijkstra with Fibonacci Heap...")

    # Custom implementation for Fibonacci Heap without generic algorithm
    fib_heap = FibonacciHeap()
    fib_heap.insert(0, src)

    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0
    visited = [False] * graph.num_vertices

    start_time = time.time()

    while fib_heap.min_node is not None:
        min_node = fib_heap.extract_min()
        if min_node is None:
            break
        
        u = min_node.vertex
        
        if visited[u]:
            continue
        visited[u] = True

        for v, weight in graph.edges[u]:
            new_dist = distances[u] + weight
            if new_dist < distances[v]:
                distances[v] = new_dist
                fib_heap.insert(new_dist, v)

    end_time = time.time()
    execution_time = (end_time - start_time) * 1000

    save_results(distances, execution_time, r'dataset/fibonacci_op.txt', 'Fibonacci Heap')
    print("Output written to fibonacci_op.txt")
