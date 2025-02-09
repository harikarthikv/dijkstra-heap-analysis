import time 
import os

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = [[] for _ in range(num_vertices)]  

    def add_edge(self, u, v, weight):
        self.edges[u-1].append((v-1, weight)) 
        self.edges[v-1].append((u-1, weight))  

import math

class FibonacciNode:
    def __init__(self, key, vertex):
        self.key = key        # Distance/weight
        self.vertex = vertex  # Vertex number
        self.parent = None    # Parent node
        self.child = None     # First child
        self.left = None      # Left sibling
        self.right = None     # Right sibling
        self.degree = 0       # Number of children
        self.mark = False     # Whether node has lost a child

class FibonacciHeap:
    def __init__(self):
        self.min_node = None  # Pointer to minimum node
        self.num_nodes = 0    # Total number of nodes in heap

    def insert(self, key, vertex):       
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
        return self.min_node is None

    def extract_min(self):
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
        if node == self.min_node:
            self.min_node = node.right
        
        node.left.right = node.right
        node.right.left = node.left

    def _consolidate(self):
        #A = [None] * int(math.log(self.num_nodes) * 2 + 1)
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
def dijkstra_fibonacci(graph, src):
    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0  # Set source distance to 0

    fib_heap = FibonacciHeap()
    fib_heap.insert(0, src + 1)  # Insert source node into heap
    while not fib_heap.is_empty():
        min_node = fib_heap.extract_min()  # Extract min node (next vertex)
        u = min_node.vertex - 1  # Convert back to 0-based index

        for v, weight in graph.edges[u]:
            new_dist = distances[u] + weight  # Calculate new distance
            if new_dist < distances[v]:
                distances[v] = new_dist  # Update distance if smaller
                fib_heap.insert(new_dist, v + 1)  # Re-insert into heap for future updates
    return distances

def load_graph_from_file(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().split()
        num_vertices = int(first_line[0])
        num_edges = int(first_line[1])

        graph = Graph(num_vertices)

        edge_set = set()
        for _ in range(num_edges):
            line = f.readline().strip().split()
            if line[0] == 'a':
                u = int(line[1])
                v = int(line[2])
                w = int(line[3])

                if (u, v) not in edge_set and (v, u) not in edge_set:
                    graph.add_edge(u, v, w)  # Add edge if not already present
                    edge_set.add((u, v))  # Avoid duplicate edges
            

        src = 0  # Starting from the first vertex (0-based)

    return graph, src

def display_distances(distances, title):
    output_lines = [f"\n{title}\nVertex   Distance from Source"]
    for i, distance in enumerate(distances, 1):
        output_lines.append(f"{i:<8} {distance}")  # Display each vertex's distance
    return "\n".join(output_lines)

def write_output_to_file(output, file_path):
    with open(file_path, 'w') as f:
        f.write(output)  # Write results to output file

graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')

# Run Dijkstra's algorithm using Fibonacci Heap
print("Running Dijkstra with Fibonacci Heap...")
start_time = time.time()
distances = dijkstra_fibonacci(graph, src)
end_time = time.time()

# Prepare output
output = display_distances(distances, "Distances using Fibonacci Heap")
execution_time = (end_time - start_time) * 1000
output += f"\nExecution time using Fibonacci Heap: {execution_time:.2f} ms"  # Add execution time

# Write output to file
write_output_to_file(output, r'dataset/fibonacci_op.txt')

print("Output written to fibonacci_op.txt")
