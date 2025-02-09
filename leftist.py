import time

# Add Leftist Heap class
class LeftistHeapNode:
    def __init__(self, key, vertex):
        self.key = key
        self.vertex = vertex
        self.left = None
        self.right = None
        self.npl = 0  # Null Path Length

class LeftistHeap:
    def __init__(self):
        self.root = None

    def merge(self, h1, h2):
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
        new_node = LeftistHeapNode(key, vertex)
        self.root = self.merge(self.root, new_node)

    def remove_min(self):
        if not self.root:
            return None
        min_node = self.root
        self.root = self.merge(self.root.left, self.root.right)
        return min_node

# Update Dijkstra function to use Leftist Heap
def dijkstra(graph, src):
    num_vertices = graph.num_vertices
    distances = [float("inf")] * num_vertices
    distances[src] = 0

    leftist_heap = LeftistHeap()
    leftist_heap.insert(0, src)  # Insert the source vertex

    while leftist_heap.root:
        min_node = leftist_heap.remove_min()  # Get the vertex with the minimum distance
        current_dist, u = min_node.key, min_node.vertex
        
        if current_dist > distances[u]:
            continue
        
        for v, weight in graph.edges[u]:
            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                leftist_heap.insert(distances[v], v)  # Push updated distance to the Leftist Heap

    return distances

# Graph class
class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = [[] for _ in range(num_vertices)]  # Adjacency list

    def add_edge(self, u, v, weight):
        self.edges[u].append((v, weight))  # Add directed edge (u -> v)
        self.edges[v].append((u, weight))  # If undirected graph, add (v -> u)

# Load graph from file
def load_graph_from_file(file_path):
    with open(file_path, 'r') as f:
        # Read the first line for number of vertices and edges, ignore the second integer
        num_vertices, _ = map(int, f.readline().strip().split())
        src = 0  # Define the source vertex (default to 0, or change as needed)
        
        graph = Graph(num_vertices)

        # Process each line with 'a' format
        for line in f:
            parts = line.strip().split()
            if parts[0] == 'a':
                u = int(parts[1]) - 1  # Adjust index if needed (assuming 1-based index in file)
                v = int(parts[2]) - 1  # Adjust index if needed
                weight = int(parts[3])
                graph.add_edge(u, v, weight)
                
    return graph, src

# Display distances
def display_distances(distances, title):
    print(f"{title}")
    print("Vertex   Distance from Source")
    for i, distance in enumerate(distances):
        print(f"{i}\t\t{distance}")


# Load the graph from the file with your specified format
graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
print("Running Dijkstra with Leftist Heap...")

# Measure execution time
start_time = time.time()  # Start time
dist_binary = dijkstra(graph, src)  # Run Dijkstra's algorithm
end_time = time.time()  # End time

# Display the distances from source to each vertex

# Calculate and display the execution time in milliseconds
execution_time = (end_time - start_time) * 1000  # Convert seconds to milliseconds

# Write the distances and execution time to an output file
with open(r'dataset/leftist_op.txt', 'w') as f:
    f.write("Running Dijkstra with Leftist Heap...\n")
    f.write("Vertex   Distance from Source\n")
    for i, distance in enumerate(dist_binary):
        f.write(f"{i}\t\t{distance}\n")
    f.write(f"Execution time using Leftist Heap: {execution_time:.2f} ms\n")
print("Output written to leftist_op.txt")
