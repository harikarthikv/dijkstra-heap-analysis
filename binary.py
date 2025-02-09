import time  

class BinaryMinHeap:
    def __init__(self):
        self.heap = []

    def insert(self, key, vertex):
        self.heap.append((key, vertex))
        self._heapify_up(len(self.heap) - 1)

    def extract_min(self):
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        min_element = self.heap[0]
        self.heap[0] = self.heap.pop()  
        self._heapify_down(0)
        return min_element

    def _heapify_up(self, index):
        parent = (index - 1) // 2
        if index > 0 and self.heap[index][0] < self.heap[parent][0]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self._heapify_up(parent)

    def _heapify_down(self, index):
        smallest = index
        left = 2 * index + 1
        right = 2 * index + 2

        if left < len(self.heap) and self.heap[left][0] < self.heap[smallest][0]:
            smallest = left
        if right < len(self.heap) and self.heap[right][0] < self.heap[smallest][0]:
            smallest = right

        if smallest != index:
            self.heap[index], self.heap[smallest] = self.heap[smallest], self.heap[index]
            self._heapify_down(smallest)


def dijkstra_binary(graph, src):
    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0
    min_heap = BinaryMinHeap()
    min_heap.insert(0, src)

    while min_heap.heap:
        current_distance, u = min_heap.extract_min()
        if current_distance > distances[u]:
            continue

        for v, weight in graph.edges[u]:
            distance = current_distance + weight
            if distance < distances[v]:
                distances[v] = distance
                min_heap.insert(distance, v)

    return distances

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = [[] for _ in range(num_vertices)]

    def add_edge(self, u, v, weight):
        self.edges[u].append((v, weight))
        self.edges[v].append((u, weight))

def load_graph_from_file(file_path):
    with open(file_path, 'r') as f:
        num_vertices, _ = map(int, f.readline().strip().split())
        src = 0  
        graph = Graph(num_vertices)
        for line in f:
            parts = line.strip().split()
            if parts[0] == 'a':
                u = int(parts[1]) - 1
                v = int(parts[2]) - 1
                weight = int(parts[3])
                graph.add_edge(u, v, weight)
    return graph, src

def display_distances(distances, title):
    print(f"{title}")
    print("Vertex   Distance from Source")
    for i, distance in enumerate(distances):
        print(f"{i}\t\t{distance}")

graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
print("Running Dijkstra with Custom Binary Min-Heap...")

start_time = time.time()
dist_binary = dijkstra_binary(graph, src)
end_time = time.time()

execution_time = (end_time - start_time) * 1000
with open(r'dataset/binary_op.txt', 'w') as f:
    f.write("Vertex  Distance from Source\n")
    for i, distance in enumerate(dist_binary):
        f.write(f"{i}\t\t{distance}\n")
    f.write(f"Execution time: {execution_time:.2f} ms\n")
print("Output written to binary_op.txt")
