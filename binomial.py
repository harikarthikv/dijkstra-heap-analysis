import time

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.edges = [[] for _ in range(num_vertices)]  # Adjacency list

    def add_edge(self, u, v, weight):
        self.edges[u-1].append((v-1, weight))  # Add edge (u, v) with weight
        self.edges[v-1].append((u-1, weight))  # For undirected graph

class BinomialNode:
    def __init__(self, key, vertex):
        self.key = key  # Node key (distance)
        self.vertex = vertex  # Node vertex
        self.degree = 0  # Node degree
        self.parent = None  # Parent node
        self.child = None  # Child node
        self.sibling = None  # Sibling node

class BinomialHeap:
    def __init__(self):
        self.head = None  # Heap root

    def link(self, y, z):
        y.parent = z  # Link y to z
        y.sibling = z.child  # y becomes a child of z
        z.child = y
        z.degree += 1  # Increment degree

    def merge(self, other):
        if not self.head:  # If heap is empty, take other heap
            self.head = other.head
            return
        if not other.head:  # If other heap is empty, do nothing
            return

        merged = None
        h1 = self.head
        h2 = other.head
        if h1.degree <= h2.degree:
            merged = h1
            h1 = h1.sibling
        else:
            merged = h2
            h2 = h2.sibling
        current = merged

        # Merge heaps based on degree
        while h1 and h2:
            if h1.degree <= h2.degree:
                current.sibling = h1
                h1 = h1.sibling
            else:
                current.sibling = h2
                h2 = h2.sibling
            current = current.sibling

        # Attach remaining nodes
        if h1:
            current.sibling = h1
        if h2:
            current.sibling = h2

        self.head = merged
        self._consolidate()  # Consolidate heap

    def _consolidate(self):
        if not self.head:  # If heap is empty, return
            return

        degree_table = {}  # Table to store nodes by degree
        current = self.head
        min_key = float('inf')
        min_node = None

        while current:
            degree = current.degree
            next_node = current.sibling
            # Consolidate nodes with same degree
            while degree in degree_table:
                other = degree_table[degree]
                if current.key > other.key:  # Ensure smaller key stays
                    current, other = other, current
                self.link(other, current)
                degree_table.pop(degree)
                degree += 1
            degree_table[degree] = current  # Update degree table
            if current.key < min_key:
                min_key = current.key  # Track min key
                min_node = current
            current = next_node

        self.head = None
        last = None
        # Rebuild heap from degree table
        for degree in sorted(degree_table.keys()):
            node = degree_table[degree]
            node.sibling = None
            if not self.head:
                self.head = node
            else:
                last.sibling = node
            last = node

    def insert(self, key, vertex):
        new_heap = BinomialHeap()  # Create new heap
        new_heap.head = BinomialNode(key, vertex)
        self.merge(new_heap)  # Merge into current heap

    def extract_min(self):
        if not self.head:  # If heap is empty, return None
            return None, None

        min_node = self.head  # Find minimum node
        min_prev = None
        prev = None
        current = self.head

        # Find node with smallest key
        while current:
            if current.key < min_node.key:
                min_node = current
                min_prev = prev
            prev = current
            current = current.sibling

        if min_prev:
            min_prev.sibling = min_node.sibling  # Remove min node from heap
        else:
            self.head = min_node.sibling

        # Merge children of min node
        if min_node.child:
            child_heap = BinomialHeap()
            current = min_node.child
            prev = None
            while current:
                next_node = current.sibling
                current.sibling = prev  # Reverse the child list
                current.parent = None
                prev = current
                current = next_node
            child_heap.head = prev
            self.merge(child_heap)

        return min_node.key, min_node.vertex  # Return min node key and vertex

def dijkstra_binomial(graph, src):
    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0  # Set source distance to 0

    heap = BinomialHeap()
    heap.insert(0, src)  # Insert source into heap

    while heap.head:
        current_dist, u = heap.extract_min()  # Extract min distance node
        if current_dist > distances[u]:
            continue  # Skip if not optimal

        for v, weight in graph.edges[u]:
            new_dist = current_dist + weight
            if new_dist < distances[v]:  # Relaxation step
                distances[v] = new_dist
                heap.insert(new_dist, v)  # Insert updated distance into heap

    return distances

def load_graph_from_file(file_path):
    with open(file_path, 'r') as f:
        first_line = f.readline().strip().split()
        num_vertices = int(first_line[0])
        num_edges = int(first_line[1])

        graph = Graph(num_vertices)

        edge_set = set()  # Set to track unique edges
        for _ in range(num_edges):
            line = f.readline().strip().split()
            if line[0] == 'a':
                u = int(line[1])
                v = int(line[2])
                w = int(line[3])

                if (u, v) not in edge_set and (v, u) not in edge_set:
                    graph.add_edge(u, v, w)  # Add edge to graph
                    edge_set.add((u, v))  # Add to edge set

        src = 0  # Default source vertex

    return graph, src

def display_distances(distances, title):
    output_lines = [f"\n{title}\nVertex   Distance from Source"]
    for i, distance in enumerate(distances, 1):
        output_lines.append(f"{i:<8} {distance}")  # Format output
    return "\n".join(output_lines)

def write_output_to_file(output, file_path):
    with open(file_path, 'w') as f:
        f.write(output)  # Write output to file

graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
print("Running Dijkstra with Binomial Heap...")
start_time = time.time()
distances = dijkstra_binomial(graph, src)
end_time = time.time()

output = display_distances(distances, "Distances using Binomial Heap")
execution_time = (end_time - start_time) * 1000  # Calculate execution time
output += f"\nExecution time using Binomial Heap: {execution_time:.2f} ms"

write_output_to_file(output, r'dataset/binomial_op.txt')

print("Output written to binomial_op.txt")
