import time

class AVLNode:
    def __init__(self, key, vertex):
        self.key = key  
        self.vertex = vertex  
        self.left = None
        self.right = None
        self.height = 1  

class AVLTree:
    def __init__(self):
        self.root = None

    def height(self, node):
        if not node:
            return 0
        return node.height

    def balance_factor(self, node):
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)

    def right_rotate(self, y):
        x = y.left
        
        y.left = x.right
        x.right = y

        y.height = max(self.height(y.left), self.height(y.right)) + 1
        x.height = max(self.height(x.left), self.height(x.right)) + 1

        return x

    def left_rotate(self, x):
        y = x.right

        x.right = y.left
        y.left = x

        x.height = max(self.height(x.left), self.height(x.right)) + 1
        y.height = max(self.height(y.left), self.height(y.right)) + 1

        return y

    def balance(self, node):
        if not node:
            return node

        node.height = max(self.height(node.left), self.height(node.right)) + 1

        balance = self.balance_factor(node)

        if balance > 1 and self.balance_factor(node.left) >= 0: # right rotation
            return self.right_rotate(node)

        if balance < -1 and self.balance_factor(node.right) <= 0: # left rotation
            return self.left_rotate(node)

        if balance > 1 and self.balance_factor(node.left) < 0: # left rotation followed by right rotation
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def insert(self, node, key, vertex):
        if not node:
            return AVLNode(key, vertex)

        if key < node.key:
            node.left = self.insert(node.left, key, vertex)
        else:
            node.right = self.insert(node.right, key, vertex)

        return self.balance(node)
  
    def extract_min(self, node):
        if node.left is None:
            return node.right  
        node.left = self.extract_min(node.left)  
        return self.balance(node)    

    def get_min(self, node):
        if node.left is None:
            return node
        return self.get_min(node.left)

    def delete(self, root, key):
        if not root:
            return root

        if key < root.key:
            root.left = self.delete(root.left, key)
        elif key > root.key:
            root.right = self.delete(root.right, key)
        else:
            if not root.left: # no left child
                return root.right
            elif not root.right: # no right child
                return root.left

            min_node = self.get_min(root.right)
            root.key, root.vertex = min_node.key, min_node.vertex
            root.right = self.delete(root.right, min_node.key)

        root.height = max(self.height(root.left), self.height(root.right)) + 1
        balance = self.balance_factor(root)

        if abs(balance) > 1:
            root = self.balance(root)

        return root

class Graph:
    def __init__(self, num_vertices):
        self.num_vertices = num_vertices
        self.graph = [[] for _ in range(num_vertices)]  

    def add_edge(self, u, v, weight):
        self.graph[u].append((v, weight))
        self.graph[v].append((u, weight))  



def dijkstra(graph, src):
    num_vertices = graph.num_vertices
    
    distances = [float("inf")] * num_vertices
    distances[src] = 0
    
    avl_tree = AVLTree()
    avl_tree.root = avl_tree.insert(avl_tree.root, 0, src)  

    while avl_tree.root:
        min_node = avl_tree.get_min(avl_tree.root)
        current_dist, u = min_node.key, min_node.vertex
        avl_tree.root = avl_tree.extract_min(avl_tree.root)  

        if current_dist > distances[u]:
            continue
        
        for v, weight in graph.edges[u]:

            if distances[u] + weight < distances[v]:
                distances[v] = distances[u] + weight
                avl_tree.root = avl_tree.insert(avl_tree.root, distances[v], v)  
                
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
print("Running Dijkstra with AVL Heap...")

start_time = time.time()  
dist_binary = dijkstra(graph, src)  
end_time = time.time()  

execution_time = (end_time - start_time) * 1000  

with open(r'dataset/avl_op.txt', 'w') as f:
    f.write("Running Dijkstra with AVL Heap...\n")
    f.write("Vertex   Distance from Source\n")
    for i, distance in enumerate(dist_binary):
        f.write(f"{i}\t\t{distance}\n")
    f.write(f"Execution time using AVL Heap: {execution_time:.2f} ms\n")
print("Output written to avl_op.txt")
