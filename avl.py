"""
AVL Tree implementation of Dijkstra's algorithm.
"""

from dijkstra import load_graph_from_file, run_dijkstra_benchmark


class AVLNode:
    """Node in AVL Tree."""
    
    def __init__(self, key, vertex):
        self.key = key
        self.vertex = vertex
        self.left = None
        self.right = None
        self.height = 1


class AVLTree:
    """Self-balancing binary search tree for priority queue."""
    
    def __init__(self):
        self.root = None

    def height(self, node):
        """Get height of node."""
        if not node:
            return 0
        return node.height

    def balance_factor(self, node):
        """Calculate balance factor of node."""
        if not node:
            return 0
        return self.height(node.left) - self.height(node.right)

    def right_rotate(self, y):
        """Perform right rotation."""
        x = y.left
        y.left = x.right
        x.right = y
        y.height = max(self.height(y.left), self.height(y.right)) + 1
        x.height = max(self.height(x.left), self.height(x.right)) + 1
        return x

    def left_rotate(self, x):
        """Perform left rotation."""
        y = x.right
        x.right = y.left
        y.left = x
        x.height = max(self.height(x.left), self.height(x.right)) + 1
        y.height = max(self.height(y.left), self.height(y.right)) + 1
        return y

    def balance(self, node):
        """Balance node and apply rotations if necessary."""
        if not node:
            return node

        node.height = max(self.height(node.left), self.height(node.right)) + 1
        balance = self.balance_factor(node)

        if balance > 1 and self.balance_factor(node.left) >= 0:
            return self.right_rotate(node)

        if balance < -1 and self.balance_factor(node.right) <= 0:
            return self.left_rotate(node)

        if balance > 1 and self.balance_factor(node.left) < 0:
            node.left = self.left_rotate(node.left)
            return self.right_rotate(node)

        if balance < -1 and self.balance_factor(node.right) > 0:
            node.right = self.right_rotate(node.right)
            return self.left_rotate(node)

        return node

    def insert(self, node, key, vertex):
        """Insert key-vertex pair into AVL tree."""
        if not node:
            return AVLNode(key, vertex)

        if key < node.key:
            node.left = self.insert(node.left, key, vertex)
        else:
            node.right = self.insert(node.right, key, vertex)

        return self.balance(node)

    def extract_min(self, node):
        """Extract minimum element (leftmost node)."""
        if node.left is None:
            return node.right
        node.left = self.extract_min(node.left)
        return self.balance(node)

    def get_min(self, node):
        """Get node with minimum key."""
        if node.left is None:
            return node
        return self.get_min(node.left)


if __name__ == "__main__":
    graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
    print("Running Dijkstra with AVL Tree...")
    
    # Custom implementation for AVL since it doesn't use the generic interface
    import time
    avl_tree = AVLTree()
    avl_tree.root = avl_tree.insert(avl_tree.root, 0, src)
    
    INF = float('inf')
    distances = [INF] * graph.num_vertices
    distances[src] = 0
    
    start_time = time.time()
    
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
    
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000
    
    from dijkstra import save_results
    save_results(distances, execution_time, r'dataset/avl_op.txt', 'AVL Tree')
    print("Output written to avl_op.txt")

