"""
Binomial Heap implementation of Dijkstra's algorithm.
"""

from dijkstra import load_graph_from_file, run_dijkstra_benchmark


class BinomialNode:
    """Node in Binomial Heap."""
    
    def __init__(self, key, vertex):
        self.key = key
        self.vertex = vertex
        self.degree = 0
        self.parent = None
        self.child = None
        self.sibling = None


class BinomialHeap:
    """Binomial heap for efficient priority queue operations."""
    
    def __init__(self):
        self.head = None

    def link(self, y, z):
        """Link two binomial trees."""
        y.parent = z
        y.sibling = z.child
        z.child = y
        z.degree += 1

    def merge(self, other):
        """Merge two binomial heaps."""
        if not self.head:
            self.head = other.head
            return
        if not other.head:
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

        while h1 and h2:
            if h1.degree <= h2.degree:
                current.sibling = h1
                h1 = h1.sibling
            else:
                current.sibling = h2
                h2 = h2.sibling
            current = current.sibling

        if h1:
            current.sibling = h1
        if h2:
            current.sibling = h2

        self.head = merged
        self._consolidate()

    def _consolidate(self):
        """Consolidate heap to merge trees of same degree."""
        if not self.head:
            return

        degree_table = {}
        current = self.head
        min_key = float('inf')
        min_node = None

        while current:
            degree = current.degree
            next_node = current.sibling

            while degree in degree_table:
                other = degree_table[degree]
                if current.key > other.key:
                    current, other = other, current
                self.link(other, current)
                degree_table.pop(degree)
                degree += 1
            degree_table[degree] = current
            if current.key < min_key:
                min_key = current.key
                min_node = current
            current = next_node

        self.head = None
        last = None
        for degree in sorted(degree_table.keys()):
            node = degree_table[degree]
            node.sibling = None
            if not self.head:
                self.head = node
            else:
                last.sibling = node
            last = node

    def insert(self, key, vertex):
        """Insert key-vertex pair into heap."""
        new_heap = BinomialHeap()
        new_heap.head = BinomialNode(key, vertex)
        self.merge(new_heap)

    def extract_min(self):
        """Extract and return minimum element."""
        if not self.head:
            return None, None

        min_node = self.head
        min_prev = None
        prev = None
        current = self.head

        while current:
            if current.key < min_node.key:
                min_node = current
                min_prev = prev
            prev = current
            current = current.sibling

        if min_prev:
            min_prev.sibling = min_node.sibling
        else:
            self.head = min_node.sibling

        if min_node.child:
            child_heap = BinomialHeap()
            current = min_node.child
            prev = None
            while current:
                next_node = current.sibling
                current.sibling = prev
                current.parent = None
                prev = current
                current = next_node
            child_heap.head = prev
            self.merge(child_heap)

        return min_node.key, min_node.vertex


if __name__ == "__main__":
    graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
    print("Running Dijkstra with Binomial Heap...")
    
    distances, execution_time = run_dijkstra_benchmark(
        BinomialHeap,
        [],
        graph,
        src,
        r'dataset/binomial_op.txt',
        'Binomial Heap'
    )
    
    print("Output written to binomial_op.txt")
