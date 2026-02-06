"""
Binary Min-Heap implementation of Dijkstra's algorithm.
"""

from dijkstra import load_graph_from_file, run_dijkstra_benchmark


class BinaryMinHeap:
    """Efficient binary min-heap implementation for priority queue."""
    
    def __init__(self):
        self.heap = []

    def insert(self, key, vertex):
        """Insert a (key, vertex) pair and maintain heap property."""
        self.heap.append((key, vertex))
        self._heapify_up(len(self.heap) - 1)

    def extract_min(self):
        """Extract and return the minimum element."""
        if len(self.heap) == 0:
            return None
        if len(self.heap) == 1:
            return self.heap.pop()
        min_element = self.heap[0]
        self.heap[0] = self.heap.pop()
        self._heapify_down(0)
        return min_element

    def _heapify_up(self, index):
        """Move element up to maintain min-heap property."""
        parent = (index - 1) // 2
        if index > 0 and self.heap[index][0] < self.heap[parent][0]:
            self.heap[index], self.heap[parent] = self.heap[parent], self.heap[index]
            self._heapify_up(parent)

    def _heapify_down(self, index):
        """Move element down to maintain min-heap property."""
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


if __name__ == "__main__":
    graph, src = load_graph_from_file(r'dataset/dataset_ip.txt')
    print("Running Dijkstra with Custom Binary Min-Heap...")
    
    distances, execution_time = run_dijkstra_benchmark(
        BinaryMinHeap, 
        [],
        graph, 
        src, 
        r'dataset/binary_op.txt',
        'Custom Binary Min-Heap'
    )
    
    print("Output written to binary_op.txt")
