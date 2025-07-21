"""
🗺️ Pathfinding Algorithms - Graph Search Comparison
==================================================

A comprehensive comparison of pathfinding algorithms showing different
approaches to finding paths in graphs. Each algorithm has different
characteristics and use cases.

Algorithms Implemented:
1. A* - Heuristic-guided optimal search
2. Dijkstra - Guaranteed shortest path
3. BFS - Unweighted shortest path
4. DFS - Deep exploration (for comparison)

Time Complexities:
- A*: O(b^d) average, O(b^d) worst case
- Dijkstra: O((V + E) log V) with priority queue
- BFS: O(V + E) for unweighted graphs
- DFS: O(V + E) for traversal

Key Concepts:
1. Informed vs uninformed search
2. Optimality guarantees
3. Space-time tradeoffs
4. Heuristic design and admissibility
"""

import heapq
import math
import time
from collections import deque
from typing import List, Tuple, Set, Dict, Optional, Callable
from enum import Enum
import random

class CellType(Enum):
    EMPTY = 0
    WALL = 1
    START = 2
    END = 3
    VISITED = 4
    PATH = 5

class Node:
    """Node class for pathfinding algorithms"""
    
    def __init__(self, x: int, y: int, g_cost: float = 0, h_cost: float = 0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # Distance from start
        self.h_cost = h_cost  # Heuristic distance to goal
        self.f_cost = g_cost + h_cost  # Total estimated cost
        self.parent = parent
        self.visited = False
    
    def __lt__(self, other):
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class PathfindingGrid:
    """Grid-based pathfinding environment"""
    
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[CellType.EMPTY] * width for _ in range(height)]
        self.start = (0, 0)
        self.end = (width - 1, height - 1)
        
        # Statistics tracking
        self.nodes_visited = 0
        self.path_length = 0
        self.algorithm_time = 0.0
    
    def reset_stats(self):
        """Reset algorithm statistics"""
        self.nodes_visited = 0
        self.path_length = 0
        self.algorithm_time = 0.0
    
    def set_start(self, x: int, y: int):
        """Set the start position"""
        if self.is_valid_position(x, y):
            self.start = (x, y)
            self.grid[y][x] = CellType.START
    
    def set_end(self, x: int, y: int):
        """Set the end position"""
        if self.is_valid_position(x, y):
            self.end = (x, y)
            self.grid[y][x] = CellType.END
    
    def set_wall(self, x: int, y: int):
        """Set a wall at the given position"""
        if self.is_valid_position(x, y) and (x, y) not in [self.start, self.end]:
            self.grid[y][x] = CellType.WALL
    
    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is within grid bounds"""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_walkable(self, x: int, y: int) -> bool:
        """Check if position is walkable (not a wall)"""
        return (self.is_valid_position(x, y) and 
                self.grid[y][x] != CellType.WALL)
    
    def get_neighbors(self, x: int, y: int, allow_diagonal: bool = True) -> List[Tuple[int, int, float]]:
        """
        Get valid neighbors of a position
        
        Returns:
            List of tuples (x, y, distance) for each valid neighbor
        """
        neighbors = []
        
        # 4-directional movement
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        # Add diagonal movement if allowed
        if allow_diagonal:
            directions.extend([(-1, -1), (1, -1), (1, 1), (-1, 1)])
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            if self.is_walkable(new_x, new_y):
                # Calculate distance (1.0 for orthogonal, sqrt(2) for diagonal)
                distance = math.sqrt(dx * dx + dy * dy)
                neighbors.append((new_x, new_y, distance))
        
        return neighbors
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int], 
                  heuristic_type: str = "euclidean") -> float:
        """
        Calculate heuristic distance between two positions
        
        Types:
        - manhattan: |x1-x2| + |y1-y2|
        - euclidean: sqrt((x1-x2)² + (y1-y2)²)
        - diagonal: max(|x1-x2|, |y1-y2|)
        """
        x1, y1 = pos1
        x2, y2 = pos2
        
        if heuristic_type == "manhattan":
            return abs(x1 - x2) + abs(y1 - y2)
        elif heuristic_type == "euclidean":
            return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        elif heuristic_type == "diagonal":
            return max(abs(x1 - x2), abs(y1 - y2))
        else:
            raise ValueError(f"Unknown heuristic type: {heuristic_type}")
    
    def reconstruct_path(self, node: Node) -> List[Tuple[int, int]]:
        """Reconstruct path from goal node back to start"""
        path = []
        current = node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # Reverse to get start-to-goal path
    
    def clear_search_results(self):
        """Clear previous search results but keep walls and start/end"""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x] in [CellType.VISITED, CellType.PATH]:
                    self.grid[y][x] = CellType.EMPTY
        
        # Restore start and end
        self.grid[self.start[1]][self.start[0]] = CellType.START
        self.grid[self.end[1]][self.end[0]] = CellType.END
    
    def generate_random_maze(self, wall_density: float = 0.3):
        """Generate a random maze with given wall density"""
        # Clear grid
        self.grid = [[CellType.EMPTY] * self.width for _ in range(self.height)]
        
        # Add random walls
        for y in range(self.height):
            for x in range(self.width):
                if (x, y) not in [self.start, self.end] and random.random() < wall_density:
                    self.grid[y][x] = CellType.WALL
        
        # Ensure start and end are not walls
        self.grid[self.start[1]][self.start[0]] = CellType.START
        self.grid[self.end[1]][self.end[0]] = CellType.END

class PathfindingAlgorithms:
    """Collection of pathfinding algorithms"""
    
    def __init__(self, grid: PathfindingGrid):
        self.grid = grid
    
    def a_star(self, heuristic_type: str = "euclidean") -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm
        
        Uses both actual distance (g-cost) and heuristic (h-cost) to guide search.
        Guarantees optimal path if heuristic is admissible.
        """
        start_time = time.time()
        self.grid.reset_stats()
        
        start_x, start_y = self.grid.start
        end_x, end_y = self.grid.end
        
        # Priority queue of nodes to explore
        open_set = []
        heapq.heappush(open_set, Node(start_x, start_y, 0, 
                                     self.grid.heuristic(self.grid.start, self.grid.end, heuristic_type)))
        
        # Set of explored positions
        closed_set: Set[Tuple[int, int]] = set()
        
        # Dictionary to track best g-cost for each position
        g_costs: Dict[Tuple[int, int], float] = {self.grid.start: 0}
        
        while open_set:
            current_node = heapq.heappop(open_set)
            current_pos = (current_node.x, current_node.y)
            
            # Skip if already processed
            if current_pos in closed_set:
                continue
            
            # Mark as explored
            closed_set.add(current_pos)
            self.grid.nodes_visited += 1
            
            # Mark cell as visited for visualization
            if current_pos not in [self.grid.start, self.grid.end]:
                self.grid.grid[current_node.y][current_node.x] = CellType.VISITED
            
            # Check if we reached the goal
            if current_pos == self.grid.end:
                self.grid.algorithm_time = time.time() - start_time
                path = self.grid.reconstruct_path(current_node)
                self.grid.path_length = len(path)
                return path
            
            # Explore neighbors
            for neighbor_x, neighbor_y, distance in self.grid.get_neighbors(current_node.x, current_node.y):
                neighbor_pos = (neighbor_x, neighbor_y)
                
                if neighbor_pos in closed_set:
                    continue
                
                tentative_g_cost = current_node.g_cost + distance
                
                # If this path is better than any previous path to this neighbor
                if neighbor_pos not in g_costs or tentative_g_cost < g_costs[neighbor_pos]:
                    g_costs[neighbor_pos] = tentative_g_cost
                    h_cost = self.grid.heuristic(neighbor_pos, self.grid.end, heuristic_type)
                    
                    neighbor_node = Node(neighbor_x, neighbor_y, tentative_g_cost, h_cost, current_node)
                    heapq.heappush(open_set, neighbor_node)
        
        self.grid.algorithm_time = time.time() - start_time
        return None  # No path found
    
    def dijkstra(self) -> Optional[List[Tuple[int, int]]]:
        """
        Dijkstra's algorithm
        
        Explores all directions equally, guarantees shortest path.
        Essentially A* with h-cost = 0.
        """
        start_time = time.time()
        self.grid.reset_stats()
        
        start_x, start_y = self.grid.start
        
        # Priority queue with distances
        distances = {(x, y): float('inf') for x in range(self.grid.width) 
                    for y in range(self.grid.height) if self.grid.is_walkable(x, y)}
        distances[self.grid.start] = 0
        
        # Previous node tracking for path reconstruction
        previous: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {}
        
        # Priority queue: (distance, x, y)
        pq = [(0, start_x, start_y)]
        visited: Set[Tuple[int, int]] = set()
        
        while pq:
            current_distance, x, y = heapq.heappop(pq)
            current_pos = (x, y)
            
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            self.grid.nodes_visited += 1
            
            # Mark cell as visited for visualization
            if current_pos not in [self.grid.start, self.grid.end]:
                self.grid.grid[y][x] = CellType.VISITED
            
            # Check if we reached the goal
            if current_pos == self.grid.end:
                # Reconstruct path
                path = []
                current = self.grid.end
                while current is not None:
                    path.append(current)
                    current = previous.get(current)
                
                self.grid.algorithm_time = time.time() - start_time
                self.grid.path_length = len(path)
                return path[::-1]
            
            # Explore neighbors
            for neighbor_x, neighbor_y, distance in self.grid.get_neighbors(x, y):
                neighbor_pos = (neighbor_x, neighbor_y)
                
                if neighbor_pos in visited:
                    continue
                
                new_distance = current_distance + distance
                
                if new_distance < distances.get(neighbor_pos, float('inf')):
                    distances[neighbor_pos] = new_distance
                    previous[neighbor_pos] = current_pos
                    heapq.heappush(pq, (new_distance, neighbor_x, neighbor_y))
        
        self.grid.algorithm_time = time.time() - start_time
        return None  # No path found
    
    def bfs(self) -> Optional[List[Tuple[int, int]]]:
        """
        Breadth-First Search
        
        Explores all nodes at current depth before going deeper.
        Guarantees shortest path for unweighted graphs.
        """
        start_time = time.time()
        self.grid.reset_stats()
        
        queue = deque([(self.grid.start, [self.grid.start])])
        visited: Set[Tuple[int, int]] = {self.grid.start}
        
        while queue:
            current_pos, path = queue.popleft()
            x, y = current_pos
            
            self.grid.nodes_visited += 1
            
            # Mark cell as visited for visualization
            if current_pos not in [self.grid.start, self.grid.end]:
                self.grid.grid[y][x] = CellType.VISITED
            
            # Check if we reached the goal
            if current_pos == self.grid.end:
                self.grid.algorithm_time = time.time() - start_time
                self.grid.path_length = len(path)
                return path
            
            # Explore neighbors
            for neighbor_x, neighbor_y, _ in self.grid.get_neighbors(x, y, allow_diagonal=False):
                neighbor_pos = (neighbor_x, neighbor_y)
                
                if neighbor_pos not in visited:
                    visited.add(neighbor_pos)
                    new_path = path + [neighbor_pos]
                    queue.append((neighbor_pos, new_path))
        
        self.grid.algorithm_time = time.time() - start_time
        return None  # No path found
    
    def dfs(self) -> Optional[List[Tuple[int, int]]]:
        """
        Depth-First Search
        
        Explores as far as possible along each branch before backtracking.
        Does NOT guarantee shortest path, included for comparison.
        """
        start_time = time.time()
        self.grid.reset_stats()
        
        stack = [(self.grid.start, [self.grid.start])]
        visited: Set[Tuple[int, int]] = set()
        
        while stack:
            current_pos, path = stack.pop()
            
            if current_pos in visited:
                continue
            
            visited.add(current_pos)
            x, y = current_pos
            
            self.grid.nodes_visited += 1
            
            # Mark cell as visited for visualization
            if current_pos not in [self.grid.start, self.grid.end]:
                self.grid.grid[y][x] = CellType.VISITED
            
            # Check if we reached the goal
            if current_pos == self.grid.end:
                self.grid.algorithm_time = time.time() - start_time
                self.grid.path_length = len(path)
                return path
            
            # Explore neighbors (in reverse order for DFS behavior)
            neighbors = self.grid.get_neighbors(x, y, allow_diagonal=False)
            for neighbor_x, neighbor_y, _ in reversed(neighbors):
                neighbor_pos = (neighbor_x, neighbor_y)
                
                if neighbor_pos not in visited:
                    new_path = path + [neighbor_pos]
                    stack.append((neighbor_pos, new_path))
        
        self.grid.algorithm_time = time.time() - start_time
        return None  # No path found

def demonstrate_pathfinding():
    """Demonstrate and compare different pathfinding algorithms"""
    
    print("🗺️ Pathfinding Algorithms - Comprehensive Comparison")
    print("=" * 60)
    
    # Create a test grid
    grid = PathfindingGrid(20, 15)
    pathfinder = PathfindingAlgorithms(grid)
    
    # Set start and end points
    grid.set_start(1, 1)
    grid.set_end(18, 13)
    
    # Create a simple maze
    grid.generate_random_maze(wall_density=0.25)
    
    print(f"Grid: {grid.width}×{grid.height}")
    print(f"Start: {grid.start}, End: {grid.end}")
    print(f"Wall density: ~25%")
    print()
    
    # Test each algorithm
    algorithms = [
        ("A*", lambda: pathfinder.a_star("euclidean")),
        ("Dijkstra", pathfinder.dijkstra),
        ("BFS", pathfinder.bfs),
        ("DFS", pathfinder.dfs)
    ]
    
    results = []
    
    for algo_name, algo_func in algorithms:
        print(f"🔍 Testing {algo_name}:")
        print("-" * 25)
        
        # Clear previous results
        grid.clear_search_results()
        
        # Run algorithm
        path = algo_func()
        
        if path:
            # Mark path for visualization
            for x, y in path[1:-1]:  # Skip start and end
                grid.grid[y][x] = CellType.PATH
            
            print(f"✅ Path found!")
            print(f"Path length: {grid.path_length} steps")
            print(f"Nodes visited: {grid.nodes_visited}")
            print(f"Computation time: {grid.algorithm_time:.4f}s")
            
            results.append({
                'algorithm': algo_name,
                'path_length': grid.path_length,
                'nodes_visited': grid.nodes_visited,
                'time': grid.algorithm_time,
                'found_path': True
            })
        else:
            print(f"❌ No path found!")
            results.append({
                'algorithm': algo_name,
                'path_length': 0,
                'nodes_visited': grid.nodes_visited,
                'time': grid.algorithm_time,
                'found_path': False
            })
        
        print()
    
    # Comparison summary
    print("📊 Algorithm Comparison Summary:")
    print("=" * 60)
    print(f"{'Algorithm':<12} {'Path Len':<10} {'Nodes':<8} {'Time (ms)':<10} {'Optimal':<8}")
    print("-" * 60)
    
    for result in results:
        if result['found_path']:
            print(f"{result['algorithm']:<12} "
                  f"{result['path_length']:<10} "
                  f"{result['nodes_visited']:<8} "
                  f"{result['time']*1000:<10.2f} "
                  f"{'Yes' if result['algorithm'] in ['A*', 'Dijkstra', 'BFS'] else 'No':<8}")
    
    print("\n🧠 Algorithm Analysis:")
    print("- A*: Optimal with admissible heuristic, efficient")
    print("- Dijkstra: Optimal but explores more nodes than A*")
    print("- BFS: Optimal for unweighted graphs, simple")
    print("- DFS: Not optimal, good for maze generation")
    
    print("\n🎯 Use Cases:")
    print("- A*: Games, robotics, GPS navigation")
    print("- Dijkstra: Network routing, social networks")
    print("- BFS: Shortest path in unweighted graphs")
    print("- DFS: Maze solving, topological sorting")

if __name__ == "__main__":
    demonstrate_pathfinding() 