"""
🐍 Snake AI - A* Pathfinding Algorithm
==========================================

A* (A-star) is an informed search algorithm that finds the shortest path 
between nodes using heuristics to guide the search.

Time Complexity: O(b^d) where b is branching factor, d is depth
Space Complexity: O(b^d)

Key Components:
1. f(n) = g(n) + h(n)
   - g(n): actual cost from start to node n
   - h(n): heuristic estimate from node n to goal
2. Open set: nodes to be evaluated
3. Closed set: nodes already evaluated
"""

import heapq
import math
from typing import List, Tuple, Set, Optional, Dict

class Node:
    """Represents a position in the grid with pathfinding data"""
    
    def __init__(self, x: int, y: int, g_cost: float = 0, h_cost: float = 0, parent=None):
        self.x = x
        self.y = y
        self.g_cost = g_cost  # Distance from start
        self.h_cost = h_cost  # Heuristic distance to goal
        self.f_cost = g_cost + h_cost  # Total cost
        self.parent = parent
    
    def __lt__(self, other):
        """For priority queue comparison"""
        return self.f_cost < other.f_cost
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))

class SnakeAI:
    """Snake AI using A* pathfinding algorithm"""
    
    def __init__(self, grid_width: int, grid_height: int):
        self.width = grid_width
        self.height = grid_height
        self.snake_body: List[Tuple[int, int]] = []
        self.food_position: Tuple[int, int] = (0, 0)
    
    def heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Manhattan distance heuristic
        
        Manhattan distance is admissible (never overestimates) for grid-based
        movement, making A* optimal.
        """
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def get_neighbors(self, position: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get valid neighboring positions
        
        Returns positions that are:
        1. Within grid boundaries
        2. Not occupied by snake body (except tail, which will move)
        """
        x, y = position
        neighbors = []
        
        # 4-directional movement: up, right, down, left
        directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]
        
        for dx, dy in directions:
            new_x, new_y = x + dx, y + dy
            
            # Check grid boundaries
            if 0 <= new_x < self.width and 0 <= new_y < self.height:
                # Check if position is not occupied by snake body
                # (exclude tail as it will move when snake advances)
                if (new_x, new_y) not in self.snake_body[:-1]:
                    neighbors.append((new_x, new_y))
        
        return neighbors
    
    def reconstruct_path(self, goal_node: Node) -> List[Tuple[int, int]]:
        """
        Reconstruct path from goal to start using parent pointers
        
        Returns path from start to goal
        """
        path = []
        current = goal_node
        
        while current is not None:
            path.append((current.x, current.y))
            current = current.parent
        
        return path[::-1]  # Reverse to get start-to-goal path
    
    def find_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
        """
        A* pathfinding algorithm implementation
        
        Args:
            start: Starting position (snake head)
            goal: Target position (food)
            
        Returns:
            List of positions from start to goal, or None if no path exists
        """
        
        # Priority queue for open set (nodes to be evaluated)
        open_set = []
        heapq.heappush(open_set, Node(start[0], start[1], 0, self.heuristic(start, goal)))
        
        # Set of positions in open set for O(1) lookup
        open_set_positions: Set[Tuple[int, int]] = {start}
        
        # Set of evaluated positions
        closed_set: Set[Tuple[int, int]] = set()
        
        # Dictionary to store best g_cost for each position
        g_costs: Dict[Tuple[int, int], float] = {start: 0}
        
        while open_set:
            # Get node with lowest f_cost
            current_node = heapq.heappop(open_set)
            current_pos = (current_node.x, current_node.y)
            
            # Remove from open set tracking
            open_set_positions.discard(current_pos)
            
            # Add to closed set
            closed_set.add(current_pos)
            
            # Check if we reached the goal
            if current_pos == goal:
                return self.reconstruct_path(current_node)
            
            # Evaluate neighbors
            for neighbor_pos in self.get_neighbors(current_pos):
                if neighbor_pos in closed_set:
                    continue
                
                # Calculate tentative g_cost
                tentative_g_cost = current_node.g_cost + 1
                
                # If this path to neighbor is better than any previous one
                if neighbor_pos not in g_costs or tentative_g_cost < g_costs[neighbor_pos]:
                    # Update costs
                    g_costs[neighbor_pos] = tentative_g_cost
                    h_cost = self.heuristic(neighbor_pos, goal)
                    
                    # Create neighbor node
                    neighbor_node = Node(
                        neighbor_pos[0], neighbor_pos[1],
                        tentative_g_cost, h_cost, current_node
                    )
                    
                    # Add to open set if not already there
                    if neighbor_pos not in open_set_positions:
                        heapq.heappush(open_set, neighbor_node)
                        open_set_positions.add(neighbor_pos)
        
        # No path found
        return None
    
    def get_next_move(self, snake_head: Tuple[int, int], snake_body: List[Tuple[int, int]], 
                     food_pos: Tuple[int, int]) -> Optional[Tuple[int, int]]:
        """
        Get the next optimal move for the snake
        
        Args:
            snake_head: Current head position
            snake_body: List of snake body positions
            food_pos: Food position
            
        Returns:
            Next position to move to, or None if no valid move
        """
        self.snake_body = snake_body
        self.food_position = food_pos
        
        # Find path to food
        path = self.find_path(snake_head, food_pos)
        
        if path and len(path) > 1:
            # Return the next position in the path
            return path[1]
        
        # If no path to food, try to avoid collision
        # Find any safe move
        neighbors = self.get_neighbors(snake_head)
        if neighbors:
            return neighbors[0]
        
        return None

# Example usage and testing
def demonstrate_astar():
    """Demonstrate A* pathfinding with a simple example"""
    
    print("🐍 Snake AI - A* Pathfinding Demonstration")
    print("=" * 50)
    
    # Create a 10x10 grid
    ai = SnakeAI(10, 10)
    
    # Example snake configuration
    snake_head = (2, 2)
    snake_body = [(2, 2), (2, 3), (2, 4)]  # Snake going up
    food_position = (7, 7)
    
    print(f"Snake Head: {snake_head}")
    print(f"Snake Body: {snake_body}")
    print(f"Food Position: {food_position}")
    print()
    
    # Find optimal path
    path = ai.find_path(snake_head, food_position)
    
    if path:
        print("✅ Path found!")
        print(f"Path length: {len(path)} steps")
        print(f"Path: {' -> '.join([f'({x},{y})' for x, y in path])}")
        
        # Get next move
        next_move = ai.get_next_move(snake_head, snake_body, food_position)
        if next_move:
            print(f"🎯 Next move: {snake_head} -> {next_move}")
    else:
        print("❌ No path found!")
    
    print()
    print("🧠 Algorithm Analysis:")
    print("- Time Complexity: O(b^d) where b=branching factor, d=depth")
    print("- Space Complexity: O(b^d)")
    print("- Optimality: Guaranteed optimal path with admissible heuristic")
    print("- Heuristic: Manhattan distance (admissible for 4-directional grid)")

if __name__ == "__main__":
    demonstrate_astar() 