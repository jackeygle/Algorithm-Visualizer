"""
🔄 Conway's Game of Life - Cellular Automata
===========================================

Conway's Game of Life is a cellular automaton that demonstrates how complex
patterns can emerge from simple rules. Each cell follows basic rules based
on its neighbors, leading to fascinating emergent behaviors.

Time Complexity: O(n*m) per generation, where n×m is grid size
Space Complexity: O(n*m) for grid storage
Rules: Simple but produce complex emergent patterns

Key Concepts:
1. Cellular automaton with deterministic rules
2. Emergent behavior from simple local interactions
3. Pattern classification: still life, oscillators, spaceships
4. Turing completeness (can simulate any computation)
"""

import random
import time
from typing import List, Tuple, Set
import copy

class ConwayLife:
    """Conway's Game of Life implementation with pattern support"""
    
    def __init__(self, width: int = 50, height: int = 30):
        self.width = width
        self.height = height
        self.grid = [[False] * width for _ in range(height)]
        self.generation = 0
        self.population_history = []
        
        # Pattern definitions
        self.patterns = {
            'glider': [
                [False, True, False],
                [False, False, True],
                [True, True, True]
            ],
            'pulsar': [
                [False,False,True,True,True,False,False,False,True,True,True,False,False],
                [False,False,False,False,False,False,False,False,False,False,False,False,False],
                [True,False,False,False,False,True,False,True,False,False,False,False,True],
                [True,False,False,False,False,True,False,True,False,False,False,False,True],
                [True,False,False,False,False,True,False,True,False,False,False,False,True],
                [False,False,True,True,True,False,False,False,True,True,True,False,False],
                [False,False,False,False,False,False,False,False,False,False,False,False,False],
                [False,False,True,True,True,False,False,False,True,True,True,False,False],
                [True,False,False,False,False,True,False,True,False,False,False,False,True],
                [True,False,False,False,False,True,False,True,False,False,False,False,True],
                [True,False,False,False,False,True,False,True,False,False,False,False,True],
                [False,False,False,False,False,False,False,False,False,False,False,False,False],
                [False,False,True,True,True,False,False,False,True,True,True,False,False]
            ],
            'beacon': [
                [True, True, False, False],
                [True, True, False, False],
                [False, False, True, True],
                [False, False, True, True]
            ],
            'blinker': [
                [True],
                [True],
                [True]
            ],
            'block': [
                [True, True],
                [True, True]
            ],
            'toad': [
                [False, True, True, True],
                [True, True, True, False]
            ]
        }
    
    def clear(self) -> None:
        """Clear the grid and reset generation counter"""
        self.grid = [[False] * self.width for _ in range(self.height)]
        self.generation = 0
        self.population_history = []
    
    def set_cell(self, x: int, y: int, alive: bool) -> None:
        """Set the state of a specific cell"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y][x] = alive
    
    def get_cell(self, x: int, y: int) -> bool:
        """Get the state of a specific cell"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.grid[y][x]
        return False
    
    def count_neighbors(self, x: int, y: int) -> int:
        """
        Count the number of live neighbors for a cell
        
        A cell has 8 neighbors (Moore neighborhood):
        NW  N  NE
         W  X   E
        SW  S  SE
        
        Args:
            x, y: Cell coordinates
            
        Returns:
            Number of live neighbors (0-8)
        """
        count = 0
        
        # Check all 8 neighbors
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip the cell itself
                
                neighbor_x = x + dx
                neighbor_y = y + dy
                
                # Check bounds and count live neighbors
                if (0 <= neighbor_x < self.width and 
                    0 <= neighbor_y < self.height and 
                    self.grid[neighbor_y][neighbor_x]):
                    count += 1
        
        return count
    
    def apply_rules(self, x: int, y: int, neighbors: int) -> bool:
        """
        Apply Conway's Game of Life rules
        
        Rules:
        1. Live cell with 2-3 neighbors survives
        2. Dead cell with exactly 3 neighbors becomes alive
        3. All other cells die or stay dead
        
        Args:
            x, y: Cell coordinates
            neighbors: Number of live neighbors
            
        Returns:
            New state of the cell
        """
        current_state = self.grid[y][x]
        
        if current_state:  # Cell is currently alive
            if neighbors == 2 or neighbors == 3:
                return True  # Survives
            else:
                return False  # Dies (underpopulation or overpopulation)
        else:  # Cell is currently dead
            if neighbors == 3:
                return True  # Birth
            else:
                return False  # Stays dead
    
    def next_generation(self) -> None:
        """
        Calculate and apply the next generation
        
        Creates a new grid based on current state and Conway's rules.
        All cells are updated simultaneously.
        """
        # Create new grid for next generation
        new_grid = [[False] * self.width for _ in range(self.height)]
        
        # Apply rules to each cell
        for y in range(self.height):
            for x in range(self.width):
                neighbors = self.count_neighbors(x, y)
                new_grid[y][x] = self.apply_rules(x, y, neighbors)
        
        # Update grid and generation counter
        self.grid = new_grid
        self.generation += 1
        
        # Track population for analysis
        population = self.get_population()
        self.population_history.append(population)
    
    def get_population(self) -> int:
        """Get the current number of live cells"""
        return sum(sum(row) for row in self.grid)
    
    def randomize(self, density: float = 0.3) -> None:
        """
        Randomly populate the grid
        
        Args:
            density: Probability that each cell starts alive (0.0 to 1.0)
        """
        self.clear()
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = random.random() < density
    
    def load_pattern(self, pattern_name: str, start_x: int = None, start_y: int = None) -> None:
        """
        Load a predefined pattern onto the grid
        
        Args:
            pattern_name: Name of the pattern to load
            start_x, start_y: Position to place the pattern (center if None)
        """
        if pattern_name not in self.patterns:
            raise ValueError(f"Unknown pattern: {pattern_name}")
        
        pattern = self.patterns[pattern_name]
        pattern_height = len(pattern)
        pattern_width = len(pattern[0])
        
        # Default to center if no position specified
        if start_x is None:
            start_x = (self.width - pattern_width) // 2
        if start_y is None:
            start_y = (self.height - pattern_height) // 2
        
        # Place pattern on grid
        for y, row in enumerate(pattern):
            for x, cell in enumerate(row):
                grid_x = start_x + x
                grid_y = start_y + y
                if 0 <= grid_x < self.width and 0 <= grid_y < self.height:
                    self.grid[grid_y][grid_x] = cell
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """
        Get the bounding box of all live cells
        
        Returns:
            Tuple of (min_x, min_y, max_x, max_y)
        """
        min_x, min_y = self.width, self.height
        max_x, max_y = -1, -1
        
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x]:
                    min_x = min(min_x, x)
                    max_x = max(max_x, x)
                    min_y = min(min_y, y)
                    max_y = max(max_y, y)
        
        return (min_x, min_y, max_x, max_y)
    
    def is_stable(self, check_generations: int = 10) -> bool:
        """
        Check if the pattern has stabilized (repeating or static)
        
        Args:
            check_generations: Number of recent generations to analyze
            
        Returns:
            True if pattern appears stable
        """
        if len(self.population_history) < check_generations * 2:
            return False
        
        recent_populations = self.population_history[-check_generations:]
        
        # Check for static pattern (same population)
        if len(set(recent_populations)) == 1:
            return True
        
        # Check for oscillating pattern (repeating cycle)
        for period in range(2, check_generations // 2 + 1):
            if self._has_period(recent_populations, period):
                return True
        
        return False
    
    def _has_period(self, populations: List[int], period: int) -> bool:
        """Check if population list has a specific period"""
        if len(populations) < period * 2:
            return False
        
        for i in range(period, len(populations)):
            if populations[i] != populations[i - period]:
                return False
        
        return True
    
    def print_grid(self, alive_char: str = '█', dead_char: str = '·') -> None:
        """
        Print the current grid to console
        
        Args:
            alive_char: Character to represent live cells
            dead_char: Character to represent dead cells
        """
        print(f"Generation {self.generation} (Population: {self.get_population()})")
        print("┌" + "─" * self.width + "┐")
        
        for row in self.grid:
            line = "│"
            for cell in row:
                line += alive_char if cell else dead_char
            line += "│"
            print(line)
        
        print("└" + "─" * self.width + "┘")
    
    def get_statistics(self) -> dict:
        """Get various statistics about the current simulation"""
        population = self.get_population()
        density = population / (self.width * self.height)
        
        stats = {
            'generation': self.generation,
            'population': population,
            'density': density,
            'grid_size': (self.width, self.height),
            'is_stable': self.is_stable() if self.generation > 20 else False
        }
        
        if self.population_history:
            stats['max_population'] = max(self.population_history)
            stats['min_population'] = min(self.population_history)
            stats['avg_population'] = sum(self.population_history) / len(self.population_history)
        
        return stats

def demonstrate_cellular_automata():
    """Demonstrate Conway's Game of Life with various patterns"""
    
    print("🔄 Conway's Game of Life - Cellular Automata Demonstration")
    print("=" * 65)
    
    # Create a smaller grid for demonstration
    life = ConwayLife(width=30, height=15)
    
    # Test different patterns
    patterns_to_test = ['glider', 'blinker', 'beacon', 'block']
    
    for pattern_name in patterns_to_test:
        print(f"\n📋 Testing {pattern_name.capitalize()} Pattern:")
        print("-" * 40)
        
        life.clear()
        life.load_pattern(pattern_name)
        
        # Show initial state
        print("Initial state:")
        life.print_grid()
        
        # Run several generations
        for gen in range(5):
            life.next_generation()
            print(f"\nAfter generation {gen + 1}:")
            life.print_grid()
            
            # Check if stable
            if life.is_stable():
                print("✅ Pattern has stabilized!")
                break
        
        # Show statistics
        stats = life.get_statistics()
        print(f"\nStatistics:")
        print(f"- Final population: {stats['population']}")
        print(f"- Density: {stats['density']:.2%}")
        print(f"- Stable: {stats['is_stable']}")
    
    # Demonstrate random evolution
    print("\n" + "=" * 65)
    print("📋 Random Evolution Demo:")
    print("-" * 40)
    
    life.clear()
    life.randomize(density=0.3)
    
    print("Initial random state:")
    life.print_grid()
    
    print("\nEvolution over 10 generations:")
    for gen in range(10):
        life.next_generation()
        population = life.get_population()
        print(f"Gen {gen + 1:2d}: Population = {population:3d}")
        
        if population == 0:
            print("💀 All cells died!")
            break
        
        if life.is_stable():
            print("🔄 Pattern stabilized!")
            break
    
    # Final statistics
    final_stats = life.get_statistics()
    print(f"\nFinal Statistics:")
    print(f"- Generations: {final_stats['generation']}")
    print(f"- Final population: {final_stats['population']}")
    print(f"- Max population: {final_stats.get('max_population', 0)}")
    print(f"- Average population: {final_stats.get('avg_population', 0):.1f}")
    
    print("\n🧠 Algorithm Analysis:")
    print("- Time Complexity: O(n×m) per generation")
    print("- Space Complexity: O(n×m) for grid storage")
    print("- Rules: Simple local interactions → complex global behavior")
    print("- Applications: Emergence, complexity theory, artificial life")
    print("- Turing Complete: Can simulate any computable function")

if __name__ == "__main__":
    demonstrate_cellular_automata() 