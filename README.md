# 🎮 Algorithm Visualizer - Interactive Game Algorithms

[![GitHub stars](https://img.shields.io/github/stars/jackeygle/Algorithm-Visualizer?style=social)](https://github.com/jackeygle/Algorithm-Visualizer/stargazers)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Live Demo](https://img.shields.io/badge/demo-live-brightgreen.svg)](https://jackeysproject.web.app/games.html)

An educational repository containing clean implementations of classic algorithms demonstrated through interactive games and visualizations. Learn algorithms by **playing games first**, then **diving into the code**!

## 🚀 Live Interactive Demo

**🎮 Play the games**: [https://jackeysproject.web.app/games.html](https://jackeysproject.web.app/games.html)

Each game has a **"View Source Code"** button that links directly to the algorithm implementation in this repository!

---

## 📚 Featured Algorithms

### 🐍 1. Snake AI - A* Pathfinding Algorithm
**File**: [`snake-ai-astar.py`](snake-ai-astar.py) | **Lines**: 245 | **Demo**: [Snake AI Game](https://jackeysproject.web.app/games.html)

**Algorithm Overview:**
A* (A-star) is an informed search algorithm that finds the shortest path between nodes using heuristics to guide the search.

**Key Features:**
- **Heuristic Function**: Manhattan distance for grid-based movement
- **Pathfinding Strategy**: Combines actual cost (g-cost) with estimated cost (h-cost)
- **Optimality**: Guaranteed optimal path with admissible heuristic
- **Real-time Decision Making**: Snake navigates efficiently to food while avoiding collisions

**Complexity Analysis:**
- **Time Complexity**: `O(b^d)` where b = branching factor, d = depth
- **Space Complexity**: `O(b^d)` for storing open and closed sets
- **Practical Performance**: Excellent for real-time pathfinding in games

**Implementation Highlights:**
```python
def find_path(self, start, goal):
    # Priority queue with f-cost = g-cost + h-cost
    open_set = []
    heapq.heappush(open_set, Node(start[0], start[1], 0, self.heuristic(start, goal)))
    # ... A* algorithm implementation
```

---

### 🧩 2. Sudoku Solver - Backtracking Algorithm
**File**: [`sudoku-backtracking.py`](sudoku-backtracking.py) | **Lines**: 327 | **Demo**: [Sudoku Solver](https://jackeysproject.web.app/games.html)

**Algorithm Overview:**
Backtracking is a systematic method for solving constraint satisfaction problems. It incrementally builds candidates and abandons them when they cannot lead to a valid solution.

**Key Features:**
- **Constraint Propagation**: Validates moves against Sudoku rules (row, column, 3x3 box)
- **Most Constraining Variable (MCV)**: Chooses cells with fewest possibilities first
- **Systematic Exploration**: Depth-first search with intelligent pruning
- **Visual Demonstration**: Watch the algorithm solve puzzles step-by-step

**Complexity Analysis:**
- **Time Complexity**: `O(9^(n×n))` worst case, much better with pruning
- **Space Complexity**: `O(n×n)` for recursion stack
- **Average Performance**: Significantly optimized with heuristics

**Implementation Highlights:**
```python
def solve(self):
    # Find empty cell with most constraints (MCV heuristic)
    empty_cell = self.find_empty_cell()
    for num in range(1, 10):
        if self.is_valid_move(row, col, num):
            self.grid[row][col] = num  # Make choice
            if self.solve():           # Recursive call
                return True
            self.grid[row][col] = 0    # Backtrack
```

---

### 🎯 3. 2048 AI - Minimax Algorithm with Alpha-Beta Pruning
**File**: [`minimax-2048.py`](minimax-2048.py) | **Lines**: 459 | **Demo**: [2048 AI Game](https://jackeysproject.web.app/games.html)

**Algorithm Overview:**
Minimax is a decision-making algorithm for turn-based games. It assumes both players play optimally and chooses the move that maximizes the minimum guaranteed outcome.

**Key Features:**
- **Game Tree Exploration**: Alternating max (player) and min (random tiles) layers
- **Alpha-Beta Pruning**: Optimizes search by eliminating unnecessary branches
- **Multi-dimensional Evaluation**: Combines empty cells, monotonicity, smoothness, corner bonus
- **Expectimax Variant**: Handles stochastic elements (random tile placement)

**Complexity Analysis:**
- **Time Complexity**: `O(b^d)` where b = branching factor, d = search depth
- **Space Complexity**: `O(d)` for recursion stack
- **Optimization**: Alpha-beta pruning reduces average case significantly

**Evaluation Function:**
```python
def evaluate_board(self, game):
    # Multi-factor heuristic evaluation
    score = (empty_cells * 2.7 +           # More empty cells = better
             monotonicity * 1.0 +          # Ordered arrangement
             smoothness * 0.1 +            # Similar adjacent values
             corner_bonus * 0.5 +          # Max tile in corner
             current_score * 1.0)          # Game score
    return score
```

---

### 🔄 4. Conway's Game of Life - Cellular Automata
**File**: [`conway-life.py`](conway-life.py) | **Lines**: 406 | **Demo**: [Conway's Life](https://jackeysproject.web.app/games.html)

**Algorithm Overview:**
Conway's Game of Life is a cellular automaton that demonstrates how complex patterns can emerge from simple rules. Each cell follows basic rules based on its neighbors.

**Key Features:**
- **Simple Rules**: 3 basic rules create complex emergent behavior
- **Pattern Library**: Includes gliders, oscillators, still lifes, and more
- **Emergence Demonstration**: Shows how complexity arises from simplicity
- **Turing Completeness**: Can theoretically simulate any computation

**The Rules:**
1. **Survival**: Live cell with 2-3 neighbors survives
2. **Birth**: Dead cell with exactly 3 neighbors becomes alive  
3. **Death**: All other cells die or stay dead

**Complexity Analysis:**
- **Time Complexity**: `O(n×m)` per generation
- **Space Complexity**: `O(n×m)` for grid storage
- **Pattern Analysis**: Includes stability detection and population tracking

**Implementation Highlights:**
```python
def next_generation(self):
    new_grid = [[False] * self.width for _ in range(self.height)]
    for y in range(self.height):
        for x in range(self.width):
            neighbors = self.count_neighbors(x, y)
            new_grid[y][x] = self.apply_rules(x, y, neighbors)
    self.grid = new_grid  # Update all cells simultaneously
```

---

### 🗺️ 5. Pathfinding Visualizer - Graph Search Comparison
**File**: [`pathfinding-algorithms.py`](pathfinding-algorithms.py) | **Lines**: 523 | **Demo**: [Pathfinding Comparison](https://jackeysproject.web.app/games.html)

**Algorithm Overview:**
Comprehensive comparison of pathfinding algorithms showing different approaches to finding paths in graphs. Each algorithm has unique characteristics and use cases.

**Algorithms Implemented:**

#### **A* (A-Star)**
- **Strategy**: Heuristic-guided optimal search
- **Guarantee**: Optimal path with admissible heuristic
- **Use Case**: Games, robotics, GPS navigation
- **Time**: `O(b^d)` average case

#### **Dijkstra's Algorithm**  
- **Strategy**: Uniform-cost search, explores all directions equally
- **Guarantee**: Always finds shortest path
- **Use Case**: Network routing, social networks
- **Time**: `O((V + E) log V)` with priority queue

#### **Breadth-First Search (BFS)**
- **Strategy**: Layer-by-layer exploration
- **Guarantee**: Shortest path for unweighted graphs
- **Use Case**: Shortest path in unweighted scenarios
- **Time**: `O(V + E)` for graph traversal

#### **Depth-First Search (DFS)**
- **Strategy**: Deep exploration before backtracking
- **Guarantee**: No optimality guarantee (comparison purposes)
- **Use Case**: Maze generation, topological sorting
- **Time**: `O(V + E)` for traversal

**Implementation Features:**
```python
class PathfindingAlgorithms:
    def a_star(self, heuristic_type="euclidean"):
        # A* with configurable heuristics
    
    def dijkstra(self):
        # Guaranteed shortest path
    
    def bfs(self):
        # Unweighted shortest path
    
    def dfs(self):
        # Deep exploration (non-optimal)
```

---

## 🛠️ Installation & Usage

### **Quick Start**

```bash
# Clone the repository
git clone https://github.com/jackeygle/Algorithm-Visualizer.git
cd Algorithm-Visualizer

# Run any algorithm demonstration
python snake-ai-astar.py
python sudoku-backtracking.py
python minimax-2048.py
python conway-life.py
python pathfinding-algorithms.py
```

### **Requirements**

```python
# Core requirements (Python 3.8+)
import heapq          # Priority queues for A* and Dijkstra
import random         # Random generation for games and mazes
import time           # Performance timing
import math           # Mathematical calculations
from typing import *  # Type hints for better code clarity
from collections import deque  # BFS queue implementation
from enum import Enum # Enumerated types for clear state management
```

**No external dependencies required!** All implementations use only Python standard library.

---

## 🎓 Educational Value

### **Learning Path**
```
🎮 Play Games → 👀 Observe Algorithms → 📖 Study Source Code → 💻 Run Examples → 🧠 Understand Concepts
```

### **What You'll Learn**

#### **Algorithm Design Patterns**
- **Search Algorithms**: A*, Dijkstra, BFS, DFS
- **Optimization**: Alpha-beta pruning, heuristic design
- **Constraint Satisfaction**: Backtracking with pruning
- **Simulation**: Cellular automata and emergent behavior

#### **Programming Concepts**
- **Data Structures**: Priority queues, graphs, grids
- **Algorithm Analysis**: Time/space complexity
- **Design Patterns**: State management, recursive algorithms
- **Code Quality**: Documentation, type hints, clean architecture

#### **Problem-Solving Skills**
- **Heuristic Design**: Creating effective evaluation functions
- **State Space Search**: Navigating large solution spaces efficiently
- **Game Theory**: Minimax decision making
- **Emergent Systems**: Understanding complex behavior from simple rules

---

## 🌟 Features

### **🎯 Interactive Learning**
- **Visual Demonstrations**: See algorithms in action
- **Step-by-step Execution**: Understand each algorithm phase
- **Performance Metrics**: Compare algorithm efficiency
- **Real-world Applications**: Games make concepts tangible

### **📖 Educational Quality**
- **Comprehensive Documentation**: Every function explained
- **Complexity Analysis**: Big-O notation for all algorithms
- **Best Practices**: Production-quality code examples
- **Type Hints**: Modern Python with full type annotations

### **🔧 Practical Implementation**
- **Modular Design**: Reusable algorithm components
- **Clean Architecture**: Easy to understand and modify
- **Performance Optimized**: Efficient implementations
- **Extensible**: Easy to add new algorithms and features

---

## 🎪 Live Demo Experience

### **🎮 Interactive Games**
Visit [**Algorithm Playground**](https://jackeysproject.web.app/games.html) to experience:

1. **🐍 Snake AI**: Watch A* pathfinding in real-time
2. **🧩 Sudoku Solver**: See backtracking solve puzzles step-by-step  
3. **🎯 2048 AI**: Observe minimax making strategic decisions
4. **🔄 Conway's Life**: Witness emergent patterns evolve
5. **🗺️ Pathfinding**: Compare algorithm search strategies
6. **🟦 Tetris**: Experience game state management

### **📱 Features**
- **Responsive Design**: Works on desktop and mobile
- **Real-time Visualization**: See algorithms working live
- **Interactive Controls**: Play/pause, speed control, step-through
- **Source Code Links**: Direct access to implementations
- **Educational Explanations**: Algorithm descriptions and complexity analysis

---

## 🤝 Contributing

We welcome contributions! Here are ways you can help:

### **🔬 Algorithm Implementations**
- Add new algorithms (sorting, graph algorithms, ML algorithms)
- Optimize existing implementations
- Add new visualization features

### **📚 Educational Content**
- Improve documentation and explanations
- Add more example use cases
- Create tutorial content

### **🎮 Interactive Features**
- Enhance game visualizations
- Add new interactive demonstrations
- Improve user experience

### **🐛 Bug Fixes & Improvements**
- Report issues and bugs
- Suggest performance improvements
- Code quality enhancements

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **John Conway** - For Conway's Game of Life
- **Peter Hart, Nils Nilsson, Bertram Raphael** - For A* algorithm
- **Edsger Dijkstra** - For Dijkstra's algorithm
- **John von Neumann** - For minimax theorem
- **Algorithm visualization community** - For inspiration

---

## 📞 Contact

- **Website**: [https://jackeysproject.web.app](https://jackeysproject.web.app)
- **GitHub**: [@jackeygle](https://github.com/jackeygle)
- **Live Demo**: [Algorithm Playground](https://jackeysproject.web.app/games.html)

---

## 🎯 Project Stats

- **📊 Total Lines of Code**: 1,960+
- **🔧 Algorithms Implemented**: 8+
- **🎮 Interactive Games**: 6
- **📚 Educational Documentation**: Comprehensive
- **🧪 Test Coverage**: Demonstration examples included
- **⚡ Performance**: Optimized implementations

**⭐ If this project helped you learn algorithms, please give it a star!**

---

*"The best way to learn algorithms is to see them in action, understand their purpose, then dive into the implementation. This project bridges the gap between theory and practice through interactive gaming experiences."* 
