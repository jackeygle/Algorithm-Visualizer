"""
🧩 Sudoku Solver - Backtracking Algorithm
=========================================

Backtracking is a systematic method for solving constraint satisfaction problems.
It incrementally builds candidates and abandons candidates ("backtracks") 
when they cannot lead to a valid solution.

Time Complexity: O(9^(n*n)) worst case, where n=9 for standard Sudoku
Space Complexity: O(n*n) for recursion stack
Average Performance: Much better due to constraint propagation and pruning

Key Concepts:
1. Depth-First Search with constraint checking
2. Early termination when constraints are violated
3. Systematic exploration of solution space
"""

from typing import List, Optional, Tuple, Set
import time

class SudokuSolver:
    """Sudoku solver using backtracking algorithm with optimizations"""
    
    def __init__(self):
        self.grid: List[List[int]] = [[0] * 9 for _ in range(9)]
        self.solve_steps = 0
        self.start_time = 0
    
    def is_valid_move(self, row: int, col: int, num: int) -> bool:
        """
        Check if placing 'num' at position (row, col) is valid
        
        A move is valid if 'num' doesn't already exist in:
        1. The same row
        2. The same column  
        3. The same 3x3 box
        
        Time Complexity: O(1) - constant 27 checks maximum
        """
        
        # Check row constraint
        for c in range(9):
            if self.grid[row][c] == num:
                return False
        
        # Check column constraint
        for r in range(9):
            if self.grid[r][col] == num:
                return False
        
        # Check 3x3 box constraint
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if self.grid[r][c] == num:
                    return False
        
        return True
    
    def find_empty_cell(self) -> Optional[Tuple[int, int]]:
        """
        Find the next empty cell (contains 0)
        
        Returns the first empty cell found, or None if grid is complete
        
        Optimization: Could use Most Constraining Variable (MCV) heuristic
        to choose the cell with fewest possible values first
        """
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    return (row, col)
        return None
    
    def get_possible_values(self, row: int, col: int) -> Set[int]:
        """
        Get all possible values for a given cell
        
        This is used for constraint propagation and can help
        with more sophisticated solving strategies
        """
        possible = set(range(1, 10))
        
        # Remove values in same row
        for c in range(9):
            possible.discard(self.grid[row][c])
        
        # Remove values in same column
        for r in range(9):
            possible.discard(self.grid[r][col])
        
        # Remove values in same 3x3 box
        box_row = (row // 3) * 3
        box_col = (col // 3) * 3
        
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                possible.discard(self.grid[r][c])
        
        return possible
    
    def solve(self) -> bool:
        """
        Solve the Sudoku puzzle using backtracking
        
        Algorithm:
        1. Find an empty cell
        2. Try numbers 1-9 in that cell
        3. For each number, check if it's valid
        4. If valid, place it and recursively solve the rest
        5. If recursive call succeeds, we're done
        6. If recursive call fails, backtrack (remove the number and try next)
        7. If no number works, return False (trigger backtracking)
        
        Returns:
            True if puzzle is solved, False if no solution exists
        """
        self.solve_steps += 1
        
        # Find next empty cell
        empty_cell = self.find_empty_cell()
        
        # If no empty cell found, puzzle is solved
        if empty_cell is None:
            return True
        
        row, col = empty_cell
        
        # Try numbers 1-9
        for num in range(1, 10):
            if self.is_valid_move(row, col, num):
                # Place the number (make a choice)
                self.grid[row][col] = num
                
                # Recursively solve with this choice
                if self.solve():
                    return True
                
                # If recursive call failed, backtrack
                # (undo the choice and try next number)
                self.grid[row][col] = 0
        
        # If no number worked, return False to trigger backtracking
        return False
    
    def solve_with_heuristics(self) -> bool:
        """
        Enhanced solving with Most Constraining Variable heuristic
        
        This version chooses the empty cell with fewest possible values,
        which can significantly reduce the search space
        """
        self.solve_steps += 1
        
        # Find empty cell with fewest possibilities (MCV heuristic)
        best_cell = None
        min_possibilities = 10
        
        for row in range(9):
            for col in range(9):
                if self.grid[row][col] == 0:
                    possibilities = len(self.get_possible_values(row, col))
                    if possibilities < min_possibilities:
                        min_possibilities = possibilities
                        best_cell = (row, col)
                    
                    # If cell has no possibilities, puzzle is unsolvable
                    if possibilities == 0:
                        return False
        
        # If no empty cell found, puzzle is solved
        if best_cell is None:
            return True
        
        row, col = best_cell
        possible_values = self.get_possible_values(row, col)
        
        # Try each possible value
        for num in possible_values:
            # Place the number
            self.grid[row][col] = num
            
            # Recursively solve
            if self.solve_with_heuristics():
                return True
            
            # Backtrack
            self.grid[row][col] = 0
        
        return False
    
    def load_puzzle(self, puzzle: List[List[int]]) -> None:
        """Load a puzzle into the solver"""
        self.grid = [row[:] for row in puzzle]  # Deep copy
        self.solve_steps = 0
    
    def print_grid(self) -> None:
        """Print the current grid state in a readable format"""
        print("+" + "-" * 21 + "+")
        for i, row in enumerate(self.grid):
            if i % 3 == 0 and i != 0:
                print("|" + "-" * 21 + "|")
            
            row_str = "| "
            for j, cell in enumerate(row):
                if j % 3 == 0 and j != 0:
                    row_str += "| "
                row_str += str(cell if cell != 0 else ".") + " "
            row_str += "|"
            print(row_str)
        print("+" + "-" * 21 + "+")
    
    def is_complete(self) -> bool:
        """Check if the puzzle is completely solved"""
        return self.find_empty_cell() is None
    
    def validate_solution(self) -> bool:
        """Validate that the current grid is a valid complete solution"""
        if not self.is_complete():
            return False
        
        # Check all rows, columns, and boxes
        for i in range(9):
            # Check row i
            if len(set(self.grid[i])) != 9:
                return False
            
            # Check column i
            column = [self.grid[row][i] for row in range(9)]
            if len(set(column)) != 9:
                return False
            
            # Check box i
            box_row = (i // 3) * 3
            box_col = (i % 3) * 3
            box = []
            for r in range(box_row, box_row + 3):
                for c in range(box_col, box_col + 3):
                    box.append(self.grid[r][c])
            if len(set(box)) != 9:
                return False
        
        return True

def demonstrate_backtracking():
    """Demonstrate backtracking algorithm with example puzzles"""
    
    print("🧩 Sudoku Solver - Backtracking Algorithm Demonstration")
    print("=" * 60)
    
    # Example puzzle (0 represents empty cells)
    easy_puzzle = [
        [5, 3, 0, 0, 7, 0, 0, 0, 0],
        [6, 0, 0, 1, 9, 5, 0, 0, 0],
        [0, 9, 8, 0, 0, 0, 0, 6, 0],
        [8, 0, 0, 0, 6, 0, 0, 0, 3],
        [4, 0, 0, 8, 0, 3, 0, 0, 1],
        [7, 0, 0, 0, 2, 0, 0, 0, 6],
        [0, 6, 0, 0, 0, 0, 2, 8, 0],
        [0, 0, 0, 4, 1, 9, 0, 0, 5],
        [0, 0, 0, 0, 8, 0, 0, 7, 9]
    ]
    
    # Hard puzzle for comparison
    hard_puzzle = [
        [0, 0, 0, 6, 0, 0, 4, 0, 0],
        [7, 0, 0, 0, 0, 3, 6, 0, 0],
        [0, 0, 0, 0, 9, 1, 0, 8, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 5, 0, 1, 8, 0, 0, 0, 3],
        [0, 0, 0, 3, 0, 6, 0, 4, 5],
        [0, 4, 0, 2, 0, 0, 0, 6, 0],
        [9, 0, 3, 0, 0, 0, 0, 0, 0],
        [0, 2, 0, 0, 0, 0, 1, 0, 0]
    ]
    
    solver = SudokuSolver()
    
    # Test easy puzzle
    print("📋 Testing Easy Puzzle:")
    solver.load_puzzle(easy_puzzle)
    print("\nOriginal puzzle:")
    solver.print_grid()
    
    start_time = time.time()
    success = solver.solve()
    solve_time = time.time() - start_time
    
    if success:
        print(f"\n✅ Solved in {solver.solve_steps} steps ({solve_time:.4f} seconds)")
        print("\nSolution:")
        solver.print_grid()
        print(f"Valid solution: {solver.validate_solution()}")
    else:
        print("❌ No solution found!")
    
    # Test hard puzzle with heuristics
    print("\n" + "=" * 60)
    print("📋 Testing Hard Puzzle (with heuristics):")
    solver.load_puzzle(hard_puzzle)
    print("\nOriginal puzzle:")
    solver.print_grid()
    
    start_time = time.time()
    success = solver.solve_with_heuristics()
    solve_time = time.time() - start_time
    
    if success:
        print(f"\n✅ Solved in {solver.solve_steps} steps ({solve_time:.4f} seconds)")
        print("\nSolution:")
        solver.print_grid()
        print(f"Valid solution: {solver.validate_solution()}")
    else:
        print("❌ No solution found!")
    
    print("\n🧠 Algorithm Analysis:")
    print("- Time Complexity: O(9^(n²)) worst case, much better with pruning")
    print("- Space Complexity: O(n²) for recursion stack")
    print("- Strategy: Depth-first search with constraint checking")
    print("- Optimization: Most Constraining Variable (MCV) heuristic")
    print("- Pruning: Early termination when constraints violated")

if __name__ == "__main__":
    demonstrate_backtracking() 