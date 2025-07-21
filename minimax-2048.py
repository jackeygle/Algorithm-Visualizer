"""
🎯 2048 AI - Minimax Algorithm with Alpha-Beta Pruning
=====================================================

Minimax is a decision-making algorithm for turn-based games. It assumes
both players play optimally and chooses the move that maximizes the 
minimum guaranteed outcome.

Time Complexity: O(b^d) where b=branching factor, d=depth
Space Complexity: O(d) for recursion stack
With Alpha-Beta Pruning: Average case much better, worst case same

Key Concepts:
1. Game tree exploration with alternating min/max layers
2. Alpha-Beta pruning for efficiency
3. Heuristic evaluation function for non-terminal states
4. Expectimax variant for stochastic games (random tile placement)
"""

import random
import time
from typing import List, Tuple, Optional
import copy

class Game2048:
    """2048 game implementation with AI using Minimax algorithm"""
    
    def __init__(self):
        self.grid = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.move_count = 0
        
        # Initialize with two random tiles
        self.add_random_tile()
        self.add_random_tile()
    
    def clone(self) -> 'Game2048':
        """Create a deep copy of the game state"""
        new_game = Game2048()
        new_game.grid = [row[:] for row in self.grid]
        new_game.score = self.score
        new_game.move_count = self.move_count
        return new_game
    
    def add_random_tile(self) -> bool:
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = [(r, c) for r in range(4) for c in range(4) if self.grid[r][c] == 0]
        
        if not empty_cells:
            return False
        
        r, c = random.choice(empty_cells)
        self.grid[r][c] = 2 if random.random() < 0.9 else 4
        return True
    
    def slide_and_merge_row(self, row: List[int]) -> Tuple[List[int], int]:
        """
        Slide and merge a single row to the left
        
        Returns:
            Tuple of (new_row, score_gained)
        """
        # Remove zeros
        filtered = [x for x in row if x != 0]
        
        # Merge adjacent equal tiles
        merged = []
        score_gained = 0
        i = 0
        
        while i < len(filtered):
            if i + 1 < len(filtered) and filtered[i] == filtered[i + 1]:
                # Merge tiles
                merged_value = filtered[i] * 2
                merged.append(merged_value)
                score_gained += merged_value
                i += 2  # Skip next tile
            else:
                merged.append(filtered[i])
                i += 1
        
        # Pad with zeros
        while len(merged) < 4:
            merged.append(0)
        
        return merged, score_gained
    
    def move(self, direction: str) -> bool:
        """
        Execute a move in the given direction
        
        Args:
            direction: 'left', 'right', 'up', or 'down'
            
        Returns:
            True if move was valid and changed the board, False otherwise
        """
        original_grid = [row[:] for row in self.grid]
        total_score_gained = 0
        
        if direction == 'left':
            for r in range(4):
                self.grid[r], score_gained = self.slide_and_merge_row(self.grid[r])
                total_score_gained += score_gained
        
        elif direction == 'right':
            for r in range(4):
                reversed_row = self.grid[r][::-1]
                new_row, score_gained = self.slide_and_merge_row(reversed_row)
                self.grid[r] = new_row[::-1]
                total_score_gained += score_gained
        
        elif direction == 'up':
            for c in range(4):
                column = [self.grid[r][c] for r in range(4)]
                new_column, score_gained = self.slide_and_merge_row(column)
                for r in range(4):
                    self.grid[r][c] = new_column[r]
                total_score_gained += score_gained
        
        elif direction == 'down':
            for c in range(4):
                column = [self.grid[r][c] for r in range(4)][::-1]
                new_column, score_gained = self.slide_and_merge_row(column)
                new_column = new_column[::-1]
                for r in range(4):
                    self.grid[r][c] = new_column[r]
                total_score_gained += score_gained
        
        # Check if board changed
        if self.grid != original_grid:
            self.score += total_score_gained
            self.move_count += 1
            return True
        
        return False
    
    def get_available_moves(self) -> List[str]:
        """Get list of valid moves that would change the board"""
        moves = []
        directions = ['left', 'right', 'up', 'down']
        
        for direction in directions:
            game_copy = self.clone()
            if game_copy.move(direction):
                moves.append(direction)
        
        return moves
    
    def is_game_over(self) -> bool:
        """Check if no moves are available"""
        return len(self.get_available_moves()) == 0
    
    def get_empty_cells(self) -> List[Tuple[int, int]]:
        """Get list of empty cell coordinates"""
        return [(r, c) for r in range(4) for c in range(4) if self.grid[r][c] == 0]

class MinimaxAI:
    """Minimax AI for 2048 with multiple evaluation heuristics"""
    
    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth
        self.nodes_evaluated = 0
    
    def evaluate_board(self, game: Game2048) -> float:
        """
        Heuristic evaluation function for board position
        
        Combines multiple factors:
        1. Empty cells (more empty = better)
        2. Monotonicity (values increase/decrease in order)
        3. Smoothness (adjacent cells have similar values)
        4. Corner bonus (highest tile in corner)
        5. Current score
        """
        grid = game.grid
        
        # Weight factors
        empty_weight = 2.7
        monotonicity_weight = 1.0
        smoothness_weight = 0.1
        corner_weight = 0.5
        score_weight = 1.0
        
        # 1. Empty cells score
        empty_cells = len(game.get_empty_cells())
        empty_score = empty_cells * empty_weight
        
        # 2. Monotonicity score
        monotonicity_score = self.calculate_monotonicity(grid)
        
        # 3. Smoothness score
        smoothness_score = self.calculate_smoothness(grid)
        
        # 4. Corner bonus
        corner_score = self.calculate_corner_bonus(grid)
        
        # 5. Current game score
        score_bonus = game.score * score_weight
        
        total_score = (empty_score + 
                      monotonicity_score * monotonicity_weight +
                      smoothness_score * smoothness_weight +
                      corner_score * corner_weight +
                      score_bonus)
        
        return total_score
    
    def calculate_monotonicity(self, grid: List[List[int]]) -> float:
        """
        Calculate monotonicity score
        Higher when rows/columns are monotonic (increasing or decreasing)
        """
        totals = [0, 0, 0, 0]  # up, down, left, right
        
        # Check rows (left/right monotonicity)
        for r in range(4):
            current = 0
            next_pos = 1
            while next_pos < 4:
                while next_pos < 4 and grid[r][next_pos] == 0:
                    next_pos += 1
                if next_pos >= 4:
                    next_pos -= 1
                
                current_value = grid[r][current] if grid[r][current] != 0 else 0
                next_value = grid[r][next_pos] if grid[r][next_pos] != 0 else 0
                
                if current_value > next_value:
                    totals[0] += next_value - current_value
                elif next_value > current_value:
                    totals[1] += current_value - next_value
                
                current = next_pos
                next_pos += 1
        
        # Check columns (up/down monotonicity)
        for c in range(4):
            current = 0
            next_pos = 1
            while next_pos < 4:
                while next_pos < 4 and grid[next_pos][c] == 0:
                    next_pos += 1
                if next_pos >= 4:
                    next_pos -= 1
                
                current_value = grid[current][c] if grid[current][c] != 0 else 0
                next_value = grid[next_pos][c] if grid[next_pos][c] != 0 else 0
                
                if current_value > next_value:
                    totals[2] += next_value - current_value
                elif next_value > current_value:
                    totals[3] += current_value - next_value
                
                current = next_pos
                next_pos += 1
        
        return max(totals[0], totals[1]) + max(totals[2], totals[3])
    
    def calculate_smoothness(self, grid: List[List[int]]) -> float:
        """
        Calculate smoothness score
        Higher when adjacent cells have similar values
        """
        smoothness = 0
        
        for r in range(4):
            for c in range(4):
                if grid[r][c] != 0:
                    value = grid[r][c]
                    target_value = int(value ** 0.5) if value > 0 else 0
                    
                    # Check right neighbor
                    if c < 3 and grid[r][c + 1] != 0:
                        neighbor_value = int(grid[r][c + 1] ** 0.5)
                        smoothness -= abs(target_value - neighbor_value)
                    
                    # Check down neighbor
                    if r < 3 and grid[r + 1][c] != 0:
                        neighbor_value = int(grid[r + 1][c] ** 0.5)
                        smoothness -= abs(target_value - neighbor_value)
        
        return smoothness
    
    def calculate_corner_bonus(self, grid: List[List[int]]) -> float:
        """
        Calculate corner bonus
        Reward having the maximum tile in a corner
        """
        max_tile = max(max(row) for row in grid)
        
        corners = [grid[0][0], grid[0][3], grid[3][0], grid[3][3]]
        
        if max_tile in corners:
            return max_tile
        
        return 0
    
    def minimax(self, game: Game2048, depth: int, alpha: float, beta: float, 
                maximizing_player: bool) -> float:
        """
        Minimax algorithm with alpha-beta pruning
        
        Args:
            game: Current game state
            depth: Current search depth
            alpha: Alpha value for pruning
            beta: Beta value for pruning
            maximizing_player: True for player turn, False for computer turn
            
        Returns:
            Best evaluation score for this position
        """
        self.nodes_evaluated += 1
        
        # Terminal conditions
        if depth == 0 or game.is_game_over():
            return self.evaluate_board(game)
        
        if maximizing_player:
            # Player's turn - maximize score
            max_eval = float('-inf')
            
            for move in game.get_available_moves():
                game_copy = game.clone()
                game_copy.move(move)
                
                eval_score = self.minimax(game_copy, depth - 1, alpha, beta, False)
                max_eval = max(max_eval, eval_score)
                
                # Alpha-beta pruning
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Beta cutoff
            
            return max_eval
        
        else:
            # Computer's turn (random tile placement) - minimize score
            min_eval = float('inf')
            empty_cells = game.get_empty_cells()
            
            if not empty_cells:
                return self.evaluate_board(game)
            
            # Consider placing 2 or 4 in each empty cell
            for r, c in empty_cells:
                for tile_value in [2, 4]:
                    game_copy = game.clone()
                    game_copy.grid[r][c] = tile_value
                    
                    # Weight by probability (90% for 2, 10% for 4)
                    probability = 0.9 if tile_value == 2 else 0.1
                    eval_score = self.minimax(game_copy, depth - 1, alpha, beta, True)
                    weighted_score = eval_score * probability
                    
                    min_eval = min(min_eval, weighted_score)
                    
                    # Alpha-beta pruning
                    beta = min(beta, weighted_score)
                    if beta <= alpha:
                        break
                
                if beta <= alpha:
                    break
            
            return min_eval
    
    def get_best_move(self, game: Game2048) -> Optional[str]:
        """
        Get the best move using minimax algorithm
        
        Returns:
            Best move direction, or None if no moves available
        """
        self.nodes_evaluated = 0
        available_moves = game.get_available_moves()
        
        if not available_moves:
            return None
        
        best_move = None
        best_score = float('-inf')
        
        for move in available_moves:
            game_copy = game.clone()
            game_copy.move(move)
            
            score = self.minimax(game_copy, self.max_depth - 1, 
                               float('-inf'), float('inf'), False)
            
            if score > best_score:
                best_score = score
                best_move = move
        
        return best_move

def demonstrate_minimax():
    """Demonstrate minimax algorithm with 2048"""
    
    print("🎯 2048 AI - Minimax Algorithm Demonstration")
    print("=" * 55)
    
    # Create game and AI
    game = Game2048()
    ai = MinimaxAI(max_depth=4)
    
    print("Initial board:")
    for row in game.grid:
        print([f"{cell:4d}" if cell != 0 else "   ." for cell in row])
    print()
    
    # Play several moves
    moves_played = 0
    max_moves = 10
    
    while moves_played < max_moves and not game.is_game_over():
        print(f"Move {moves_played + 1}:")
        print(f"Score: {game.score}")
        
        # Get AI move
        start_time = time.time()
        best_move = ai.get_best_move(game)
        think_time = time.time() - start_time
        
        if best_move:
            print(f"AI chooses: {best_move}")
            print(f"Nodes evaluated: {ai.nodes_evaluated}")
            print(f"Think time: {think_time:.3f}s")
            
            # Execute move
            game.move(best_move)
            game.add_random_tile()
            
            # Show board
            for row in game.grid:
                print([f"{cell:4d}" if cell != 0 else "   ." for cell in row])
            print()
            
            moves_played += 1
        else:
            print("No moves available!")
            break
    
    max_tile = max(max(row) for row in game.grid)
    print(f"Final Results:")
    print(f"Score: {game.score}")
    print(f"Max tile: {max_tile}")
    print(f"Moves played: {moves_played}")
    
    print("\n🧠 Algorithm Analysis:")
    print("- Time Complexity: O(b^d) where b=branching factor, d=depth")
    print("- Space Complexity: O(d) for recursion stack")
    print("- Strategy: Minimax with expectimax for stochastic elements")
    print("- Optimization: Alpha-beta pruning reduces search space")
    print("- Heuristics: Empty cells, monotonicity, smoothness, corners")

if __name__ == "__main__":
    demonstrate_minimax() 