from unsloth import FastLanguageModel, PatchFastRL
PatchFastRL("GRPO", FastLanguageModel)

from unsloth import is_bfloat16_supported
import torch
import re
from datasets import load_dataset, Dataset
import numpy as np
from typing import List, Dict, Tuple
from trl import GRPOConfig, GRPOTrainer
from vllm import SamplingParams

HF_TOKEN = "hf_XXXXXX" # Replace with your Hugging Face token

max_seq_length = 8192 # Can increase for longer reasoning traces
lora_rank = 256 # Larger rank = smarter, but slower

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "Qwen/Qwen2.5-0.5B-Instruct", #"meta-llama/Llama-3.2-1B-Instruct", #Qwen2.5-3B-Instruct",
    max_seq_length = max_seq_length,
    load_in_4bit = True, # False for LoRA 16bit
    fast_inference = True, # Enable vLLM fast inference
    max_lora_rank = lora_rank,
    gpu_memory_utilization = 0.5, # Reduce if out of memory
)

model = FastLanguageModel.get_peft_model(
    model,
    r = lora_rank, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ], # Remove QKVO if out of memory
    lora_alpha = lora_rank,
    use_gradient_checkpointing = "unsloth", # Enable long context finetuning
    random_state = 1999,
)

# --- System Prompt and XML Format ---
SYSTEM_PROMPT = """You are a master Connect Four strategist whose goal is to win while preventing your opponent from winning. The game is played on a 6x7 grid (columns a–g, rows 1–6 with 1 at the bottom) where pieces drop to the lowest available spot.

Board:
- Represented as a list of occupied cells in the format: <column><row>(<piece>), e.g., 'a1(O)'.
- For example: 'a1(O), a2(X), b1(O)' indicates that cell a1 has an O, a2 has an X, and b1 has an O.
- An empty board is shown as 'Empty Board'.
- Win by connecting 4 pieces in any direction (horizontal, vertical, or diagonal).

Strategy:
1. Identify taken positions, and empty positions.
2. Find and execute winning moves.
3. If There isn't a winning move, then block your opponent’s potential wins.
4. Control the center and set up future moves.

Respond in XML:
<reasoning>
Explain your thought process, focusing on your winning move, how you block your opponent, and your strategic plans.
</reasoning>
<move>
Specify the column letter (a–g) for your next move.
</move>
"""

def extract_xml_move(text: str) -> str:
    """
    Extracts the move (a single column letter a–g) from the XML format
    using an improved regex. This function is kept simple for reuse.
    """
    import re
    match = re.search(r'<move>\s*([a-g])\s*</move>', text)
    if match:
        return match.group(1)
    return ""

def convert_moves_to_coordinate_list(moves_list: List[str]) -> str:
    """
    Converts a list of moves to a coordinate list representation.
    Each move is formatted as <column><row>(<piece>).
    Returns "Empty" if no moves are present.
    """
    # Create an empty 6x7 grid (row 1 is at index 0)
    grid = [['.' for _ in range(7)] for _ in range(6)]
    
    for i, move in enumerate(moves_list):
        if not move:
            continue
        col = ord(move[0]) - ord('a')
        # Find the lowest available row in this column:
        for row in range(6):
            if grid[row][col] == '.':
                grid[row][col] = 'X' if i % 2 == 0 else 'O'
                break
    
    # Build coordinate list: Only include cells with a piece.
    coords = []
    for row in range(6):
        for col in range(7):
            if grid[row][col] != '.':
                # Convert row index to board row number (row 0 -> 1, etc.)
                coords.append(f"{chr(col + ord('a'))}{row+1}({grid[row][col]})")
    
    return ", ".join(coords) if coords else "Empty Board"

def parse_coordinate_list(board_str: str) -> List[List[str]]:
    """
    Converts a coordinate list representation (e.g., "a1(O), a2(X), b1(O)")
    into a 6x7 grid (list of lists) with row index 0 as the bottom.
    """
    grid = [['.' for _ in range(7)] for _ in range(6)]
    if not board_str.strip():
        return grid
    coords = board_str.split(",")
    for coord in coords:
        coord = coord.strip()
        # Expecting format: a1(O)
        if len(coord) < 4:
            continue
        col_letter = coord[0]
        try:
            row_number = int(coord[1])
        except ValueError:
            continue
        piece = coord[3]  # The piece inside the parentheses
        col = ord(col_letter) - ord('a')
        row = row_number - 1
        if 0 <= row < 6 and 0 <= col < 7:
            grid[row][col] = piece
    return grid

def get_available_positions(board_moves: List[str]) -> str:
    """Returns all available positions for each column in a clear format,
    reconstructing the board from a list of move strings."""
    # Initialize empty grid ('.' means empty)
    grid = [['.' for _ in range(7)] for _ in range(6)]
    
    # Fill in taken positions from the moves using the move index for parity
    for i, move in enumerate(board_moves):
        if len(move) >= 2:
            col = ord(move[0]) - ord('a')
            row = int(move[1]) - 1
            if 0 <= row < 6 and 0 <= col < 7:
                grid[row][col] = 'X' if i % 2 == 0 else 'O'
    
    # Find all available positions in each column
    available = []
    for col in range(7):
        col_letter = chr(ord('a') + col)
        positions = []
        for row in range(6):
            if grid[row][col] == '.':
                positions.append(f"{col_letter}{row + 1}")
        
        if positions:
            available.append(f"Column {col_letter}: {', '.join(positions)}")
        else:
            available.append(f"Column {col_letter}: Full")
    
    return "\n  ".join(available)

def check_win(board: List[List[str]], piece: str) -> bool:
    """Enhanced win checking with all directions."""
    directions = [(0,1), (1,0), (1,1), (1,-1)]  # horizontal, vertical, diagonals
    rows, cols = len(board), len(board[0])
    
    for r in range(rows):
        for c in range(cols):
            if board[r][c] != piece:
                continue
            for dr, dc in directions:
                count = 1
                for i in range(1, 4):
                    nr, nc = r + dr*i, c + dc*i
                    if not (0 <= nr < rows and 0 <= nc < cols):
                        break
                    if board[nr][nc] != piece:
                        break
                    count += 1
                if count >= 4:
                    return True
    return False

def create_enhanced_training_example(game_text: str, outcome: str) -> List[Dict]:
    """Enhanced training example with better game state representation."""
    examples = []
    turns = game_text.strip().split(' ')
    board_moves = []
    x_moves = []
    o_moves = []
    
    for turn_idx in range(0, len(turns), 3):
        # Process first player's move (X)
        if turn_idx + 1 < len(turns):
            move = turns[turn_idx + 1]
            current_board = convert_moves_to_coordinate_list(board_moves)
            next_positions = get_available_positions(board_moves)
            
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""Game State:
- You are playing as: X
- Your previous moves: {', '.join(x_moves)}
- Opponent's moves: {', '.join(o_moves)}
- Current board state: {current_board}
- Next available position per column:
  {next_positions}

Make your move."""}
            ]
            examples.append({
                "prompt": prompt,
                "answer": move[0],
                "board_state": current_board,
                "player": 1
            })
            board_moves.append(move)
            x_moves.append(move)
        
        # Process second player's move (O)
        if turn_idx + 2 < len(turns):
            move = turns[turn_idx + 2]
            current_board = convert_moves_to_coordinate_list(board_moves)
            next_positions = get_available_positions(board_moves)
            
            prompt = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"""Game State:
- You are playing as: O
- Your previous moves: {', '.join(o_moves)}
- Opponent's moves: {', '.join(x_moves)}
- Current board state: {current_board}
- Next available position per column:
  {next_positions}

Make your move."""}
            ]
            examples.append({
                "prompt": prompt,
                "answer": move[0],
                "board_state": current_board,
                "player": 2
            })
            board_moves.append(move)
            o_moves.append(move)
    
    return examples

# Dataset creation
def connect_four_dataset(split="train") -> Dataset:
    """Enhanced dataset creation with coordinate list board representation."""
    data = load_dataset("Lyte/ConnectFour-T10", split=split)
    all_examples = []
    
    for item in data:
        game_text = item['text']
        examples = create_enhanced_training_example(game_text, item['outcome']) #create_training_example(game_text, item['outcome'])
        all_examples.extend(examples)

    # Filter by length
    filtered_examples = [ex for ex in all_examples 
                        if sum(len(tokenizer.encode(turn["content"])) 
                        for turn in ex['prompt']) + len(tokenizer.encode(ex['answer'])) <= max_seq_length]

    return Dataset.from_list(filtered_examples)

train_dataset = connect_four_dataset("train") # this creates the v3 dataset 
#train_dataset = load_dataset("Lyte/ConnectFour-Training-Data_v3", split="train")
#train_dataset = load_dataset("Lyte/ConnectFour-Training-Data_v2", split="train")
#train_dataset = load_dataset('Lyte/ConnectFour-Training-Data', split="train") # this is the data uploaded after connect_four_dataset was created to save time #connect_four_dataset("train")

def evaluate_win_paths(board: List[List[str]], row: int, col: int, player: str) -> float:
    """
    Evaluates the number and quality of potential winning paths (lines of 2 or 3)
    created or extended by placing a piece at (row, col).

    Args:
        board: The 6x7 game board grid AFTER the move has been made.
        row: The row index (0-5) of the placed piece.
        col: The column index (0-6) of the placed piece.
        player: The player ('X' or 'O') who made the move.

    Returns:
        A float score representing the threat level created by the move.
        Higher scores indicate more/better potential winning lines.
    """
    score = 0.0
    rows, cols = 6, 7
    # Directions: Horizontal, Vertical, Diagonal Up-Right, Diagonal Up-Left
    directions = [(0, 1), (1, 0), (1, 1), (1, -1)]

    for dr, dc in directions:
        # For each direction, check the line formed including the new piece
        line_pieces = 1  # Count the piece just placed
        open_ends = 0

        # Check positive direction (e.g., right, up, up-right, up-left)
        for i in range(1, 4):
            nr, nc = row + dr * i, col + dc * i
            if not (0 <= nr < rows and 0 <= nc < cols):
                break  # Out of bounds
            if board[nr][nc] == player:
                line_pieces += 1
            elif board[nr][nc] == '.':
                open_ends += 1
                break  # Stop counting pieces in this direction, but note the open space
            else:
                break  # Opponent piece blocks this direction

        # Check negative direction (e.g., left, down, down-left, down-right)
        for i in range(1, 4):
            nr, nc = row - dr * i, col - dc * i
            if not (0 <= nr < rows and 0 <= nc < cols):
                break  # Out of bounds
            if board[nr][nc] == player:
                line_pieces += 1
            elif board[nr][nc] == '.':
                open_ends += 1
                break  # Stop counting pieces in this direction, but note the open space
            else:
                break  # Opponent piece blocks this direction

        # --- Score the potential of this line ---
        # Note: An immediate win (line_pieces >= 4) should ideally be caught
        # by check_win before calling evaluation functions, but we handle it.

        if line_pieces >= 4:
             # This move created an immediate win. Give very high score.
             # The main winning_reward_func should prioritize this even more.
             score += 10.0 # Assign a high base score for winning path creation
        elif line_pieces == 3:
            # Creates a line of 3. Threat level depends on open ends.
            if open_ends == 1:
                score += 1.5  # Threat: Connect Four possible next turn if space is taken.
            elif open_ends >= 2:
                score += 2.5  # Strong Threat: Open three-in-a-row, winnable in two ways.
        elif line_pieces == 2:
            # Creates a line of 2. Potential depends on open ends.
            if open_ends >= 2:
                # Needs two more pieces, but space exists. Good setup potential.
                score += 0.75 # Open two-in-a-row.

    # --- Bonus for placing above own piece (Vertical setup) ---
    if row > 0 and board[row-1][col] == player:
        score += 0.1 # Small bonus for vertical stacking

    # --- Bonus for placing next to own piece (Horizontal setup) ---
    if col > 0 and board[row][col-1] == player: # Check left
        score += 0.05
    if col < cols - 1 and board[row][col+1] == player: # Check right
        score += 0.05

    return score

def evaluate_trap_potential(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Evaluates the potential for creating traps (forcing opponent into a losing position)."""
    score = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Temporarily place our piece
    board[row][col] = player
    
    # Check all columns for potential traps
    for test_col in range(7):
        # Find landing row in this column
        test_row = 0
        while test_row < 6 and board[test_row][test_col] == '.':
            test_row += 1
        test_row -= 1
        
        if test_row >= 0:  # Valid move
            # Place opponent piece
            board[test_row][test_col] = opponent
            
            # Check if we can win on our next move after opponent plays
            for next_col in range(7):
                next_row = 0
                while next_row < 6 and board[next_row][next_col] == '.':
                    next_row += 1
                next_row -= 1
                
                if next_row >= 0:  # Valid move
                    # Try our next move
                    board[next_row][next_col] = player
                    if check_win(board, player):
                        score += 2.0  # Found a winning move after opponent's move
                        board[next_row][next_col] = '.'
                        break  # One winning move is enough for this column
                    board[next_row][next_col] = '.'
            
            # Reset opponent's piece
            board[test_row][test_col] = '.'
    
    # Restore original position
    board[row][col] = '.'
    
    return score

def strategic_winning_reward_func(prompts, completions: List[List[Dict]], answer: List[str], **kwargs) -> List[float]:
    """Enhanced strategic reward function with better game state understanding."""
    rewards = []
    
    for prompt, completion, ans in zip(prompts, completions, answer):
        try:
            predicted_move = extract_xml_move(completion[0]['content'])
            if not predicted_move or len(predicted_move) != 1 or not ('a' <= predicted_move[0] <= 'g'):
                rewards.append(-1.0)
                continue
            
            # Extract game state information
            content = prompt[1]['content']
            player = 'X' if 'You are playing as: X' in content else 'O'
            board_state = re.search(r'Current board state: (.*?)\n', content).group(1)
            current_board = parse_coordinate_list(board_state)
            
            # Extract next available position for the chosen column
            col = ord(predicted_move[0]) - ord('a')
            next_positions = re.search(r'Next available position per column:\n(.*?)(?=\n\nMake)', content, re.DOTALL).group(1)
            target_position = re.search(f'{predicted_move}: {predicted_move}(\d+)', next_positions)
            
            if not target_position:
                rewards.append(-1.0)  # Column is full or invalid
                continue
                
            row = int(target_position.group(1)) - 1
            
            # Apply move
            current_board[row][col] = player
            
            # Strategic evaluation
            reward = evaluate_strategic_position(current_board, row, col, player)
            
            rewards.append(reward)
            
        except Exception as e:
            rewards.append(-1.0)
            
    return rewards

def evaluate_strategic_position(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Enhanced position evaluation with stage-specific strategic concepts."""
    reward = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Calculate total pieces and board fill percentage for dynamic game stage determination
    total_pieces = sum(1 for r in board for c in r if c != '.')
    board_fill_percentage = total_pieces / 42  # 42 is total cells in a 6x7 board
    
    # Determine game stage dynamically
    # Early: <30% filled, Mid: 30-70% filled, Late: >70% filled
    game_stage = "early" if board_fill_percentage < 0.3 else "mid" if board_fill_percentage < 0.7 else "late"
    
    # Immediate win or block is always highest priority regardless of game stage
    if check_win(board, player):
        return 10.0  # Win is always the best move
    
    # Block opponent's win
    board[row][col] = opponent  # Temporarily place opponent's piece
    if check_win(board, opponent):
        reward += 5.0  # High value for blocking
    board[row][col] = player  # Restore player's piece
    
    # Apply stage-specific evaluation
    if game_stage == "early":
        reward += early_game_eval(board, row, col, player)
    elif game_stage == "mid":
        reward += mid_game_eval(board, row, col, player)
    else:  # late game
        reward += late_game_eval(board, row, col, player)
    
    return reward

def early_game_eval(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Early game strategy focuses on center control, development, and avoiding premature threats."""
    score = 0.0
    
    # Moderate center column bonus
    if col == 3:  # Column d (center)
        score += 0.05
    elif col in [2, 4]:  # Columns c and e (near center)
        score += 0.03
    
    # Encourage base development - pieces in bottom rows provide foundation
    if row <= 1:  # Bottom two rows
        score += 0.02
    
    # Encourage move diversity to avoid predictability
    # Count existing pieces in this column
    pieces_in_column = sum(1 for r in range(6) if board[r][col] != '.')
    if pieces_in_column >= 2:
        score -= 0.03  # Slight penalty for stacking too many pieces in one column early
    
    # Early connection potential without overcommitting
    score += evaluate_early_connections(board, row, col, player)
    
    # Avoid moves that help opponent create threats
    score += evaluate_defensive_positioning(board, row, col, player) * 0.5  # Lower weight in early game
    
    return score

def mid_game_eval(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Mid game strategy focuses on creating threats, blocking opponent threats, and building structures."""
    score = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Stronger emphasis on creating own threats
    score += evaluate_win_paths(board, row, col, player) * 1.5
    
    # Moderate emphasis on blocking opponent's developing threats
    score += evaluate_opponent_threats(board, row, col, player, opponent) * 1.2
    
    # Trap creation becomes valuable in mid-game
    score += evaluate_trap_potential(board, row, col, player) * 1.0
    
    # Building connected structures for future advantage
    score += evaluate_connected_structures(board, row, col, player) * 0.8
    
    # Control of key positions (height advantage)
    if row >= 2 and sum(1 for r in range(row) if board[r][col] != '.') == row:
        score += 0.3  # Reward building on solid foundation
    
    return score

def late_game_eval(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Late game strategy focuses on forcing wins, preventing opponent wins, and tactical play."""
    score = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Heavy emphasis on creating immediate threats
    score += evaluate_win_paths(board, row, col, player) * 2.5
    
    # Creating multiple simultaneous threats is crucial
    double_threats = evaluate_double_threats(board, row, col, player)
    if double_threats > 0:
        score += double_threats * 3.0
    
    # Blocking opponent's winning paths
    score += evaluate_opponent_threats(board, row, col, player, opponent) * 2.0
    
    # Forced move sequences
    score += evaluate_forcing_sequences(board, row, col, player) * 2.0
    
    # Penalize moves that create opportunities for opponent
    score -= evaluate_opponent_opportunities(board, row, col, player, opponent) * 1.5
    
    return score

def evaluate_early_connections(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Evaluates potential for early piece connections without overcommitting."""
    score = 0.0
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    
    for dr, dc in directions:
        # Look one step in each direction
        for direction in [-1, 1]:
            r, c = row + dr * direction, col + dc * direction
            if 0 <= r < 6 and 0 <= c < 7:
                if board[r][c] == player:
                    score += 0.02  # Small bonus for connecting pieces
                elif board[r][c] == '.':
                    score += 0.01  # Smaller bonus for potential future connection
    
    return score

def evaluate_defensive_positioning(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Evaluates if a move avoids creating easy threats for opponent."""
    score = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Check if placing here would allow opponent to place above and create a threat
    if row < 5:  # Not top row
        # Temporarily place our piece
        board[row][col] = player
        
        # Check if opponent could place above
        board[row+1][col] = opponent
        
        # See if this creates a threat for opponent
        threat_value = evaluate_win_paths(board, row+1, col, opponent)
        if threat_value > 1.0:
            score -= 0.5  # Penalize moves that give opponent easy threats
        
        # Reset board
        board[row+1][col] = '.'
        board[row][col] = '.'
    
    return score

def evaluate_opponent_threats(board: List[List[str]], row: int, col: int, player: str, opponent: str) -> float:
    """Evaluates how well a move blocks opponent's developing threats."""
    score = 0.0
    
    # Save current state
    original = board[row][col]
    
    # For each of opponent's possible next moves
    for test_col in range(7):
        # Find where piece would land in this column
        test_row = 0
        while test_row < 6 and board[test_row][test_col] == '.':
            test_row += 1
        test_row -= 1
        
        if test_row >= 0:  # Valid move
            # Place opponent piece
            board[test_row][test_col] = opponent
            
            # Check for win or strong threat
            if check_win(board, opponent):
                score += 2.0  # Blocking an immediate threat
            else:
                # Check for developing threats
                threat_level = evaluate_win_paths(board, test_row, test_col, opponent)
                if threat_level > 1.0:
                    score += 0.5  # Blocking a developing threat
            
            # Restore board
            board[test_row][test_col] = '.'
    
    # Restore original state
    board[row][col] = original
    
    return score

def evaluate_connected_structures(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Evaluates creation of connected piece structures (2 or 3 in a row with space to extend)."""
    score = 0.0
    directions = [(0,1), (1,0), (1,1), (1,-1)]
    
    for dr, dc in directions:
        # Check both directions
        for direction in [-1, 1]:
            connected = 0
            spaces = 0
            
            # Count connected pieces and spaces in this direction
            for i in range(1, 4):  # Look up to 3 steps away
                r = row + dr * direction * i
                c = col + dc * direction * i
                
                if 0 <= r < 6 and 0 <= c < 7:
                    if board[r][c] == player:
                        connected += 1
                    elif board[r][c] == '.':
                        spaces += 1
                        break
                    else:
                        break
            
            # Score based on structure
            if connected == 2 and spaces >= 1:
                score += 0.4  # Three in a row with space to extend
            elif connected == 1 and spaces >= 2:
                score += 0.2  # Two in a row with space to extend
    
    return score

def evaluate_double_threats(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Evaluates if a move creates multiple winning threats simultaneously."""
    threats = 0
    
    # Save current position
    original = board[row][col]
    
    # Place player's piece
    board[row][col] = player
    
    # Find all winning moves for next turn
    for test_col in range(7):
        # Find landing row in this column
        test_row = 0
        while test_row < 6 and board[test_row][test_col] == '.':
            test_row += 1
        test_row -= 1
        
        if test_row >= 0:  # Valid move
            board[test_row][test_col] = player
            if check_win(board, player):
                threats += 1
            board[test_row][test_col] = '.'
    
    # Restore board
    board[row][col] = original
    
    return threats

def evaluate_forcing_sequences(board: List[List[str]], row: int, col: int, player: str) -> float:
    """Evaluates if a move forces opponent to play in a particular way, leading to advantage."""
    score = 0.0
    opponent = 'O' if player == 'X' else 'X'
    
    # Place our piece
    board[row][col] = player
    
    # Count forced responses (moves opponent must make to prevent loss)
    forced_responses = 0
    forced_col = -1
    
    for test_col in range(7):
        # Find landing row
        test_row = 0
        while test_row < 6 and board[test_row][test_col] == '.':
            test_row += 1
        test_row -= 1
        
        if test_row >= 0:  # Valid move
            # Place our piece in next move
            board[test_row][test_col] = player
            
            # If this creates a winning position
            if check_win(board, player):
                forced_responses += 1
                forced_col = test_col
            
            # Reset
            board[test_row][test_col] = '.'
    
    # If there's exactly one forced response
    if forced_responses == 1 and forced_col >= 0:
        # Check if this forced move gives us advantage
        force_row = 0
        while force_row < 6 and board[force_row][forced_col] == '.':
            force_row += 1
        force_row -= 1
        
        # Place opponent's forced move
        board[force_row][forced_col] = opponent
        
        # See if we can create a follow-up threat
        for follow_col in range(7):
            follow_row = 0
            while follow_row < 6 and board[follow_row][follow_col] == '.':
                follow_row += 1
            follow_row -= 1
            
            if follow_row >= 0:
                board[follow_row][follow_col] = player
                if check_win(board, player):
                    score += 2.0  # Significant bonus for forcing sequence
                board[follow_row][follow_col] = '.'
        
        # Reset forced move
        board[force_row][forced_col] = '.'
    
    # Restore original position
    board[row][col] = '.'
    
    return score

def evaluate_opponent_opportunities(board: List[List[str]], row: int, col: int, player: str, opponent: str) -> float:
    """Evaluates if a move creates opportunities for opponent."""
    opportunity_score = 0.0
    
    # Place our piece
    board[row][col] = player
    
    # If this move allows opponent to place above
    if row < 5:  # Not top row
        # Opponent places above
        board[row+1][col] = opponent
        
        # Check if this creates threats for opponent
        threat_value = evaluate_win_paths(board, row+1, col, opponent)
        if threat_value > 1.5:
            opportunity_score += 1.0
        
        # Reset
        board[row+1][col] = '.'
    
    # Restore original
    board[row][col] = '.'
    
    return opportunity_score

def validate_xml_format(response: str) -> float:
    """
    Checks if <reasoning> and <move> tags exist correctly.
    - Rewards 1.0 for perfect format.
    - Penalizes missing or extra occurrences.
    """
    expected_counts = {"<reasoning>": 1, "</reasoning>": 1, "<move>": 1, "</move>": 1}
    actual_counts = {tag: response.count(tag) for tag in expected_counts}

    reward = 1.0
    for tag, expected in expected_counts.items():
        actual = actual_counts.get(tag, 0)
        if actual != expected:
            reward -= 0.25 * abs(actual - expected)

    return reward

def length_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Rewards responses between 150 and 512 tokens."""
    rewards = []
    
    for completion in completions:
        response = completion[0]['content']
        num_tokens = len(tokenizer.encode(response))
        
        if 200 <= num_tokens <= 512:
            rewards.append(2.0)
        elif num_tokens < 200:
            penalty = (200 - num_tokens) / 200
            rewards.append(-penalty)
        else:  # num_tokens > 512
            penalty = ((num_tokens - 512) / 512) + 0.1
            rewards.append(-penalty)
    
    return rewards

'''def format_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    """Checks for <reasoning> and <move> tags."""
    rewards = []
    for completion in completions:
        response = completion[0]['content']
        if "<reasoning>" in response and "</reasoning>" in response and "<move>" in response and "</move>" in response:
            rewards.append(0.5)
        else:
            rewards.append(-0.5)
    return rewards'''

def strict_format_reward_func(completions, **kwargs) -> List[float]:
    """Strict format reward using regex."""
    rewards = []
    pattern = r"^<reasoning>\n(.*?)\n</reasoning>\n<move>\n(.*?)\n</move>$"
    for completion in completions:
        response = completion[0]["content"]
        match = re.match(pattern, response, re.DOTALL)
        if match:
            rewards.append(1.0)
        else:
            rewards.append(-0.5)
    return rewards

def strict_move_format_reward_func(completions: List[List[Dict]], **kwargs) -> List[float]:
    """
    Strictly enforces that the XML move is correctly formatted:
    - A single letter (a–g) enclosed within <move> and </move>
    - No extra non-whitespace content after the closing tag.
    
    Returns a reward of 2.0 for a perfect format; otherwise, a penalty.
    """
    rewards = []
    for completion in completions:
        response = completion[0]['content']
        move = extract_xml_move(response)
        # Check if a move was extracted and that there's no extra content after </move>
        closing_index = response.find("</move>")
        if move and closing_index != -1 and response[closing_index + len("</move>"):].strip() == "":
            rewards.append(2.0)
        else:
            rewards.append(-2.0)
    return rewards

def xml_count_reward_func(completions, **kwargs) -> List[float]:
    """Ensures the correct number of XML tags appear in the response."""
    rewards = []
    for completion in completions:
        response = completion[0]["content"]
        rewards.append(validate_xml_format(response))
    return rewards

train_dataset = train_dataset.shuffle(seed=1999)

model_path = "Lyte/QuadConnect2.5-0.5B-v0.0.9b"

training_args = GRPOConfig(
    use_vllm = True, # use vLLM for fast inference!
    learning_rate = 5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type = "cosine",
    optim = "adamw_8bit",
    logging_steps = 1,
    bf16 = is_bfloat16_supported(),
    fp16 = not is_bfloat16_supported(),
    per_device_train_batch_size = 8,
    gradient_accumulation_steps = 8, # Increase to 4 for smoother training
    num_generations = 8, # Decrease if out of memory
    max_prompt_length = 1024,
    max_completion_length = 768,
    #num_train_epochs = 1, # Set to 1 for a full training run
    max_steps = 250,
    save_steps = 50,
    max_grad_norm = 0.1,
    report_to = "tensorboard", # Can use Weights & Biases
    output_dir = model_path,
    logging_dir=model_path
)

# Update the trainer with new reward functions
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        #validity_reward_func,
        strategic_winning_reward_func,
        #format_reward_func,
        strict_format_reward_func,
        strict_move_format_reward_func,
        xml_count_reward_func,
        length_reward_func,
    ],
    args=training_args,
    train_dataset=train_dataset,
)
print("Training Started")
trainer.train()

moves = []
prompt = f"Game State:\n- You are playing as: X\n- Your previous moves: \n- Opponent's moves: b1\n- Current board state: b1(O)\n- Next available position per column:  \nColumn a: a1, a2, a3, a4, a5, a6  \nColumn b: b2, b3, b4, b5, b6  \nColumn c: c1, c2, c3, c4, c5, c6  \nColumn d: d1, d2, d3, d4, d5, d6  \nColumn e: e1, e2, e3, e4, e5, e6  \nColumn f: f1, f2, f3, f4, f5, f6  \nColumn g: g1, g2, g3, g4, g5, g6\n\nMake your move."
text = tokenizer.apply_chat_template(
    [{"role" : "system", "content" : SYSTEM_PROMPT}, {"role" : "user", "content" : prompt}],
    tokenize = False, add_generation_prompt = True)

sampling_params = SamplingParams(
    temperature = 0.8,
    top_p = 0.95,
    max_tokens = 1024,
)
output = model.fast_generate(
    text,
    sampling_params = sampling_params,
    lora_request = model.load_lora("grpo_saved_lora"),
)[0].outputs[0].text

print(output)

model.save_lora("grpo_saved_lora_v0.0.8b")

trainer.repo_id = model_path
trainer.push_to_hub(model_path)

# Merge to 16bit
if True: model.save_pretrained_merged("model", tokenizer, save_method = "merged_16bit",)
if True: model.push_to_hub_merged(model_path, tokenizer, save_method = "merged_16bit", token = HF_TOKEN)

# Save to multiple GGUF options - much faster if you want multiple!
if True:
    model.push_to_hub_gguf(
        model_path, # Change hf to your username!
        tokenizer,
        quantization_method = ["q8_0"],
        token = HF_TOKEN,
    )