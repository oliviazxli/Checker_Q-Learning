import sys, os, pygame
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    # Importing game logic, AI components, and GUI
    from Board_operations import Board, check_jump_required, update_board, check_win, check_tie, get_legal_moves
    from Q_Learning import DQNAgent, encode_board
    from bitboard_converter import convert_to_bitboard
    from Gui import Gui
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit()


def main():
    # Initialize Pygame and set up the display window
    pygame.init()
    size = (800, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Checkers")
    clock = pygame.time.Clock()

    # Initialize the board data and the Graphical User Interface (GUI)
    board_obj = Board()
    # Gui(board, window_size, clock, screen, start_player)
    gui = Gui(board_obj.board, size, clock, screen, 2)

    # 1. Start Menu: Let the human choose their side
    human_color = gui.draw_start_menu()  # 1 = White, 2 = Black
    ai_color = 2 if human_color == 1 else 1

    # 2. Initialize the DQNAgent
    agent = DQNAgent(player_id=ai_color)

    # Model Selection: Prioritize the 'best' saved model; fall back to 'last' if best is missing
    model_path = "dqn_best.pth" if os.path.exists("dqn_best.pth") else "dqn_last.pth"

    if os.path.exists(model_path):
        try:
            agent.load_model(model_path)
            print(f"Successfully loaded {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Troubleshooting: Delete old .pth files and retrain to ensure architecture compatibility.")
            sys.exit()
    else:
        print("No model file found. AI will play using random initialization.")

    # Set Epsilon to 0 for pure competitive mode (no random exploration)
    agent.epsilon = 0.0

    # 3. Game Start Logic: Black pieces (ID 2) always move first in Checkers
    current_player = 2
    game_history = []  # Used for detecting ties/draws via board repetition
    running = True

    while running:
        # Standard Pygame event loop to handle window closing
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        # Update board rendering
        gui.draw()
        # Record the current board state in bitboard format for tie detection
        game_history.append(convert_to_bitboard(board_obj.board))

        # Turn Management
        if current_player == human_color:
            # Human Player Turn
            # turn_continues is True if a multi-jump is required
            turn_continues, chosen_move = gui.choose_action()
            if chosen_move:
                # Highlight the move on the GUI (start and end blocks)
                gui.red_blocks = [chosen_move[0], chosen_move[1]]
                if not turn_continues:
                    current_player = ai_color
        else:
            # AI Player Turn
            gui.current_turn_text = "AI Thinking..."
            gui.draw()
            pygame.time.delay(600)

            # Fetch all legal moves for the AI based on current board state
            moves = get_legal_moves(board_obj.board, ai_color)
            if not moves:
                # If no moves are available, the AI loses
                gui.win_messsage = "YOU WIN!"
                gui.draw()
                break

            # AI predicts the best move using the loaded Dueling DQN model
            action = agent.choose_action(board_obj.board, moves)
            # Apply move to board; is_jump returns True if a piece was captured
            is_jump = update_board(action[0], action[1], board_obj.board)

            # Highlight the AI's move in blue
            gui.blue_blocks = [action[0], action[1]]

            # Multi-jump Logic: If the AI jumped, check if the same piece can jump again
            if not is_jump or not check_jump_required(board_obj.board, ai_color, action[1]):
                current_player = human_color

        # Win/Loss/Tie Conditions
        winner = check_win(board_obj.board, current_player)
        if winner != 0:
            gui.win_messsage = "YOU WIN!" if winner == human_color else "AI WINS!"
            gui.draw()
            break

        # Check for draws (e.g., repeating board states or insufficient material)
        if check_tie(game_history):
            gui.win_messsage = "DRAW!"
            gui.draw()
            break

        clock.tick(60)

    # End of game delay to allow user to see the result
    print("Game Over. Closing in 5 seconds...")
    pygame.time.wait(5000)
    pygame.quit()


if __name__ == "__main__":
    main()