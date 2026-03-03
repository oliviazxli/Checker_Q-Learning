"""
AI vs AI — Watch two DQN agents play checkers against each other via Pygame.
"""

import sys
import os
import argparse
import pygame
from pathlib import Path

current_dir = Path(__file__).resolve().parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))

try:
    from Board_operations import (Board, check_jump_required, update_board,
                                   check_win, check_tie, get_legal_moves)
    from Q_Learning import DQNAgent
    from bitboard_converter import convert_to_bitboard
    from Gui import Gui
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit()


# Load a DQN agent from a model file
def load_agent(player_id: int, model_path: str) -> DQNAgent:
    agent = DQNAgent(player_id=player_id)
    if os.path.exists(model_path):
        try:
            agent.load_model(model_path)
            print(f"  Player {player_id} loaded: {model_path}")
        except Exception as e:
            print(f"  Player {player_id} load failed ({e}), using random weights.")
    else:
        print(f"  Player {player_id}: '{model_path}' not found, using random weights.")
    agent.epsilon = 0.0   # Pure exploitation — no random moves during exhibition
    return agent


# Play a single game, return the winner (1, 2, or 0 for draw)
def play_one_game(agent1: DQNAgent, agent2: DQNAgent,
                  gui: Gui, board_obj: Board,
                  delay_ms: int, clock: pygame.time.Clock) -> int:

    board_obj.__init__()                    # Reset board
    gui.board = board_obj.board             # Point GUI at fresh board
    gui.win_messsage = ""

    current_player = 1
    game_history = []
    agents = {1: agent1, 2: agent2}

    while True:
        # Handle quit event so the window stays responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        gui.draw()
        game_history.append(convert_to_bitboard(board_obj.board))

        # Draw by repetition
        if check_tie(game_history):
            gui.win_messsage = "DRAW!"
            gui.draw()
            pygame.time.wait(1500)
            return 0

        agent = agents[current_player]
        moves = get_legal_moves(board_obj.board, current_player)

        # No legal moves → current player loses
        if not moves:
            winner = 3 - current_player
            msg    = "WHITE WINS!" if winner == 1 else "BLACK WINS!"
            gui.win_messsage = msg
            gui.draw()
            pygame.time.wait(1500)
            return winner

        # Choose and apply move
        pygame.time.delay(delay_ms)
        action = agent.choose_action(board_obj.board, moves)
        is_jump = update_board(action[0], action[1], board_obj.board)

        # Highlight the moved piece
        # Assuming gui.white_blocks corresponds to Player 1 moves
        if current_player == 1:
            gui.blue_blocks = [action[0], action[1]]
        else:
            # Changed from red_blocks to white_blocks to stay consistent
            gui.white_blocks = [action[0], action[1]]

        # Win check
        next_player = 3 - current_player
        winner = check_win(board_obj.board, next_player)
        if winner != 0:
            msg = "WHITE WINS!" if winner == 1 else "BLACK WINS!"
            gui.win_messsage = msg
            gui.draw()
            pygame.time.wait(1500)
            return winner

        # Multi-jump: same player moves again if jump continues
        if is_jump and check_jump_required(board_obj.board, current_player, action[1]):
            continue   # Don't switch player — chain jump

        current_player = next_player
        clock.tick(60)


# Main
def main():
    parser = argparse.ArgumentParser(description="Checkers AI vs AI")
    parser.add_argument("--p1",    default="dqn_best.pth", help="Model path for player 1 ")
    parser.add_argument("--p2",    default="dqn_last.pth", help="Model path for player 2 ")
    parser.add_argument("--games", type=int, default=1,    help="Number of games to play")
    parser.add_argument("--delay", type=int, default=500,  help="Delay between moves in ms")
    args = parser.parse_args()

    print("=== Checkers: AI vs AI ===")
    print(f"  P1 (black): {args.p1}")
    print(f"  P2 (white): {args.p2}")
    print(f"  Games:      {args.games}")
    print(f"  Move delay: {args.delay} ms")
    print()

    agent1 = load_agent(1, args.p1)
    agent2 = load_agent(2, args.p2)

    pygame.init()
    size = (800, 600)
    screen = pygame.display.set_mode(size)
    pygame.display.set_caption("Checkers")
    clock = pygame.time.Clock()

    board_obj = Board()
    # Pass player_id=0 so Gui knows neither side is human
    gui = Gui(board_obj.board, size, clock, screen, 0)

    # Score tracking
    wins = {1: 0, 2: 0, 0: 0}

    for game_num in range(1, args.games + 1):
        print(f"Game {game_num}/{args.games} ...", end=" ", flush=True)
        winner = play_one_game(agent1, agent2, gui, board_obj, args.delay, clock)
        wins[winner] += 1

        label = {1: "White wins", 2: "Black wins", 0: "Draw"}[winner]
        print(label)

    # Final summary
    print()
    print(f"  White wins: {wins[1]}")
    print(f"  Black wins: {wins[2]}")
    print(f"  Draws:      {wins[0]}")

    pygame.time.wait(3000)
    pygame.quit()


if __name__ == "__main__":

    main()
