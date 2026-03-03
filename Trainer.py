import os, random, torch
from Board_operations import Board, update_board, check_win, get_legal_moves
from Q_Learning import DQNAgent, encode_board

def get_reward(old_board, new_board, winner, player_id):
    # Terminal Rewards
    if winner == player_id:
        return 100.0  # Reward for winning the game
    if winner != 0 and winner != 3:
        return -100.0  # Severe penalty for losing

    reward = -0.1  # Step penalty to prevent infinite loops or stalling
    pid, eid = player_id, 3 - player_id

    def get_stats(b, p):
        pawns, kings = [], []
        for r in range(8):
            for c in range(8):
                if b[r][c] == p:
                    pawns.append((r, c))
                elif b[r][c] == p + 2:
                    kings.append((r, c))
        return pawns, kings

    o_p, o_k = get_stats(old_board, pid)
    n_p, n_k = get_stats(new_board, pid)
    eo_p, eo_k = get_stats(old_board, eid)
    en_p, en_k = get_stats(new_board, eid)

    # 2. Capture Logic, Force AI to seek multi-jump sequences
    captured = (len(eo_p) + len(eo_k)) - (len(en_p) + len(en_k))
    if captured > 1:
        reward += (captured ** 2) * 3.0  # Exponential growth: 2 pieces = 12 pts, 3 pieces = 27 pts
    elif captured == 1:
        reward += 2.0  # Standard reward for single capture

    # 3. King Promotion Logic, increase incentive for promotion
    new_kings = len(n_k) - len(o_k)
    reward += 15.0 * new_kings  # Promotion is worth roughly 5 captured pieces

    # 4. Baseline Sprint Reward (Non-linear progressive reward)
    for (r, c) in n_p:
        # Distance to target baseline (0-7)
        dist = r if player_id == 1 else (7 - r)
        # The closer to the baseline, the higher the reward per step
        # Logic: Pieces near the baseline possess potential value close to a King
        reward += (7 - dist) ** 2 / 8.0

    # 5. Defense Logic, Addresses the issue of sacrificing pieces unnecessarily
    lost = (len(o_p) + len(o_k)) - (len(n_p) + len(n_k))
    # Penalty for losing a piece outweighs the reward for taking one
    reward -= 5.0 * lost
    return reward

# Increased episodes to accommodate complex reward structure
def train(episodes=40000):
    agent = DQNAgent(player_id=1)
    opp_agent = DQNAgent(player_id=2)
    best_avg_reward = -float('inf')
    batch_reward_sum = 0.0

    if os.path.exists("dqn_best.pth"):
        agent.load_model("dqn_best.pth")

    for ep in range(1, episodes + 1):
        board_obj = Board()
        board = board_obj.board
        ai_side = random.choice([1, 2])
        agent.player_id = ai_side
        # Black pieces move first
        curr = 2

        # Self-Play model update
        if ep > 10000 and ep % 500 == 0 and os.path.exists("dqn_best.pth"):
            opp_agent.load_model("dqn_best.pth")
            opp_agent.epsilon = 0.1

        # Increase max steps per game
        for step in range(250):
            moves = get_legal_moves(board, curr)
            if not moves: break

            if curr == ai_side:
                s_feat = encode_board([r[:] for r in board], ai_side)
                action = agent.choose_action(board, moves)
                old_b = [r[:] for r in board]
                update_board(action[0], action[1], board)

                win = check_win(board, 3 - ai_side)
                rew = get_reward(old_b, board, win, ai_side)
                batch_reward_sum += rew

                agent.learn(s_feat, rew, encode_board(board, ai_side), win != 0)
                if win != 0: break
            else:
                # Hybrid Opponent: 50% chance of Self-Play after 10,000 episodes
                if ep > 10000 and random.random() < 0.5:
                    act = opp_agent.choose_action(board, moves)
                else:
                    act = random.choice(moves)
                update_board(act[0], act[1], board)
                if check_win(board, ai_side) != 0:
                    break
            curr = 3 - curr

        if ep % 500 == 0: agent.target_model.load_state_dict(agent.model.state_dict())
        agent.epsilon = max(0.05, agent.epsilon * 0.99985)  # Slower decay to allow more exploration

        if ep % 100 == 0:
            avg_r = batch_reward_sum / 100
            print(f"Ep {ep:5d} | Eps: {agent.epsilon:.3f} | AvgR: {avg_r:+.2f}")
            agent.save_model("dqn_last.pth")
            if avg_r > best_avg_reward and ep > 3000:
                best_avg_reward = avg_r
                agent.save_model("dqn_best.pth")
                print(f"  >>> [SAVED BEST] {best_avg_reward:.2f}")
            batch_reward_sum = 0.0


if __name__ == "__main__":
    train(episodes=10000)