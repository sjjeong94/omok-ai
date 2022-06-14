import time
import omok
import numpy as np
import onnxruntime
from tqdm import tqdm


def rollout(agent1_path, agent2_path):
    agent1 = omok.OmokAgent(agent1_path)
    agent2 = omok.OmokAgent(agent2_path)

    num_games = 1000
    count = 0
    win1 = 0
    win2 = 0
    tie = 0

    env = omok.Omok()

    for _ in tqdm(range(num_games)):
        env.reset()

        while True:
            state = env.get_state()
            player = env.get_player()

            if player == 1:
                action = agent1(state, player)
            else:
                action = agent2(state, player)

            result = env(action)
            if result:
                break

        count += 1
        winner = env.get_winner()
        if winner == 1:
            win1 += 1
        elif winner == 2:
            win2 += 1
        else:
            tie += 1

    print('Player 1 Win -> ', win1 / num_games)
    print('Player 2 Win -> ', win2 / num_games)
    print('Tie          -> ', tie / num_games)


def rollout_view(agent1_path, agent2_path):
    agent1 = omok.OmokAgent(agent1_path)
    agent2 = omok.OmokAgent(agent2_path)

    game = omok.OmokGame()

    while game():

        state = game.env.get_state()
        player = game.env.get_player()

        if player == 1:
            action = agent1(state, player)
        else:
            action = agent2(state, player)

        result = game.env(action)


if __name__ == '__main__':
    rollout_view(
        agent1_path='logs/model.onnx',
        agent2_path='logs/model.onnx',
    )
