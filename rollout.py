import time
import omok
import numpy as np
import onnxruntime
from tqdm import tqdm


class Agent:
    def __init__(self, model_path):
        self.session = onnxruntime.InferenceSession(model_path)

    def __call__(self, state, player):
        if np.random.rand() < 0.5:
            # Random Transpose
            state_t = state.T
            action_t = self.inference(state_t, player)
            y, x = divmod(action_t, 15)
            action = x * 15 + y
        else:
            action = self.inference(state, player)
        return action

    def inference(self, state, player):
        opponent = player ^ 3
        board = np.int8(state == player) - np.int8(state == opponent)
        x = board.astype(np.float32)
        x = np.reshape(x, (1, 1, 15, 15))
        outs = self.session.run(None, {'input': x})
        action = np.argmax(outs[0].squeeze())
        return action


def rollout(agent1_path, agent2_path):
    agent1 = Agent(agent1_path)
    agent2 = Agent(agent2_path)

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
    agent1 = Agent(agent1_path)
    agent2 = Agent(agent2_path)

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
    rollout(
        agent1_path='logs/model_t_augment_10.onnx',
        agent2_path='logs/model_t_augment_20.onnx'
    )
    '''
    rollout_view(
        agent1_path='logs/model_t_augment_10.onnx',
        agent2_path='logs/model_t_augment_20.onnx',
    )
    '''
