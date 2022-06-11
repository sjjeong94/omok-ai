import torch
import numpy as np
import omok
import onnxruntime

import models


def play(agent_player=2):
    net = models.Tower()
    net.load_state_dict(torch.load('logs/model_t_augment_10.pth'))
    net = net.eval()

    game = omok.OmokGame()
    while game():
        if game.env.get_player() == agent_player:
            state = game.env.get_state()
            player = game.env.get_player()
            opponent = player ^ 3
            board = np.int8(state == player) - np.int8(state == opponent)
            x = board.astype(np.float32)
            x = torch.from_numpy(x)
            x = x.unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                out = net(x).squeeze().numpy()
                action = np.argmax(out)
            result = game.env(action)


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


def play_onnx(agent_player=2):
    agent = Agent('./logs/model_t_augment_20.onnx')

    game = omok.OmokGame()
    while game():
        if game.env.get_player() == agent_player:
            state = game.env.get_state()
            player = game.env.get_player()
            action = agent(state, player)
            result = game.env(action)


if __name__ == '__main__':
    # play()
    play_onnx()
