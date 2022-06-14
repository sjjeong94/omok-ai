import torch
import numpy as np
import omok
import onnxruntime

import models


def play(agent_player=2):
    net = models.Tower()
    net.load_state_dict(torch.load('logs/model.pth'))
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


def play_onnx(agent_player=2):
    agent = omok.OmokAgent('./logs/model.onnx')

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
