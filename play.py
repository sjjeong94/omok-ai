import torch
import numpy as np
import omok
import onnxruntime

import models


def play(agent=2):
    net = models.Tower()
    net.load_state_dict(torch.load('logs/model_t_augment_10.pth'))
    net = net.eval()

    game = omok.OmokGame()
    while game():
        if game.env.get_player() == agent:
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


def play_onnx(agent=2):
    ort_session = onnxruntime.InferenceSession('./logs/model.onnx')

    game = omok.OmokGame()
    while game():
        if game.env.get_player() == agent:
            state = game.env.get_state()
            player = game.env.get_player()
            opponent = player ^ 3
            board = np.int8(state == player) - np.int8(state == opponent)

            x = board.astype(np.float32)
            x = np.reshape(x, (1, 1, 15, 15))
            ort_inputs = {ort_session.get_inputs()[0].name: x}
            ort_outs = ort_session.run(None, ort_inputs)
            out = ort_outs[0].squeeze()
            action = np.argmax(out)
            result = game.env(action)


if __name__ == '__main__':
    play_onnx(agent=2)
