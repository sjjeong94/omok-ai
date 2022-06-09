import time
import omok
import torch
import numpy as np

import models


def test():

    net = models.TestNet()
    net.load_state_dict(torch.load('logs/model.pth'))
    net = net.eval()

    env = omok.Omok()
    env.show_state()

    while True:
        state = env.get_state()
        player = env.get_player()
        opponent = player ^ 3
        board = np.int8(state == player) - np.int8(state == opponent)
        x = board.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            out = net(x).squeeze().numpy()
            action = np.argmax(out)
        result = env(action)
        env.show_state()
        time.sleep(0.1)
        if result:
            break

    print('Winner: ', env.get_winner())


if __name__ == '__main__':
    test()
