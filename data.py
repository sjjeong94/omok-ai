import json
import omok
import numpy as np
from tqdm import tqdm

import augment


def make_dataset(
    data_path='data/data.json',
    save_path='data/data.npy',
):
    print("Make Dataset...")

    with open(data_path, 'r') as f:
        pack = json.load(f)

    env = omok.Omok()
    boards = []
    moves = []

    for i in tqdm(range(len(pack))):
        data = pack[i]
        env.reset()
        for move in data['moves']:
            state = env.get_state()
            player = env.get_player()
            opponent = player ^ 3
            board = np.int8(state == player) - np.int8(state == opponent)

            boards.append(board)
            moves.append(move)

            env(move)

    boards = np.asarray(boards)
    moves = np.asarray(moves)

    with open(save_path, 'wb') as f:
        np.save(f, boards)
        np.save(f, moves)


def augment_data(
    data_path='data/data.npy',
    save_path='data/data_augmented.npy',
):
    with open(data_path, 'rb') as f:
        states = np.load(f)
        moves = np.load(f)

    states_augmented = []
    moves_augmented = []
    for i in tqdm(range(len(states))):
        state = states[i]
        move = moves[i]
        s, m = augment.augment(state, move)
        states_augmented.extend(s)
        moves_augmented.extend(m)

    states_augmented = np.asarray(states_augmented)
    moves_augmented = np.asarray(moves_augmented)

    with open(save_path, 'wb') as f:
        np.save(f, states_augmented)
        np.save(f, moves_augmented)


def data():
    make_dataset()
    augment_data()


if __name__ == '__main__':
    data()
