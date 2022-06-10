import os
import json
import omok
import zipfile
import numpy as np
from tqdm import tqdm
from urllib import request

import augment

download_links = [
    'https://gomocup.org/static/tournaments/2018/results/gomocup2018results.zip',
    'https://gomocup.org/static/tournaments/2019/results/gomocup2019results.zip',
    'https://gomocup.org/static/tournaments/2020/results/gomocup2020results.zip',
    'https://gomocup.org/static/tournaments/2021/results/gomocup2021results.zip',
]


def prepare_data(
    data_root='data/gomocup',
):
    print("Download Data...")
    os.makedirs(data_root, exist_ok=True)
    file_names = []
    for download_link in download_links:
        file_name = download_link.split('/')[-1]
        file_names.append(file_name)
        data_path = os.path.join(data_root, file_name)
        request.urlretrieve(download_link, data_path)

    print("Unzip Data...")
    paths = []
    for file_name in file_names:
        data_path = os.path.join(data_root, file_name)
        print(data_path)
        path = data_path.split('.')[0]
        paths.append(path)
        print(path)
        with zipfile.ZipFile(data_path, 'r') as zip_ref:
            zip_ref.extractall(path)

    data_names = []
    for path in paths:
        data_path = os.path.join(path, 'Renju')
        for data_file in sorted(os.listdir(data_path)):
            if data_file.split('.')[-1] == 'psq':
                data_names.append(os.path.join(data_path, data_file))

    print("Convert Data...")
    SIZE = 15
    env = omok.Omok()
    logs = []
    for data_name in data_names:
        with open(data_name, 'r') as f:
            lines = f.read().splitlines()

        env.reset()
        for i in range(1, len(lines)):
            line = lines[i]
            if ',' not in line:
                break
            m = list(map(int, line.split(',')))
            move = (m[0]-1)*SIZE + (m[1]-1)
            result = env(move)
        log = env.get_log()
        logs.append(log)

    save_name = os.path.join(data_root, 'data.json')
    with open(save_name, 'w') as f:
        json.dump(logs, f, separators=(',', ':'))


def make_dataset(
    data_root='data/gomocup',
    save_root='data',
):
    print("Make Dataset...")

    with open(os.path.join(data_root, 'data.json'), 'r') as f:
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

    save_name = os.path.join(save_root, 'data.npy')
    with open(save_name, 'wb') as f:
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
    for state, move in tqdm(zip(states, moves)):
        s, m = augment.augment(state, move)
        states_augmented.extend(s)
        moves_augmented.extend(m)

    states_augmented = np.asarray(states_augmented)
    moves_augmented = np.asarray(moves_augmented)

    with open(save_path, 'wb') as f:
        np.save(f, states_augmented)
        np.save(f, moves_augmented)


def data():
    prepare_data()
    make_dataset()
    augment_data()


if __name__ == '__main__':
    data()
