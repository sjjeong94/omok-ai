import os
import json
import omok
import zipfile
from urllib import request

download_links = [
    'https://gomocup.org/static/tournaments/2018/results/gomocup2018results.zip',
    'https://gomocup.org/static/tournaments/2019/results/gomocup2019results.zip',
    'https://gomocup.org/static/tournaments/2020/results/gomocup2020results.zip',
    'https://gomocup.org/static/tournaments/2021/results/gomocup2021results.zip',
]


def download_data(
    data_root='data_gomocup',
    save_path='data/data.json'
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

    dirname, basename = os.path.split(save_path)
    os.makedirs(dirname)
    with open(save_path, 'w') as f:
        json.dump(logs, f, separators=(',', ':'))


if __name__ == '__main__':
    download_data()
