import os
import time
import random
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import models


def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as f:
            self.states = np.load(f).astype(np.float32)
            self.moves = np.load(f)

    def __len__(self):
        return len(self.moves)

    def __getitem__(self, idx):
        s = self.states[idx]
        m = self.moves[idx]
        return s, m


def train(
    data_path='./data/data_augmented.npy',
    logs_path='./logs',
    seed=1234,
    epochs=10,
    batch_size=128,
    learning_rate=0.001,
):
    set_seed(seed)
    os.makedirs(logs_path, exist_ok=True)

    train_dataset = Dataset(data_path)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = models.Tower()
    net = net.to(device)
    print(net)

    optimizer = torch.optim.AdamW(net.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    print('%12s %12s %12s %12s' %
          ('Epoch', 'Time', 'Train Loss', 'Train Acc.'))

    time_total = 0
    for epoch in range(epochs):
        # Train
        t0 = time.time()
        net = net.train()
        losses = 0
        corrects = 0
        for x, y in tqdm(train_loader):
            x = x.to(device).unsqueeze(1)  # add channel dimension
            y = y.to(device)

            optimizer.zero_grad()
            out = net(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            losses += loss.detach()
            correct = torch.sum(torch.argmax(out, axis=1) == y)
            corrects += correct.detach()
        loss_train = losses / len(train_loader)
        acc_train = corrects / len(train_loader.dataset)
        t1 = time.time()
        time_train = t1 - t0

        time_total += time_train
        print('%12d %12.4f %12.4f %12.4f' %
              (epoch+1, time_total, loss_train, acc_train))

    # Save the model
    model_file = os.path.join(logs_path, 'model.pth')
    torch.save(net.cpu().state_dict(), model_file)
    print('Model -> ', model_file)


if __name__ == '__main__':
    train(epochs=1)
