import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import RandomDataset
from net import Net



def main():
    random_dataset = RandomDataset(length=64, data_shape=(4, 128, 128))
    random_dataloader = DataLoader(random_dataset, batch_size=4, shuffle=True)
    net = Net()
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    loss_func = nn.CrossEntropyLoss()
    epochs = 10
    for epoch in range(epochs):
        for idx, (data, label) in enumerate(random_dataloader):
            output = net(data)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('epoch: {}, idx: {}, loss: {}'.format(epoch, idx, loss.item()))
    
    print('finished')

if __name__ == '__main__':
    start_time = time.time()
    main()
    print('time: {} sec'.format(time.time() - start_time))