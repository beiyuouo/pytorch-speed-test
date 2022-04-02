import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from dataset import RandomDataset
from net import Net
from setting import config



def main():
    print('device: {}'.format(config.device))
    

    random_dataset = RandomDataset(length=64, data_shape=(4, 128, 128))
    random_dataloader = DataLoader(random_dataset, batch_size=config.batch_size, shuffle=True)
    net = Net()
    net.to(config.device)
    print(net)
    optimizer = torch.optim.Adam(net.parameters(), lr=config.lr)
    loss_func = nn.CrossEntropyLoss()
    
    start_time = time.time()
    for epoch in range(config.epochs):
        for idx, (data, label) in enumerate(random_dataloader):
            data = data.to(config.device)
            label = label.to(config.device)
            output = net(data)
            loss = loss_func(output, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if idx % 10 == 0:
                print('epoch: {}, idx: {}, loss: {}'.format(epoch, idx, loss.item()))
    
    print('finished')
    print('time: {} sec'.format(time.time() - start_time))

if __name__ == '__main__':
    main()