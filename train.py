import Net
from LineDataset import LineDataset
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch

if __name__ == '__main__':
    dataset_file_path = 'samples_0.txt'
    line_dataset = LineDataset(dataset_file_path)
    line_dataloader = DataLoader(line_dataset, batch_size=32, shuffle=True, num_workers=4)
    model = Net.Net1()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3)
    for epoch in range(5):
        for ibatch, sample_batched in enumerate(line_dataloader):
            imgs = sample_batched['image']
            labels = sample_batched['label']
            labels = labels.float()
            pre_labels = model(imgs)
            loss = criterion(pre_labels, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if ibatch % 20 == 0:
                print('Epoch [{}/ 500] Batch [{}], loss: {:.6f}'
                      .format(epoch + 1, ibatch, loss.item()))
        torch.save(model, './model/base_model_{}.pth')
