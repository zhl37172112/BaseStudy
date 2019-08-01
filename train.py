import Net
from LineDataset import *
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import torch
from torchvision import transforms
from torch.autograd import Variable
import os

def train_step(model, batched_sample, class_num):
    imgs = batched_sample['image'].cuda()
    labels = batched_sample['label'].long()
    temp_batch_size = labels.size()[0]
    ont_hot_labels = Variable(torch.zeros(temp_batch_size, class_num).scatter_(1, labels, 1).float().cuda())
    imgs = Variable(imgs)
    pre_labels = model(imgs)
    classify_loss = criterion(pre_labels, ont_hot_labels)
    det_loss = 0
    det_ratio = 0.001
    # det_ratio = 0
    det_losses = []
    for weight_name, weight_value in model.named_parameters():
        if weight_name.startswith('conv') and weight_name.endswith('weight'):
            weight2d = weight_value.view(weight_value.shape[0], -1)
            weight_norm = torch.unsqueeze(torch.norm(weight2d, dim=1), 0)
            weight_norm_muti = weight_norm.t().mm(weight_norm)

            muti_weight = weight2d.mm(weight2d.t())
            if weight_name not in inv_eyes:
                inv_eyes[weight_name] = (1 - torch.eye(muti_weight.shape[0], muti_weight.shape[1])).cuda()
            curr_det_loss = torch.sum(torch.abs(muti_weight * inv_eyes[weight_name] / weight_norm_muti))
            det_loss += curr_det_loss
            det_losses.append(curr_det_loss)
            pass
    det_loss = det_loss * det_ratio
    optimizer.zero_grad()
    # loss = classify_loss + det_loss
    loss = classify_loss
    loss.backward()
    upgrade_optimizer_step(optimizer, loss)
    # optimizer.step()
    return loss, classify_loss, det_loss, det_losses

def upgrade_optimizer_step(optimizer, loss):
    for group in optimizer.param_groups:
        weight_decay = group['weight_decay']
        momentum = group['momentum']
        dampening = group['dampening']
        nesterov = group['nesterov']

        for p in group['params']:
            if p.grad is None:
                continue
            d_p = torch.sign(p.grad.data) * torch.abs(p.data)
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            # if momentum != 0:
            #     param_state = self.state[p]
            #     if 'momentum_buffer' not in param_state:
            #         buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
            #         buf.mul_(momentum).add_(d_p)
            #     else:
            #         buf = param_state['momentum_buffer']
            #         buf.mul_(momentum).add_(1 - dampening, d_p)
            #     if nesterov:
            #         d_p = d_p.add(momentum, buf)
            #     else:
            #         d_p = buf
            p.data.add_(-group['lr'] * abs(loss.cpu().item()) * 5, d_p)

def test(model, data_loader):
    right_num = torch.Tensor([0]).cuda()
    all_num = torch.Tensor([0]).cuda()
    for batched_sample in data_loader:
        imgs = batched_sample['image'].cuda()
        labels = batched_sample['label'].long().cuda()
        pred_lables = model(imgs)
        pred_lables = torch.argmax(pred_lables, 1, keepdim=True)
        right_num += (pred_lables == labels).sum()
        all_num += labels.shape[0]
    accuracy = (right_num / all_num).cpu()
    return accuracy

if __name__ == '__main__':
    trainset_path = 'samples_0.txt'
    testset_path = 'samples_1.txt'
    ckpt_dir = './check_point'
    composed = transforms.Compose([Normalize(),
                                   ToTensor()])
    batch_size = 32
    class_num = 2
    inv_eyes = {}
    train_dataset = LineDataset(trainset_path, transform=composed)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_dataset = LineDataset(testset_path, transform=composed)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    model = Net.Net1().cuda()
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    epoch_size = 100
    for epoch in range(epoch_size):
        for ibatch, batched_sample in enumerate(train_dataloader):
            loss, classify_loss, det_loss, det_losses = train_step(model, batched_sample, class_num)
            # params = model.state_dict()
            if ibatch % 20 == 0:
                print('Epoch [{} / {}] Batch [{}], loss: {:.6f}, classify_loss: {:.6f}, det_loss: {:.6f}'
                      .format(epoch + 1, epoch_size, ibatch, loss.item(), classify_loss.item(), det_loss.item()), end='')
                for i, part_det_loss in enumerate(det_losses):
                    print(', det_loss_{}: {:.6f}'.format(i, part_det_loss), end='')
                print()
        if (epoch + 1) % 20 == 0:
            if not os.path.isdir(ckpt_dir):
                os.makedirs(ckpt_dir)
            model_path = os.path.join(ckpt_dir, 'base_model_{}.pth'.format(epoch + 1))
            torch.save(model, model_path)
            accuracy = test(model, test_dataloader)
            print('Epoch [{} / {}], accuracy: {:.3f}'
                      .format(epoch + 1, epoch_size, accuracy.item()))