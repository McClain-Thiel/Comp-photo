import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import png
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from colormap.colors import Color, hex2rgb
from sklearn.metrics import average_precision_score as ap_score
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from sklearn import preprocessing


from dataset import FacadeDataset

N_CLASS=5
#torch.nn.ConvTranspose2d

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.n_class = N_CLASS
        self.layers = nn.Sequential(
            #########################################
            ###        TODO: Add more layers      ###
            #########################################
            nn.Conv2d(3, self.n_class, 1, padding=0),
            nn.ReLU(inplace=True)
        )
        self.encoder = nn.Sequential(
            # conv layer 1
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True),
            # conv layer 2
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2, ceil_mode=True)
            )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 256, 5),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(256, 256, 1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(),
            nn.Conv2d(256, self.n_class, 1)
            )


    def forward(self, x):
        temp = self.encoder(x)
        output = self.decoder(temp)
        output = F.interpolate(output, (256,256), mode='bicubic')
        return output

class u_net(nn.Module):
    def __init__(self):
        super(u_net, self).__init__()
        self.n_class = N_CLASS
        self.encoder0 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1))

        self.encoder1 = nn.Sequential(
            nn.MaxPool2d(2),            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.encoder2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.Conv2d(128,128,3,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.upsample1 = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.upconv1 = nn.Conv2d(256, 64, 3, padding=1)

        self.upsample2 = nn.Upsample(scale_factor=2, mode = 'bilinear', align_corners=True)
        self.upconv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.classify = nn.Conv2d(64, self.n_class, kernel_size = 3, padding=1)

    def forward(self, x):
         skip0 = self.encoder0(x)
         skip1 = self.encoder1(skip0)
         skip2 = self.encoder2(skip1)

         #first skip
         temp = self.upsample1(skip2)
         temp = torch.cat([temp, skip1], dim=1)
         out1 = self.upconv1(temp)

         #second skip
         temp2 = self.upsample2(out1)
         temp2 = torch.cat([temp2, skip0], dim=1)
         out2 = self.upconv2(temp2)

         output = self.classify(out2)
         return output


def custom_loss(input_, target, weight = None, size_average=True):
    n, c, h, w = input_.size()
    log_p = F.log_softmax(input_, dim=1)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)
    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, reduction='sum')
    if size_average:
        loss /= mask.data.sum()
    return loss

def save_label(label, path):
    '''
    Function for ploting labels.
    '''
    colormap = [
        '#000000',
        '#0080FF',
        '#80FF80',
        '#FF8000',
        '#FF0000',
    ]
    assert(np.max(label)<len(colormap))
    colors = [hex2rgb(color, normalise=False) for color in colormap]
    w = png.Writer(label.shape[1], label.shape[0], palette=colors, bitdepth=4)
    with open(path, 'wb') as f:
        w.write(f, label)

def train(trainloader, net, criterion, optimizer, device, epoch):
    '''
    Function for training.
    '''
    start = time.clock()
    running_loss = 0.0
    net = net.train()
    for images, labels in tqdm(trainloader):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = net(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        running_loss = loss.item()
    end = time.clock()
    print('[epoch %d] loss: %.3f elapsed time %.3f' %
          (epoch, running_loss, end-start))

def test(testloader, net, criterion, device):
    '''
    Function for testing.
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)
            loss = criterion(output, labels)
            losses += loss.item()
            cnt += 1
    print(losses / cnt)
    return (losses/cnt)

def cal_AP(testloader, net, criterion, device):
    '''
    Calculate Average Precision
    '''
    losses = 0.
    cnt = 0
    with torch.no_grad():
        net = net.eval()
        preds = [[] for _ in range(5)]
        heatmaps = [[] for _ in range(5)]
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images).cpu().numpy()
            print(labels.shape, output.shape)
            for c in range(5):
                preds[c].append(output[:, c].reshape(-1))
                heatmaps[c].append(labels[:, c].cpu().numpy().reshape(-1))

        aps = []
        for c in range(5):
            preds[c] = np.concatenate(preds[c])
            heatmaps[c] = np.concatenate(heatmaps[c])
            if heatmaps[c].max() == 0:
                ap = float('nan')
            else:
                print(heatmaps[c].shape, preds[c].shape)
                ap = ap_score(heatmaps[c], preds[c])
                aps.append(ap)
            print("AP = {}".format(ap))

    # print(losses / cnt)
    return None

def get_result(testloader, net, device, folder='output_train'):
    result = []
    cnt = 1
    with torch.no_grad():
        net = net.eval()
        cnt = 0
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device)
            output = net(images)[0].cpu().numpy()
            c, h, w = output.shape
            assert(c == N_CLASS)
            y = np.zeros((h,w)).astype('uint8')
            for i in range(N_CLASS):
                mask = output[i]>0.5
                y[mask] = i

            gt = labels.cpu().data.numpy().astype('uint8')
            save_label(y, './{}/y{}.png'.format(folder, cnt))
            save_label(gt, './{}/gt{}.png'.format(folder, cnt))
            plt.imsave(
                './{}/x{}.png'.format(folder, cnt),
                ((images[0].cpu().data.numpy()+1)*128).astype(np.uint8).transpose(1,2,0))

            cnt += 1

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # TODO change data_range to include all train/evaluation/test data.
    # TODO adjust batch_size.
    train_data = FacadeDataset(flag='train', data_range=(0,20), onehot=False)
    train_loader = DataLoader(train_data, batch_size=1)
    val_data = FacadeDataset(flag='train', data_range=(20,30), onehot=False)
    val_loader = DataLoader(val_data, batch_size=1)

    #test_data = FacadeDataset(flag='test_dev', data_range=(0,10), onehot=False)
    #test_loader = DataLoader(test_data, batch_size=1)

    ap_data = FacadeDataset(flag='test_dev', data_range=(0,10), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1)

    name = 'starter_net'
    net = u_net().to(device)
    criterion = nn.CrossEntropyLoss() #custom_loss #TODO decide loss
    optimizer = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-5)

    print('\nStart training')
    for epoch in range(1): #TODO decide epochs
        print('-----------------Epoch = %d-----------------' % (epoch+1))
        train(train_loader, net, criterion, optimizer, device, epoch+1)
        # TODO create your evaluation set, load the evaluation set and test on evaluation set
        evaluation_loader = val_loader
        test(evaluation_loader, net, criterion, device)

    print('\nFinished Training, Testing on test set')
    test(test_loader, net, criterion, device)

    print("Saving...")
    torch.save(net.state_dict(), './models/model_{}.pth'.format(name))
    print('\nGenerating Unlabeled Result')

    result = get_result(val_loader, net, device, folder='output_test')
    cal_AP(ap_loader, net, criterion, device)

def eval_pretrained():
    """I trained a network on a VM and then wanted to retest it locally"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    val_data = FacadeDataset(flag='train', data_range=(700, 900), onehot=False)
    val_loader = DataLoader(val_data, batch_size=1)
    ap_data = FacadeDataset(flag='test_dev', data_range=(0,100), onehot=True)
    ap_loader = DataLoader(ap_data, batch_size=1)

    criterion = nn.CrossEntropyLoss()

    model_weight_path = 'vm_net.pth'

    net = u_net()
    net.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))


    result = get_result(val_loader, net, device, folder='output_test')
    cal_AP(ap_loader, net, criterion, device)



if __name__ == "__main__":
    eval_pretrained()
