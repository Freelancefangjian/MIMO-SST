import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from DataSet import DataSet
from config import FLAGES
import torch.utils.data as Data
def l2_penaalty(w):
    return (w**2).sum()/2
import numpy as np
import time
import Model
import math
def PSNR(img1, img2):
    mse_sum  = (img1  - img2 ).pow(2)
    mse_loss = mse_sum.mean(2).mean(2)
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return mse_loss, 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
now = time.strftime("%Y-%m-%d-%H-%M-%S",time.localtime(time.time()))
class fftLoss(nn.Module):
    def __init__(self):
        super(fftLoss, self).__init__()

    def forward(self, x, y):
        diff = torch.fft.fft2(x.to('cuda:0')) - torch.fft.fft2(y.to('cuda:0'))
        loss = torch.mean(abs(diff))
        return loss
from torchsummary import summary
if __name__ == '__main__':
    #freeze_support()
    dataset = DataSet(FLAGES.pan_size, FLAGES.ms_size, FLAGES.img_path, FLAGES.data_path, FLAGES.batch_size,
                      FLAGES.stride)
    #HR = np.transpose(dataset.gt, [3, 1, 2])
    MSI = torch.from_numpy(np.transpose(dataset.pan.astype(np.float32), [0,3, 1, 2]))
    HSI = torch.from_numpy(np.transpose(dataset.ms.astype(np.float32), [0,3, 1, 2]))
    GT = torch.from_numpy(np.transpose(dataset.gt.astype(np.float32), [0,3, 1, 2]))

    torch_dataset = Data.TensorDataset(MSI, HSI, GT)
    loader = Data.DataLoader(dataset = torch_dataset, batch_size=16, shuffle=True, num_workers=4)

    device = torch.device("cuda:0")
    net = Model.Net()
    #net.load_state_dict(torch.load("./model/state_dicr_460.pkl"))
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    net.to(device)
    import torch.optim as optim

    criterion = nn.L1Loss().to(device)
    WEIGHT_DECAY = 1e-8
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=WEIGHT_DECAY)
    criterion_fft = fftLoss()
    for epoch in range(1001):  # loop over the dataset multiple times

        running_loss = 0.0
        mpsnr = 0.0
        for i, data in enumerate(loader, 0):
            # get the inputs
            MSI1, HSI1, GT1 = data
            MSI1 = MSI1.type(torch.FloatTensor)
            HSI1 = HSI1.type(torch.FloatTensor)
            GT1 = GT1.type(torch.FloatTensor)

            MSI1 = MSI1.cuda(device)
            HSI1 = HSI1.cuda(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs3, outputs2, outputs1 = net(MSI1, HSI1)
            GT1 = GT1.cuda(device)
            GT2 = F.interpolate(GT1, scale_factor=0.5)
            GT3 = F.interpolate(GT2, scale_factor=0.5)
            loss_fft = criterion_fft(outputs1, GT1) + criterion_fft(outputs2, GT2) + criterion_fft(outputs3, GT3)
            loss1 = criterion(outputs1, GT1) + criterion(outputs2, GT2) + criterion(outputs3, GT3)
            loss = loss1 + 0.01 * loss_fft
            mse, psnr = PSNR(outputs1, GT1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            mpsnr += psnr
        if epoch % 10 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.7f PSNR:%.3f' %
                    (epoch + 1, i + 1, running_loss/(i + 1), mpsnr/(i + 1)))
        if epoch % 100 == 0:  # print every 2000 mini-batches
            torch.save(net.state_dict(), './model/state_dicr_{}.pkl'.format(epoch))

    print('Finished Training')