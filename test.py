import scipy.io as sio
import numpy as np
import Model
import os
import torch
import math
def PSNR(img1, img2):
    mse_sum  = (img1  - img2 ).pow(2)
    mse_loss = mse_sum.mean(2).mean(2)
    mse = mse_sum.mean()                     #.pow(2).mean()
    if mse < 1.0e-10:
        return 100
    PIXEL_MAX = 1
    # print(mse)
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
from thop import profile
if __name__ == '__main__':
    net = Model.Net().cuda()
    print('# generator parameters:', sum(param.numel() for param in net.parameters()))
    net.load_state_dict(torch.load("./model/state_dicr_0.pkl"))
    test_path = 'G:\\OneDrive - njust.edu.cn\\Hyperspectral_Image_Fusion_Benchmarkx8\\CAVE\\MW-DAN\\test'
    psnr1 = np.zeros(12)
    for i in range(12):
        ind = i + 1
        path = str(i + 1) + '.mat'
        print('processing for %d' % ind)
        source_hs_path = os.path.join(test_path, 'hs', path)
        data = sio.loadmat(source_hs_path)
        data = torch.FloatTensor(data['I'].astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()
        source_ms_path = os.path.join(test_path, 'ms', path)
        data1 = sio.loadmat(source_ms_path)
        source_gt_path = os.path.join(test_path, 'gt', path)
        GT = sio.loadmat(source_gt_path)
        GT = torch.FloatTensor(GT['I'].astype(np.float32)).permute(2, 0, 1).unsqueeze(0)
        GT = np.transpose(GT, [0, 2, 3, 1])
        data1 = torch.FloatTensor(data1['I'].astype(np.float32)).permute(2, 0, 1).unsqueeze(0).cuda()
        with torch.no_grad():
            data_get3, data_get2, data_get1 = net(data1, data)
            flops, params = profile(net, inputs=(input,))
            print(f"FLOPs: {flops}")
        data_get = data_get1.cpu().detach()
        data_get = np.transpose(data_get, [0, 2, 3, 1])
        psnr1[i] = PSNR(data_get, GT)
        data_get = np.reshape(data_get, (512, 512, 31))
        data_get = np.array(data_get.numpy(), dtype=np.float64)

        sio.savemat('./get/eval_%d.mat' % ind, {'b': data_get})
    print(np.mean(psnr1))