from cvc_model.architecture import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse
import os
from torchvision import transforms
import time

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--testset_dir', type=str, default='../dataset/test')
    parser.add_argument('--dataset', type=str, default='KITTI2015')#middlebury KITTI2012 KITTI2015 Flick1024_test
    parser.add_argument('--scale_factor', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()

def test(test_loader, cfg):
    net = CVCnet(upscale=cfg.scale_factor)
    device = torch.device("cuda")
    net = net.to(device)

    cudnn.benchmark = True
    # pretrained_dict = torch.load('./log/x' + str(cfg.scale_factor) + '/PASSRnet_x' + str(cfg.scale_factor) + '.pth')
    pretrained_dict = torch.load('./log/x4/CVCnet_x4.pth')
    net.load_state_dict(pretrained_dict)

    psnr_list = []
    ssim_list = []
    time_list=[]

    with torch.no_grad():
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(test_loader):
            HR_left, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(
                LR_right).to(cfg.device)
            video_name = test_loader.dataset.file_list[idx_iter]

            time_start = time.time()
            SR_left = net(LR_left,LR_right)
            time_end = time.time()

            SR_left = torch.clamp(SR_left, 0, 1)

            SR_left_np = np.array(torch.squeeze(SR_left[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
            HR_left_np = np.array(torch.squeeze(HR_left[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))

            PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
            SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True, data_range=1)
            # SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
            psnr_list.append(PSNR)
            ssim_list.append(SSIM)
            time_list.append(time_end-time_start)
            ## save results
            if not os.path.exists('results/' + cfg.dataset):
                os.mkdir('results/' + cfg.dataset)
            if not os.path.exists('results/' + cfg.dataset + '/' + video_name):
                os.mkdir('results/' + cfg.dataset + '/' + video_name)
            SR_left_img = transforms.ToPILImage()(torch.squeeze(SR_left.data.cpu(), 0))
            SR_left_img.save('results/' + cfg.dataset + '/' + video_name + '/img_0.png')

        ## print results
        print(cfg.dataset + ' mean psnr:', float(np.array(psnr_list).mean()), 'mean ssim:',
              float(np.array(ssim_list).mean()), '  mean time:', np.array(time_list).mean())

def main(cfg):
    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)
    result = test(test_loader, cfg)
    return result

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)
