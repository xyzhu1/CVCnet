from cvc_model.architecture import *
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn
from utils import *
import argparse,os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] ="0"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scale_factor", type=int, default=2)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=2e-4, help='initial learning rate')
    parser.add_argument('--gamma', type=float, default=0.5, help='')
    parser.add_argument('--n_epochs', type=int, default=180, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=30, help='number of epochs to update learning rate')
    parser.add_argument('--trainset_dir', type=str, default='../dataset/Flickr1024_trainset/Flickr1024_patches')#patches

    parser.add_argument('--testset_dir', type=str, default='../dataset/test')
    parser.add_argument('--dataset', type=str, default='KITTI2015')

    return parser.parse_args()


def train(train_loader, cfg):
    ###################################
    net = CVCnet(upscale=cfg.scale_factor)
    device = torch.device("cuda")
    net = net.to(device)
    ##################################

    #################### 计算参数数量 #######################
    total_params = sum(p.numel() for p in net.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in net.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    #######################################################

    net.apply(weights_init_xavier)
    #cudnn.benchmark = True

    criterion_mse = torch.nn.MSELoss().to(cfg.device)
    criterion_L1 = L1Loss()
    optimizer = torch.optim.Adam([paras for paras in net.parameters() if paras.requires_grad == True], lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.n_steps, gamma=cfg.gamma)

    psnr_epoch = []
    loss_epoch = []
    loss_list = []
    psnr_list = []

    psnr_now=0
    psnr_max=0
    psnr_max_epoch=0
    epoch_now=1

    test_set = TestSetLoader(dataset_dir=cfg.testset_dir + '/' + cfg.dataset, scale_factor=cfg.scale_factor)
    test_loader = DataLoader(dataset=test_set, num_workers=1, batch_size=1, shuffle=False)

    for idx_epoch in range(cfg.n_epochs):
        # print("lr_rate: ", optimizer.state_dict()['param_groups'][0]['lr'])
        net.train()
        # with tqdm(total=len(train_loader), desc='Epoch: [%d/%d]' % (idx_epoch+1, cfg.n_epochs),
        #           miniters=1) as t:
        for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(train_loader):
            b, c, h, w = LR_left.shape
            HR_left, LR_left, LR_right  = Variable(HR_left).to(cfg.device), Variable(LR_left).to(cfg.device), Variable(LR_right).to(cfg.device)

            SR_left = net(LR_left,LR_right)

            ### loss_SR
            loss_SR = criterion_mse(SR_left, HR_left)

            ### losses
            loss = loss_SR

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            psnr_epoch.append(cal_psnr(HR_left[:,:,:,64:].data.cpu(), SR_left[:,:,:,64:].data.cpu()))
            loss_epoch.append(loss.data.cpu())
                # t.update()
        scheduler.step()

        if idx_epoch % 1 == 0:
            loss_list.append(float(np.array(loss_epoch).mean()))
            psnr_list.append(float(np.array(psnr_epoch).mean()))
            # print('Epoch----%5d, loss---%f, PSNR---%f' % (idx_epoch + 1, float(np.array(loss_epoch).mean()), float(np.array(psnr_epoch).mean())))

            psnr_Validation = []
            ssim_Validation = []

            ###########验证
            net.eval()
            with torch.no_grad():
                for idx_iter, (HR_left, _, LR_left, LR_right) in enumerate(test_loader):
                    HR_left, LR_left, LR_right = Variable(HR_left).to(cfg.device), Variable(LR_left).to(
                        cfg.device), Variable(LR_right).to(cfg.device)

                    video_name = test_loader.dataset.file_list[idx_iter]

                    SR_left = net(LR_left,LR_right)
                    SR_left = torch.clamp(SR_left, 0, 1)

                    SR_left_np = np.array(torch.squeeze(SR_left[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))
                    HR_left_np = np.array(torch.squeeze(HR_left[:, :, :, 64:].data.cpu(), 0).permute(1, 2, 0))

                    PSNR = measure.compare_psnr(HR_left_np, SR_left_np)
                    SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True, data_range=1)
                    # SSIM = measure.compare_ssim(HR_left_np, SR_left_np, multichannel=True)
                    psnr_Validation.append(PSNR)
                    ssim_Validation.append(SSIM)

                ## print results
                psnr_now = float(np.array(psnr_Validation).mean())
                ssim_now = float(np.array(ssim_Validation).mean())
                if psnr_now >= psnr_max:
                    psnr_max = psnr_now
                    psnr_max_epoch = idx_epoch + 1
                    print(cfg.dataset + '    PSNR_now:', "%.3f" % psnr_now, '    SSIM_now:', "%.3f" % ssim_now,
                          '    PSNR_max:', "%.3f" % psnr_max, '   current_epoch:', idx_epoch + 1, '   max_epoch:',
                          psnr_max_epoch, '    lr_rate:',
                          optimizer.state_dict()['param_groups'][0]['lr'])
                else:
                    print(cfg.dataset + '    PSNR_now:', "%.3f" % psnr_now, '    SSIM_now:', "%.3f" % ssim_now,
                          '    PSNR_max:', "%.3f" % psnr_max, '   current_epoch:', idx_epoch + 1, '   max_epoch:',
                          psnr_max_epoch, '    lr_rate:',
                          optimizer.state_dict()['param_groups'][0]['lr'])
            # print('current_epoch:', idx_epoch + 1,)


            save_path = 'log/x' + str(cfg.scale_factor) + '/'
            filename = 'PASSRnet_x' + str(cfg.scale_factor) + '_epoch' + str(idx_epoch + 1) + '.pth'
            torch.save(net.state_dict(), os.path.join(save_path, filename))

            psnr_epoch = []
            loss_epoch = []

def main(cfg):
    seed=0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    train_set = TrainSetLoader(dataset_dir=cfg.trainset_dir, cfg=cfg)
    train_loader = DataLoader(dataset=train_set, num_workers=4, batch_size=cfg.batch_size, shuffle=True)
    train(train_loader, cfg)

if __name__ == '__main__':
    cfg = parse_args()
    main(cfg)