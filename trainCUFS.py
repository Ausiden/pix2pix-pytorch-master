
# @File  : train2.py
# @Author: Jehovah
# @Date  : 18-9-18
# @Desc  : 


#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : train.py
# @Author: Jehovah
# @Date  : 18-6-4
# @Desc  :

from torch.utils.data import DataLoader
import os
import tensorboardX
import torchvision.utils as vutils
import torchvision.transforms as transforms
import torch
import time
from data_loader2 import *
from fid_score import calculate_fid_given_paths
# from data import *
from networks import *
from pix2pix_model import *
import option
opt = option.init()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuid  # 指定gpu

class GANLoss(nn.Module):
    def __init__(self, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

def train():
    model_name = 'pix2pix_CUFS'
    train_writer = tensorboardX.SummaryWriter(os.path.join("./logs", model_name))
    # data_loader = MyDataset(opt, "/train")
    data_loader = MyDataset(opt)
    dataset_size = len(data_loader)
    print('trainA images = %d' % dataset_size)

    train_loader = torch.utils.data.DataLoader(dataset=data_loader, batch_size=opt.batchSize, shuffle=True, num_workers=2)

    test_set = MyDataset(opt, isTrain=1)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=1, shuffle=False,
                                              num_workers=8)


    net_G = Generator(opt.input_nc, opt.output_nc)

    net_D = Discriminator(opt.input_nc, opt.output_nc)

    net_G.cuda()
    net_D.cuda()

    net_G.apply(weights_init)
    net_D.apply(weights_init)

    print(net_G)
    print(net_D)

    criterionGAN = GANLoss()
    critertion1 = nn.L1Loss()

    optimizerG = torch.optim.Adam(net_G.parameters(), lr=opt.lr, betas=(opt.bata, 0.999))
    optimizerD = torch.optim.Adam(net_D.parameters(), lr=opt.lr, betas=(opt.bata, 0.999))


    net_D.train()
    net_G.train()

    for epoch in range(1, opt.niter+1):
        epoch_start_time = time.time()
        for i, image in enumerate(train_loader):
            imgA = image['A']
            imgB = image['B']

            real_A = imgA.cuda()
            fake_B = net_G(real_A)
            real_B = imgB.cuda()
            net_D.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            pred_fake = net_D(fake_AB.detach())

            loss_D_fake = criterionGAN(pred_fake, False)

            real_AB = torch.cat((real_A, real_B), 1)
            pred_real = net_D(real_AB)
            loss_D_real = criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            loss_D.backward()

            optimizerD.step()

            # netG
            net_G.zero_grad()
            fake_AB = torch.cat((real_A, fake_B), 1)
            out_put = net_D(fake_AB)
            loss_G_GAN = criterionGAN(out_put, True)

            loss_G_L1 = critertion1(fake_B, real_B) * opt.lamb
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optimizerG.step()
            if i % 100 == 0:
                print('[%d/%d][%d/%d] LOSS_D: %.4f LOSS_G: %.4f LOSS_L1: %.4f' % (epoch, opt.niter, i, len(train_loader), loss_D, loss_G, loss_G_L1))
                print('LOSS_real: %.4f LOSS_fake: %.4f' % (loss_D_real, loss_D_fake))
        print('Time Taken: %d sec' % (time.time() - epoch_start_time))
        train_writer.add_scalar('loss/D', float(loss_D), epoch)
        train_writer.add_scalar('loss/G', float(loss_G), epoch)
        train_writer.add_scalar('loss/G_L1', float(loss_G_L1), epoch)

        if epoch % 5 == 0:
            vutils.save_image(fake_B.data, '%s/fake_samples_epoch_%03d.png' % (opt.sample,epoch), normalize=True)
            # test(epoch,net_G,test_loader)
        if epoch >= 200:
            if not os.path.exists(opt.checkpoints):
                os.makedirs(opt.checkpoints)
            if epoch % 20 == 0:
                test(epoch, net_G, test_loader)
                torch.save(net_G.state_dict(), opt.checkpoints + '/net_G_ins' + str(epoch) + '.pth')
                torch.save(net_D.state_dict(), opt.checkpoints + '/net_D_ins' + str(epoch) + '.pth')
                print("saved model at epoch " + str(epoch))
    print("save net")
    if not os.path.exists(opt.checkpoints):
        os.makedirs(opt.checkpoints)
    torch.save(net_G.state_dict(), opt.checkpoints+'/net_G_ins.pth')
    torch.save(net_D.state_dict(), opt.checkpoints+'/net_D_ins.pth')


def test(epoch, netG, test_data):
    f = open(opt.FID, 'a')
    if not os.path.exists('%s/%d' % (opt.output, epoch)):
        os.mkdir('%s/%d' % (opt.output, epoch))

    print('test_data len = %d' % len(test_data))
    for i, image in enumerate(test_data):
        imgA = image['A']
        A_path=image['A_path']
        real_A = imgA.cuda()
        fake_B = netG(real_A)
        name=A_path[0].split('/')
        vutils.save_image(fake_B[:, :, 3:253, 28:228],'%s/%d/%s' % (opt.output,epoch,name[-1]), normalize=True, scale_each=True)

    fid_value = calculate_fid_given_paths('%s/%d' % (opt.output,epoch),'/home/lixiang/dataset/photosketch/sketch/test', 8, opt.gpuid != '', 2048)
    print(str(epoch) + " saved")
    str1 = str(epoch) + ':' + str(fid_value)+'\n'
    f.write(str1)
    f.close()
    print('calculate FID:%.4f' %fid_value)


def draw_FIDdata():
    import matplotlib.pyplot as plt
    import numpy as np
    with open('FID_CUFS.txt') as f:
        x, y = [], []
        # plt.ylim(24, 29)
        for line in f.readlines():
            x.append(int(line.split(':')[0]))
            y.append(float(line.split(':')[1]))
        plt.plot(np.array(x), np.array(y))
        plt.xlabel('epochs')
        plt.ylabel('FID')

        plt.show()


def calculate_FID_avg():
    with open('FID_CUFS.txt') as f:
        total = 0
        data = f.readlines()
        for line in data:
            total = total + float(line.split(':')[1])
        print('%.2f' % float(total / len(data)))


if __name__ == '__main__':
    train()
    # draw_FIDdata()
    # calculate_FID_avg()
    pass
