#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : test.py
# @Author: Jehovah
# @Date  : 18-6-7
# @Desc  : 

import os
import torchvision.utils as vutils
import option
import torch
from data_loader2 import *
from networks import *
from pix2pix_model import *
from fid_score import calculate_fid_given_paths


os.environ["CUDA_VISIBLE_DEVICES"] = '0'  # 指定gpu

if __name__ == '__main__':
    opt = option.init()
    net_G = Generator(opt.input_nc, opt.output_nc)
    net_G.load_state_dict(torch.load(opt.checkpoints+'/net_G_ins500.pth'))
    net_G.cuda()
    print("net_G loaded")
    # dataloader = MyDataset(opt, '/test', isTrain=1)
    dataloader = MyDataset(opt, isTrain=1)
    imgNum = len(dataloader)
    print(len(dataloader))

    test_loader = torch.utils.data.DataLoader(dataset=dataloader, batch_size=opt.batchSize, shuffle=True, num_workers=2)

    fakeB = torch.FloatTensor(imgNum, opt.output_nc, opt.fineSize, opt.fineSize)
    A = torch.FloatTensor(imgNum, opt.output_nc, opt.fineSize, opt.fineSize)
    realB = torch.FloatTensor(imgNum, opt.output_nc, opt.fineSize, opt.fineSize)

    save_dir = os.path.join(opt.output, 'fakeB500')
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i, image in enumerate(test_loader):
        imgA = image[0]
        # imgB = image['A']

        real_A = Variable(imgA.cuda())
        # real_B = Variable(imgB.cuda())

        fake_B = net_G(real_A)

        # fakeB[i, :, :, :] = fake_B.data
        # realB[i, :, :, :] = real_B.data

        print("%d.jpg generate completed" % (i+88))


        # vutils.save_image(fake_B[:, :, 3:253, 28:228],
        #                   '%s/fakeB/%d.png' % (opt.output, i),
        #                   normalize=True,
        #                   scale_each=True)
        vutils.save_image(fake_B[:, :, 3:253, 28:228],'%s/%d.png' % (save_dir, i),normalize=True,scale_each=True)

    fid_value = calculate_fid_given_paths(save_dir, opt.dataroot + '/CUHK/Testing Images/sketches', 8, opt.gpuid != '',2048)
    print(fid_value)
    # vutils.save_image(realB,
    #                   '%s/realB_8.png' % (opt.outf),
    #                   normalize=True,
    #                   scale_each=True)
