import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from common import *
from networks import *
import torch
import torch.nn as nn
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
import torch.nn.functional as F
from torchsummary import summary
import cv2
from utils import *

Nx = 512
Ny = 512
z = 17526
wavelength = 532e-3
deltaX = 3.45
deltaY = 3.45
# optical parameters

hologram_path = "./Figures/air_2/hologram.png"
background_path = './Figures/air_2/bg.png'
p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=[600,244,512,512], dtype='float32')
bg = import_image(background_path, modifiers=p1)
p2 = PreprocessHologram(background=bg)
p3 = ConvertToTensor()
hologram = import_image(hologram_path, modifiers=[p1, p2, p3])




# plt.figure(figsize=(20,15))
# plt.imshow(np.squeeze(hologram), cmap='gray')

phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
eta = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
# plt.figure(figsize=(20,15))
# plt.imshow(np.squeeze(np.abs(eta)), cmap='gray')


criterion_1 = RECLoss()
model = Net().cuda()
optimer_1 = optim.Adam(model.parameters(), lr=5e-3)


device = torch.device("cuda")
epoch_1 = 5000
epoch_2 = 2000
period = 100
eta = torch.from_numpy(np.concatenate([np.real(eta)[np.newaxis,np.newaxis,:,:], np.imag(eta)[np.newaxis,np.newaxis,:,:]], axis = 1))
holo = torch.from_numpy(np.concatenate([np.real(hologram)[np.newaxis,np.newaxis,:,:], np.imag(hologram)[np.newaxis,np.newaxis,:,:]], axis = 1))

for i in range(epoch_1):
    in_img = eta.to(device)
    target = holo.to(device)

    out = model(in_img)
    l1_loss = criterion_1(out, target)
    loss = l1_loss


    optimer_1.zero_grad()
    loss.backward()
    optimer_1.step()

    print('epoch [{}/{}]     Loss: {}'.format(i+1, epoch_1, l1_loss.cpu().data.numpy()))
    if ((i) % period) == 0:
        outtemp = out.cpu().data.squeeze(0).squeeze(1)
        outtemp = outtemp
        plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
        plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
        amplitude = np.array(plotout)
        amplitude = amplitude.astype('float32')*255
        cv2.imwrite("./results/Amplitude/iter%d.bmp"%(i), amplitude)

        plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
        plotout_p = Phase_unwrapping(plotout_p)
        plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
        phase = np.array(plotout_p)
        phase = phase.astype('float32')*255
        cv2.imwrite("./results/Phase/iter%d.bmp"%(i), phase)

outtemp = out.cpu().data.squeeze(0).squeeze(1)
outtemp = outtemp
plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
amplitude = np.array(plotout)
amplitude = amplitude.astype('float32')*255
cv2.imwrite("./results/Amplitude/final.bmp", amplitude)


plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
plotout_p = Phase_unwrapping(plotout_p)
plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
phase = np.array(plotout_p)
phase = phase.astype('float32')*255
cv2.imwrite("./results/Phase/final.bmp", phase)
