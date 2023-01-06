import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"
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

# optical parameters
def calculate_tv(img):
    amp = np.abs(img)
    w,h = amp.shape
    tv_w = np.square(amp[1:,:]-amp[0:w-1,:]).sum()
    tv_h = np.square(amp[:,1:]-amp[:,0:h-1]).sum()
    tv = np.sqrt(tv_w+tv_h)
    return tv

Nx = 1024
Ny = 1024
#

wavelength = 650e-3
deltaX = 3.45
deltaY = 3.45
# optical parameters
background_path = './Figures/MPsample/bg3.bmp'
hologram_path = "./Figures/MPsample/mp_4samples.bmp"

z = 19090
p1 = ImageToArray(bit_depth=16, channel='gray', crop_window=[0,0,1024,1024], dtype='float32')


bg = import_image(background_path, modifiers=p1)
p2 = PreprocessHologram(background=bg)
p3 = ConvertToTensor()
hologram = import_image(hologram_path, modifiers=[p1, p2, p3])


plt.figure(figsize=(20,15))
plt.imshow(np.squeeze(hologram), cmap='gray')
phase = propagator(Nx,Ny,z,wavelength,deltaX,deltaY)
eta = np.fft.ifft2(np.fft.fft2(hologram)*np.fft.fftshift(np.conj(phase)))
plt.figure(figsize=(20,15))
plt.imshow(np.squeeze(np.abs(eta)), cmap='gray')
criterion_1 = RECLoss(Nx,Ny,z,wavelength,deltaX,deltaY)
model = Net()
optimer_1 = optim.Adam(model.parameters(), lr=5e-3)


device = torch.device("cuda")
model = model.to(device)
epoch_1 = 10000
epoch_2 = 2000
period = 100
eta = torch.from_numpy(np.concatenate([np.real(eta)[np.newaxis,np.newaxis,:,:], np.imag(eta)[np.newaxis,np.newaxis,:,:]], axis = 1))
holo = torch.cat([hologram[None,None,:,:],torch.zeros_like(hologram)[None,None,:,:]],dim=1)

for i in range(epoch_1):
    ckpt = {
        "param" : model.state_dict(),
        "last_epoch" : i,
        "optimizer" : optimer_1.state_dict()
    }
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
        cv2.imwrite("./results/MP_amp/iter%d.bmp"%(i), amplitude)

        plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
        plotout_p = Phase_unwrapping(plotout_p)
        plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
        phase = np.array(plotout_p)
        phase = phase.astype('float32')*255
        cv2.imwrite("./results/MP_phase/iter%d.bmp"%(i), phase)

        torch.save(ckpt,"./results/MP_last.pt")

outtemp = out.cpu().data.squeeze(0).squeeze(1)
outtemp = outtemp
plotout = torch.sqrt(outtemp[0,:,:]**2 + outtemp[1,:,:]**2)
plotout = (plotout - torch.min(plotout))/(torch.max(plotout)-torch.min(plotout))
amplitude = np.array(plotout)
amplitude = amplitude.astype('float32')*255
cv2.imwrite("./results/MP_amp/final.bmp", amplitude)


plotout_p = (torch.atan(outtemp[1,:,:]/outtemp[0,:,:])).numpy()
plotout_p = Phase_unwrapping(plotout_p)
plotout_p = (plotout_p - np.min(plotout_p))/(np.max(plotout_p)-np.min(plotout_p))
phase = np.array(plotout_p)
phase = phase.astype('float32')*255
cv2.imwrite("./results/MP_phase/final.bmp", phase)