import cv2
import numpy as np
import torch

from utils.data import Imgs
from utils.myUtils import forNestingList


class AutoPad:
    def __init__(self, imgs, multiple):
        self.N, self.C, self.H, self.W = imgs.size()
        self.HPad = ((self.H - 1) // multiple + 1) * multiple
        self.WPad = ((self.W - 1) // multiple + 1) * multiple

    def pad(self, imgs):
        def _pad(img):
            imgPad = torch.zeros([self.N, self.C, self.HPad, self.WPad], dtype=img.dtype,
                                 device=img.device.type)
            imgPad[:, :, (self.HPad - self.H):, (self.WPad - self.W):] = img
            return imgPad

        return forNestingList(imgs, _pad)

    def unpad(self, imgs):
        return forNestingList(imgs, lambda img: img[:, :, (self.HPad - self.H):, (self.WPad - self.W):])


def flipLR(ims):
    # Flip among W dimension. For NCHW data type.
    if type(ims) in (tuple, list):
        return forNestingList(ims, flipLR)
    elif type(ims) is Imgs:
        for name in ims.keys():
            ims[name] = flipLR(ims[name])
        return ims
    else:
        return ims.flip(-1)


def assertDisp(dispL=None, dispR=None):
    if (dispL is None or dispL.numel() == 0) and (dispR is None or dispR.numel() == 0):
        raise Exception('No disp input!')


def gray2rgb(im):
    if im.dim() == 2:
        im = im.unsqueeze(0)
    if im.dim() == 3:
        im = im.unsqueeze(1)
    return im.repeat(1, 3, 1, 1)


def gray2color(im):
    if im.dim() == 4 and im.size(1) == 1:
        im = im.squeeze(1)
    if im.dim() == 3 and im.size(0) >= 1:
        imReturn = torch.zeros([im.size(0), 3, im.size(1), im.size(2)], dtype=torch.uint8)
        for i in range(im.size(0)):
            imReturn[i, :, :, :] = gray2color(im[i, :, :])
        return imReturn
    elif im.dim() == 2:
        im = (im.numpy() * 255).astype(np.uint8)
        im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
        im = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
        return im
    else:
        raise Exception('Error: Input of gray2color must have one channel!')


def quantize(img, range):
    return img.clamp(0, range) / (range)


def savePreprocessRGB(im):
    output = im.squeeze()
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    output = (output * 255).astype('uint8')
    return output


def savePreprocessDisp(disp, dispScale=256):
    dispOut = disp.squeeze()
    dispOut = dispOut.data.cpu().numpy()
    dispOut = (dispOut * dispScale).astype('uint16')
    return dispOut