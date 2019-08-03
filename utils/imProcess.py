import collections
import os

import cv2
import numpy as np
import skimage.io
import torch
import imageio


from utils.myUtils import forNestingList, checkDir


class AutoPad:
    def __init__(self, imgs, multiple, scale=1):
        self.N, self.C, self.H, self.W = imgs.size()
        self.HPad = ((self.H - 1) // multiple + 1) * multiple
        self.WPad = ((self.W - 1) // multiple + 1) * multiple
        self.scale = scale

    def pad(self, imgs):
        def _pad(img):
            imgPad = torch.zeros([self.N, self.C, self.HPad, self.WPad], dtype=img.dtype,
                                 device=img.device.type)
            imgPad[:, :, (self.HPad - self.H):, (self.WPad - self.W):] = img
            return imgPad

        return forNestingList(imgs, _pad)

    def unpad(self, imgs):
        return forNestingList(imgs, lambda img: img[:, :, (self.HPad - self.H) * self.scale:, (self.WPad - self.W) * self.scale:])


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


# def gray2color(im):
#     if im.dim() == 4 and im.size(1) == 1:
#         im = im.squeeze(1)
#     if im.dim() == 3 and im.size(0) >= 1:
#         imReturn = torch.zeros([im.size(0), 3, im.size(1), im.size(2)], dtype=torch.uint8)
#         for i in range(im.size(0)):
#             imReturn[i, :, :, :] = gray2color(im[i, :, :])
#         return imReturn
#     elif im.dim() == 2:
#         im = (im.numpy() * 255).astype(np.uint8)
#         im = cv2.applyColorMap(im, cv2.COLORMAP_JET)
#         im = torch.from_numpy(cv2.cvtColor(im, cv2.COLOR_BGR2RGB).transpose((2, 0, 1)))
#         return im
#     else:
#         raise Exception('Error: Input of gray2color must have one channel!')


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


class Imgs(collections.OrderedDict):
    def __init__(self, imgs=None):
        if type(imgs) in (list, tuple):
            super().__init__(imgs)
        else:
            super().__init__()
        if isinstance(imgs, Imgs):
            self.update(imgs)
        self.range = {}

    def cpu(self):
        for name in self.keys():
            self[name] = self[name].cpu()
        return self

    def update(self, imgs, suffix=''):
        assert type(imgs) is Imgs
        for name in imgs.keys():
            self[name + suffix] = imgs[name]
            self.range[name + suffix] = imgs.range[name]

    def clone(self):
        r = Imgs()
        for name, value in self.items():
            r[name] = value.clone()
            r.range[name] = self.range[name]
        return r

    def addImg(self, name: str, img, range=1):
        if img is not None:
            self.range[name] = range
            self[name] = img

    def getImgPair(self, name: str, suffix=''):
        return (self[name + 'L' + suffix], self[name + 'R' + suffix])

    def getIt(self, it: int):
        output = Imgs()
        for key in self.keys():
            if key.endswith('_%d' % it):
                output[key[:key.find('_')]] = self[key]
        return output

    def logPrepare(self):
        for name in self.keys():
            self[name] /= self.range[name]

    def _savePrepare(self):
        for name in self.keys():
            self[name] = self[name][0]
            if 'Disp' in name:
                if self.range[name] == 384:
                    self[name] = savePreprocessDisp(self[name], dispScale=170)
                else:
                    self[name] = savePreprocessDisp(self[name])
            elif 'Rgb':
                self[name] = savePreprocessRGB(self[name])

        # scan for SR input/output
        addedImgs = Imgs()
        for name in self.keys():
            if name.startswith('inputSr'):
                interName = 'bicubicSr' + name[len('inputSr'):]
                interImg = cv2.resize(self[name], None, fx=2, fy=2, interpolation = cv2.INTER_CUBIC)
                addedImgs.addImg(interName, interImg, self.range[name])
        self.update(addedImgs)

    def save(self, dir, name):
        self._savePrepare()
        checkDir(dir)
        for folder, value in self.items():
            checkDir(os.path.join(dir, folder))
            saveDir = os.path.join(dir, folder, name + '.png')
            skimage.io.imsave(saveDir, value)
            print('saved to: %s' % saveDir)

        ## save gif for sr
        gifFrames = collections.OrderedDict()
        for key in self.keys():
            if key.startswith('bicubicSr'):
                gifFrames[key] = self[key]
                outputName = 'output' + key[len('bicubic'):]
                gifFrames[outputName] = self[outputName]

        # add title
        if len(gifFrames) > 0:
            frames=[]
            framesCrop=[]
            for title, im in gifFrames.items():
                frames.append(putTitle(im, title))
                cropCenter = [l // 2 for l in im.shape]
                framesCrop.append(putTitle(
                    im[(cropCenter[0] - 100):(cropCenter[0] + 100), (cropCenter[0] - 100):(cropCenter[0] + 100)], title))
            checkDir(os.path.join(dir, 'compareSr'))
            saveDir = os.path.join(dir, 'compareSr', name + '.gif')
            imageio.mimsave(saveDir, frames, 'GIF', duration=1)
            print('saved to: %s' % saveDir)

            checkDir(os.path.join(dir, 'compareSrCrop'))
            saveDir = os.path.join(dir, 'compareSrCrop', name + '.gif')
            imageio.mimsave(saveDir, framesCrop, 'GIF', duration=1)
            print('saved to: %s' % saveDir)


def putTitle(im, title):
    fontScale = im.shape[0] // 500 + 1
    thickness = fontScale
    fontFace = cv2.FONT_HERSHEY_DUPLEX
    textSize = cv2.getTextSize(title, fontFace, fontScale, 3)
    textCoor = (im.shape[1] // 2 - textSize[0][0] // 2, im.shape[0] - textSize[0][1])
    imTitle = cv2.putText(np.ascontiguousarray(im, dtype=np.uint8), title, textCoor,
                          fontFace, fontScale, (255, 255, 255), int(thickness * 1.5 + 1), cv2.LINE_AA)
    imTitle = cv2.putText(np.ascontiguousarray(imTitle, dtype=np.uint8), title, textCoor,
                          fontFace, fontScale, (255, 0, 0), thickness, cv2.LINE_AA)
    return imTitle
