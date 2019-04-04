import torch
import os
import argparse
from tensorboardX import SummaryWriter
import collections
import cv2
import numpy as np
import random
import time
import copy
import sys

class NameValues(collections.OrderedDict):
    def __init__(self, seq=(), prefix='', suffix=''):
        super().__init__()
        for name, value in seq:
            if value is not None:
                super().__setitem__(prefix + name + suffix, value)

    def clone(self):
        return copy.deepcopy(self)

    def update(self, nameValues, suffix=''):
        for name in nameValues.keys():
            self[name + suffix] = nameValues[name]

    def strPrint(self, prefix='', suffix=''):
        strReturn = ''
        for name, value in super().items():
            if name.find('outlier') != -1:
                unit = '%'
            else:
                unit = ''
            strReturn += '%s: ' % (prefix + name + suffix)

            def addValue(value):
                s = ''
                if type(value) in (list, tuple):
                    for v in value:
                        s += addValue(v)
                else:
                    s += '%.3f%s, ' % (value, unit)
                return s

            strReturn += addValue(value)

        return strReturn

    def strSuffix(self, prefix='', suffix=''):
        sSuffix = ''
        for name, value in super().items():
            sSuffix += '_%s' % (prefix + name + suffix)

            def addValue(sAppend, values):
                if type(values) == int:
                    return sAppend + '_' + str(values)
                elif type(values) == float:
                    return sAppend + '_%.1f' % values
                elif type(values) in (list, tuple):
                    for v in values:
                        sAppend = addValue(sAppend, v)
                    return sAppend
                else:
                    raise Exception('Error: Type of values should be in int, float, list, tuple!')

            sSuffix = addValue(sSuffix, value)

        return sSuffix


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
        return forNestingList(imgs, lambda img: img[:, (self.HPad - self.H):, (self.WPad - self.W):])


class Imgs(collections.OrderedDict):
    def __init__(self, imgs=None):
        super().__init__()
        self._range = {}
        if imgs is not None:
            if isinstance(imgs, Imgs):
                self.update(imgs)
            else:
                raise Exception(f'Error: No rule to initialize Imgs with type {type(imgs)}')

    def update(self, imgs, suffix=''):
        assert type(imgs) is Imgs
        for name in imgs.keys():
            self[name + suffix] = imgs[name]
            self._range[name + suffix] = imgs._range[name]

    def clone(self):
        r = Imgs()
        for name, value in self.items():
            r[name] = value.clone()
            r._range[name] = self._range[name]
        return r

    def getImg(self, name: str, prefix: str, side: str = ''):
        return self.get(prefix + name + side)

    def addImg(self, name: str, img, prefix: str, range=1, side: str = ''):
        if img is not None:
            self._range[prefix + name + side] = range
            self[prefix + name + side] = img

    def logPrepare(self):
        for name in self.keys():
            self[name] /= self._range[name]


class Loss(NameValues):
    def __init__(self, seq=(), prefix='', suffix=''):
        super().__init__(seq=seq, prefix=prefix, suffix=suffix)
        self.nAccum = 1

    def getLoss(self, name: str, prefix: str = 'loss', side: str = ''):
        return self[prefix + name + side]

    def addLoss(self, loss, name: str, prefix: str = 'loss', side: str = ''):
        self[prefix + name + side] = loss

    def accumuate(self, loss):
        for key in self.keys():
            self[key] += loss[key]
        self.nAccum += 1

    def avg(self):
        for key in self.keys():
            self[key] /= self.nAccum
        self.nAccum = 1


class Experiment:
    def __init__(self, model, stage, args):
        self.args = args
        self.model = model
        self.epoch = 0
        self.iteration = 0
        self.globalStep = 0

        # load from checkpoint, if success will overwrite dirs below
        self.chkpointDir = None
        self.chkpointFolder = None
        self.logFolder = None

        # if resume, results will be saved to where the loaded checkpoint is.
        if args.resume:
            if args.chkpoint is None:
                raise Exception('Error: No checkpoint to resume!')
            elif len(args.chkpoint) > 1:
                raise Exception('Error: Cannot resume multi-checkpoints model!')
            else:
                args.chkpoint = args.chkpoint[0]
        # if not resume, result will be saved to new folder
        else:
            # auto experiment naming
            saveFolderSuffix = NameValues((
                ('loadScale', args.loadScale),
                ('trainCrop', args.trainCrop),
                ('batchSize', args.batchSize),
                ('lossWeights', args.lossWeights),
            ))
            startTime = time.localtime(time.time())
            newFolderName = time.strftime('%y%m%d%H%M%S_', startTime) \
                            + self.__class__.__name__ \
                            + saveFolderSuffix.strSuffix()
            newFolderName += '_' + args.dataset
            if args.outputFolder is not None:
                stage = os.path.join(args.outputFolder, stage)
            self.chkpointFolder = os.path.join('logs', stage, newFolderName)
            checkDir(self.chkpointFolder)
            self.logFolder = os.path.join(self.chkpointFolder, 'logs')
        self.load(args.chkpoint)
        self.logger = Logger(folder=self.logFolder)

    def load(self, chkpointDir):
        if chkpointDir is not None:
            print('Loading checkpoint from %s' % chkpointDir)
        else:
            print('No checkpoint specified. Will initialize weights randomly.')
        # get checkpoint file dirs
        chkpointDir = scanCheckpoint(chkpointDir)

        # update checkpointDir
        self.chkpointDir = chkpointDir
        if self.args.resume:
            self.chkpointFolder, _ = os.path.split(self.chkpointDir)
            self.logFolder = os.path.join(self.chkpointFolder, 'logs')

        epoch, _ = self.model.load(self.chkpointDir)

        self.epoch = epoch + 1 if self.args.resume and epoch is not None else 0

    def save(self, epoch, iteration, info=None):
        # update checkpointDir
        self.chkpointDir = os.path.join(self.chkpointFolder,
                                        'checkpoint_epoch_%04d_it_%05d.tar' % (epoch, iteration))
        print('Saving model to: ' + self.chkpointDir)
        saveDict = {
            'epoch': epoch,
            'iteration': iteration,
        }
        if info is not None:
            saveDict.update(info)
        self.model.save(self.chkpointDir, info=saveDict)

    def log(self, mkFile, info=None):
        writeMessage = ''
        writeMessage += '---------------------- %s ----------------------\n\n' % \
                        time.asctime(time.localtime(time.time()))
        writeMessage += 'bash param: '
        for arg in sys.argv:
            writeMessage += arg + ' '
        writeMessage += '\n\n'

        baseInfos = (
                     ('checkpoint', self.chkpointDir),
                     ('evalFcn', self.args.evalFcn),
                     ('epoch', self.epoch),
                     ('iteration', self.iteration),
                     ('globalStep', self.globalStep),
                     )
        for pairs, title in zip((baseInfos, info),
                                ('basic info:', 'additional info:')):
            if len(pairs) > 0:
                writeMessage += title + '\n\n'
                for (name, value) in pairs:
                    if value is not None:
                        writeMessage += '- ' + name + ': ' + str(value) + '\n'
                writeMessage += '\n'

        with open(os.path.join(self.chkpointFolder, mkFile), "a") as log:
            log.write(writeMessage)

        self.logger.writer.add_text('testPrint/epochs', writeMessage,
                                    global_step=self.globalStep)


# Flip among W dimension. For NCHW data type.
def flipLR(ims):
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


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


class DefaultParser:
    def __init__(self, description='Stereo'):
        self.parser = argparse.ArgumentParser(description=description)

    def seed(self):
        self.parser.add_argument('--seed', type=int, default=1, metavar='S',
                                 help='random seed (default: 1)')
        return self

    # model
    def outputFolder(self):
        self.parser.add_argument('--outputFolder', type=str, default=None,
                                 help='output checkpoints and logs to foleder logs/outputFolder')
        return self

    def maxDisp(self):
        self.parser.add_argument('--maxDisp', type=int, default=192,
                                 help='maximum disparity of unscaled model (or dataset in some module test)')
        return self

    def dispScale(self):
        self.parser.add_argument('--dispScale', type=float, default=1,
                                 help='scale disparity when training (gtDisp/dispscale) and predicting (outputDisp*dispscale')
        return self

    def model(self):
        self.parser.add_argument('--model', default='PSMNet',
                                 help='select model')
        return self

    def chkpoint(self):
        self.parser.add_argument('--chkpoint', type=str, default=None, nargs='+',
                                 help='checkpoint(s) of model(s) to load')
        return self

    def noCuda(self):
        self.parser.add_argument('--noCuda', action='store_true', default=False,
                                 help='enables CUDA training')
        return self

    # logging
    def nSampleLog(self):
        self.parser.add_argument('--nSampleLog', type=int, default=1,
                                 help='number of disparity maps to log')
        return self

    # datasets
    def dataset(self):
        self.parser.add_argument('--dataset', type=str, default='sceneflow',
                                 help='(sceneflow/kitti2012/kitti2015/carla_kitti)')
        return self

    def dataPath(self):
        self.parser.add_argument('--dataPath', default=None, help='folder of dataset')
        return self

    def loadScale(self):
        self.parser.add_argument('--loadScale', type=float, default=[1], nargs='+',
                                 help='scaling applied to data during loading')
        return self

    # training
    def batchSize(self):
        self.parser.add_argument('--batchSize', type=int, default=[0], nargs='+',
                                 help='training and testing batch size')
        return self

    def trainCrop(self):
        self.parser.add_argument('--trainCrop', type=int, default=(256, 512), nargs=2,
                                 help='size of random crop (H x W) applied to data during training')
        return self

    def logEvery(self):
        self.parser.add_argument('--logEvery', type=int, default=10,
                                 help='log every log_every iterations. set to 0 to stop logging')
        return self

    def saveEvery(self):
        self.parser.add_argument('--saveEvery', type=int, default=1,
                                 help='save every save_every epochs; '
                                      'set to -1 to train without saving; '
                                      'set to 0 to save after the last epoch.')
        return self

    def testEvery(self):
        self.parser.add_argument('--testEvery', type=int, default=1,
                                 help='test every test_every epochs. '
                                      '> 0 will not test before training. '
                                      '= 0 will test before training and after final epoch. '
                                      '< 0 will test before training')
        return self

    def epochs(self):
        self.parser.add_argument('--epochs', type=int, default=10,
                                 help='number of epochs to train')
        return self

    def lr(self):
        self.parser.add_argument('--lr', type=float, default=[0.001], help='', nargs='+')
        return self

    def lossWeights(self):
        self.parser.add_argument('--lossWeights', type=float, default=[1], nargs='+',
                                 help='weights of losses if model have multiple losses')
        return self

    def resume(self):
        self.parser.add_argument('--resume', action='store_true', default=False,
                                 help='resume specified training '
                                      '(or save evaluation results to old folder)'
                                      ' else save/log into a new folders')
        return self

    # evaluation
    def evalFcn(self):
        self.parser.add_argument('--evalFcn', type=str, default='outlier',
                                 help='evaluation function used in testing')
        return self

    def validSetSample(self):
        self.parser.add_argument('--validSetSample', type=float, default=1,
                                 help='test with part of valid set')
        return self

    # submission
    def subType(self):
        self.parser.add_argument('--subType', type=str, default=None,
                                 help='dataset type used for submission (eval/test)')
        return self

    # half precision
    def half(self):
        self.parser.add_argument('--half', action='store_true', default=False,
                                 help='enables half precision')
        return self

    # SRdispStereoRefine specified param
    def itRefine(self):
        self.parser.add_argument('--itRefine', type=int, default=1,
                                 help='iterations of refining process')
        return self

    def parse(self):
        args = self.parser.parse_args()

        if hasattr(args, 'noCuda'):
            args.cuda = not args.noCuda and torch.cuda.is_available()

        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            if hasattr(args, 'cuda') and args.cuda:
                torch.cuda.manual_seed(args.seed)

        return args


def struct2dict(struct):
    argsDict = dict((name, getattr(struct, name)) for name in dir(struct)
                    if not name.startswith('__') and not callable(getattr(struct, name)))
    return argsDict


def adjustLearningRate(optimizer, epoch, lr):
    if len(lr) % 2 == 0:
        raise Exception('lr setting should be like \'0.001 300 0.0001 \'')
    nThres = len(lr) // 2 + 1
    for iThres in range(nThres):
        lrThres = lr[2 * iThres]
        if iThres < nThres - 1:
            epochThres = lr[2 * iThres + 1]
            if epoch <= epochThres:
                lr = lrThres
                break
        else:
            lr = lrThres
    print('lr = %f' % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


class Logger:
    def __init__(self, folder=None):
        self.writer = None
        self._folder = None
        if folder is not None:
            self.set(folder)

    def __del__(self):
        if self.writer is not None:
            self.writer.close()

    def set(self, folder):
        if self.writer is not None:
            if folder != self._folder:
                self.writer.close()
        self.writer = SummaryWriter(folder)
        self._folder = folder

    def logImages(self, imags: Imgs, prefix, global_step=None, n=0):
        imags.logPrepare()
        for key, value in imags.items():
            self.logImage(value, prefix + key, 1, global_step, n)

    # Log First n ims into tensorboard
    # Log All ims if n == 0
    def logImage(self, im, name, range, global_step=None, n=0):
        n = min(n, im.size(0))
        if n > 0 and im.dim() > 2:
            im = im[:n]
        if im.dim() == 3 or im.dim() == 2 or (im.dim() == 4 and im.size(1) == 1):
            im = im / range
            im = im.clamp(0, 1)
            im = gray2rgb(im.cpu())
        self.writer.add_images(name, im, global_step=global_step)


class Batch:
    def __init__(self, batch, cuda=None, half=None):
        # for different param types
        if type(batch) in (list, tuple):
            self._assertLen(len(batch))
            self.batch = batch[:]  # deattach with initial list
        elif type(batch) is Batch:
            self.batch = batch.batch[:]
            if cuda is None:
                cuda = batch.cuda
            if half is None:
                half = batch.half
        elif type(batch) is int:
            self._assertLen(batch)
            if batch % 4 != 0:
                raise Exception(f'Error: input batch with length {len(batch)} doesnot match required 4n!')
            self.batch = [None] * batch
        else:
            raise Exception('Error: batch must be class list, tuple, Batch or int!')

        # default params
        if cuda is None:
            cuda = False
        if half is None:
            half = False

        self.half = half
        self.cuda = cuda

        # convert type
        self.batch = forNestingList(self.batch, lambda im: im if im is not None and im.numel() else None)
        if half:
            self.batch = forNestingList(self.batch, lambda im: im.half() if im is not None else None)
        if cuda:
            self.batch = forNestingList(self.batch, lambda im: im.cuda() if im is not None else None)

        # assert nan
        def assertData(t):
            if t is not None and torch.isnan(t).any():
                raise Exception('Error: Data has nan in it')

        forNestingList(self.batch, assertData)

    def _assertLen(self, len):
        assert len % 4 == 0

    def assertLen(self, length):
        if type(length) in (list, tuple):
            for l in length:
                self.assertLen(l)
        else:
            assert len(self) == length

    def assertScales(self, nScales):
        if type(nScales) in (list, tuple):
            for nScale in nScales:
                self.assertScales(nScale)
        else:
            self.assertLen(nScales * 4)

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, item):
        return self.batch[item]

    def __setitem__(self, key, value):
        self.batch[key] = value

    def detach(self):
        return Batch(self)

    def scaleBatch(self, scale):
        return Batch(self.batch[scale * 4: (scale + 1) * 4], cuda=self.cuda, half=self.half)

    def lastScaleBatch(self):
        return Batch(self.batch[-4:], cuda=self.cuda, half=self.half)

    def firstScaleBatch(self):
        return Batch(self.batch[:4], cuda=self.cuda, half=self.half)

    def highResRGBs(self, set=None):
        if set is not None:
            self.batch[0:2] = set
        return self.batch[0:2]

    def highResDisps(self, set=None):
        if set is not None:
            self.batch[2:4] = set
        return self.batch[2:4]

    def lowResRGBs(self, set=None):
        if set is not None:
            self.batch[4:6] = set
        return self.batch[4:6]

    def lowResDisps(self, set=None):
        if set is not None:
            self.batch[6:8] = set
        return self.batch[6:8]

    def lowestResRGBs(self, set=None):
        if set is not None:
            self.batch[-4:-2] = set
        return self.batch[-4:-2]

    def lowestResDisps(self, set=None):
        if set is not None:
            self.batch[-2:] = set
        return self.batch[-2:]

    def allRGBs(self, set=None):
        if set is not None:
            self.batch[0::4] = set[:len(set) // 2]
            self.batch[1::4] = set[len(set) // 2:]
        return self.batch[0::4] + self.batch[1::4]

    def allDisps(self, set=None):
        if set is not None:
            self.batch[2::4] = set[:len(set) // 2]
            self.batch[3::4] = set[len(set) // 2:]
        return self.batch[2::4] + self.batch[3::4]


def forNestingList(l, fcn):
    if type(l) in (list, tuple):
        l = [forNestingList(e, fcn) for e in l]
        return l
    else:
        return fcn(l)


def scanCheckpoint(checkpointDirs):
    if type(checkpointDirs) in (list, tuple):
        checkpointDirs = [scanCheckpoint(dir) for dir in checkpointDirs]
    else:
        # if checkpoint is folder
        if os.path.isdir(checkpointDirs):
            filenames = [d for d in os.listdir(checkpointDirs) if os.path.isfile(os.path.join(checkpointDirs, d))]
            filenames.sort()
            latestCheckpointName = None
            latestEpoch = None

            def _getEpoch(name):
                try:
                    keywords = name.split('_')
                    epoch = keywords[keywords.index('epoch') + 1]
                    return int(epoch)
                except ValueError:
                    return None

            for filename in filenames:
                if any(filename.endswith(extension) for extension in ('.tar', '.pt')):
                    if latestCheckpointName is None:
                        latestCheckpointName = filename
                        latestEpoch = _getEpoch(filename)
                    else:
                        epoch = _getEpoch(filename)
                        if epoch > latestEpoch or epoch is None:
                            latestCheckpointName = filename
                            latestEpoch = epoch
            checkpointDirs = os.path.join(checkpointDirs, latestCheckpointName)

    return checkpointDirs


def getSuffix(checkpointDirOrFolder):
    if type(checkpointDirOrFolder) is str or \
            (type(checkpointDirOrFolder) in (list, tuple) and len(checkpointDirOrFolder) == 1):
        checkpointDir = scanCheckpoint(checkpointDirOrFolder[0])
        checkpointFolder, _ = os.path.split(checkpointDir)
        checkpointFolder = checkpointFolder.split('/')[-1]
        saveFolderSuffix = checkpointFolder.split('_')[2:]
        saveFolderSuffix = ['_' + suffix for suffix in saveFolderSuffix]
        saveFolderSuffix = ''.join(saveFolderSuffix)
    else:
        saveFolderSuffix = ''
    return saveFolderSuffix


def depth(l):
    if type(l) in (tuple, list):
        return 1 + max(depth(item) for item in l)
    else:
        return 0


class Filter:
    def __init__(self, weight=0.1):
        self.weight = weight
        self.old = None

    def __call__(self, x):
        self.old = x if self.old is None else self.old * (1 - self.weight) + x * self.weight
        return self.old


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


def shuffleLists(lists):
    c = list(zip(*lists))
    random.shuffle(c)
    lists = list(zip(*c))
    return lists
