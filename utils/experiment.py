import argparse
import os
import sys
import time

from comet_ml import Experiment as CometExp
import torch
from tensorboardX import SummaryWriter

from utils.data import NameValues
from utils.myUtils import struct2dict, getattrNE, checkDir
from utils.imProcess import gray2rgb, Imgs


class Experiment:
    def __init__(self, model, stage, args):
        self.cometExp = CometExp(project_name='srstereo',
                                 auto_metric_logging=False,
                                 auto_param_logging=False,
                                 log_code=False,
                                 disabled=args.noComet)
        self.cometExp.log_parameters(dic=struct2dict(args), prefix='args')
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
        if args.resume == 'toOld':
            if args.chkpoint is None:
                raise Exception('Error: No checkpoint to resume!')
            elif len(args.chkpoint) > 1:
                raise Exception('Error: Cannot resume multi-checkpoints model!')
        # if not resume, result will be saved to new folder
        else:
            # auto experiment naming
            saveFolderSuffix = NameValues((
                ('model', args.model),
                ('loadScale', getattrNE(args, 'loadScale')),
                ('trainCrop', getattrNE(args, 'trainCrop')),
                ('batchSize', getattrNE(args, 'batchSize')),
                ('lossWeights', getattrNE(args, 'lossWeights')),
            ))
            startTime = time.strftime('%y%m%d%H%M%S', time.localtime(time.time()))
            self.cometExp.log_parameter(name='startTime', value=startTime)
            newFolderName = startTime + saveFolderSuffix.strSuffix()
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
        if self.args.resume == 'toOld':
            self.chkpointFolder, _ = os.path.split(self.chkpointDir[0])
            self.logFolder = os.path.join(self.chkpointFolder, 'logs')

        epoch, _ = self.model.load(self.chkpointDir)

        self.epoch = epoch + 1 if self.args.resume is not None and epoch is not None else 0

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
        self.parser.add_argument('--model', type=str, default=None, nargs='+',
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

    def mask(self):
        self.parser.add_argument('--mask', type=int, default=None, nargs='+',
                                 help='mask for dataloader')
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
        self.parser.add_argument('--lossWeights', type=float, default=1, nargs='+',
                                 help='weights of losses if model have multiple losses')
        return self

    def resume(self):
        self.parser.add_argument('--resume', type=str, default=None,
                                 help='resume specified training '
                                      '(or save evaluation results to old folder)'
                                      ' toNew/toOld')
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

    def noComet(self):
        self.parser.add_argument('--noComet', action='store_true', default=False,
                                 help='disable comet logging')
        return self

    def argument(self):
        self.parser.add_argument('--argument', type=float, default=None, nargs=2,
                                 help='scaling range (from, to) of argumentation when training')
        return self

    def parse(self):
        args = self.parser.parse_args()

        if hasattr(args, 'noCuda'):
            args.cuda = not args.noCuda and torch.cuda.is_available()

        if hasattr(args, 'model') and args.model is not None and len(args.model) == 1:
            args.model = args.model[0]

        if hasattr(args, 'seed'):
            torch.manual_seed(args.seed)
            if hasattr(args, 'cuda') and args.cuda:
                torch.cuda.manual_seed(args.seed)

        return args


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


def scanCheckpoint(checkpointDirs):
    if checkpointDirs is None:
        return None
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
                if any(filename.endswith(extension) for extension in ('.tar', '.pt', '.ckpt')):
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


def checkStateDict(model: torch.nn.Module, stateDict: dict, strict=False, possiblePrefix=None):
    writeModelDict = model.state_dict()
    selectModelDict = {}
    for name, value in stateDict.items():
        if possiblePrefix is not None:
            if type(possiblePrefix) is str:
                if name.startswith(possiblePrefix):
                    name = name[len(possiblePrefix):]
            elif type(possiblePrefix) in (list, tuple):
                for prefix in possiblePrefix:
                    if name.startswith(prefix):
                        name = name[len(prefix):]
                        break
        if name in writeModelDict and writeModelDict[name].size() == value.size():
            selectModelDict[name] = value
        else:
            message = 'Warning! While copying the parameter named {}, ' \
                      'whose dimensions in the model are {} and ' \
                      'whose dimensions in the checkpoint are {}.' \
                .format(
                name, writeModelDict[name].size() if name in writeModelDict else '(Not found)',
                value.size()
            )
            if strict:
                raise Exception(message)
            else:
                print(message)
    writeModelDict.update(selectModelDict)
    return writeModelDict


