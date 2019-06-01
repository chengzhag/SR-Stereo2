import utils.experiment
import torch.utils.data as data
import random
from PIL import Image
import numpy as np
import utils.data
from utils import python_pfm as pfm
import torchvision.transforms as transforms
import operator
import torch
import os


def rgbLoader(path):
    return Image.open(path).convert('RGB')


def pfmLoader(path):
    return pfm.readPFM(path)[0]


def grayLoader(path):
    return Image.open(path)


class myImageFloder(data.Dataset):
    # trainCrop = (W, H)
    def __init__(self, inputLdirs=None, inputRdirs=None, gtLdirs=None, gtRdirs=None,
                 cropSize=(256, 512), kitti=False, loadScale=(1,), mode='training',
                 mask=(1, 1, 1, 1), dispScale=1, argument=None):
        self.mask = mask
        self.mode = mode
        self.dirs = (inputLdirs, inputRdirs, gtLdirs, gtRdirs)
        self.inputLoader = rgbLoader
        self.gtLoader = grayLoader if kitti else pfmLoader
        self.cropSize = cropSize
        self.dispScale = dispScale
        self.loadScale = loadScale
        self.argument = argument

    def __getitem__(self, index):
        def scale(im, method, scaleRatios):
            w, h = im.size
            ims = []
            for r in scaleRatios:
                ims.append(im.resize((round(w * r), round(h * r)), method))
            return ims

        def testCrop(im):
            w, h = im.size
            wCrop, hCrop = self.cropSize
            return im.crop((w - wCrop, h - hCrop, w, h))

        class RandomCrop:
            def __init__(self, trainCrop):
                self.hCrop, self.wCrop = trainCrop
                self.x1 = None
                self.y1 = None

            def __call__(self, input):
                w, h = input.size
                if self.x1 is None: self.x1 = random.randint(0, w - self.wCrop)
                if self.y1 is None: self.y1 = random.randint(0, h - self.hCrop)
                return input.crop((self.x1, self.y1, self.x1 + self.wCrop, self.y1 + self.hCrop))

        class RandomScale:
            def __init__(self, scaleFrom, scaleTo):
                self.scale = random.uniform(scaleFrom, scaleTo)

            def __call__(self, method, input):
                output = scale(input, method, [self.scale])
                return output[0]

        class RandomRotate:
            def __init__(self, rotateFrom, rotateTo):
                self.rotate = random.uniform(rotateFrom, rotateTo)

            def __call__(self, method, input):
                output = input.rotate(self.rotate, method)
                return output

        def getPatch():
            if self.cropSize is not None:
                randomCrop = RandomCrop(trainCrop=self.cropSize)
            if self.argument is not None:
                randomScale = RandomScale(scaleFrom=self.argument[0], scaleTo=self.argument[1])
                # randomRotate = RandomRotate(rotateFrom=-30, rotateTo=30)

            def loadIm(dirsIndex, loader, scaleRatios, isRGBorDepth):
                ims = []
                if not self.mask[dirsIndex] or self.dirs[dirsIndex] is None:
                    return [np.array([], dtype=np.float32), ] * len(self.loadScale)
                im0 = loader(self.dirs[dirsIndex][index])
                if type(im0) == np.ndarray:
                    im0 = Image.fromarray(im0)

                # scale first to reduce time consumption
                scaleMethod = Image.ANTIALIAS if isRGBorDepth else Image.NEAREST
                # rotateMethod = Image.BICUBIC if isRGBorDepth else Image.NEAREST
                im0 = scale(im0, scaleMethod, (scaleRatios[0],))[0]
                ims.append(im0)

                multiScales = []
                if len(scaleRatios) > 1:
                    for i in range(1, len(scaleRatios)):
                        multiScales.append(scaleRatios[i] / scaleRatios[0])

                if self.mode == 'PIL':
                    pass
                else:
                    if self.mode == 'rawScaledTensor':
                        pass
                    elif self.mode in ('training', 'testing', 'submission'):
                        if self.mode == 'training':
                            # random scale
                            if self.argument is not None:
                                ims[0] = randomScale(method=scaleMethod, input=ims[0])
                                # ims[0] = randomRotate(method=rotateMethod, input=ims[0])
                            # random crop
                            ims[0] = randomCrop(ims[0])
                        elif self.mode == 'testing':
                            if self.cropSize is not None:
                                # crop to the same size
                                ims[0] = testCrop(ims[0])
                        elif self.mode == 'submission':
                            # do no crop
                            pass
                        else:
                            raise Exception('No stats \'%s\'' % self.mode)
                        # scale to different sizes specified by scaleRatios
                    else:
                        raise Exception('No mode %s!' % self.mode)

                    # scale to different sizes specified by scaleRatios
                    ims += scale(ims[0], scaleMethod, multiScales)
                    ims = [transforms.ToTensor()(im) for im in ims]
                    if not isRGBorDepth and any([torch.sum(im > 0) < 100 for im in ims]):
                        # print('Note: Crop has no data, recropping...')
                        return None
                    if any([torch.isnan(im).any() for im in ims]):
                        return None

                ims = [np.ascontiguousarray(im, dtype=np.float32) for im, scaleRatio in zip(ims, scaleRatios)]
                if not isRGBorDepth:
                    if self.argument is not None:
                        ims = [im * randomScale.scale for im in ims]
                    ims = [im / self.dispScale * scaleRatio for im, scaleRatio in zip(ims, scaleRatios)]
                return ims

            gtL = loadIm(2, self.gtLoader, self.loadScale, False)
            if gtL is None:
                return None
            gtR = loadIm(3, self.gtLoader, self.loadScale, False)
            if gtR is None:
                return None

            inputL = loadIm(0, self.inputLoader, self.loadScale, True)
            if inputL is None:
                return None
            inputR = loadIm(1, self.inputLoader, self.loadScale, True)
            if inputR is None:
                return None

            outputs = [inputL, inputR, gtL, gtR]
            return outputs

        while True:
            outputs = getPatch()
            if outputs is not None:
                # for iIms, ims in enumerate(outputs):
                #     iCompare = iIms + 1 if iIms % 2 == 0 else iIms - 1
                #     for iScale in range(len(ims)):
                #         if ims[iScale].size == 0:
                #             ims[iScale] = np.zeros_like(outputs[iCompare][iScale])
                r = [im for scale in zip(*outputs) for im in scale]
                return tuple(r)

    def __len__(self):
        for dirs in self.dirs:
            if dirs is not None:
                return len(dirs)
        raise Exception('Empty dataloader!')

    def name(self, index):
        for dirs in self.dirs:
            if dirs is not None:
                return dirs[index].split('/')[-1]
        raise Exception('Empty dataloader!')


# cropScale: Defaultly set to loadScale to remain ratio between loaded image and cropped image.
# loadScale: A list of scale to load. Will return 4 * len(loadScale) images. Should be decreasing values.
def getDataLoader(dataPath, dataset='sceneflow', trainCrop=(256, 512), batchSizes=(0, 0),
                  loadScale=(1,), mode='normal', mask=None, validSetSample=1, argument=None):
    if mask is None:
        mask = (1, 1, 1, 1)

    # import listing file fcn according to param dataset
    if dataset == 'sceneflow':
        from dataloader import listSceneFlowFiles as listFile
    elif dataset == 'kitti2012':
        if mode == 'subTest':
            from dataloader import listKitti2012Sub as listFile
        else:
            from dataloader import listKitti2012Files as listFile
    elif dataset in ('kitti2015', 'kitti2015dense'):
        if mode == 'subTest':
            from dataloader import listKitti2015Sub as listFile
        else:
            from dataloader import listKitti2015Files as listFile
    elif dataset == 'carla':
        from dataloader import listCarlaFiles as listFile
    else:
        raise Exception('No dataloader for dataset \'%s\'!' % dataset)

    # split returned tuple into training and testing
    paths = list(listFile.dataloader(dataPath))
    pathsTrain = paths[0:4]
    pathsTest = paths[4:8] if len(paths) == 8 else None

    # sample first validSetSample dirs from pathsTest
    if validSetSample < 1:
        pathsTest = list(zip(*pathsTest))
        pathsTest = pathsTest[:round(len(pathsTest) * validSetSample)]
        pathsTest = list(zip(*pathsTest))

    # special params setting for kitti dataset
    kitti = dataset in ('kitti2012', 'kitti2015', 'kitti2015_dense')
    if dataset in ('kitti2012', 'kitti2015'):
        mask = [a and b for a, b in zip(mask, (1, 1, 1, 0))]
        dispScale = 256
        kittiScale = 1
    elif dataset == 'kitti2015_dense':
        dispScale = 170
        kittiScale = 2
    else:
        dispScale = 1
        kittiScale = 1

    testCrop = (round(1232 * loadScale[0] * kittiScale), round(368 * loadScale[0] * kittiScale)) if kitti else None

    if mode in ('subTrain', 'subEval', 'subTrainEval', 'subTest'):
        if mode == 'subTrain':
            pathsTest = pathsTrain
        elif mode == 'subEval':
            pass
        elif mode == 'subTrainEval':
            pathsTest = [dirsTrain + dirsEval if dirsTrain is not None else None for dirsTrain, dirsEval in
                         zip(pathsTrain, pathsTest)]
        elif mode == 'subTest':
            pathsTest = pathsTrain
        pathsTrain = None
        mode = 'submission'

    if mode == 'trainSub':
        pathsTrain = [dirsTrain + dirsEval if dirsTrain is not None else None for dirsTrain, dirsEval in
                      zip(pathsTrain, pathsTest)]
        pathsTest = None
        mode = 'training'

    trainImgLoader = torch.utils.data.DataLoader(
        myImageFloder(*pathsTrain,
                      trainCrop,
                      kitti=kitti,
                      loadScale=loadScale,
                      mode=mode,
                      mask=mask,
                      dispScale=dispScale,
                      argument=argument),
        batch_size=batchSizes[0], shuffle=True, num_workers=4, drop_last=False
    ) if batchSizes[0] > 0 and pathsTrain is not None else None

    testImgLoader = torch.utils.data.DataLoader(
        myImageFloder(*pathsTest,
                      cropSize=testCrop if mode in ('testing', 'training') else trainCrop,
                      kitti=kitti,
                      loadScale=loadScale,
                      mode='testing' if mode == 'training' else mode,
                      mask=mask,
                      dispScale=dispScale,
                      argument=None),
        batch_size=batchSizes[1], shuffle=False, num_workers=4, drop_last=False
    ) if batchSizes[1] > 0 and pathsTest is not None else None

    # Add dataset info to imgLoader objects
    # For KITTI, evaluation should exclude zero disparity pixels. A flag kitti will be added to imgLoader.
    for imgLoader in (trainImgLoader, testImgLoader):
        if imgLoader is not None:
            imgLoader.kitti = kitti
            imgLoader.loadScale = loadScale
            imgLoader.trainCrop = trainCrop
            imgLoader.datapath = dataPath
            imgLoader.batchSizes = batchSizes

    return trainImgLoader, testImgLoader


def main():
    from utils import myUtils

    # Arguments
    args = utils.experiment.DefaultParser(description='DataLoader module test') \
        .outputFolder().maxDisp().seed().dataPath().loadScale().nSampleLog().dataset().parse()

    # Dataset
    trainImgLoader, testImgLoader = getDataLoader(dataPath=args.dataPath, dataset=args.dataset,
                                                  batchSizes=(1, 1),
                                                  loadScale=args.loadScale,
                                                  mode='rawScaledTensor')

    # Log samples
    logFolder = [folder for folder in args.dataPath.split('/') if folder != '']
    logFolder[-1] += '_listTest'
    logger = utils.experiment.Logger(os.path.join(*logFolder))

    def logSamplesFrom(imgLoader, tag):
        if imgLoader is not None:
            for iSample, sample in enumerate(imgLoader, 1):
                batch = utils.data.Batch(sample)
                print('sample %d' % iSample)
                for iScale, scale in enumerate(args.loadScale):
                    for name, im in zip(('inputL', 'inputR', 'gtL', 'gtR'), batch.scaleBatch(iScale)):
                        if im is not None:
                            name = tag + '/' + name + '_' + str(scale)
                            print('logging ' + name)
                            range = args.maxDisp if im.size(1) == 1 else 255
                            logger.logImage(im, name, range, global_step=iSample, n=1)

                if iSample >= args.nSampleLog:
                    break

    logSamplesFrom(trainImgLoader, 'trainImgLoader')
    logSamplesFrom(testImgLoader, 'testImgLoader')


if __name__ == '__main__':
    main()
