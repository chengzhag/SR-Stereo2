from utils import myUtils
import os
from models import Stereo
from evaluation.Stereo_eval import Evaluation
from train.Train import Train as Base
import dataloader


class Train(Base):

    def _trainIt(self, batch: myUtils.Batch):
        loss, outputs = self.experiment.model.train(batch=batch.detach())
        for disp, input, side in zip(batch.lowestResDisps(), batch.lowestResRGBs(), ('L', 'R')):
            outputs.addImg(name='gtDisp' + side, img=disp, range=self.experiment.model.outMaxDisp)
            outputs.addImg(name='inputRgb' + side, img=input)

        return loss, outputs.cpu()


def main():
    # Arguments
    args = myUtils.DefaultParser(description='evaluate Stereo net or SR-Stereo net') \
        .outputFolder().maxDisp().dispScale().model().dataPath() \
        .chkpoint().noCuda().seed().evalFcn().nSampleLog().dataset() \
        .loadScale().mask().batchSize().trainCrop().logEvery().testEvery() \
        .saveEvery().epochs().lr().lossWeights().subType() \
        .half().resume().itRefine().validSetSample().noComet().parse()

    # Dataset
    trainImgLoader, testImgLoader = dataloader.getDataLoader(dataPath=args.dataPath,
                                                             dataset=args.dataset,
                                                             trainCrop=args.trainCrop,
                                                             batchSizes=args.batchSize,
                                                             loadScale=args.loadScale,
                                                             mode='training' if args.subType is None else args.subType,
                                                             validSetSample=args.validSetSample,
                                                             mask=args.mask)

    # Model
    stereo = Stereo.getModel(
        args.model,
        kitti=trainImgLoader.kitti,
        maxDisp=args.maxDisp,
        dispScale=args.dispScale,
        cuda=args.cuda,
        half=args.half)
    if hasattr(stereo, 'itRefine'):
        stereo.itRefine = args.itRefine

    # Train
    stage, _ = os.path.splitext(os.path.basename(__file__))
    experiment = myUtils.Experiment(model=stereo, stage=stage, args=args)
    test = Evaluation(experiment=experiment, testImgLoader=testImgLoader)
    train = Train(test=test, trainImgLoader=trainImgLoader)
    train()


if __name__ == '__main__':
    main()
