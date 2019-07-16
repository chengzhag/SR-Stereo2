import utils.experiment
import utils.data
from utils import myUtils
import os
from models import SR
from evaluation.SR_eval import Evaluation
from train.Train import Train as Base
import dataloader


class Train(Base):

    def _trainIt(self, batch: utils.data.Batch):
        loss, outputs = self.experiment.model.train(batch=batch.detach())

        for input, gt, side in zip(batch.lowResRGBs(), batch.highResRGBs(), ('L', 'R')):
            outputs.addImg(name='inputSr' + side, img=input)
            outputs.addImg(name='gtSr' + side, img=gt)

        return loss, outputs.cpu()


def main():
    # Arguments
    args = utils.experiment.DefaultParser(description='train or finetune SR net') \
        .outputFolder().dataPath().model().chkpoint().noCuda().seed().evalFcn() \
        .nSampleLog().dataset().loadScale().trainCrop().batchSize().logEvery() \
        .testEvery().saveEvery().epochs().lr().half().lossWeights().resume().subType() \
        .validSetSample().noComet().argument().mask().parse()

    # Dataset
    mask = SR.getMask(args.model)
    trainImgLoader, testImgLoader = dataloader.getDataLoader(
        dataPath=args.dataPath,
        dataset=args.dataset,
        trainCrop=args.trainCrop,
        batchSizes=args.batchSize,
        loadScale=(args.loadScale[0], args.loadScale[0] / 2),
        mode='training' if args.subType is None else args.subType,
        validSetSample=args.validSetSample,
        mask=mask if args.mask is None else args.mask,
        argument=args.argument)

    # Model
    sr = SR.getModel(args.model, cuda=args.cuda, half=args.half)

    # Train
    stage, _ = os.path.splitext(os.path.basename(__file__))
    experiment = utils.experiment.Experiment(model=sr, stage=stage, args=args)
    test = Evaluation(experiment=experiment, testImgLoader=testImgLoader)
    train = Train(test=test, trainImgLoader=trainImgLoader)
    train()


if __name__ == '__main__':
    main()
