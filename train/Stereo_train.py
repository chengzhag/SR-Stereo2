import torch.utils.data
import os
from models import Stereo
from evaluation import Stereo_eval
from utils import myUtils
from train.Train import Train as Base


class Train(Base):
    def __init__(self, model, args, trainImgLoader):
        stage, _ = os.path.splitext(os.path.basename(__file__))
        super().__init__(model=model, stage=stage, args=args, trainImgLoader=trainImgLoader)

    def _trainIt(self, batch: myUtils.Batch):
        loss, outputs = self.model.train(batch=batch.detach(),
                                         kitti=self.trainImgLoader.kitti,)
        for disp, input, side in zip(batch.lowestResDisps(), batch.lowestResRGBs(), ('L', 'R')):
            outputs.addImg('Disp', disp, range=self.model.outMaxDisp, prefix='gt', side=side)
            outputs.addImg('Rgb', input, prefix='input', side=side)

        return loss, outputs


def main():
    import dataloader

    # Arguments
    args = myUtils.DefaultParser(description='evaluate Stereo net or SR-Stereo net') \
        .outputFolder().maxDisp().dispScale().model().dataPath() \
        .chkpoint().noCuda().seed().evalFcn().nSampleLog().dataset() \
        .loadScale().batchSize().trainCrop().logEvery().testEvery() \
        .saveEvery().epochs().lr().lossWeights().subType() \
        .half().resume().itRefine().validSetSample().parse()

    # Dataset
    trainImgLoader, _ = dataloader.getDataLoader(dataPath=args.dataPath,
                                                 dataset=args.dataset,
                                                 trainCrop=args.trainCrop,
                                                 batchSizes=args.batchSize,
                                                 loadScale=args.loadScale,
                                                 mode='training' if args.subType is None else args.subType,
                                                 validSetSample=args.validSetSample)

    stereo = getattr(Stereo, args.model)(
        maxDisp=args.maxDisp, dispScale=args.dispScale, cuda=args.cuda, half=args.half)
    if hasattr(stereo, 'itRefine'):
        stereo.itRefine = args.itRefine

    # Train
    train = Train(model=stereo, args=args, trainImgLoader=trainImgLoader)
    train()


if __name__ == '__main__':
    main()
