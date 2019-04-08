import os
from models import SR
from utils import myUtils
from evaluation.Evaluation import Evaluation as Base

class Evaluation(Base):

    def _evalIt(self, batch: myUtils.Batch):
        loss, outputs = self.experiment.model.test(batch=batch.detach(),
                                                   evalType=self.experiment.args.evalFcn)
        imgs = batch.lowResRGBs() + batch.highResRGBs()

        for imsSide, side in zip((imgs[0::2], imgs[1::2]), ('L', 'R')):
            for name, im in zip(('input', 'gt'), imsSide):
                outputs.addImg(name=name + side, img=im)

        return loss, outputs.cpu()


def main():
    import dataloader

    # Arguments
    args = myUtils.DefaultParser(description='evaluate SR net') \
        .outputFolder().dataPath().model().chkpoint().noCuda().seed() \
        .evalFcn().nSampleLog().dataset().loadScale().batchSize() \
        .half().resume().validSetSample().noComet().subType().parse()

    # Dataset
    if args.model in ('EDSR',):
        mask = (1, 1, 0, 0)
    elif args.model in ('EDSRdisp',):
        mask = (1, 1, 1, 1)
    else:
        raise Exception('Error: No model named \'%s\'!' % args.model)
    _, testImgLoader = dataloader.getDataLoader(dataPath=args.dataPath,
                                                dataset=args.dataset,
                                                batchSizes=args.batchSize,
                                                loadScale=(args.loadScale[0], args.loadScale[0] / 2),
                                                mode='testing' if args.subType is None else args.subType,
                                                validSetSample=args.validSetSample,
                                                mask=mask)

    # Model
    sr = getattr(SR, args.model)(cuda=args.cuda, half=args.half)

    # Test
    stage, _ = os.path.splitext(os.path.basename(__file__))
    experiment = myUtils.Experiment(model=sr, stage=stage, args=args)
    test = Evaluation(experiment=experiment, testImgLoader=testImgLoader)
    test()


if __name__ == '__main__':
    main()
