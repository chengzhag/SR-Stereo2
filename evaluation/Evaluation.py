import time
from utils import myUtils


class Evaluation:
    def __init__(self, experiment: myUtils.Experiment, testImgLoader):
        self.experiment = experiment
        self.testImgLoader = testImgLoader

    def _evalIt(self, batch: myUtils.Batch) -> (myUtils.Loss, myUtils.Imgs):
        # return scores, outputs
        return None, None

    def __call__(self):
        # Evaluate
        ticETC = time.time()
        timeFilter = myUtils.Filter()
        totalTestLoss = None
        avgTestLoss = myUtils.Loss()
        for batchIdx, batch in enumerate(self.testImgLoader, 1):
            batch = myUtils.Batch(batch, cuda=self.experiment.model.cuda, half=self.experiment.model.half)

            loss, imgs = self._evalIt(batch)

            if totalTestLoss is None:
                totalTestLoss = loss
            else:
                totalTestLoss.accumuate(loss)

            if batchIdx == 1:
                self.experiment.logger.logImages(imgs, 'testImages/', self.experiment.globalStep,
                                                 self.experiment.args.nSampleLog)

            timeLeft = timeFilter((time.time() - ticETC) / 3600 * (len(self.testImgLoader) - batchIdx))

            avgTestLoss = totalTestLoss.clone()
            avgTestLoss.avg()

            # print info
            printMessage = 'it %d/%d, %s%sleft %.2fh' % (
                batchIdx, len(self.testImgLoader),
                loss.strPrint(), avgTestLoss.strPrint(suffix='Avg'), timeLeft)
            print(printMessage)
            self.experiment.logger.writer.add_text('testPrint/iterations', printMessage,
                                                   global_step=self.experiment.globalStep)

            ticETC = time.time()

        # Log
        for name, value in avgTestLoss.items():
            self.experiment.logger.writer.add_scalar('testLosses/' + name, value, self.experiment.globalStep)

        for name, value in (('data', self.testImgLoader.datapath),
                            ('loadScale', self.testImgLoader.loadScale),
                            ('trainCrop', self.testImgLoader.trainCrop)):
            avgTestLoss.addLoss(loss=value, name=name)
        self.experiment.log(mkFile='test_results.md', info=avgTestLoss.items())

        return avgTestLoss
