import time

import utils.data
import utils.experiment
import utils.imProcess
from utils import myUtils
import os


class Evaluation:
    def __init__(self, experiment: utils.experiment.Experiment, testImgLoader):
        self.experiment = experiment
        self.testImgLoader = testImgLoader

    def _evalIt(self, batch: utils.data.Batch) -> (utils.data.NameValues, utils.imProcess.Imgs):
        # return scores, outputs
        return None, None

    def __call__(self):
        # Evaluate
        ticETC = time.time()
        timeFilter = myUtils.Filter()
        totalTestLoss = None
        avgTestLoss = utils.data.NameValues()
        for batchIdx, batch in enumerate(self.testImgLoader, 1):
            batch = utils.data.Batch(batch, cuda=self.experiment.model.cuda, half=self.experiment.model.half)

            loss, imgs = self._evalIt(batch)

            if totalTestLoss is None:
                totalTestLoss = loss
            else:
                totalTestLoss.accumuate(loss)

            if self.experiment.args.subType is not None and self.experiment.args.subType.startswith('sub'):
                name = self.testImgLoader.dataset.name(batchIdx - 1)
                name, extension = os.path.splitext(name)
                imgs.clone().save(dir=os.path.join(self.experiment.chkpointFolder, 'submission'), name=name)

            if batchIdx == 1:
                self.experiment.logger.logImages(imgs.clone(), 'test/', self.experiment.epoch,
                                                 self.experiment.args.nSampleLog)
                if not self.experiment.cometExp.disabled:
                    imgs.clone().save(dir='temp', name='temp')
                for name in imgs.keys():
                    self.experiment.cometExp.set_step(self.experiment.epoch)
                    self.experiment.cometExp.log_image(os.path.join('temp', name, 'temp.png'), name, overwrite=True)

            timeLeft = timeFilter((time.time() - ticETC) / 3600 * (len(self.testImgLoader) - batchIdx))

            avgTestLoss = totalTestLoss.clone()
            avgTestLoss.avg()

            # print info
            printMessage = 'it %d/%d, %s%sleft %.2fh' % (
                batchIdx, len(self.testImgLoader),
                loss.strPrint(), avgTestLoss.strPrint(suffix='Avg'), timeLeft)
            print(printMessage)
            self.experiment.logger.writer.add_text('test/iterations', printMessage,
                                                   global_step=self.experiment.epoch)

            ticETC = time.time()

        # Log
        for name, value in avgTestLoss.items():
            self.experiment.logger.writer.add_scalar('test/' + name, value, self.experiment.epoch)
        self.experiment.cometExp.log_metrics(avgTestLoss, prefix='test', step=self.experiment.epoch)
        for name, value in (('data', self.testImgLoader.datapath),
                            ('loadScale', self.testImgLoader.loadScale),
                            ('trainCrop', self.testImgLoader.trainCrop)):
            avgTestLoss[name] = value
        self.experiment.log(mkFile='test_results.md', info=avgTestLoss.items())

        return avgTestLoss
