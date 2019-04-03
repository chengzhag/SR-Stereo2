import time
import os
from utils import myUtils
import sys


class Evaluation(myUtils.Experiment):
    def __init__(self, model, stage, args, testImgLoader):
        super().__init__(model=model, stage=stage, args=args)
        self.testImgLoader = testImgLoader

    def _evalIt(self, batch: myUtils.Batch) -> (myUtils.Loss, myUtils.Imgs):
        # return scores, outputs
        return None, None

    def __call__(self):
        # Evaluate
        ticETC = time.time()
        timeFilter = myUtils.Filter()
        totalTestLoss = None
        avgTestLoss = None
        for batchIdx, batch in enumerate(self.testImgLoader, 1):
            batch = myUtils.Batch(batch, cuda=self.model.cuda, half=self.model.half)

            loss, imgs = self._evalIt(batch)

            if totalTestLoss is None:
                totalTestLoss = loss
            else:
                totalTestLoss.accumuate(loss)

            if batchIdx == 1:
                self.logger.logImages(imgs, 'testImages/', self.globalStep, self.args.nSampleLog)

            timeLeft = timeFilter((time.time() - ticETC) / 3600 * (len(self.testImgLoader) - batchIdx))

            avgTestLoss = totalTestLoss.clone()
            avgTestLoss.avg()

            # print info
            printMessage = 'it %d/%d, %s%sleft %.2fh' % (
                batchIdx, len(self.testImgLoader),
                loss.strPrint(), avgTestLoss.strPrint(suffix='Avg'), timeLeft)
            print(printMessage)
            self.logger.writer.add_text('testPrint/iterations', printMessage,
                                        global_step=self.globalStep)

            ticETC = time.time()

        # Log
        logTime = time.asctime(time.localtime(time.time()))
        logDir = os.path.join(self.chkpointFolder, 'test_results.md')

        writeMessage = ''
        writeMessage += '---------------------- %s ----------------------\n\n' % logTime
        writeMessage += 'bash param: '
        for arg in sys.argv:
            writeMessage += arg + ' '
        writeMessage += '\n\n'

        baseInfos = (('data', self.testImgLoader.datapath),
                     ('loadScale', self.testImgLoader.loadScale),
                     ('trainCrop', self.testImgLoader.trainCrop),
                     ('checkpoint', self.chkpointDir),
                     ('evalFcn', self.args.evalFcn),
                     ('epoch', self.epoch),
                     ('iteration', self.iteration),
                     ('globalStep', self.globalStep),
                     )
        for pairs, title in zip((baseInfos, avgTestLoss.items()),
                                ('basic info:', 'test results:')):
            if len(pairs) > 0:
                writeMessage += title + '\n\n'
                for (name, value) in pairs:
                    if value is not None:
                        writeMessage += '- ' + name + ': ' + str(value) + '\n'
                writeMessage += '\n'

        with open(logDir, "a") as log:
            log.write(writeMessage)

        self.logger.writer.add_text('testPrint/epochs', writeMessage,
                                    global_step=self.globalStep)

        for name, value in avgTestLoss.items():
            self.logger.writer.add_scalar('testLosses/' + name, value, self.globalStep)

        return avgTestLoss
