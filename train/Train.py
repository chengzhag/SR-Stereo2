import utils.experiment
import time
import utils.data
import utils.imProcess
from utils import myUtils
from evaluation.Evaluation import Evaluation
import torch


class Train:
    def __init__(self, test: Evaluation, trainImgLoader):
        self.experiment = test.experiment
        self.test = test
        self.trainImgLoader = trainImgLoader
        self.experiment.globalStep = (self.experiment.epoch - 1) * len(self.trainImgLoader) \
            if self.experiment.epoch > 0 else 0
        self.experiment.model.setLossWeights(self.experiment.args.lossWeights)

    def _trainIt(self, batch: utils.data.Batch) -> (utils.data.NameValues, utils.imProcess.Imgs):
        # return scores, outputs
        return None, None

    def __call__(self):
        self.experiment.log(mkFile='train_info.md',
                            info=(('data', self.trainImgLoader.datapath),
                                  ('loadScale', self.trainImgLoader.loadScale),
                                  ('trainCrop', self.trainImgLoader.trainCrop),
                                  ('epochs', self.experiment.args.epochs),
                                  ('lr', self.experiment.args.lr),
                                  ('logEvery', self.experiment.args.logEvery),
                                  ('testEvery', self.experiment.args.testEvery)))
        timeFilter = myUtils.Filter()
        avgPeriodLoss = None
        for self.experiment.epoch in list(range(self.experiment.epoch, self.experiment.args.epochs + 1)):
            self.experiment.cometExp.log_current_epoch(self.experiment.epoch)
            if self.experiment.epoch > 0:
                # Train
                print('This is %d-th epoch' % (self.experiment.epoch))
                lrNow = utils.experiment.adjustLearningRate(
                    self.experiment.model.optimizer, self.experiment.epoch, self.experiment.args.lr)

                # iteration
                ticETC = time.time()
                for self.experiment.iteration, batch in enumerate(self.trainImgLoader, 1):
                    batch = utils.data.Batch(batch, cuda=self.experiment.model.cuda, half=self.experiment.model.half)

                    self.experiment.globalStep += 1

                    loss, imgs = self._trainIt(batch)

                    if avgPeriodLoss is None:
                        avgPeriodLoss = loss.clone()
                    else:
                        avgPeriodLoss.accumuate(loss)

                    # print
                    timeLeft = timeFilter((time.time() - ticETC) / 3600 * (
                            (self.experiment.args.epochs - self.experiment.epoch + 1) * len(
                        self.trainImgLoader) - self.experiment.iteration))
                    printMessage = 'globalIt %d/%d, it %d/%d, epoch %d/%d, %sleft %.2fh' % (
                        self.experiment.globalStep,
                        len(self.trainImgLoader) * self.experiment.args.epochs,
                        self.experiment.iteration,
                        len(self.trainImgLoader),
                        self.experiment.epoch,
                        self.experiment.args.epochs,
                        loss.strPrint(''),
                        timeLeft)
                    self.experiment.logger.writer.add_text('train/iterations',
                                                           printMessage,
                                                           global_step=self.experiment.globalStep)
                    print(printMessage)
                    ticETC = time.time()

                    # Log
                    if self.experiment.args.logEvery > 0 \
                            and self.experiment.globalStep % self.experiment.args.logEvery == 0:
                        avgPeriodLoss.avg()
                        avgPeriodLoss['lr'] = lrNow
                        avgPeriodLoss['ETC'] = timeLeft
                        self.experiment.cometExp.log_metrics(avgPeriodLoss, prefix='train',
                                                             step=self.experiment.globalStep)
                        for name, value in avgPeriodLoss.items():
                            self.experiment.logger.writer.add_scalar(
                                'train/' + name, value, self.experiment.globalStep)
                        avgPeriodLoss = None

                        self.experiment.logger.logImages(imgs, 'train/', self.experiment.globalStep,
                                                         self.experiment.args.nSampleLog)

                printMessage = 'epoch %d done' % (self.experiment.epoch)
                print(printMessage)
                self.experiment.logger.writer.add_text(
                    'train/epochs', printMessage, global_step=self.experiment.globalStep)

            # save
            if (self.experiment.args.saveEvery > 0 and self.experiment.epoch % self.experiment.args.saveEvery == 0) \
                    or (self.experiment.args.saveEvery == 0 and self.experiment.epoch == self.experiment.args.epochs) \
                    or (self.experiment.epoch == self.experiment.args.epochs):
                self.experiment.save(epoch=self.experiment.epoch, iteration=self.experiment.iteration)

            # test
            if (
                    ((self.experiment.args.testEvery > 0
                     and self.experiment.epoch > 0
                     and self.experiment.epoch % self.experiment.args.testEvery == 0)
                    or (self.experiment.args.testEvery == 0
                        and (self.experiment.epoch == 0 or self.experiment.epoch == self.experiment.args.epochs))
                    or (self.experiment.args.testEvery < 0
                        and (-self.experiment.epoch) % self.experiment.args.testEvery == 0))
                    and self.test.testImgLoader is not None
            ):
                torch.cuda.empty_cache()
                self.test.__call__()
                torch.cuda.empty_cache()

            self.experiment.cometExp.log_epoch_end(
                epoch_cnt=self.experiment.args.epochs, step=self.experiment.epoch)
        if self.test.testImgLoader is not None:
            self.test.estimateFlops()
