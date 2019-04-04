import time
from utils import myUtils
from evaluation.Evaluation import Evaluation


class Train:
    def __init__(self, test: Evaluation, trainImgLoader):
        self.experiment = test.experiment
        self.test = test
        self.trainImgLoader = trainImgLoader
        self.experiment.globalStep = (self.experiment.epoch - 1) * len(self.trainImgLoader) if self.experiment.epoch > 0 else 0

    def _trainIt(self, batch: myUtils.Batch) -> (myUtils.Loss, myUtils.Imgs):
        # return scores, outputs
        return None, None

    def __call__(self):
        self.experiment.log(mkFile='train_results.md',
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
                self.lrNow = myUtils.adjustLearningRate(
                    self.experiment.model.optimizer, self.experiment.epoch, self.experiment.args.lr)

                # iteration
                ticETC = time.time()
                for batchIdx, batch in enumerate(self.trainImgLoader, 1):
                    batch = myUtils.Batch(batch, cuda=self.experiment.model.cuda, half=self.experiment.model.half)

                    self.experiment.globalStep += 1

                    loss, imgs = self._trainIt(batch)

                    if avgPeriodLoss is None:
                        avgPeriodLoss = loss
                    else:
                        avgPeriodLoss.accumuate(loss)

                    # Log
                    if self.experiment.args.logEvery > 0 \
                            and self.experiment.globalStep % self.experiment.args.logEvery == 0:
                        avgPeriodLoss.avg()
                        for name, value in avgPeriodLoss.items():
                            self.experiment.logger.writer.add_scalar('trainLosses/' + name, value, self.experiment.globalStep)
                        self.experiment.cometExp.log_metrics(avgPeriodLoss, prefix='trainLosses',
                                                             step=self.experiment.globalStep)
                        avgPeriodLoss = None

                        self.experiment.logger.logImages(imgs, 'trainImages/', self.experiment.globalStep,
                                                         self.experiment.args.nSampleLog)

                    # print
                    timeLeft = timeFilter((time.time() - ticETC) / 3600 * (
                            (self.experiment.args.epochs - self.experiment.epoch + 1) * len(
                        self.trainImgLoader) - batchIdx))
                    printMessage = 'globalIt %d/%d, it %d/%d, epoch %d/%d, %sleft %.2fh' % (
                        self.experiment.globalStep, len(self.trainImgLoader) * self.experiment.args.epochs,
                        batchIdx, len(self.trainImgLoader),
                        self.experiment.epoch, self.experiment.args.epochs,
                        loss.strPrint(''), timeLeft)
                    self.experiment.logger.writer.add_text('trainPrint/iterations', printMessage,
                                                           global_step=self.experiment.globalStep)
                    print(printMessage)
                    ticETC = time.time()

                printMessage = 'epoch %d done' % (self.experiment.epoch)
                print(printMessage)
                self.experiment.logger.writer.add_text(
                    'trainPrint/epochs', printMessage, global_step=self.experiment.globalStep)

            # save
            if (self.experiment.args.saveEvery > 0 and self.experiment.epoch % self.experiment.args.saveEvery == 0) \
                    or (self.experiment.args.saveEvery == 0 and self.experiment.epoch == self.experiment.args.epochs):
                self.experiment.save(epoch=self.experiment.epoch, iteration=self.experiment.iteration)

            # test
            if (
                    (self.experiment.args.testEvery > 0
                     and self.experiment.epoch > 0
                     and self.experiment.epoch % self.experiment.args.testEvery == 0)
                    or (self.experiment.args.testEvery == 0
                        and (self.experiment.epoch == 0 or self.experiment.epoch == self.experiment.args.epochs))
                    or (self.experiment.args.testEvery < 0
                        and (-self.experiment.epoch) % self.experiment.args.testEvery == 0)
                    and self.test.testImgLoader is not None
            ):
                self.test.__call__()

            self.experiment.cometExp.log_epoch_end(
                epoch_cnt=self.experiment.args.epochs, step=self.experiment.globalStep)
