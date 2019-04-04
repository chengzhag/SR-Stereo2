import time
from utils import myUtils


class Train(myUtils.Experiment):
    def __init__(self, model, stage, args, trainImgLoader):
        super().__init__(model=model, stage=stage, args=args)
        self.trainImgLoader = trainImgLoader
        self.globalStep = (self.epoch - 1) * len(self.trainImgLoader) if self.epoch > 0 else 0

    def _trainIt(self, batch: myUtils.Batch) -> (myUtils.Loss, myUtils.Imgs):
        # return scores, outputs
        return None, None

    def __call__(self):
        self.log(mkFile='train_results.md',
                 info=(('data', self.trainImgLoader.datapath),
                       ('loadScale', self.trainImgLoader.loadScale),
                       ('trainCrop', self.trainImgLoader.trainCrop),
                       ('epochs', self.args.epochs),
                       ('lr', self.args.lr),
                       ('logEvery', self.args.logEvery),
                       ('testEvery', self.args.testEvery)))
        timeFilter = myUtils.Filter()
        avgPeriodLoss = None
        for self.epoch in list(range(self.epoch, self.args.epochs + 1)):
            if self.epoch > 0:
                # Train
                print('This is %d-th epoch' % (self.epoch))
                self.lrNow = myUtils.adjustLearningRate(self.model.optimizer, self.epoch, self.args.lr)

                # iteration
                ticETC = time.time()
                for batchIdx, batch in enumerate(self.trainImgLoader, 1):
                    batch = myUtils.Batch(batch, cuda=self.model.cuda, half=self.model.half)

                    self.globalStep += 1

                    loss, imgs = self._trainIt(batch)

                    if avgPeriodLoss is None:
                        avgPeriodLoss = loss
                    else:
                        avgPeriodLoss.accumuate(loss)

                    if self.args.logEvery > 0 and self.globalStep % self.args.logEvery == 0:
                        avgPeriodLoss.avg()
                        for name, value in avgPeriodLoss.items():
                            self.logger.writer.add_scalar('trainLosses/' + name, value, self.globalStep)
                        avgPeriodLoss = None

                        self.logger.logImages(imgs, 'trainImages/', self.globalStep, self.args.nSampleLog)

                    timeLeft = timeFilter((time.time() - ticETC) / 3600 * (
                            (self.args.epochs - self.epoch + 1) * len(self.trainImgLoader) - batchIdx))
                    printMessage = 'globalIt %d/%d, it %d/%d, epoch %d/%d, %sleft %.2fh' % (
                        self.globalStep, len(self.trainImgLoader) * self.args.epochs,
                        batchIdx, len(self.trainImgLoader),
                        self.epoch, self.args.epochs,
                        loss.strPrint(''), timeLeft)
                    self.logger.writer.add_text('trainPrint/iterations', printMessage,
                                                global_step=self.globalStep)
                    print(printMessage)
                    ticETC = time.time()

                printMessage = 'epoch %d done' % (self.epoch)
                print(printMessage)
                self.logger.writer.add_text('trainPrint/epochs', printMessage, global_step=self.globalStep)

            # save
            if (self.args.saveEvery > 0 and self.epoch % self.args.saveEvery == 0) \
                    or (self.args.saveEvery == 0 and self.epoch == self.args.epochs):
                self.save(epoch=self.epoch, iteration=self.iteration)

            # test
            if ((self.args.testEvery > 0 and self.epoch > 0 and self.epoch % self.args.testEvery == 0)
                    or (self.args.testEvery == 0 and (self.epoch == 0 or self.epoch == self.args.epochs))
                    or (self.args.testEvery < 0 and (-self.epoch) % self.args.testEvery == 0)):
                pass
