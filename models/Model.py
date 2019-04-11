import torch
from utils import myUtils
from apex import amp


# manage loss, training, predicting, testing, loading, saving of general models
class Model:
    def __init__(self, cuda=True, half=False):
        self.cuda = cuda
        self.half = half
        self.ampHandle = amp.init(half)

        self.model = None
        self.optimizer = None
        self.lossWeights = None

    def initModel(self):
        pass

    def packOutputs(self, outputs, imgs: myUtils.Imgs = None) -> myUtils.Imgs:
        if imgs is None:
            imgs = myUtils.Imgs()
        return imgs

    def setLossWeights(self, lossWeights):
        self.lossWeights = lossWeights

    def loss(self, output, gt):
        pass

    def trainOneSide(self, input, gt):
        self.model.train()
        self.optimizer.zero_grad()
        rawOutput = self.model.forward(*input)
        loss = self.loss(output=rawOutput, gt=gt)
        with self.ampHandle.scale_loss(loss['loss'], self.optimizer) as scaledLoss:
            scaledLoss.backward()
        self.optimizer.step()
        output = self.packOutputs(rawOutput)
        return loss.dataItem(), output

    def train(self, batch: myUtils.Batch):
        pass

    def predict(self, batch: myUtils.Batch):
        pass

    def load(self, chkpointDir: str) -> (int, int):
        if chkpointDir is None:
            return None, None
        if type(chkpointDir) is list:
            if len(chkpointDir) > 1:
                raise Exception('Error: One model can only load one checkpoint!')
            else:
                chkpointDir = chkpointDir[0]

        loadStateDict = torch.load(chkpointDir)
        # for EDSR and PSMNet compatibility
        writeModelDict = loadStateDict.get('state_dict', loadStateDict)
        try:
            self.model.load_state_dict(writeModelDict)
        except RuntimeError:
            self.model.module.load_state_dict(writeModelDict)

        if 'optimizer' in loadStateDict.keys() and self.optimizer is not None:
            self.optimizer.load_state_dict(loadStateDict['optimizer'])

        epoch = loadStateDict.get('epoch')
        iteration = loadStateDict.get('iteration')
        print(f'Checkpoint epoch {epoch}, iteration {iteration}')
        return epoch, iteration

    def nParams(self):
        return sum([p.data.nelement() for p in self.model.parameters()])

    def save(self, chkpointDir: str, info=None):
        saveDict = {
            'state_dict': self.model.state_dict()
        }
        if info is not None:
            saveDict.update(info)
        if self.optimizer is not None:
            saveDict['optimizer'] = self.optimizer.state_dict()
        torch.save(saveDict, chkpointDir)
