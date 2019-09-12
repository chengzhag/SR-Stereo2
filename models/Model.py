import utils.experiment
import torch
import utils.data
import utils.imProcess
from utils import myUtils
from apex import amp
from thop import profile


# manage loss, training, predicting, testing, loading, saving of general models
class Model:
    def __init__(self, cuda=True, half=False):
        self.cuda = cuda
        self.half = half

        self.model = None
        self.optimizer = None
        self.lossWeights = None

    def initModel(self):
        pass

    def getParamNum(self, show=True):
        total_num = sum(p.numel() for p in self.model.parameters())
        trainable_num = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        if show:
            print(f'model: {self.__class__.__name__}, Total params: {total_num}, Trainable params: {trainable_num}')
        return total_num, trainable_num

    def getFlops(self, inputs, show=True):
        flops, params = profile(myUtils.getNNmoduleFromModel(self), inputs=inputs)
        if show:
            print(f'model: {self.__class__.__name__}, flops: {flops}, params: {params}')
        return flops, params

    def packOutputs(self, outputs, imgs: utils.imProcess.Imgs = None) -> utils.imProcess.Imgs:
        if imgs is None:
            imgs = utils.imProcess.Imgs()
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
        with amp.scale_loss(loss['loss'], self.optimizer) as scaledLoss:
            scaledLoss.backward()
        self.optimizer.step()
        output = self.packOutputs(rawOutput)
        return loss.dataItem(), output

    def trainBothSides(self, inputs, gts):
        pass

    def train(self, batch: utils.data.Batch):
        pass

    def predict(self, batch: utils.data.Batch):
        pass

    def load(self, chkpointDir: str, strict=False) -> (int, int):
        if chkpointDir in (None, 'None'):
            return None, None
        if type(chkpointDir) is list:
            if len(chkpointDir) > 1:
                raise Exception('Error: One model can only load one checkpoint!')
            else:
                chkpointDir = chkpointDir[0]

        loadStateDict = torch.load(chkpointDir)
        # for EDSR and PSMNet compatibility
        writeModelDict = loadStateDict.get('state_dict', loadStateDict)
        writeModelDict = loadStateDict.get('model', writeModelDict)
        startsWithModule = all([key.startswith('module') for key in writeModelDict.keys()])
        hasModule = hasattr(self.model, 'module')
        if startsWithModule:
            writeModelDict = {key[len('module.'):]:value for key, value in writeModelDict.items()}
        if hasModule:
            self.model.module.load_state_dict(writeModelDict, strict=strict)
        else:
            self.model.load_state_dict(writeModelDict, strict=strict)

        if 'optimizer' in loadStateDict.keys() and self.optimizer is not None:
            try:
                self.optimizer.load_state_dict(loadStateDict['optimizer'])
            except ValueError as error:
                print('Optimizer not loaded: ' + error.args[0])

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
