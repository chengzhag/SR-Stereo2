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

    def loss(self, outputs, gts):
        pass

    def train(self, batch: myUtils.Batch):
        self.model.train()

    def predict(self, batch: myUtils.Batch):
        self.model.eval()

    def load(self, chkpointDir: str) -> (int, int):
        chkpointDir = myUtils.scanCheckpoint(chkpointDir)

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
