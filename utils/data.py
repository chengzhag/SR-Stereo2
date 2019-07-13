import collections
import copy

import torch

from utils.imProcess import Imgs
from utils.myUtils import forNestingList
import numpy as np


class NameValues(collections.OrderedDict):
    def __init__(self, seq=(), prefix='', suffix=''):
        super().__init__()
        for name, value in seq:
            if value is not None:
                super().__setitem__(prefix + name + suffix, value)
        self.nAccum = 1

    def clone(self):
        return copy.deepcopy(self)

    def dataItem(self):
        for key in self.keys():
            self[key] = self[key].data.item()
        return self

    def update(self, nameValues, suffix=''):
        for name in nameValues.keys():
            self[name + suffix] = nameValues[name]

    def add(self, other):
        for key in other.keys():
            if key in self.keys():
                if other[key] == other[key]:
                    self[key] += other[key]
                else:
                    print('Warning: NaN added!')
            else:
                self[key] = other[key]
        return self

    def div(self, other):
        for key in self.keys():
            self[key] /= other
        return self

    def __add__(self, other):
        output = NameValues()
        for key in set(self.keys()) | set(other.keys()):
            output[key] = self.get(key, 0) + other.get(key, 0)
        return output

    def __truediv__(self, other):
        output = NameValues()
        for key in self.keys():
            output[key] = self.get(key, 0) / other
        return output

    def accumuate(self, nameValues):
        self.add(nameValues)
        self.nAccum += 1

    def avg(self):
        for key in self.keys():
            self[key] /= self.nAccum
        self.nAccum = 1

    def strPrint(self, prefix='', suffix=''):
        strReturn = ''
        for name, value in super().items():
            if name.find('outlier') != -1:
                unit = '%'
            else:
                unit = ''
            strReturn += '%s: ' % (prefix + name + suffix)

            def addValue(value):
                s = ''
                if type(value) in (list, tuple):
                    for v in value:
                        s += addValue(v)
                else:
                    s += '%.3f%s, ' % (value, unit)
                return s

            strReturn += addValue(value)

        return strReturn

    def strSuffix(self, prefix='', suffix=''):
        sSuffix = ''
        for name, value in super().items():
            sSuffix += '_%s' % (prefix + name + suffix)

            def addValue(sAppend, values):
                if type(values) == int:
                    return sAppend + '_' + str(values)
                elif type(values) == float:
                    return sAppend + '_%.1f' % values
                elif type(values) == str:
                    return sAppend + '_' + values
                elif type(values) in (list, tuple):
                    for v in values:
                        sAppend = addValue(sAppend, v)
                    return sAppend
                else:
                    raise Exception('Error: Type of values should be in int, float, list, tuple!')

            sSuffix = addValue(sSuffix, value)

        return sSuffix


class Batch:
    def __init__(self, batch, cuda=None, half=None):
        # for different param types
        if type(batch) in (list, tuple):
            self._assertLen(len(batch))
            if type(batch[0]) is torch.Tensor:
                self.batch = batch[:]  # deattach with initial list
            elif type(batch[0]) is np.ndarray:
                self.batch = [torch.from_numpy(x[np.newaxis, :]) for x in batch]
            else:
                raise Exception(f'Error: type {type(batch[0]).__name__} of batch elements has no match!')
        elif type(batch) is Batch:
            self.batch = batch.batch[:]
            if cuda is None:
                cuda = batch.cuda
            if half is None:
                half = batch.half
        elif type(batch) is int:
            self._assertLen(batch)
            if batch % 4 != 0 or batch > 8:
                raise Exception(f'Error: input batch with length {len(batch)} doesnot match required 4n <= 8!')
            self.batch = [None] * batch
        else:
            raise Exception('Error: batch must be class list, tuple, Batch or int!')

        # default params
        if cuda is None:
            cuda = False
        if half is None:
            half = False

        self.half = half
        self.cuda = cuda

        # convert type
        self.batch = forNestingList(self.batch, lambda im: im if im is not None and im.numel() else None)
        if half:
            self.batch = forNestingList(self.batch, lambda im: im.half() if im is not None else None)
        if cuda:
            self.batch = forNestingList(self.batch, lambda im: im.cuda() if im is not None else None)

        # assert nan
        def assertData(t):
            if t is not None and torch.isnan(t).any():
                raise Exception('Error: Data has nan in it')

        forNestingList(self.batch, assertData)

    def _assertLen(self, len):
        assert len % 4 == 0 and len <= 8

    def assertLen(self, length):
        if type(length) in (list, tuple):
            assert len(self) in length
        else:
            assert len(self) == length

    def assertScales(self, nScales, strict=True):
        try:
            if type(nScales) in (list, tuple):
                self.assertLen([nScale * 4 for nScale in nScales])
            else:
                self.assertLen(nScales * 4)
        except:
            if strict:
                raise
            else:
                return False
        return True

    def __len__(self):
        return len(self.batch)

    def __getitem__(self, item):
        return self.batch[item]

    def __setitem__(self, key, value):
        self.batch[key] = value

    def detach(self):
        return Batch(self)

    def scaleBatch(self, scale):
        return Batch(self.batch[scale * 4: (scale + 1) * 4], cuda=self.cuda, half=self.half)

    def lastScaleBatch(self):
        return Batch(self.batch[-4:], cuda=self.cuda, half=self.half)

    def firstScaleBatch(self):
        return self.scaleBatch(0)

    def oneResRGBs(self, set=None):
        self.assertScales(1)
        return self.highestResRGBs(set)

    def oneResDisps(self, set=None):
        self.assertScales(1)
        return self.highestResDisps(set)

    def highResRGBs(self, set=None, strict=True):
        if not self.assertScales(2, strict=strict):
            return None, None
        return self.highestResRGBs(set)

    def highResDisps(self, set=None, strict=True):
        if not self.assertScales(2, strict=strict):
            return None, None
        return self.highestResDisps(set)

    def lowResRGBs(self, set=None, strict=True):
        if not self.assertScales(2, strict=strict):
            return None, None
        return self.lowestResRGBs(set)

    def lowResDisps(self, set=None, strict=True):
        if not self.assertScales(2, strict=strict):
            return None, None
        return self.lowestResDisps(set)

    def highestResRGBs(self, set=None):
        if set is not None:
            self.batch[0:2] = set
        return self.batch[0:2]

    def highestResDisps(self, set=None):
        if set is not None:
            self.batch[2:4] = set
        return self.batch[2:4]

    def lowestResRGBs(self, set=None):
        if set is not None:
            self.batch[-4:-2] = set
        return self.batch[-4:-2]

    def lowestResDisps(self, set=None):
        if set is not None:
            self.batch[-2:] = set
        return self.batch[-2:]

    def allRGBs(self, set=None):
        if set is not None:
            self.batch[0::4] = set[:len(set) // 2]
            self.batch[1::4] = set[len(set) // 2:]
        return self.batch[0::4] + self.batch[1::4]

    def allDisps(self, set=None):
        if set is not None:
            self.batch[2::4] = set[:len(set) // 2]
            self.batch[3::4] = set[len(set) // 2:]
        return self.batch[2::4] + self.batch[3::4]


def packWarpTo(warpTos, outputs: Imgs):
    for warpTo, side in zip(warpTos, ('L', 'R')):
        outputs.addImg(name='warpTo' + side, img=warpTo)