import os
import random


def getattrNE(object, name, default=None):
    try:
        return getattr(object, name, default)
    except:
        return None


def checkDir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def struct2dict(struct):
    argsDict = dict((name, getattr(struct, name)) for name in dir(struct)
                    if not name.startswith('_') and not callable(getattr(struct, name)))
    return argsDict


def forNestingList(l, fcn):
    if type(l) in (list, tuple):
        l = [forNestingList(e, fcn) for e in l]
        return l
    else:
        return fcn(l)


def depth(l):
    if type(l) in (tuple, list):
        return 1 + max(depth(item) for item in l)
    else:
        return 0


class Filter:
    def __init__(self, weight=0.1):
        self.weight = weight
        self.old = None

    def __call__(self, x):
        self.old = x if self.old is None else self.old * (1 - self.weight) + x * self.weight
        return self.old


def shuffleLists(lists):
    c = list(zip(*lists))
    random.shuffle(c)
    lists = list(zip(*c))
    return lists


def getNNmoduleFromModel(model):
    model = model.model
    if hasattr(model, 'module'):
        model = model.module
    return model

