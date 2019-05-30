import torch

def create_model(opt):
    print(opt.model)
    if opt.model == 'wsupervised':
        from .T2model import T2NetModel
        model = T2NetModel()
    elif opt.model == 'supervised':
        from .TaskModel import TNetModel
        model = TNetModel()
    elif opt.model == 'commutative':
        from .CommutativeModel import CommutativeModel
        model = CommutativeModel()
    elif opt.model == 'test':
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'supervised_seg':
        from .SegModel import SegModel
        model = SegModel()
    elif opt.model == 'commutative_seg':
        from .ComSegModel import ComSegModel
        model = ComSegModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model


models = {}
def register_model(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def get_model(name, num_cls=10, **args):
    net = models[name](num_cls=num_cls, **args)
    if torch.cuda.is_available():
        net = net.cuda()
    return net
