
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
        from .FCN8sModel import FCN8sModel
        model = FCN8sModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created." % (model.name()))
    return model