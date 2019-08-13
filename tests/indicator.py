
def indicator():
    import sys
    sys.path.insert(0, '../code/')
    from lme_forecast_general import LME
    import repeat_utils as rutils
    import numpy as np

    dimensions = [5, 4, 3, 2]
    indi = [5, 1, 1, 2]

    y = np.random.randn(np.prod(dimensions))
    model = LME(dimensions, 0, y, [], indi, [], [], False, [])
    Z = rutils.kronecker(indi, dimensions, 0)
    x = np.random.randn(np.prod(indi))

    ok = (np.linalg.norm(model.X(x) - Z.dot(x)) < 1e-10) and \
         (np.linalg.norm(model.XT(y) - np.transpose(Z).dot(y)) < 1e-10)
    return ok
