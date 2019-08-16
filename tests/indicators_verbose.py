
def indicators_verbose():
    import sys
    sys.path.insert(0, '../code/')
    from lme_forecast_verbose import LME
    import repeat_utils as rutils
    import numpy as np

    dimensions = [5, 4, 3, 2]
    indi = [[5, 1, 1, 2], [1,4,1,1]]
    indi_bool = {'indi0':[True, False, False, True],
                 'indi1':[False, True, False, False]}

    y = np.random.randn(np.prod(dimensions))
    model = LME(dimensions, 0, y, {}, indi_bool, [], False, {})
    Z1 = rutils.kronecker(indi[0], dimensions, 0)
    Z2 = rutils.kronecker(indi[1], dimensions, 0)
    Z = np.hstack((Z1, Z2))
    x = np.random.randn(np.prod(indi[0]) + np.prod(indi[1]))

    ok = (np.linalg.norm(model.X(x) - Z.dot(x)) < 1e-10) and \
         (np.linalg.norm(model.XT(y) - np.transpose(Z).dot(y)) < 1e-10)
    return ok
