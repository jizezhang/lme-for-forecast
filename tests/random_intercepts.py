def random_intercepts():
    import sys
    sys.path.insert(0, '../lme/')
    from lme_forecast_general import LME
    import rutils
    import numpy as np

    dimensions = [5, 4, 3, 2]
    ran_intercepts = [[4,1,1],[1,3,1]]

    y = np.random.randn(np.prod(dimensions))
    model = LME(dimensions, 1, y, [], [], [], True,
               [(None, [True, True, False, False]), (None, [True, False, True, False])])
    Z = []
    Z.append(rutils.kronecker(ran_intercepts[0], dimensions, 1))
    Z.append(rutils.kronecker(ran_intercepts[1], dimensions, 1))
    Z1 = np.tile(np.hstack(Z),(dimensions[0],1))
    model.buildZ()
    ok = np.linalg.norm(Z1-model.Z) == 0.0
    return ok
