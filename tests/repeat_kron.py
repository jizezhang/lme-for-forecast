def repeat_kron():
    import sys
    sys.path.insert(0, '../code/')
    import numpy as np
    import repeat_utils as rutils

    full_dims = [5,4,3,2]
    dims = [1,4,1,2]
    values = np.random.randn(np.prod(dims))

    Z = rutils.kronecker(dims, full_dims, 0)
    vals = rutils.repeat(values, dims, full_dims)
    y = np.random.randn(np.prod(full_dims))
    vals2 = rutils.repeatTranspose(y, dims, full_dims)

    ok = np.linalg.norm(Z.dot(values) - vals) < 1e-10 \
           and np.linalg.norm(np.transpose(Z).dot(y) - vals2) < 1e-10
    #print('ok = ', ok)
    return ok
