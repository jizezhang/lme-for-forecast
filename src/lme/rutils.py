import numpy as np


def repeat(values, dims, full_dims):
    """
    to compute function value
    """
    assert len(values) == np.prod(dims)
    cumprod = 1
    for i in range(len(full_dims)-1,-1,-1):
        if dims[i] == 1:
            values = np.tile(values.reshape((-1,cumprod)),(1,full_dims[i])).reshape(-1)
        cumprod *= full_dims[i]
    assert values.shape[0] == np.prod(full_dims)
    return values


def repeatTranspose(y, dims, full_dims):
    """
    to compute jacobian
    """
    values = [y]
    for i in range(len(full_dims)):
        if dims[i] == 1:
            values = [np.sum(x.reshape((full_dims[i],-1)),axis=0) for x in values]
        else:
            temp = []
            ext = temp.extend
            for x in values:
                ext(np.split(x, full_dims[i]))
            values = temp
    if len(values) == 1:
        assert np.prod(dims) == 1
        return np.array(list(values))
    assert len(np.squeeze(values)) == np.prod(dims)
    return np.squeeze(values)


def kronecker(dims, full_dims, start=0):
    """
    build Z matrix using kronecker product
    """
    # def recurse(i):
    #     if i == self.nd:
    #         return [1]
    #     if dims[i] == 1:
    #         return np.tile(recurse(i+1),(self.dimensions[i],1))
    #     else:
    #         return np.kron(np.identity(self.dimensions[i]), recurse(i+1))
    # Z = recurse(0)
    Z = [1]
    for i in range(len(dims)-1,-1,-1):
        if dims[i] == 1:
            Z = np.tile(Z,(full_dims[i+start],1))
        else:
            Z = np.kron(np.identity(full_dims[i+start]),Z)
    assert Z.shape[0] == np.prod(full_dims[start:])
    assert Z.shape[1] == np.prod(dims)

    return Z
