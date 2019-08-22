import numpy as np
import xarray

def saveDraws(beta_samples, u_samples, global_effects_names,
              random_effects_dims, random_effects_names, coords_dict={}):
    all_coords = {}
    assert beta_samples.shape[0] == len(global_effects_names)
    beta_xr = xarray.DataArray(beta_samples, dims=('cov','draw'),
                               coords={'cov': global_effects_names,
                                       'draw': np.arange(1, beta_samples.shape[1]+1)})
    all_coords.update({'cov': global_effects_names,
            'draw': np.arange(1, beta_samples.shape[1]+1)})
    assert len(u_samples) == len(random_effects_dims)
    u_xrs = []
    for i in range(len(u_samples)):
        dim_names = random_effects_dims[i] + ['draw']
        dim_sizes = u_samples[i].shape
        assert len(dim_names) == len(dim_sizes)
        coords = {dim_names[j]:np.arange(1, dim_sizes[j]+1) for j in range(len(dim_names))}
        coords.update(coords_dict)
        all_coords.update(coords)
        u_xrs.append(xarray.DataArray(u_samples[i], dims=tuple(dim_names), coords=coords))
    data_vars = {random_effects_names[i]:u_xrs[i] for i in range(len(random_effects_names))}
    data_vars['beta_global'] = beta_xr
    #all_coords.update(coords_dict)

    return xarray.Dataset(data_vars, all_coords)
