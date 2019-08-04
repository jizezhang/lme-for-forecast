import numpy as np


def readDataset(ds,n_year=28, n_sex=2, n_age=23, n_loc=195, check=False):
    ds = ds.sortby(['location_id','age_group_id','sex_id','year_id'])
    values_check = []
    if check:
        df = ds.to_dataframe().sort_values(['location_id','age_group_id','sex_id','year_id'])
        values_check = df['value'].values
    values = ds.transpose('location_id','age_group_id','sex_id','year_id').to_array().values.squeeze().reshape(-1)
    if check:
        assert np.linalg.norm(values - values_check) == 0.0
    if len(ds.year_id.values) == 1:
        print('number of years == 1')
        values = np.repeat(values,n_year)
    if len(ds.sex_id.values) == 1:
        print('number of sexes == 1')
        values = np.tile(values.reshape((-1,n_year)),(1,n_sex)).reshape(-1)
    if len(ds.age_group_id.values) == 1:
        print('number of age groups == 1')
        values = np.tile(values.reshape((-1,n_year*n_sex)),(1,n_age)).reshape(-1)
    if len(ds.location_id.values) == 1:
        print('number of locations == 1')
        values = np.tile(values,n_loc)
    assert values.shape[0] == n_year*n_sex*n_age*n_loc
    return values
