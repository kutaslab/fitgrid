import pandas as pd
import patsy
from tqdm import tqdm
from statsmodels.stats.outliers_influence import (
    variance_inflation_factor as vif,
)


def get_vifs(epochs, RHS):
    def get_single_vif(group, RHS):
        dmatrix = patsy.dmatrix(formula_like=RHS, data=group)
        vifs = {
            name: vif(dmatrix, index)
            for name, index in dmatrix.design_info.column_name_indexes.items()
        }
        return pd.Series(vifs)

    tqdm.pandas(desc="Time")

    return epochs._snapshots.progress_apply(get_single_vif, RHS=RHS)




def _ols_to_rerps(fg_ols, ci_alpha=.05):
    """scrape model and parameter values out of OLS fitgrids for plotting

    Follows the same format as the lmer fit scraper

    Parameters
    ----------
    fg_ols : fitgrid.LMFitGrid

    ci_alpha : float {.05}
       alpha for confidence interval


    Returns
    -------
    rerps_df : pd.DataFrame
       index.names = [`Time`, `model`, `param`, `key`]
       columns are the `fg_ols` columns


    Notes
    -----
    The `rerps_df` dataframe is row and column indexed the same
    as for lmer ERPs

    """

    # TO DO _check_grid guard grid to ensure fits, modles, data
    # etc. are same everywhere

    index_names = ['Time', 'model', 'param', 'key']
    key_labels = [
        '2.5_ci',
        '97.5_ci',
        'AIC',
        'DF',
        'Estimate',
        'P-val',
        'SE',
        # 'Sig',
        'T-stat',
        # 'has_warning',
    ]

    # fetch the model info, progpagate for parameters and index
    rhs = fg_ols[
        0,
        fg_ols._grid.columns[0]
    ].model.formula.iat[0, 0].split('~')[1]

    model_vals = []
    model_key_attrs = [("DF", "df_resid"), ("AIC", "aic")]
    for (key, attr) in model_key_attrs:
        vals = None
        vals = getattr(fg_ols, attr).copy()
        if vals is None:
            raise AttributeError(f"model: {rhs} attribute: {attr}")
        vals['key'] = key
        model_vals.append(vals)
    model_vals = pd.concat(model_vals)
    model_vals['model'] = rhs

    param_names = fg_ols.params.index.get_level_values(-1).unique()
    pmvs = []
    for p in param_names:
        pmv = model_vals.copy()
        pmv['param'] = p
        pmvs.append(pmv)
    pmvs = pd.concat(pmvs).reset_index().set_index(index_names)

    # lookup the param_name specifc info for this bundle
    rerps = []

    # select model point estimates
    sv_attrs = [
        ('Estimate', 'params'),  # coefficient value
        ('SE', 'bse'),
        ('P-val', 'pvalues'),
        ('T-stat', 'tvalues'),
    ]

    for idx, (key, attr) in enumerate(sv_attrs):
        attr_vals = getattr(fg_ols, attr).copy()  # ! don't mod the _grid
        if attr_vals is None:
            raise AttributeError(f"not found: {attr}")

        attr_vals.index = attr_vals.index.rename(['Time', 'param'])
        attr_vals['model'] = rhs
        attr_vals['key'] = key

        # update list of param bundles
        rerps.append(attr_vals.reset_index().set_index(index_names))

    # special handling for confidence interval
    ci_bounds = [
        f"{bound:.1f}_ci"
        for bound in [100 * (1 + (b * (1 - ci_alpha))) / 2.0 for b in [-1, 1]]
    ]
    cis = fg_ols.conf_int(alpha=ci_alpha)
    cis.index = cis.index.rename(['Time', 'param', 'key'])
    cis.index = cis.index.set_levels(ci_bounds, 'key')
    cis['model'] = rhs
    rerps.append(cis.reset_index().set_index(index_names))

    rerps_df = pd.concat(rerps)

    # add the parmeter model info
    rerps_df = pd.concat([rerps_df, pmvs]).sort_index()

    assert rerps_df.index.names == index_names
    assert set(key_labels) == set(rerps_df.index.levels[-1])
    return rerps_df
