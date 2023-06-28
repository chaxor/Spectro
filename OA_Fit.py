import numpy as np
from scipy import optimize
from globals_and_params import *
from SWCNT_Transition import *

from Spectral_Model import *
from data_retrieval import *


def full_OA_fit(
        E=default_X,
        absorbance=default_abs,
        params=default_params,
        nms=nm_list()):  # , fit_param_list=default_params.T['Value']['fit_param_list']):
    fit_param_list = params.T['Value']['fit_param_list']
    NL_param = params.T['Value'][fit_param_list].values

    NL_param_bounds = list(zip(
        params.T['Min'][fit_param_list].values,
        params.T['Max'][fit_param_list].values))

    x, f, d = optimize.fmin_tnc(
        func=residual, x0=NL_param, args=(
            E, absorbance, nms, params), bounds=NL_param_bounds, approx_grad=True)  # , factr = 10)
    # fmin_l_bfgs_b
    resid, combined_struct, soln_struct, full_model, swcnt_model = NNLS_OA_fit(
        E, absorbance, nms, params)

    return x, f, d, soln_struct, residual, combined_struct, full_model, swcnt_model


def residual(NL_param, *args):
    E = args[0]
    absorbance = args[1]
    nms = args[2]
    params = args[3]

    fit_param_list = params.ix['Value', 'fit_param_list']

    # Unpack to variables becuase it is required by scipy?
    for i, param_name in enumerate(fit_param_list):
        locals()[param_name] = NL_param[i]

    # Set new values for non linear parameters into the param structure to
    # carry to all of the functions
    params.ix['Value', fit_param_list] = NL_param
    # print(NL_param)
    # print(params.ix['Value', fit_param_list])
    resid, combined_struct, soln_struct, full_model, swcnt_model = NNLS_OA_fit(
        E, absorbance, nms, params)

    #residual_array = full_model - absorbance

    return resid  # np.sum(residual_array**2)


def NNLS_swcnt_fit(
        E=default_X,
        absorbance=default_abs,
        nms=nm_list(),
        params=default_params):
    """
    Non-Negative Linear Least Squares Regression to fit an OA spectra *including* background contributions from pi-plasmons,
    graphitic carbon, and M-point transitions
        Input:
            SWCNT_struct := Structure which includes the absorption profiles and concentrations of SWCNTs
                            Includes theory calculated SWCNT absorption profiles normalized to 1 mg/ml
                            Each column corresponds to a different (n,m)

            XY_experiment := A numpy record array (ndarray) containing XY values for an absorption spectrum
                            This absorption spectrum must have the same number of values as the number of rows in the SWCNT_matrix
                            The values within the absorption spectra must be created to match in the excitation energy as well.

        Output:
            scwnt_soln := solution vector containing the concentration of each species in mg / ml
            bkg_soln := solution vector containing the amplitudes for each of the background components
            residual := residual from the sum of squares
    """

    swcnt_struct_ = norm_SWCNT_struct(E, params)

    swcnt_soln, swcnt_resid = optimize.nnls(
        swcnt_struct_.as_matrix(), absorbance)

    swcnt_soln_struct = pd.Series(swcnt_soln, index=swcnt_struct_.columns, name='Concentration (ug/ml)')

    swcnt_model = np.inner(swcnt_struct_.as_matrix(), swcnt_soln)

    return swcnt_struct_, swcnt_resid, swcnt_soln_struct, swcnt_model


def NNLS_bkg_fit(
        E=default_X,
        absorbance=default_abs,
        nms=nm_list(),
        params=default_params):
    """
    Non-Negative Linear Least Squares Regression to fit an OA spectra *including* background contributions from pi-plasmons,
    graphitic carbon, and M-point transitions
        Input:
            SWCNT_struct := Structure which includes the absorption profiles and concentrations of SWCNTs
                            Includes theory calculated SWCNT absorption profiles normalized to 1 mg/ml
                            Each column corresponds to a different (n,m)

            XY_experiment := A numpy record array (ndarray) containing XY values for an absorption spectrum
                            This absorption spectrum must have the same number of values as the number of rows in the SWCNT_matrix
                            The values within the absorption spectra must be created to match in the excitation energy as well.

        Output:
            scwnt_soln := solution vector containing the concentration of each species in mg / ml
            bkg_soln := solution vector containing the amplitudes for each of the background components
            residual := residual from the sum of squares
    """

    bkg_struct_ = bkg_struct(E, params)

    bkg_soln, bkg_resid = optimize.nnls(bkg_struct_.as_matrix(), absorbance)

    bkg_soln_struct = pd.Series(bkg_soln, index=bkg_struct_.columns)

    bkg_model = np.inner(bkg_struct_.as_matrix(), bkg_soln)

    return bkg_struct_, bkg_resid, bkg_soln_struct, bkg_model


def NNLS_OA_fit(
        E=default_X,
        absorbance=default_abs,
        nms=nm_list(),
        params=default_params):
    """
    Non-Negative Linear Least Squares Regression to fit an OA spectra *including* background contributions from pi-plasmons,
    graphitic carbon, and M-point transitions
        Input:
            SWCNT_struct := Structure which includes the absorption profiles and concentrations of SWCNTs
                            Includes theory calculated SWCNT absorption profiles normalized to 1 mg/ml
                            Each column corresponds to a different (n,m)

            XY_experiment := A numpy record array (ndarray) containing XY values for an absorption spectrum
                            This absorption spectrum must have the same number of values as the number of rows in the SWCNT_matrix
                            The values within the absorption spectra must be created to match in the excitation energy as well.

        Output:
            scwnt_soln := solution vector containing the concentration of each species in mg / ml
            bkg_soln := solution vector containing the amplitudes for each of the background components
            residual := residual from the sum of squares
    """

    bkg_struct_, bkg_resid, bkg_soln_struct, bkg_model = NNLS_bkg_fit(
        E, absorbance, nms, params)
    swcnt_struct_, swcnt_resid, swcnt_soln_struct, swcnt_model = NNLS_swcnt_fit(
        E, absorbance - bkg_model, nms, params)

    resid = 4 * bkg_resid + swcnt_resid
    prev_resid = 0
    # while (prev_resid - resid)**2 > 10:
    #     prev_resid = resid
    #     #bkg_model_previous = bkg_model
    #     swcnt_model_previous = swcnt_model
    #     bkg_struct_, bkg_resid, bkg_soln_struct, bkg_model = NNLS_bkg_fit(
    #         E, absorbance - swcnt_model_previous, nms, params)
    #     swcnt_struct_, swcnt_resid, swcnt_soln_struct, swcnt_model = NNLS_swcnt_fit(
    #         E, absorbance - bkg_model, nms, params)

    #     resid = 15 * bkg_resid + swcnt_resid

    combined_struct = swcnt_struct_.join(bkg_struct_, how='right')

    combined_matrix = combined_struct.as_matrix()
    combined_soln = pd.concat([swcnt_soln_struct, bkg_soln_struct])
    full_model = np.inner(combined_matrix, combined_soln)
    swcnt_model = np.inner(swcnt_struct_.as_matrix(), combined_soln[:len(nms)])

    resid = bkg_resid + swcnt_resid

    return resid, combined_struct, swcnt_soln_struct, full_model, swcnt_model
