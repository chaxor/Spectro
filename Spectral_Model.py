import pandas as pd
import scipy.optimize as optimize
from SWCNT_Transition import *
from Spectroscopic_Functions import *
from globals_and_params import *

def nm_list(
    lowestNM=default_lowestNM,
    highestNM=default_highestNM,
     exclude_list=[]):
    """
    List of Possible SWCNTs within a Range of nm Values
        Format: (n,m) (not string)
        Units: --
        Reference --
    """

    nm_list = list()

    for n in range(0, highestNM + 1):
        for m in range(0, highestNM + 1):
            # Get rid of some of the combinations
            if(n < lowestNM and m < lowestNM): break
            elif(n < m): break
            elif((n, m) in exclude_list): break

            # Combinations of interest to create structure
            else:
                nm_list.append((n, m))

    return nm_list

def spectral_index(E):
    spectral_indices = pd.MultiIndex.from_tuples(
        list(zip(E, WAVELENGTH_TO_ENERGY_CONVERSION / E, E / (WAVELENGTH_TO_ENERGY_CONVERSION * 10**(-7)))),
        names=['Energy (eV)', 'Wavelength (nm)', 'Wavenumber (cm^-1)'])

    return spectral_indices

def convert_dict_multicolumn(dict_for_df, is_swcnt=True):
    df=pd.DataFrame(dict_for_df)
    identifiers=['Species Type', 'Electronic Type', 'Species']
    if is_swcnt:
        tuple_of_identifiers=[
    ("SWCNT", SWCNT(
        n, m).electronic_type_str, SWCNT(
            n, m).strNM) for n, m in df.columns]
    else:
        tuple_of_identifiers=[
    ('Background',
    'Background',
     name) for name in df.columns]

    col_index=pd.MultiIndex.from_tuples(
    tuple_of_identifiers,
     names=identifiers)

    new_df=pd.DataFrame(df.as_matrix(), index=df.index, columns=col_index)
    return new_df


class Absorption_Spectra:
    """
    Structure for each Normalized SWCNT Absorbance (Spectra for 1 ug/ml solution of (n, m) SWCNT)

    struct:

        Pandas Data Frame Type of Structure
        Key: ( "(n, m)", "M"/"S1"/"S2" )
        Value: Pandas Series of Normalized Spectrum

        To get just the metallic species data frame:
            SWCNT_struct.T.xs('M', level='Electronic Type').T

        To get one of the 'X' arrays ['Energy (eV)', 'Wavelength (nm)']:
            SWCNT_struct.index.get_level_values('Energy (eV)')

        To get a list of the electronic types, or the second level of the columns:
            SWCNT_struct.columns.get_level_values('Electronic Type')

    norm_abs:
            
        Absorbance of 1 ug/ml solution of SWCNT
        Psuedo Mass Absorption Coefficient
        ! 1 cm long attenuation length is assumed !
        The output spectra from this funciton for each SWCNT should match the experimental spectra obtained from a 1 ug/ml solution of that SWCNT type in a 1 cm long cuvette.
        (Cross Section is Energy Dependent) --> (Mass Absorption Coefficient is Energy Dependent) --> Output is an Array (numpy ndarray)
            Units: cm^3 / ug (inverse density)
            Reference 1

    """

    def __init__(self, E=default_X, absorbance=default_abs, params=default_params, nms=nm_list()):
        self.E = E
        self.absorbance = absorbance
        self.params = params
        self.nms = nms

        # Normalized absorbance structures
        self.norm_swcnt_abs_dict = {(n, m): self.norm_abs(SWCNT(n, m))  for n, m in nms}
        self.norm_swcnt_abs_struct = convert_dict_multicolumn(self.norm_swcnt_abs_dict, is_swcnt=True)
        self.norm_bkg_abs_struct = self.bkg_struct()

        self.norm_total_abs_struct = pd.concat([self.norm_swcnt_abs_struct, self.norm_bkg_abs_struct], axis=1)

        # Solutions and fit residuals initialization
        self.total_soln, self.total_resid = optimize.nnls(self.norm_total_abs_struct.as_matrix(), self.absorbance)

        # initialize
        self.bkg_soln = np.ones(len(self.norm_bkg_abs_struct.columns))
        self.bkg_resid = 0
        self.swcnt_soln = np.ones(len(self.norm_swcnt_abs_struct.columns))
        self.swcnt_resid = 0
        self.resid = self.bkg_resid + self.swcnt_resid
    
    @property
    def swcnt_abs_struct(self):
        return self.norm_swcnt_abs_struct.as_matrix()*self.swcnt_soln

    #    self.swcnt_abs_struct = self.norm_swcnt_abs_struct.as_matrix()*self.swcnt_soln#property(lambda self: np.inner(self.norm_swcnt_abs_struct.as_matrix(), self.swcnt_soln))
     #   print self.swcnt_abs_struct.shape
    @property
    def bkg_abs_struct(self):
        return self.norm_bkg_abs_struct.as_matrix()*self.bkg_soln
    #self.bkg_abs_struct = self.norm_bkg_abs_struct.as_matrix()*self.bkg_soln#property(lambda self: np.inner(self.norm_bkg_abs_struct.as_matrix(), self.bkg_soln))

    @property
    def total_abs_struct(self):
        return self.norm_total_abs_struct.as_matrix()*self.total_soln
    #self.total_abs_struct = self.norm_total_abs_struct.as_matrix()*self.total_soln#property(lambda self: np.inner(self.norm_total_abs_struct.as_matrix(), self.total_soln))

    @property 
    def bkg_soln_struct(self):
        return pd.DataFrame(self.bkg_soln, index= self.norm_bkg_abs_struct.columns)
    #self.bkg_soln_struct = pd.DataFrame(self.bkg_soln, index= self.norm_bkg_abs_struct.columns)

    @property 
    def swcnt_soln_struct(self):
        return pd.DataFrame(self.swcnt_soln, index=self.norm_swcnt_abs_struct.columns)
    #self.swcnt_soln_struct = pd.DataFrame(self.swcnt_soln, index=self.norm_swcnt_abs_struct.columns)

    @property
    def swcnt_model(self):
        return self.swcnt_abs_struct.sum(axis=1)
    #self.swcnt_model = self.swcnt_abs_struct.sum(axis=1)
    
    @property
    def background_model(self):
        return self.bkg_abs_struct.sum(axis=1)
    #self.background_model = self.bkg_abs_struct.sum(axis=1)

    @property
    def total_model(self):
        return self.swcnt_model + self.background_model

    def norm_abs(self, swcnt):
        # Absorption Cross Section = Mass Absorption Coefficient * atomic molar mass / Avagadros Number
        #-> sigma = J * MW_C / NA
        #-> J = NA / MW_C * sigma
        # We have cross section from paper, so we solve for J
        # Mass Absorption Coefficient = Attenuation Coefficient / Density
        # Divded by 1,000,000 to convert from cm^3 / g to cm^3 / ug
        swcnt_line = self.abs_transition_struct(swcnt).sum(axis=1)
        return NA / MW_C * swcnt_line / 1000000

    def abs_transition_struct(self, swcnt):
        E = self.E
        d = dict()
        for trans in swcnt.transitions:
            exciton = Abs_Lorentz(
                                    x=E,
                                    amp=trans.sigma_p,
                                    center=trans.Eii,
                                    width=trans.wp)
            phonon = Abs_Lorentz(
                                    x=E,
                                    amp=(
                                        trans.sigma_p /
                                        trans.a),
                                    center=(
                                        trans.Eii +
                                        trans.delta),
                                    width=(
                                        trans.b *
                                         trans.wp))
            d[trans.trans_string()] = pd.Series(
                exciton + phonon, index=spectral_index(E))
        df = pd.DataFrame(d)
        return df

    def bkg_struct(self):
        E = self.E
        pvls = self.params.T['Value']

        # These are normalized
        aCB = graphite(E, pvls['alpha_aCB'], pvls['aCBy0'], pvls['b_aCB'])
        N134 = graphite(E, pvls['alpha_N134'], pvls['N134y0'], pvls['b_N134'])

        # Normalized to amplitude of 1
        GL = Voigt(E, 1., pvls['GLcenter'], pvls['GLFWHM'], pvls['GLshape'])
        PP = Voigt(E, 1., pvls['PPcenter'], pvls['PPFWHM'], pvls['PPshape'])

        # add constant background
        y0 = np.ones(len(E))

        bkg_dict = dict()
        bkg_dict['aCB'] = pd.Series(aCB, index=spectral_index(E))
        bkg_dict['N134'] = pd.Series(N134, index=spectral_index(E))
        bkg_dict['GL'] = pd.Series(GL, index=spectral_index(E))
        bkg_dict['PP'] = pd.Series(PP, index=spectral_index(E))
        bkg_dict['y0'] = pd.Series(y0, index=spectral_index(E))

        bkg_structure = convert_dict_multicolumn(bkg_dict, is_swcnt=False)

        return bkg_structure

    def full_fit():
        return

    def iter_nnls_fit(self, tolerance=1.0, background_wieght=10**10):
        # shouldn't need this if property() works like it's supposed to in the initializer?
        #self.swcnt_model = self.swcnt_abs_struct.as_matrix() self.absorbance)
        high_num = 100000000000000000000
        previous_resid = high_num
        while (self.resid - previous_resid)**2 > tolerance:
            previous_resid = self.resid

            if previous_resid == high_num:
                new_Y = self.swcnt_model*0
            # remove the simulated CNT absorption from the absorbance before fitting to estimate background 
            new_Y = self.absorbance - self.swcnt_model
            self.bkg_soln, self.bkg_resid = optimize.nnls(self.norm_bkg_abs_struct.as_matrix(), new_Y)
            # print(self.bkg_soln)
            # shouldn't need this if property() works like it's supposed to in the initializer?
            #self.background_model = np.inner(self.norm_bkg_struct.as_matrix(), self.bkg_soln)

            # remove the calculated background model before fitting
            new_Y = (self.absorbance - self.background_model)
            self.swcnt_soln, self.swcnt_resid = optimize.nnls(self.norm_swcnt_abs_struct.as_matrix(), new_Y)
            # shouldn't need this if property() works like it's supposed to in the initializer?
            #self.swcnt_model = np.inner(self.swcnt_abs_struct.as_matrix(), self.absorbance)

            self.resid = background_wieght * self.bkg_resid + self.swcnt_resid

        # shouldn't need this if property() works like it's supposed to in the initializer?
        #self.bkg_soln_struct = pd.DataFrame(bkg_soln, index=bkg_abs_struct.columns)
        #self.swcnt_soln_struct = pd.DataFrame(swcnt_soln, index=swcnt_abs_struct.columns)
            


class Photoluminescence_Spectra:
    def __init__(E=default_X, params=default_params, nms=nm_list()):
        self.E = E
        self.params = params
        self.nms = nms


    def QY_transition_struct(swcnt, E=default_X, params=default_params):

        d = dict()
        # Exciton and Phonon Sideband for each transition:
        for trans in swcnt.transitions:
            exciton = Lorentzian(
        x=E,
        amp=trans.QY,
        center=trans.Eii,
         width=trans.wp)
            d[trans.trans_string()] = pd.Series(exciton, index=spectral_index(E))

        return pd.DataFrame(d)


    def norm_QY(swcnt, E=default_X, params=default_params):

        # A model line for each nanotube is the sum of each of the transitions
        # that can occur for the nanotube
        QY_line = QY_transition_struct(swcnt, E, params).sum(axis=1)

        return QY_line


    def norm_PL(swcnt, E=default_X, params=default_params):
        absorb = norm_abs(swcnt, E, params)
        QY = norm_QY(swcnt, E, params)
        matrix = np.outer(absorb, QY)

        return pd.DataFrame(matrix, index=absorb.index, columns=QY.index)

#
# 1) Liu, Kaihui, Xiaoping Hong, Sangkook Choi, Chenhao Jin, Rodrigo B. Capaz, Jihoon Kim, Shaul Aloni et al.
#    "Systematic Determination of Absolute Absorption Cross-section of Individual Carbon Nanotubes."
#    arXiv preprint arXiv:1311.3328 (2013).
#
# 2) Liu, Kaihui, Jack Deslippe, Fajun Xiao, Rodrigo B. Capaz, Xiaoping Hong, Shaul Aloni, Alex Zettl et al.
#    "An atlas of carbon nanotbe optical transitions."
#    Nature nanotechnology 7, no. 5 (2012): 325-329.
#
# 3) Dresselhaus, Mildred S., G. Dresselhaus, R. Saito, and A. Jorio.
#    "Raman spectroscopy of carbon nanotubes."
#    Physics Reports 409, no. 2 (2005): 47-99.
#
# 4) Larsen, Brian A., Pravas Deria, Josh M. Holt, Ian N. Stanton, Michael J. Heben, Michael J. Therien, and Jeffrey L. Blackburn.
#    "Effect of solvent polarity and electrophilicity on quantum yields and solvatochromic shifts of single-walled carbon nanotube photoluminescence."
#    Journal of the American Chemical Society 134, no. 30 (2012): 12485-12491.
#
# 5) Silvera-Batista, Carlos A., Randy K. Wang, Philip Weinberg, and Kirk J. Ziegler.
#    "Solvatochromic shifts of single-walled carbon nanotubes in nonpolar microenvironments."
#    Physical Chemistry Chemical Physics 12, no. 26 (2010): 6990-6998.
#
# 6) E Haroz Thesis RIce University
#
# 6) Perspectives on Carbon Nanotubes and Graphene Raman Spectroscopy
