
from globals_and_params import *
from Spectroscopic_Functions import *
import numpy as np
import scipy.signal

##########################################################################
##########################################################################

# SWCNT structure


class SWCNT:

    """
        A single walled carbon nanotube object

    Attributes:

        ch
            Chiral Vector Length (Circumference of SWCNT)
                Units: Angstroms
                Reference 3

        dt
            SWCNT Diameter
                Units: nm
                Reference 3

        theta
            Chiral Angle
                Units: Radians
                Reference 3


        electronic_type
            Electronic type test:
                0 - Metallic
                1 - Semiconducting type 1 (S1)
                2 - Semiconducting type 2 (S2)
                Reference 3

        mod_type
            Alternative Nomenclature to Electronic Type:
                0 - Metallic
                1 - mod type 1 <=> semiconducting type 2 (MOD1 <=> S2)
                2 - mod type 2 <=> semiconducting type 1 (MOD2 <=> S1)
                Reference 3

        strNM
            Returns the Dresselhaus Nomenclature "(n,m)" for each SWCNT
                Reference 3
    """

    def __init__(self, n, m, params=default_params):
        self.n = float(n)
        self.m = float(m)

        # Chiral Vector
        self.ch = np.sqrt(
            3) * CCdist * np.sqrt(self.n**2 + self.m**2 + self.n * self.m)

        # Tube Diameter
        # /10 --> nanometers from angstroms
        self.dt = self.ch / np.pi / 10

        # Chiral Angle
        self.theta = np.arctan(np.sqrt(3) * self.m / (self.m + 2 * self.n))

        self.electronic_type = np.mod(2 * self.n + self.m, 3)

        self.mod_type = np.mod(self.n - self.m, 3)

        # Semiconductors have 7 transitions, metals have 3
        self.transition_list = [
            list(range(
                1, 8)), list(range(
                1, 4))][
            self.electronic_type == 0]

        # If metallic and not armchair (chiral metallic) k list is [0,1], else
        # 0
        self.k_list = [
            [0], [
                0, 1]][
            self.electronic_type == 0 and self.n != self.m]

        self.transitions = [
            Transition(
                self,
                i,
                k,
                params) for i in self.transition_list for k in self.k_list]

        self.strNM = "(" + str(int(self.n)) + "," + str(int(self.m)) + ")"

        self.electronic_type_str = [
            'S',
            'M'][
            self.electronic_type == 0]  # if self.electronic_type!=0 else 'M'

##########################################################################
##########################################################################

# Optical / Electronic Transition Funcitons


class Transition:

    """
    Optical Transition Object

    Attributes:

    p
                Optical Transition Index
                Units: Dimensionless
                Reference 2
    k_p
                Length of vector from the K point (in polar coordinates)
                Units: 1/nm
                Reference 2
    theta_p
                Angle of wave vector around K point
                Units: radians
                Reference 2

    Eii_vacc
                Excitonic Energy for Optical Transition from Valence Band (i) to Conduction Band (i)
                Two sub bands are given for metallic chiral CNTs: k=0, k=1
                Units: eV
                Reference 2

    Eii
                Excitonic Energy for Optical Transition from Valence Band (i) to Conduction Band (i)
                Two sub bands are given for metallic chiral CNTs: k=0, k=1
                Adjusted for Environmental Shifts
                Units: eV
                Reference 4, 5

    sigma_p
                Optical Osciallator Strength
                Units: eV * cm^2 / atom
                Reference 1

    wp
                Half Line Width of Excitonic Optical Transition Peak
                Units: eV
                Reference 1

    a
                Oscillator Strength Term for High Energy Sideband Feature in Excitonic Transition
                Units: Dimensionless
                Reference 1

    b
                Energy Broadening Term for High Energy Sideband Feature in Excitonic Transition
                Units: Dimensionless
                Reference 1

    delta
                Energy Offset Term for High Energy Sideband Feature in Excitonic Transition
                Units: eV
                Reference 1
    """

    def __init__(self, parent, i, k=0, params=default_params):

        self.SWCNT = parent
        self.i = i
        self.k = k
        self.p = self.p()
        self.k_p = 2 * self.p / (3 * self.SWCNT.dt)
        self.theta_p = self.theta_p()

        for param_name in params:
            p_val = params[param_name]['Value']

            # If value is iterable and from the
            # 'Atlas of carbon nanotubes paper', it is likely
            # a list of values for determining transition properties
            try:
                if hasattr(p_val, '__iter__') and 'An atlas of carbon nanotube optical transitions' in params[
                        param_name]['Reference']:
                    vars(self)[param_name] = p_val[self.p - 1]
                else:
                    vars(self)[param_name] = p_val
            except:
                vars(self)[param_name] = p_val

        linear_Dirac_cone = self.alpha_supp_info * self.k_p
        electron_interaction = self.beta_supp_info * \
            self.k_p * np.log10(1.5 * self.k_p)
        trigonal_warping = (self.k_p**2) * (self.eta_supp_info + \
                            self.gamma_supp_info * np.cos(3 * self.theta_p)) * np.cos(3 * self.theta_p) 
        self.Eii_vacc = linear_Dirac_cone + \
            electron_interaction + trigonal_warping

        self.sigma_p = 45.9 / ((self.p + 7.5) * self.SWCNT.dt) * 10**-18

        # [value_if_false, value_if_true][0/1 or false/true]
        # if electronic type is not 0 , then semiconducting, so
        # [metallic, semiconducting]
        self.wp = self.Eii * [0.0214, 0.0194][self.SWCNT.electronic_type != 0]
        self.a = [
            0.976 +
            self.SWCNT.dt *
            0.186,
            4.673 -
            self.SWCNT.dt *
            0.747][
            self.SWCNT.electronic_type != 0]
        self.b = [
            3.065 -
            self.SWCNT.dt *
            0.257,
            0.970 +
            self.SWCNT.dt *
            0.256][
            self.SWCNT.electronic_type != 0]
        self.delta = [
            0.175 -
            self.SWCNT.dt *
            0.0147,
            0.273 -
            self.SWCNT.dt *
            0.041][
            self.SWCNT.electronic_type != 0]

        self.Eb = self.delta
        #self.Eb11 = Transition(parent, 1, k, params).Eb

        self.QY = [4.1 * np.exp(-0.0030 * self.Eb), 4.2 * np.exp(-0.0037 * self.Eb)][
            self.SWCNT.electronic_type == 2] if self.SWCNT.electronic_type != 0 else 0

    def p(self):
        i = self.i
        # Metallic CNT
        if(self.SWCNT.mod_type == 0):
            p = 3 * i

        # Semiconducting CNT
        elif(self.SWCNT.mod_type == 1 or self.SWCNT.mod_type == 2):
            # i = even
            if(i % 2 == 0):
                p = i + (i / 2) - 1
            # i = odd
            elif(i % 2 == 1):
                p = i + int(i / 2)
        else:
            print("Error in electronic type")
        return int(p)

    def theta_p(self):
        i = self.i
        k = self.k
        mod = self.SWCNT.mod_type
        theta = self.SWCNT.theta

        # Metallic CNT
        if(mod == 0):
            # Lower energy sub band
            if(k == 0):
                theta_p = theta + np.pi
            # Higher energy sub band
            elif(k == 1):
                theta_p = theta
        # If Semiconducting
        elif(mod == 1):
            theta_p = theta + i * np.pi
        elif(mod == 2):
            theta_p = theta + (i + 1) * np.pi
        return theta_p

    @property
    def Eii(self):

        Eii_shift = -self.Dswcntsolvent * \
            Onsager_diff(self.epsilon, self.eta) / (self.Eii_vacc**3 * self.SWCNT.dt**5)

        Eii = self.Eii_vacc + Eii_shift
        return Eii

    def trans_string(self):
        return str(self.i) + str(self.i) + "(" + str(self.k) + ")"


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
