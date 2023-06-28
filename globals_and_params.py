import numpy as np
import pandas as pd
import ast
# Defaults
default_lowestNM = 5
default_highestNM = 12
default_X = np.linspace(0.9, 6.0, 900)
default_abs = np.linspace(0.0, 4.0, 900)
default_params = pd.read_csv('Default_params.csv', index_col=0)

for col in default_params:
    for row in ['Value', 'Min', 'Max']:
        try:
            default_params[col][row] = ast.literal_eval(
                default_params[col][row])
        except ValueError:
            pass

# Fundamental Constants
# 1.421 is the C-C distance in angstroms
CCdist = 1.421

# Avagardo's Number
NA = 6.0221413 * 10**23

# Atomic Molar Mass of Carbon
MW_C = 12  # g/mol

# Planck's Constant
hbar = 6.58211928 * 10**-16  # eV*s

# E=hc/lambda , and hc=1240 eV*nm
WAVELENGTH_TO_ENERGY_CONVERSION = 1239.842  # eV*nm
