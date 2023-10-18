# Spectra.py
#
# Models for spectra analysis.
# David Lister
# October 2023
#

# Naming convention is:
#   - s_name for single models
#   - c_name for compound models
#   - fm_name for functions that build up the single models
#   - fd_name for the domain functions
#   - fg_name for the initial guess functions

import numpy as np
from scipy import special
import model_classes
import common

ureg = common.ureg


def fm_prob_pair_separation(x, params):
    """
    Calculates probability distribution of the distances between pairs of donors in a crystal
    :param x: Pair distance in nm
    :param params: Dictionary containing Nd, in units of per cm^3
    :return: Probability of a donor having this pair separation
    """
    nd = params["nd"] * ureg.cm**-3  # Using units here because this can be error-prone
    R_0 = np.power((4/3) * np.pi * nd, -1/3)
    R_0 = R_0.to("nm").magnitude
    R = x
    return 3 * R**2 / R_0**3 * np.exp(-1 * R**3 / R_0**3)


def fm_pair_kittel_R_from_E_tb(delta_energy, params):
    """
    Function that determines pair spacing given an energy shift, using Kittel's tight binding approximation.
    :param delta_energy: Energy shift from bound state, in eV. Only valid  in (-2* E_bind, 0).
    :param params: r_bohr and E_bind, in nm and eV respectively.
    :return: Pair separation, in nm
    """
    r_bohr = params['r_bohr']  # Exciton bohr radius in nm
    E_bind = params['E_bind']  # Exciton donor binding energy, in eV

    inner = delta_energy / (2 * np.exp(1) * E_bind)
    return r_bohr * (-np.real(special.lambertw(inner, k=-1)) - 1)


def fm_pair_kittel_dRdE_tb(delta_energy, params):
    """
    Function that the change in pair spacing corresponding to a change in energy, using Kittel's tight binding approximation.
    :param x: Energy, in eV.
    :param params: r_bohr and E_peak and E_bind, in nm and eV respectively.
    :return: Pair separation, in nm
    """
    r_bohr = params['r_bohr']  # Exciton bohr radius in nm
    E_bind = params['E_bind']  # Exciton donor binding energy, in eV

    inner = delta_energy / (2 * np.exp(1) * E_bind)
    lambertW = np.real(special.lambertw(inner, k=-1))
    return -r_bohr * lambertW / (delta_energy * (lambertW + 1))



def fm_pair_kittel_model(x, params):
    """
    An exciton pair model of inhomogeneous photoluminescence broadening.
    :param x: Energy, in eV. Depending on the domain, it can be the full, left or right side of the tail. Note that only the left tail is physically interpetable with this model.
    :param params: A dict containing r_bohr, E_peak, E_bind, and nd and scale in nm, eV and cm^-3 respectively.
    :return: Expected counts based on the model.
    """
    E_peak = params['E_peak']  # Peak center position
    scale = params['scale']  # Scaling factor
    scale = np.abs(scale)

    delta_energy = x - E_peak
    delta_energy = -1 * np.abs(delta_energy)  # the sub-functions all assume energy shift from isolated bound state
    mask = delta_energy > -1e-6
    delta_energy[mask] = -1e-6  # avoid zero values, smoothly

    r_from_E = fm_pair_kittel_R_from_E_tb(delta_energy, params)
    p_of_E = fm_prob_pair_separation(r_from_E, params)
    drdE = fm_pair_kittel_dRdE_tb(delta_energy, params)
    return p_of_E * drdE * scale


def fm_two_level_thermal_probability(delta_energy, params):
    """
    Probability of being in a state of a two-level system. Negative values x correspond to the lower energy state and
    positive values correspond to the higher energy state. The magnitude is the distance the state is from the midpoint.
    :param x: Delta energy from the midpoint of the two-level system. eV.
    :param params: T for the temperature in kelvin.
    :return: Probability of being in that state.
    """
    T = np.abs(params['T'])
    kb = 8.617555e-5  # eV / K
    beta = 1/(kb * T)
    Z = 1 + np.exp(2 * beta * delta_energy)
    return 1/Z


def fm_pair_kittel_model_thermal(x, params):
    """
    An exciton pair model of inhomogeneous photoluminescence broadening.
    :param x: Energy, in eV. Depending on the domain, it can be the full, left or right side of the tail. Note that only the left tail is physically interpetable with this model.
    :param params: A dict containing r_bohr, E_peak, E_bind, nd, T and scale in nm, eV, cm^-3 and K respectively.
    :return: Expected counts based on the model.
    """
    E_peak = params['E_peak']  # Peak center position
    scale = params['scale']  # Scaling factor
    scale = np.abs(scale)

    delta_energy_signed = x - E_peak
    delta_energy = -1 * np.abs(delta_energy_signed)  # the sub-functions all assume energy shift from isolated bound state
    mask = delta_energy > -1e-6
    delta_energy[mask] = -1e-6  # avoid zero values, smoothly

    r_from_E = fm_pair_kittel_R_from_E_tb(delta_energy, params)
    p_of_E = fm_prob_pair_separation(r_from_E, params)
    drdE = fm_pair_kittel_dRdE_tb(delta_energy, params)
    thermal_factor = fm_two_level_thermal_probability(delta_energy_signed, params)
    return p_of_E * drdE * scale * thermal_factor


def fd_pair_full_domain(x, params):
    """
    Kittel model domain masking function.
    :param x: Energy in eV
    :param params: E_bind and E_peak in eV
    :return:
    """
    E_peak = params['E_peak']  # Peak position
    E_bind = params['E_bind']  # Exciton donor binding energy, in eV
    mask = x > E_peak - 2*E_bind
    mask = np.logical_and(mask, x < E_peak + 2*E_bind)
    return mask


def fd_pair_lower_side_domain(x, params):
    """
    Kittel model domain masking function.
    :param x: Energy in eV
    :param params: E_bind and E_peak in eV
    :return:
    """
    E_peak = params['E_peak']  # Peak position
    E_bind = params['E_bind']  # Exciton donor binding energy, in eV
    mask = x > E_peak - 2*E_bind
    mask = np.logical_and(mask, x < E_peak)
    return mask


def fd_all(x, params):
    return np.full(x.shape, True)


s_pair_kittel_model_lower_side = model_classes.SingleModel(fm_pair_kittel_model, fd_pair_lower_side_domain, ('E_bind', "E_peak", "r_bohr", "nd", "scale"))
s_pair_kittel_model_thermal = model_classes.SingleModel(fm_pair_kittel_model_thermal, fd_pair_full_domain, ('E_bind', "E_peak", "r_bohr", "nd", "T", "scale"))

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 2000)
    T_list = [4, 70, 300]
    for T in T_list:
        plt.plot(x, fm_two_level_thermal_probability(x, {"T": T}), label=f"T = {T}K")

    plt.legend()
    plt.show()
