"""Tests for all TI-based estimators in ``alchemlyb``.

"""
import pytest

import pandas as pd

from alchemlyb.parsing import gmx
from alchemlyb.parsing import amber
from alchemlyb.preprocessing.subsampling import uncorrelate_dhdl
import alchemtest.gmx
import alchemtest.amber
from numpy.testing import assert_almost_equal
import pandas as pd


def gmx_benzene_coul_dHdl():
    dataset = alchemtest.gmx.load_benzene()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['Coulomb']])

    return dHdl

def gmx_benzene_vdw_dHdl():
    dataset = alchemtest.gmx.load_benzene()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['VDW']])

    return dHdl

def gmx_expanded_ensemble_case_1_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_1()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_expanded_ensemble_case_2_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_2()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_expanded_ensemble_case_3_dHdl():
    dataset = alchemtest.gmx.load_expanded_ensemble_case_3()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_water_particle_with_total_energy_dHdl():
    dataset = alchemtest.gmx.load_water_particle_with_total_energy()

    dHdl = [gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']]

    return dHdl

def gmx_water_particle_with_potential_energy_dHdl():
    dataset = alchemtest.gmx.load_water_particle_with_potential_energy()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def gmx_water_particle_without_energy_dHdl():
    dataset = alchemtest.gmx.load_water_particle_without_energy()

    dHdl = pd.concat([gmx.extract_dHdl(filename, T=300)
                      for filename in dataset['data']['AllStates']])

    return dHdl

def amber_simplesolvated_charge_dHdl():
    dataset = alchemtest.amber.load_simplesolvated()

    dHdl = pd.concat([amber.extract_dHdl(filename)
                      for filename in dataset['data']['charge']])

    return dHdl

def amber_simplesolvated_vdw_dHdl():
    dataset = alchemtest.amber.load_simplesolvated()

    dHdl = pd.concat([amber.extract_dHdl(filename)
                      for filename in dataset['data']['vdw']])

    return dHdl



def test_autocorrelation_analysis():
    """Test that the reduced potential is calculated correctly when no energy is given.

    """

    # Load dataset
    dataset = gmx_water_particle_with_total_energy_dHdl()

    # Uncorrelate using dhddls
    df = uncorrelate_dhdl(dataset, dataset)

    # Check if the sum of values on the diagonal has the correct value
    assert_almost_equal(df.iloc[0][0], -7.4585, decimal=3)