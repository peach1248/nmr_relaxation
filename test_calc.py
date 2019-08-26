#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import pytest
import numpy as np

from RelaxationCurve import RelaxationCurveCalculation


def test_init():
    r = RelaxationCurveCalculation(i=2.5)
    assert r.i == 2.5


def test_matrix():
    r = RelaxationCurveCalculation(i=0.5)
    r.make_matrix()

    ip = np.array([[0, 1],
                   [0, 0]])
    im = np.array([[0, 0],
                   [1, 0]])

    assert (r.ip == ip).all() and (r.im == im).all()


def test_hamiltonian():
    r = RelaxationCurveCalculation(i=0.5, gyr=1, h0=1, nuq=0)
    hamiltonian = np.array([[-0.5, 0],
                            [0, 0.5]])
    assert (np.abs(r.Hamiltonian() - hamiltonian) < 1e-16).all()


class TestEnergyLevelRelaxation:
    @pytest.mark.parametrize("auto", [True, False])
    def test_no_error(self, auto):
        r = RelaxationCurveCalculation(i=0.5, h0=0, nuq=0)
        r.formula(auto_initial=auto)

    @pytest.mark.parametrize("auto", [True, False])
    def test_energy_level_i_one_half(self, auto):
        r = RelaxationCurveCalculation(i=0.5, h0=1, gyr=1, nuq=0)
        r.EnergyLevel()
        assert (r.level == np.array([-0.5, 0.5])).all()
        assert (r.v == np.array([[1, 0],
                                [0, 1]])
                ).all()
        r.get_resonance_and_relaxation(get_initial_value=auto)
        assert (np.abs(r.fres - np.array([1])) < 1e-6).all()
        assert (np.abs(
            r.c
            - np.array([[1]])) < 1e-6
        ).all()
        assert (np.abs(
            r.evw
            - np.array([[-1]])) < 1e-6
        ).all()

    @pytest.mark.parametrize("auto", [True, False])
    def test_energy_level_i_three_half(self, auto):
        r = RelaxationCurveCalculation(i=1.5, h0=1, gyr=1, nuq=0.1)
        r.EnergyLevel()
        assert (np.abs(r.level - np.array([-1.45, -0.55, 0.45, 1.55])) < 1e-6).all()
        assert (np.abs(r.v - np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])) < 1e-6
                ).all()

        r.get_resonance_and_relaxation(get_initial_value=auto)
        assert (np.abs(r.fres - np.array([0.9, 1, 1.1])) < 1e-6).all()
        assert (np.abs(
            r.c
            - np.array([[0.1, 0.5, 0.4],
                        [0.1, 0.0, 0.9],
                        [0.1, 0.5, 0.4]])) < 1e-6
                ).all()

    @pytest.mark.parametrize("auto", [True, False])
    def test_energy_level_nqr_i_three_half(self, auto):
        r = RelaxationCurveCalculation(i=1.5, h0=0, nuq=10, eta=0)
        r.EnergyLevel()
        assert (r.level == np.array([-5, -5, 5, 5])).all()
        r.get_resonance_and_relaxation(get_initial_value=auto)
        assert (np.abs(r.fres - np.array([10])) < 1e-6).all()
        assert (np.abs(
            r.c - np.array([[1]])) < 1e-6
        ).all()
        assert (np.abs(
            r.evw - np.array([[-3]])) < 1e-6
        ).all()

    @pytest.mark.parametrize("auto", [True, False])
    def test_energy_level_nqr_i_five_half(self, auto):
        r = RelaxationCurveCalculation(i=2.5, h0=0, nuq=10, eta=0)
        r.EnergyLevel()
        r.get_resonance_and_relaxation(get_initial_value=auto)
        assert (np.abs(
            r.c - np.array([[3/28, 25/28],
                            [3/7, 4/7]])) < 1e-6
        ).all()
        assert (np.abs(
            r.evw - np.array([[-3, -10]])) < 1e-6
        ).all()

    @pytest.mark.parametrize("auto", [True, False])
    def test_energy_level_nqr_i_three(self, auto):
        r = RelaxationCurveCalculation(i=3, h0=0, nuq=10, eta=0)
        r.EnergyLevel()
        r.get_resonance_and_relaxation(get_initial_value=auto)
        assert (np.abs(
            r.c - np.array([[0.0079365, 0.10823, 0.88384],
                            [0.10714, 0.41558, 0.47727],
                            [0.29762, 0.64935, 0.05303]])) < 1e-5
        ).all()
        assert (np.abs(
            r.evw - np.array([[-3, -10, -21]])) < 1e-5
        ).all()

    @pytest.mark.parametrize("auto", [True, False])
    def test_energy_level_nmr_i_five_half(self, auto):
        r = RelaxationCurveCalculation(i=2.5, gyr=1, h0=10, nuq=1, eta=0)
        r.EnergyLevel()
        r.get_resonance_and_relaxation(get_initial_value=auto)
        assert (np.abs(
            r.c - np.array([[0.028571, 0.2143, 0.3996, 0.2857, 0.0714],
                            [0.028571, 0.05357, 0.0249987, 0.4464187, 0.4463875],
                            [0.02857, 0, 0.17778, 0, 0.793667],
                            [0.028571, 0.05357, 0.0249987, 0.4464187, 0.4463875],
                            [0.028571, 0.2143, 0.3996, 0.2857, 0.0714]])) < 1e-3
        ).all()
        assert (np.abs(
            r.evw - np.array([[-1, -3, -6, -10, -15]])) < 1e-5
        ).all()


def main():
    pass


if __name__ == "__main__":
    main()
