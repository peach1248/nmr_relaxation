#!/usr/bin/env python3
# -*- coding: utf-8 -*-


from RelaxationCurve import RelaxationCurveCalculation
import numpy as np


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
    def test_no_error(self):
        r = RelaxationCurveCalculation(i=0.5, h0=0, nuq=0)
        r.formula()

    def test_energy_level(self):
        r = RelaxationCurveCalculation(i=0.5, h0=1, gyr=1, nuq=0)
        r.EnergyLevel()
        assert (r.level == np.array([-0.5, 0.5])).all()
        assert (r.v == np.array([[1, 0],
                                [0, 1]])
                ).all()

    def test_energy_level_i_three_half(self):
        r = RelaxationCurveCalculation(i=1.5, h0=1, gyr=1, nuq=0.1)
        r.EnergyLevel()
        assert (np.abs(r.level - np.array([-1.45, -0.55, 0.45, 1.55])) < 1e-6).all()
        assert (np.abs(r.v - np.array([[1, 0, 0, 0],
                                       [0, 1, 0, 0],
                                       [0, 0, 1, 0],
                                       [0, 0, 0, 1]])) < 1e-6
                ).all()

        r.get_resonance_and_relaxation()
        assert (np.abs(r.fres - np.array([0.9, 1, 1.1])) < 1e-6).all()
        assert (np.abs(
            r.c
            - np.array([[0.1, 0.5, 0.4],
                        [0.1, 0.0, 0.9],
                        [0.1, 0.5, 0.4]])) < 1e-6
                ).all()

    def test_energy_level_nqr(self):
        r = RelaxationCurveCalculation(i=1.5, h0=0, nuq=10, eta=0)
        r.EnergyLevel()
        assert (r.level == np.array([-5, -5, 5, 5])).all()


def main():
    pass


if __name__ == "__main__":
    main()
