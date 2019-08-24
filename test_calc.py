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


def test_energy_level():
    r = RelaxationCurveCalculation(i=0.5, h0=1, gyr=1, nuq=0)
    r.EnergyLevel()
    assert (r.level == np.array([-0.5, 0.5])).all()
    assert (r.v == np.array([[1, 0],
                            [0, 1]])
            ).all()


def test_energy_level_nqr():
    r = RelaxationCurveCalculation(i=1.5, h0=0, nuq=10, eta=0)
    r.EnergyLevel()
    assert (r.level == np.array([-5, -5, 5, 5])).all()


def main():
    pass


if __name__ == "__main__":
    main()
