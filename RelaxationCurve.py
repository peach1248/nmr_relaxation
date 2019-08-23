#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np


class RelaxationCurveCalculation:
    """緩和曲線を計算する
        メソッド
        make_matrix(): 与えられたパラメータから必要な行列を計算する. 変数を設定したときに実行する.
        Hamiltonian(): Hamiltonianを計算し, numpyの行列で返す.
        EnergyLevel(): 核spinのエネルギー固有値self.levelと固有状態self.vを計算する.
        TransitionProbability(v): 固有状態vに対して遷移確率の行列を計算する.
        FSP(): 共鳴周波数を計算する. 有意な遷移のみを取り出し, self.fres, self.intensに返す.
        some_case(): 自動初期条件設定. NQRの場合に対応.
        RelaxationCurve(): 緩和関数の計算.
        RelaxationCurveInitial(n0): 初期条件n0を与えた時の緩和関数の計算.
        formula(fid, n0): fidに緩和関数を書きだす. 初期条件n0を与えることもできる.
        """

    def __init__(self,
                 i=3.5,
                 gyr=10.0,
                 h0=1,
                 k_shift=0,
                 nuq=1.000,
                 eta=0,
                 theta=0,
                 phi=0,
                 intens_lim=1e-5,
                 eps=2e-16,
                 eps_c=1e-6):
        """デフォルトの初期条件を与える.
        実際の使用時にはここで与えた変数を外部から与えて, 次のmake_matrixを行う"""
        pi = np.pi
        self.i = i  # 核spin
        self.gyr = gyr  # MHz/T
        self.h0 = h0  # T
        self.k_shift = k_shift * 1e-2 # 単位は%
        self.nuq = nuq  # MHz
        self.eta = eta  #
        self.theta = theta * pi / 180.0  # 電場勾配の最大主軸をz方向，
        self.phi = phi * pi / 180.0  # 磁場方向を極座標(theta,phi)で表す．単位はrad.
        self.intens_lim = intens_lim  # NMR中心線の信号強度のintens_lim倍未満の線は無視
        self.eps = eps
        self.eps_c = eps_c  # 指数関数の係数がこの値以上であれば表示する
        self.make_matrix()

    def make_matrix(self):
        """与えられたパラメータから行列及び次元を決定する.
        パラメータを与えたら実行せよ."""
        self.dim = int(2 * self.i + 1)
        self.ii = self.i * (self.i + 1)
        m_spin = np.arange(self.i, -self.i - 1.0, -1.0)

        self.ip = np.diag(np.sqrt(self.ii - (m_spin[0:-1]) * (m_spin[0:-1] - 1.0)), 1)
        self.im = np.conj(self.ip.T)
        self.ix = +0.5 * (self.ip + self.im)
        self.iy = -0.5j * (self.ip - self.im)
        self.iz = np.diag(m_spin)

    def Hamiltonian(self):
        nuq = np.abs(self.nuq)
        H_Z = -self.gyr * self.h0 * (1 + self.k_shift) * (self.iz * np.cos(self.theta)
                                                          + (self.ix * np.cos(self.phi) + self.iy * np.sin(
                    self.phi)) * np.sin(self.theta))
        H_Q = nuq / 6.0 \
              * (3 * self.iz @ self.iz - self.ii * np.identity(self.dim)
                 + 0.5 * self.eta * (self.ip @ self.ip + self.im @ self.im))

        return H_Z + H_Q

    def EnergyLevel(self):
        """
          diagonalizing Hamiltonian
          :return: self.level: eigen values (ascending order, 昇順)
                   self.v:     eigen states
        """
        h = self.Hamiltonian()
        self.level, self.v = np.linalg.eigh(h, 'L')

        sort_index = np.argsort(self.level)
        self.level = self.level[sort_index]
        self.v = self.v[:, sort_index]

    def TransitionProbability(self, v):
        wx = np.conj(v.T) @ self.ix @ v
        wy = np.conj(v.T) @ self.iy @ v
        wz = np.conj(v.T) @ self.iz @ v
        w = np.real(wx * np.conj(wx)
                    + wy * np.conj(wy)
                    + wz * np.conj(wz)) - self.ii * np.identity(self.dim)
        return w

    def FSP(self):
        """
         FSPの計算
        :return: self.fres: 共鳴周波数のnp.array
                 self.intens: intensityのnp.array
                 self.
        """
        self.diagonalize_w_and_get_initial()
        # あらゆるエネルギー固有値の差を作る
        fres = (np.abs(np.tril(self.level[:, np.newaxis] - self.level, -1)) -
                np.triu(np.ones((self.dim_ev, self.dim_ev)))).flatten()
        intens = (np.tril(self.w, -1)).flatten() * 2.0 / np.ceil(self.ii)
        # 有意なintensityかつ正の共鳴周波数の場合にTrue
        resind = (intens >= self.intens_lim) * (fres > 0.0)
        # 有意な遷移のみ取り出して周波数でソート
        self.w_sort_idx = np.argsort(fres[resind])
        self.fres = fres[resind][self.w_sort_idx]
        self.intens = intens[resind][self.w_sort_idx]
        return resind

    def diagonalize_w_and_get_initial(self):
        """
          diagonalizing transition matrix
        :return: self.evw: eigen value of transition probability
                 self.a:   eigen vectors
                 self.c0:  initial values for distribution of nucleus
        """
        self.w = self.TransitionProbability(self.v)
        self.evw, self.a = np.linalg.eigh(self.w)

        self.w_sort_idx = np.argsort(self.evw)[-1::-1]  # 遷移確率の固有値 降順
        self.evw = self.evw[self.w_sort_idx]
        self.a = self.a[:, self.w_sort_idx]

        # 場合分け
        # todo 初期値、計算するRの差の設定を精密化
        # 共鳴周波数に合わせた初期値の設定
        case = self.cases()
        if case == "integer NQR":  # NQR，I: 整数，η=0
            self.w = np.vstack((np.zeros(self.dim), self.w))
            self.w = np.hstack((np.zeros((self.dim + 1, 1)), self.w))
            self.a = np.vstack((2.0 * self.a[0, :], self.a[1::2, :] + self.a[2::2, :]))
            # 係数の補正項
            self.c0 = np.vstack((np.array([1 / 6.0]),
                                 0.25 * np.ones(((self.dim - 3) // 2, 1))))

        elif case == "half integer NQR":  # NQR，I: 半奇数
            self.a = self.a[0::2, :] + self.a[1::2, :]
            self.c0 = 0.25
        else:  # NMR, または分裂なし
            self.c0 = 0.5
            self.dim_ev = self.dim
            return

        self.level = self.level[::2]  # 縮退エネルギーをまとめる
        # 遷移確率の縮退準位に関する和 (intensity に寄与)
        self.w = self.w[::2, ::2] + self.w[::2, 1::2] + self.w[1::2, ::2] + self.w[1::2, 1::2]
        self.dim_ev = (self.dim + 1) // 2

    def cases(self):
        f_nmr = self.gyr * self.h0 * (1 + self.k_shift)
        # NQR case
        is_nqr = np.abs(f_nmr) < self.eps * np.abs(self.nuq)
        is_eta_zero_in_integer_case = (self.dim % 2) * np.abs(self.eta) < self.eps
        if is_nqr and is_eta_zero_in_integer_case:
            if self.dim % 2 == 1:
                # NQR，I: 整数，η=0
                return "integer NQR"
            else:
                # NQR，I: 半奇数
                return "half integer NQR"
        else:
            # NMR, または分裂なし
            return "NMR"

    def RelaxationCurve(self, resind):
        # あらゆる固有ベクトルの差を作る
        # 第二項のindex [0, 0, ,...,0 ,..., 2I+1, ..., 2I+1]
        idx_t = np.arange(0, self.dim_ev, 1, dtype=np.int8)
        x, y = np.meshgrid(idx_t, idx_t)
        self.c = self.a[x.flatten(), 1:] - self.a[y.flatten(), 1:]
        self.c = self.c[resind, :][self.w_sort_idx, :]
        self.c = self.c0 * (self.c * self.c)
        cind = (np.max(self.c, axis=0) > self.eps_c)
        self.evw = self.evw[1:][cind]
        self.c = self.c[:, cind]

    def RelaxationCurveInitial(self, n0, resind):
        """初期値のリスト n0に対して緩和曲線を計算する. """
        self.dim_ev = self.dim
        # 係数行列の計算
        c1 = self.a.T @ n0

        # meshgrid が圧倒的に早い
        idx_t = np.arange(0, self.dim_ev, 1, dtype=np.int8)
        x, y = np.meshgrid(idx_t, idx_t)
        c2 = self.a[x.flatten(), :] - self.a[y.flatten(), :]

        print(resind.shape)
        print(c2.shape)
        # 有意な遷移のみ残す.
        c2 = c2[resind, :][self.w_sort_idx, :]

        # self.c[i,j] = sum_d (C_ni-C_nj)*C_nd*n_d(0)
        # self.c[nu,n]は遷移nuの緩和の, n番目の固有値の係数. 準位(j,k)間の遷移とは nu = j*2I + k の関係にある.
        self.c = (c2 @ np.diag(c1))[:, 1:]
        # self.cの形は[self.w_sort_idx.size, dim_ev]

        cind = np.max(self.c, axis=0) > self.eps_c
        self.evw = self.evw[1:][cind]
        self.c = self.c[:, cind]

        # self.resind = self.c.sum(axis=1) > 0
        # self.fres = self.fres[self.resind]
        # self.intens = self.intens[self.resind]
        # self.c = self.c[self.resind, :]
        # self.w_sort_idx = np.argsort(self.fres)

        # extract the formula with all positive coefficients
        resind = (self.c > 0).prod(axis=1) == 1
        self.fres = self.fres[resind]
        self.intens = self.intens[resind]
        self.c = self.c[resind, :]
        self.w_sort_idx = np.argsort(self.fres)

        # 規格化
        self.c = RelaxationCurveCalculation.row_normalize(self.c)

    @staticmethod
    def row_normalize(x):
        """np行列xを受け取り, 行ごとに規格化(sum|・| = 1)して返す."""
        return np.linalg.inv(np.diag(abs(x).sum(axis=1))) @ x

    def formula(self, n0=None):
        self.EnergyLevel()
        resonance_index = self.FSP()
        if n0 is None:
            self.RelaxationCurve(resind=resonance_index)
        else:
            self.RelaxationCurveInitial(n0, resind=resonance_index)
        return self.formula_write()

    def formula_write(self):
        fid = []

        #       pi = np.pi
        #       fid.append("""# NMR relaxation curve
        #           # I={0:.1f}, gamma={1:.5f} MHz/T, H0={2:.5f} T, K={3:.5f}%
        #           # nuQ={4:.5f} MHz, eta={5:.5f}, theta={6:.5f}deg, phi={7:.5f}deg
        #           """.format(self.i, self.gyr, self.h0, 100.0 * self.k_shift, self.nuq, self.eta,
        #                      self.theta / pi * 180.0, self.phi / pi * 180))

        # formula style
        fid.append("[f(MHz) intensity]\n")
        for i in range(self.fres.size):
            fid.append("[{0:g} {1:g}] ".format(self.fres[i], self.intens[i]))
            printed = 0
            for j in range(self.evw.size):
                if abs(self.c[i, j]) > self.eps_c:
                    if printed == 0:
                        printed = 1
                    else:
                        fid.append(" + ")

                    fid.append("{0:g} * exp({1:g} * x/T1)".format(self.c[i, j], self.evw[j] / 1000))
            fid.append("\n")
        return ' '.join(fid)

    def list(self, fid):
        for j in range(self.evw.size):
            fid.write(" {0:f}".format(self.evw[j]))

        fid.write("\n")

        for i in range(self.w_sort_idx.size):
            fid.write("[{0:f} {1:f}]".format(self.fres[i], self.intens[i]))
            for j in range(self.evw.size):
                fid.write(" {0:f}".format(self.c[i, j]))

            fid.write("\n")

