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
                 f_resolution=1e-2,
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
        self.f_resolution = f_resolution
        self.eps_c = eps_c  # 指数関数の係数がこの値以上であれば表示する
        self.make_matrix()

    def make_matrix(self):
        """
          determining dimension & nuclear spin matrices
          don't for get to execute this function when you give parameter manually
          (e.g.
            r = RelaxationCurveCalculation()
            r.i = 3.5
            r.make_matrix()
          )
        与えられたパラメータから行列及び次元を決定する.
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
        """
          calculate Hamiltonian from initialized parameters
        :return: Hamiltonian
        """
        nuq = np.abs(self.nuq)
        hz = -self.gyr * self.h0 * (1 + self.k_shift) * (
                self.iz * np.cos(self.theta)
                + (self.ix * np.cos(self.phi) + self.iy * np.sin(self.phi)) * np.sin(self.theta)
        )
        hq = nuq / 6.0 * (
                3 * self.iz @ self.iz - self.ii * np.identity(self.dim)
                + 0.5 * self.eta * (self.ip @ self.ip + self.im @ self.im)
        )
        return hz + hq

    def EnergyLevel(self):
        """
          diagonalizing nuclear spin Hamiltonian
          :return: self.level: eigen values (ascending order, 昇順)
                   self.v:     eigen states
        """
        h = self.Hamiltonian()
        self.level, self.v = np.linalg.eigh(h, 'L')

        sort_index = np.argsort(self.level)
        self.level = self.level[sort_index]
        self.v = self.v[:, sort_index]

    def TransitionProbability(self, v):
        v_transpose = np.conj(v.T)
        wx = v_transpose @ self.ix @ v
        wy = v_transpose @ self.iy @ v
        wz = v_transpose @ self.iz @ v
        w = np.real(wx * np.conj(wx)
                    + wy * np.conj(wy)
                    + wz * np.conj(wz)) - self.ii * np.identity(self.dim)
        return w

    def diagonalize_w_and_get_initial(self):
        """
          diagonalizing transition matrix
        :return: self.evw: eigen value of transition probability
                 self.a:   eigen vectors
                 self.c0:  initial values for distribution of nucleus
        """
        self.w = self.TransitionProbability(self.v)
        self.evw, self.a = np.linalg.eigh(self.w)

        idx = np.argsort(self.evw)[-1::-1]  # 遷移確率の固有値 降順
        self.evw = self.evw[idx]
        self.a = self.a[:, idx]

        # 場合分け
        case = self.cases()
        self.dim_ev = self.dim
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

    def get_resonance_and_relaxation(self, n0=None, get_initial_value=False):
        """
        共鳴周波数と強度, 緩和関数の係数行列を求める
        :return: self.fres: 共鳴周波数のnp.array
                 self.intens: intensityのnp.array
                 self.w     : 遷移行列の固有値
                 self.c     : 緩和関数の係数行列[resonance, evw]
        """
        self.diagonalize_w_and_get_initial()
        # あらゆるエネルギー固有値の差を作る
        self.fres = np.abs(np.tril(self.level[:, np.newaxis] - self.level, -1))
        self.intens = np.tril(self.w, -1) * 2.0 / np.ceil(self.ii)

        if get_initial_value and np.abs(self.h0*self.gyr) > 1e-6 :
            """
                共鳴周波数から初期条件を計算する.
                計算コストはかかるが, この中では最も正しい計算
            """
            # 共鳴周波数の重複を得る.
            # idx = [[i, ...][j, ...]] はi <-> j 遷移が有意であるようなindex全体(重複なし, i>j).
            decimal = -int(np.floor(np.log10(self.f_resolution)))
            self.fres = np.round(self.fres, decimals=decimal)
            # self なしは一時変数. 編集した後self つきに代入する.
            fres = np.unique(self.fres)
            fres = fres[np.argsort(fres)][1:]
            intens = np.zeros(fres.size)
            n0s = np.zeros((fres.size, self.dim))
            for f_i, f in enumerate(fres):
                idx = np.where(self.fres == f)
                n0s[f_i][idx[0]] += self.intens[idx]
                n0s[f_i][idx[1]] -= self.intens[idx]
                intens[f_i] += self.intens[idx].sum()
                self.intens[idx] = 0

            # 係数行列cを作る
            # self.c[resind][n] = (sum_d C_nd * n_d[resind](t=0))**2
            if n0s.size > 0:
                # 0行目はdummy
                self.c = np.zeros(n0s[0].size)
                for n0 in n0s:
                    c1 = self.a.T @ n0
                    self.c = np.vstack((self.c,
                                        c1*c1))
                # dummy消去
                self.c = self.c[1:, :]

                # まとめ
                self.fres = fres
                self.intens = intens
                self.evw = self.evw[1:]  # 固有値0除く
                self.c = self.c[:, 1:]
            else:
                self.c = np.empty(0)
            # 有意な遷移のみ取り出して周波数でソート
            # resind = (self.intens >= self.intens_lim) * (self.fres > 0.0)
            self.clean_resonance_by(
                condition=lambda i=None, f=None, **kargs: (i >= self.intens_lim) * (f > 0.0)
            )

        else:
            # 係数行列の計算
            # self.c[i,j] = sum_d (C_ni-C_nj)*C_nd*n_d(0)
            # self.c[nu,n]は遷移nuの緩和の, n番目の固有値の係数. 準位(j,k)間の遷移とは nu = j*2I + k の関係にある.
            # self.cの形は[self.fres.size, dim_ev]

            # あらゆる固有ベクトルの差を作る
            # 第二項のindex [0, 0, ,...,0 ,..., 2I+1, ..., 2I+1]
            idx_t = np.arange(0, self.dim_ev, step=1, dtype=np.int8)
            x, y = np.meshgrid(idx_t, idx_t)
            self.c = self.a[x.flatten(), :] - self.a[y.flatten(), :]

            # 有意なintensityかつ正の共鳴周波数の場合にTrue
            self.fres -= np.triu(np.ones((self.dim_ev, self.dim_ev)))
            self.fres = self.fres.flatten()
            self.intens = self.intens.flatten()

            # 有意な遷移のみ取り出して周波数でソート
            # resind = (self.intens >= self.intens_lim) * (self.fres > 0.0)
            self.clean_resonance_by(
                condition=lambda i=None, f=None, **kargs: (i >= self.intens_lim) * (f > 0.0)
            )

            # 遷移を間引いてから計算した方が計算コストが低い
            if n0 is None:
                self.c = self.c0 * (self.c * self.c)
            else:
                c1 = self.a.T @ n0
                self.c = self.c @ np.diag(c1)

            # 固有値0を除く
            self.evw = self.evw[1:]
            self.c = self.c[:, 1:]

        # 係数が有意な固有値のみ残す
        if self.c.size == 0:
            self.evw = np.empty(0)
        else:
            self.clean_w_by(
                condition=lambda c=None, **kwargs: np.max(self.c, axis=0) > self.eps_c
            )

        if n0 is not None:
            # self.resind = self.c.sum(axis=1) > 0
            # self.fres = self.fres[self.resind]
            # self.intens = self.intens[self.resind]
            # self.c = self.c[self.resind, :]
            # self.w_sort_idx = np.argsort(self.fres)

            # extract the formula with all positive coefficients
            self.clean_resonance_by(
                condition=lambda c=None, **kwargs: (c > 0).prod(axis=1) == 1
            )

            # 規格化
            self.c = RelaxationCurveCalculation.row_normalize(self.c)

    @staticmethod
    def row_normalize(x):
        """np行列xを受け取り, 行ごとに規格化(sum|・| = 1)して返す."""
        return np.linalg.inv(np.diag(abs(x).sum(axis=1))) @ x

    def clean_w_by(self, condition=None):
        if condition is None:
            return
        idx = condition(
            c=self.c, w=self.evw
        )
        self.evw = self.evw[idx]
        self.c = self.c[:, idx]

    def clean_resonance_by(self, condition=None):
        if condition is None:
            return
        idx = condition(
            f=self.fres, i=self.intens, c=self.c
        )
        self.fres = self.fres[idx]
        sort_idx = np.argsort(self.fres)
        self.fres = self.fres[sort_idx]
        self.intens = self.intens[idx][sort_idx]
        self.c = self.c[idx, :][sort_idx, :]

    def formula(self, n0=None, auto_initial=False):
        self.EnergyLevel()
        self.get_resonance_and_relaxation(n0=n0, get_initial_value=auto_initial)
        return self.formula_write()

    def formula_write(self):
        #       pi = np.pi
        #       fid.append("""# NMR relaxation curve
        #           # I={0:.1f}, gamma={1:.5f} MHz/T, H0={2:.5f} T, K={3:.5f}%
        #           # nuQ={4:.5f} MHz, eta={5:.5f}, theta={6:.5f}deg, phi={7:.5f}deg
        #           """.format(self.i, self.gyr, self.h0, 100.0 * self.k_shift, self.nuq, self.eta,
        #                      self.theta / pi * 180.0, self.phi / pi * 180))

        # formula style
        fid = ["[f(MHz) intensity]\n"]
        for i in range(self.fres.size):
            fid.append("[{0:g} {1:g}] ".format(self.fres[i], self.intens[i]))
            printed = False
            for j in range(self.evw.size):
                if abs(self.c[i, j]) > self.eps_c:
                    if not printed:
                        printed = True
                    else:
                        fid.append(" + ")

                    fid.append("{0:g} * exp({1:g} * x/T1)".format(self.c[i, j], self.evw[j] / 1000))
            fid.append("\n")
        return ' '.join(fid)

    def list(self, fid):
        for j in range(self.evw.size):
            fid.write(" {0:f}".format(self.evw[j]))
        fid.write("\n")

        for i in range(self.fres.size):
            fid.write("[{0:f} {1:f}]".format(self.fres[i], self.intens[i]))
            for j in range(self.evw.size):
                fid.write(" {0:f}".format(self.c[i, j]))
            fid.write("\n")


def main():
    r = RelaxationCurveCalculation(nuq=0)
    print(
        r.formula(auto_initial=True)
    )


if __name__ == '__main__':
    main()
