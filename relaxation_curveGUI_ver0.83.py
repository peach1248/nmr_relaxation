#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import numpy as np
from PyQt5.QtWidgets import (QApplication,
                             QMainWindow, QWidget, QMessageBox, QDockWidget,
                             QCheckBox, QPushButton, QLabel,
                             QComboBox, QLineEdit, QTextEdit,
                             QHBoxLayout, QVBoxLayout, QGroupBox,
                             QAction)
from PyQt5 import QtCore

from RelaxationCurve import RelaxationCurveCalculation


"""緩和曲線計算プログラム. 
真砂さんの素晴らしいプログラムをClass化, GUI化した.
今後の予定としては
・初期値に関する修正
　・データ保持
　・NQRの場合
・T1 fitプログラム(仲嶺さんが作成中)との連携を目指す

更新履歴:
20180224 ver.0.00 作成開始. 真砂さんのプログラムをClass化した. 関数の分割はうまくいっていない.
20180225 ver.0.80 PyQtをひとまず実装.
20180301 ver.0.81 初期値入力機能実装.
20180308 ver.0.82 少々の機能追加とbug fix.
20181103 ver.0.83 PyQt5に変更
"""


class ParameterDockWidget(QWidget):
    """緩和曲線計算に必要なパラメータを入力するWidget
    オブジェクト:
    set_initial: 初期値を設定するButton
    i_combo: spin Iを選択するComboBox
    gamma: 核磁気回転比入力LineBox
    h0: 印可磁場入力LineBox
    K: Knight shift入力LineBox
    nuQ: nuQ入力LineBox
    eta: eta入力LineBox
    theta: theta入力LineBox
    phi: phi入力LineBox

    calculation_button: 計算を実行するButton; 具体的な処理はMainWindow classで指定
    """

    def __init__(self, parent=None):
        super(ParameterDockWidget, self).__init__(parent)
        # 画面幅固定
        self.setMaximumWidth(200)

        self.set_initial_check = QCheckBox('set Initial Values', self)
        self.set_initial = QPushButton('Input Initial Values')

        i_label = QLabel("I =")
        self.i_combo = QComboBox(self)
        for x in ["1/2", "3/2", "5/2", "7/2", "9/2"]:
            self.i_combo.addItem(x)
        i_box = QHBoxLayout()
        i_box.addWidget(i_label)
        i_box.addWidget(self.i_combo)

        # 入力欄作成
        self.gamma = QLineEdit(self)
        self.h0 = QLineEdit(self)
        self.K = QLineEdit(self)
        self.nuQ = QLineEdit(self)
        self.eta = QLineEdit(self)
        self.theta = QLineEdit(self)
        self.phi = QLineEdit(self)

        objects = [self.gamma, self.h0, self.K,
                   self.nuQ, self.eta, self.theta, self.phi]
        labels = ["gamma (MHz/T)", "H0 (T)", "K (%)",
                  "nuQ (MHz)", "eta", "theta (rad)", "phi (rad)"]
        boxes = []

        for obj, l in zip(objects, labels):
            label = QLabel(l)
            box = QHBoxLayout()
            box.addWidget(label)
            box.addWidget(obj)
            boxes.append(box)

        self.calculation_button = QPushButton("Calculate")

        self.inputs_group = QGroupBox("inputs")
        group_box = QVBoxLayout()
        for x in [i_box] + boxes:
            group_box.addLayout(x)
        self.inputs_group.setLayout(group_box)

        vbox = QVBoxLayout(self)
        vbox.addWidget(self.set_initial_check)
        vbox.addWidget(self.set_initial)
        vbox.addWidget(self.inputs_group)
        vbox.addWidget(self.calculation_button)
        vbox.setAlignment(QtCore.Qt.AlignLeft)
        self.setLayout(vbox)

        self.setDefaultParameters()

    def setDefaultParameters(self):
        self.gamma.setText("1.5")
        self.h0.setText("10")
        self.K.setText("0")
        self.nuQ.setText("0")
        self.eta.setText("0")
        self.theta.setText("0")
        self.phi.setText("0")


class FunctionDockWidget(QWidget):
    """緩和曲線を表示するWidget
    オブジェクト:
    function: 緩和曲線を出力する. 一応ユーザーによる書き換えは可能"""

    def __init__(self, parent=None):
        super(FunctionDockWidget, self).__init__(parent)

        function_label = QLabel("Relaxation Curve")
        self.function = QTextEdit(self)

        vbox = QVBoxLayout(self)
        vbox.addWidget(function_label)
        vbox.addWidget(self.function)
        vbox.setAlignment(QtCore.Qt.AlignCenter)
        self.setLayout(vbox)


class InitialParameterWidget(QWidget):
    """初期値を入力するWidget: 引数nは2I+1を入れる.
    ab_button_pushed: center A or center Bの初期値を入力する.
    get_parameter: n=2I+1に対して入力された初期値をlistとして返す.
    """

    def __init__(self, n, parent=None):
        super(InitialParameterWidget, self).__init__(parent)
        self.setWindowTitle('Initial Values')
        self.setGeometry(1000, 200, 200, 300)

        hbox = QHBoxLayout()
        self.a_button = QPushButton("center A", self)
        self.b_button = QPushButton("center B", self)
        for button in [self.a_button, self.b_button]:
            hbox.addWidget(button)
            button.clicked.connect(self.ab_button_pushed)

        vbox = QVBoxLayout(self)
        vbox.addLayout(hbox)
        vbox.setAlignment(QtCore.Qt.AlignLeft)
        self.initial_text = []
        for x in range(n):
            label = QLabel("m = {}/2".format((n - 1) - x * 2))
            self.initial_text.append(QLineEdit(self))
            box = QHBoxLayout()
            box.addWidget(label)
            box.addWidget(self.initial_text[x])
            vbox.addLayout(box)

        self.setLayout(vbox)
        self.N = n

    def ab_button_pushed(self):
        """初期条件の例を与えるボタン"""
        sender = self.sender()
        if sender.text() == "center A":
            for x in range(self.N):
                t = self.initial_text[x]
                t.setText("0")
            self.initial_text[self.N // 2 - 1].setText("0.5")
            self.initial_text[self.N // 2].setText("-0.5")
        elif sender.text() == "center B":
            for x in range(self.N):
                t = self.initial_text[x]
                if x < self.N // 2:
                    t.setText("0.5")
                else:
                    t.setText("-0.5")

    def get_parameter(self, n):
        try:
            n0 = [float(self.initial_text[x].text()) for x in range(n)]
        except ValueError:
            QMessageBox.warning(self, "Message", u"Input Initial Values")
            return []

        if abs(np.sum(np.array(n0))) > 1.e-10:
            QMessageBox.warning(self, "Message", u"Sum of Initial Values is not zero")
            return []
        return n0


class AdvSettings(QWidget):
    def __init__(self, parent=None):
        super(AdvSettings, self).__init__(parent)
        self.setWindowTitle('Advanced Settings')
        self.setGeometry(500, 200, 200, 100)

        vbox = QVBoxLayout(self)
        vbox.setAlignment(QtCore.Qt.AlignLeft)
        label = QLabel("threshold of coefficient of exponential")
        self.eps_c_text = QLineEdit(self)
        box = QHBoxLayout()
        box.addWidget(label)
        box.addWidget(self.eps_c_text)
        vbox.addLayout(box)
        self.eps_c_text.textChanged.connect(self.parameter_set)

        label = QLabel("threshold of intensity of signal")
        self.int_lim_text = QLineEdit(self)
        box = QHBoxLayout()
        box.addWidget(label)
        box.addWidget(self.int_lim_text)
        vbox.addLayout(box)
        self.int_lim_text.textEdited.connect(self.parameter_set)

        self.setLayout(vbox)
        self.eps_c_text.setText("1e-2")
        self.int_lim_text.setText("1e-2")
        self.eps_c = 1e-6
        self.int_lim = 1e-5

    def parameter_set(self):
        try:
            self.eps_c = float(self.eps_c_text.text())
            self.int_lim = float(self.int_lim_text.text())
        except ValueError:
            pass


class UI(QMainWindow):
    """MainWindowの構成
    メソッド:
    calculation: 緩和曲線を計算し, 出力する.
    input_initial: 初期値入力Widgetを開く.
    closeEvent: 初期値入力Widgetを同時に閉じる."""
    def __init__(self):
        super(UI, self).__init__()
        self.setGeometry(300, 200, 700, 300)
        self.setWindowTitle('RelaxationCurveCalculation')

        # _dock: class
        # _widget: dock widgetそのもの
        self.parameter_dock = ParameterDockWidget()
        self.parameter_widget = QDockWidget("Parameters", self)
        self.parameter_widget.setWidget(self.parameter_dock)
        self.addDockWidget(QtCore.Qt.LeftDockWidgetArea, self.parameter_widget)
        self.parameter_dock.calculation_button.clicked.connect(self.calculation)
        self.parameter_dock.set_initial.clicked.connect(self.input_initial)
        self.parameter_dock.i_combo.activated.connect(self.input_initial)

        self.function_dock = FunctionDockWidget()
        self.function_widget = QDockWidget("Relaxation Curve", self)
        self.function_widget.setWidget(self.function_dock)
        self.addDockWidget(QtCore.Qt.RightDockWidgetArea, self.function_widget)

        self.initial_parameter = InitialParameterWidget(2)
        self.n0 = []
        self.advset_widget = AdvSettings()

        # menu bar
        viewParameter = QAction('Parameter', self)
        viewParameter.setShortcut('Ctrl+P')
        viewParameter.triggered.connect(self.parameter_widget.show)

        viewFunction = QAction('Function', self)
        viewFunction.setShortcut('Ctrl+F')
        viewFunction.triggered.connect(self.function_widget.show)

        advSetting = QAction('Setting more', self)
        advSetting.triggered.connect(self.advset_widget.show)

        menu_bar = self.menuBar()
        viewMenu = menu_bar.addMenu("Widget")
        viewMenu.addAction(viewParameter)
        viewMenu.addAction(viewFunction)
        optionMenu = menu_bar.addMenu("Option")
        optionMenu.addAction(advSetting)

        self.show()

    def calculation(self):
        r = RelaxationCurveCalculation()

        # パラメータの設定
        pd = self.parameter_dock
        r.gyr = float(pd.gamma.text())
        r.i = (pd.i_combo.currentIndex() * 2 + 1) / 2  # I
        r.h0 = float(pd.h0.text())
        r.k_shift = float(pd.K.text()) / 100
        r.nuq = float(pd.nuQ.text())
        r.eta = float(pd.eta.text())
        r.theta = float(pd.theta.text())
        r.phi = float(pd.phi.text())

        r.eps_c = self.advset_widget.eps_c
        r.intens_lim = self.advset_widget.int_lim

        r.make_matrix()
        try:
            if self.parameter_dock.set_initial_check.checkState():
                N = self.parameter_dock.i_combo.currentIndex() * 2 + 2  # 2I+1
                self.n0 = self.initial_parameter.get_parameter(N)
                if not self.n0:
                    return
                a = r.formula(self.n0)
            else:
                a = r.formula()
        except ValueError:
            QMessageBox.warning(self, "Message", u"Sorry but initial parameters are not available in NQR case")
        else:
            self.function_dock.function.setText(a)
            self.function_widget.show()

    def input_initial(self):
        if self.parameter_dock.set_initial_check.checkState():
            N = self.parameter_dock.i_combo.currentIndex() * 2 + 2  # 2I+1
            self.initial_parameter = InitialParameterWidget(N)
            self.initial_parameter.show()

    def closeEvent(self, event):
        self.initial_parameter.close()
        self.advset_widget.close()
        self.close()

# class RelaxationCurveCalculation:
#     """緩和曲線を計算する
#         メソッド
#         make_matrix(): 与えられたパラメータから必要な行列を計算する. 変数を設定したときに実行する.
#         Hamiltonian(): Hamiltonianを計算し, numpyの行列で返す.
#         EnergyLevel(): 核spinのエネルギー固有値self.levelと固有状態self.vを計算する.
#         TransitionProbability(v): 固有状態vに対して遷移確率の行列を計算する.
#         FSP(): 共鳴周波数を計算する. 有意な遷移のみを取り出し, self.fres, self.intensに返す.
#         some_case(): 自動初期条件設定. NQRの場合に対応.
#         RelaxationCurve(): 緩和関数の計算.
#         RelaxationCurveInitial(n0): 初期条件n0を与えた時の緩和関数の計算.
#         formula(fid, n0): fidに緩和関数を書きだす. 初期条件n0を与えることもできる.
#         """
#
#     def __init__(self):
#         """デフォルトの初期条件を与える.
#         実際の使用時にはここで与えた変数を外部から与えて, 次のmake_matrixを行う"""
#         pi = np.pi
#         self.i = 3.5  # 核spin
#         self.gyr = 10.0  # MHz/T
#         self.h0 = 10.0  # T
#         self.k_shift = 0.000  # 単位はunity
#         self.nuq = 1.000  # MHz
#         self.eta = 0.0  #
#         self.theta = 00.0 * pi / 180.0  # 電場勾配の最大主軸をz方向，
#         self.phi = 00.0 * pi / 180.0  # 磁場方向を極座標(theta,phi)で表す．単位はrad.
#         self.intens_lim = 1.0e-5  # NMR中心線の信号強度のintens_lim倍未満の線は無視
#         self.eps = 2e-16
#         self.eps_c = 1e-6  # 指数関数の係数がこの値以上であれば表示する
#         self.make_matrix()
#
#     def make_matrix(self):
#         """与えられたパラメータから行列及び次元を決定する.
#         パラメータを与えたら実行せよ."""
#         self.dim = int(2 * self.i + 1)
#         self.ii = self.i * (self.i + 1)
#         m_spin = np.arange(self.i, -self.i - 1.0, -1.0)
#
#         self.ip = np.diag(np.sqrt(self.ii - (m_spin[0:-1]) * (m_spin[0:-1] - 1.0)), 1)
#         self.im = np.conj(self.ip.T)
#         self.ix = +0.5 * (self.ip + self.im)
#         self.iy = -0.5j * (self.ip - self.im)
#         self.iz = np.diag(m_spin)
#
#     def Hamiltonian(self):
#         nuq = np.abs(self.nuq)
#         H_Z = -self.gyr * self.h0 * (1 + self.k_shift) * (self.iz * np.cos(self.theta)
#                                                           + (self.ix * np.cos(self.phi) + self.iy * np.sin(
#                     self.phi)) * np.sin(self.theta))
#         H_Q = nuq / 6.0 \
#               * (3 * np.dot(self.iz, self.iz) - self.ii * np.identity(self.dim)
#                  + 0.5 * self.eta * (np.dot(self.ip, self.ip) + np.dot(self.im, self.im)))
#
#         return H_Z + H_Q
#
#     def EnergyLevel(self):
#         h = self.Hamiltonian()
#         self.level, self.v = np.linalg.eigh(h, 'L')
#
#         sort_index = np.argsort(self.level)
#         self.level = self.level[sort_index]
#         self.v = self.v[:, sort_index]
#
#     def TransitionProbability(self, v):
#         wx = np.dot(np.dot(np.conj(v.T), self.ix), v)
#         wy = np.dot(np.dot(np.conj(v.T), self.iy), v)
#         wz = np.dot(np.dot(np.conj(v.T), self.iz), v)
#         w = np.real(wx * np.conj(wx) + wy * np.conj(wy) + wz * np.conj(wz)) - self.ii * np.identity(self.dim)
#         return w
#
#     def FSP(self):
#         self.some_case()
#         fres = (np.abs(np.tril(self.level[:, np.newaxis] - self.level, -1)) -
#                 np.triu(np.ones((self.dim_ev, self.dim_ev)))).flatten()
#         intens = (np.tril(self.w, -1)).flatten() * 2.0 / np.ceil(self.ii)
#         print(intens)
#         # 有意なIntensityかつ正の共鳴周波数の場合にTrue
#         self.resind = (intens >= self.intens_lim) * (fres > 0.0)
#         # 有意な遷移のみ取り出して周波数でソート
#         self.sortind = np.argsort(fres[self.resind])
#         self.fres = fres[self.resind][self.sortind]
#         self.intens = intens[self.resind][self.sortind]
#
#     def some_case(self):
#         # 対角化
#         self.w = self.TransitionProbability(self.v)
#         self.evw, self.a = np.linalg.eigh(self.w)
#
#         self.sortind = np.argsort(self.evw)[-1::-1]  # エネルギー固有値 降順
#         self.evw = self.evw[self.sortind]
#         self.a = self.a[:, self.sortind]
#
#         # 場合分け
#         f_nmr = self.gyr * self.h0 * (1 + self.k_shift)
#         # NQRの場合
#         if np.abs(f_nmr) < self.eps * np.abs(self.nuq) and (self.dim % 2) * np.abs(self.eta) < self.eps:
#             if self.dim % 2 == 1:  # NQR，I: 整数，η=0
#                 self.w = np.vstack((np.zeros(self.dim), self.w))
#                 self.w = np.hstack((np.zeros((self.dim + 1, 1)), self.w))
#                 self.a = np.vstack((2.0 * self.a[0, :], self.a[1::2, :] + self.a[2::2, :]))
#                 # 係数の補正項
#                 self.c0 = np.vstack(([1 / 6.0], 0.25 * np.ones(((self.dim - 3) // 2, 1))))
#
#             else:  # NQR，I: 半奇数
#                 self.a = self.a[0::2, :] + self.a[1::2, :]
#                 self.c0 = 0.25
#
#             self.level = self.level[::2]  # 縮退エネルギーをまとめる
#             # 遷移確率の縮退準位に関する和
#             self.w = self.w[::2, ::2] + self.w[::2, 1::2] + self.w[1::2, ::2] + self.w[1::2, 1::2]
#             self.dim_ev = (self.dim + 1) // 2
#         else:  # NMR, または分裂なし
#             self.c0 = 0.5
#             self.dim_ev = self.dim
#
#     def RelaxationCurve(self):
#         # ind_t = [0, ..., 2I+1, ...(2I+1回)..., 0, ..., 2I+1]
#         ind_t = list(range(self.dim_ev)) * self.dim_ev
#         # あらゆる固有ベクトルの差を作る
#         # 第二項のindex [0, 0, ,...,0 ,..., 2I+1, ..., 2I+1]
#         self.c = self.a[ind_t, 1:] - self.a[np.reshape(ind_t, (self.dim_ev, -1)).T.flatten(), 1:]
#         self.c = self.c[self.resind, :][self.sortind, :]
#         self.c = self.c0 * (self.c * self.c)
#         cind = (np.max(self.c, axis=0) > self.eps_c)
#         self.evw = self.evw[1:][cind]
#         self.c = self.c[:, cind]
#
#     def RelaxationCurveInitial(self, n0):
#         """初期値のリスト n0に対して緩和曲線を計算する. """
#         f_nmr = self.gyr * self.h0 * (1 + self.k_shift)
#         if np.abs(f_nmr) < self.eps * np.abs(self.nuq) and (self.dim % 2) * np.abs(self.eta) < self.eps:
#             """ NQRの場合 """
#             return ValueError
#         self.dim_ev = self.dim
#         # 係数行列の計算
#         c1 = np.dot(self.a.T, n0)
#
#         ind_t = list(range(self.dim_ev)) * self.dim_ev
#         c2 = self.a[ind_t, :] - self.a[np.reshape(ind_t, (self.dim_ev, -1)).T.flatten(), :]
#         # 有意な遷移のみ残す.
#         c2 = c2[self.resind, :][self.sortind, :]
#
#         # self.c[i,j] = sum_d (C_ni-C_nj)*C_nd*n_d(0)
#         # self.c[nu,n]は遷移nuの緩和の, n番目の固有値の係数. 準位(j,k)間の遷移とは nu = j*2I + k の関係にある.
#         self.c = c2.dot(np.diag(c1))[:, 1:]
#         # self.cの形は[self.w_sort_idx.size, dim_ev]
#
#         cind = (np.max(self.c, axis=0) > self.eps_c)
#         self.evw = self.evw[1:][cind]
#         self.c = self.c[:, cind]
#
#         self.resind = self.c.sum(axis=1) > 0
#         self.fres = self.fres[self.resind]
#         self.intens = self.intens[self.resind]
#         self.c = self.c[self.resind, :]
#         self.sortind = np.argsort(self.fres)
#
#         # 規格化
#         self.c = RelaxationCurveCalculation.row_normalize(self.c)
#
#     @staticmethod
#     def row_normalize(x):
#         """np行列xを受け取り, 行ごとに規格化(sum|・| = 1)して返す."""
#         return np.linalg.inv(np.diag(abs(x).sum(axis=1))).dot(x)
#
#     def formula(self, n0=None):
#         self.EnergyLevel()
#         self.FSP()
#         if n0 is None:
#             self.RelaxationCurve()
#         else:
#             self.RelaxationCurveInitial(n0)
#         return self.formula_write()
#
#     def formula_write(self):
#         fid = []
#
#  #       pi = np.pi
#  #       fid.append("""# NMR relaxation curve
#  #           # I={0:.1f}, gamma={1:.5f} MHz/T, H0={2:.5f} T, K={3:.5f}%
#  #           # nuQ={4:.5f} MHz, eta={5:.5f}, theta={6:.5f}deg, phi={7:.5f}deg
#  #           """.format(self.i, self.gyr, self.h0, 100.0 * self.k_shift, self.nuq, self.eta,
#  #                      self.theta / pi * 180.0, self.phi / pi * 180))
#
#         # formula style
#         fid.append("[f(MHz) intensity]\n")
#         for i in range(self.sortind.size):
#             fid.append("[{0:g} {1:g}] ".format(self.fres[i], self.intens[i]))
#             printed = 0
#             for j in range(self.evw.size):
#                 if abs(self.c[i, j]) > self.eps_c:
#                     if printed == 0:
#                         printed = 1
#                     else:
#                         fid.append(" + ")
#
#                     fid.append("{0:g} * exp({1:g} * x/T1)".format(self.c[i, j], self.evw[j] / 1000))
#             fid.append("\n")
#         return ' '.join(fid)
#
#     def list(self, fid):
#         for j in range(self.evw.size):
#             fid.write(" {0:f}".format(self.evw[j]))
#         fid.write("\n")
#
#         for i in range(self.sortind.size):
#             fid.write("[{0:f} {1:f}]".format(self.fres[i], self.intens[i]))
#             for j in range(self.evw.size):
#                 fid.write(" {0:f}".format(self.c[i, j]))
#             fid.write("\n")
#
#     def write_initial_simple(self, n0):
#         """初期値を入力して, 緩和関数の文字列を出力する関数を作る."""


def main():
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    # Windowの表示
    ui = UI()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
