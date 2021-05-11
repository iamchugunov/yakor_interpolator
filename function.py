import numpy as np
from cmath import sqrt
from scipy.interpolate import pchip_interpolate

from Derivatives_lin.R import dRdk_lin
from Derivatives_lin.R import d2Rdk2_lin
from Derivatives_lin.R import d2Rdv02_lin
from Derivatives_lin.R import d2Rdalpha2_lin
from Derivatives_lin.R import d2Rdkdv0_lin
from Derivatives_lin.R import d2Rdkdalpha_lin
from Derivatives_lin.R import d2Rdv0dalpha_lin
from Derivatives_lin.R import dRdv0_lin
from Derivatives_lin.R import dRdalpha_lin
from Derivatives_lin.Vr import dVrdk_lin
from Derivatives_lin.Vr import dVrdv0_lin
from Derivatives_lin.Vr import dVrdalpha_lin
from Derivatives_lin.Vr import d2Vrdk2_lin
from Derivatives_lin.Vr import d2Vrdv02_lin
from Derivatives_lin.Vr import d2Vrdalpha2_lin
from Derivatives_lin.Vr import d2Vrdkdv0_lin
from Derivatives_lin.Vr import d2Vrdkdalpha_lin
from Derivatives_lin.Vr import d2Vrdv0dalpha_lin
from Derivatives_lin.theta import dthetadk_lin
from Derivatives_lin.theta import d2thetadalpha2_lin
from Derivatives_lin.theta import d2thetadv02_lin
from Derivatives_lin.theta import d2thetadk2_lin
from Derivatives_lin.theta import dthetadv0_lin
from Derivatives_lin.theta import d2thetadkdalpha_lin
from Derivatives_lin.theta import d2thetadkdv0_lin
from Derivatives_lin.theta import dthetadalpha_lin
from Derivatives_lin.theta import d2thetadv0dalpha_lin

from Derivatives_quad.R import dRdk
from Derivatives_quad.R import dRdv0
from Derivatives_quad.R import dRdalpha
from Derivatives_quad.R import d2Rdk2
from Derivatives_quad.R import d2Rdalpha2
from Derivatives_quad.R import d2Rdkdv0
from Derivatives_quad.R import d2Rdv02
from Derivatives_quad.R import d2Rdkdalpha
from Derivatives_quad.R import d2Rdv0dalpha
from Derivatives_quad.Vr import dVrdk
from Derivatives_quad.Vr import dVrdv0
from Derivatives_quad.Vr import dVrdalpha
from Derivatives_quad.Vr import d2Vrdk2
from Derivatives_quad.Vr import d2Vrdkdv0
from Derivatives_quad.Vr import d2Vrdalpha2
from Derivatives_quad.Vr import d2Vrdkdalpha
from Derivatives_quad.Vr import d2Vrdv0dalpha
from Derivatives_quad.Vr import d2Vrdv02
from Derivatives_quad.theta import dthetadk
from Derivatives_quad.theta import d2thetadalpha2
from Derivatives_quad.theta import d2thetadv02
from Derivatives_quad.theta import d2thetadk2
from Derivatives_quad.theta import dthetadv0
from Derivatives_quad.theta import d2thetadkdalpha
from Derivatives_quad.theta import d2thetadkdv0
from Derivatives_quad.theta import dthetadalpha
from Derivatives_quad.theta import d2thetadv0dalpha

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)


# фильтрация Калманом измерений угла
def kalman_filter_theta(x_est_prev, D_x_prev, y_meas, T, ksi_theta, theta_n1):
    F = np.array([[1, T], [0, 1]])
    G = np.array([[0, 0], [0, T]])
    H = np.array([[1, 0]])
    D_ksi = ksi_theta ** 2
    D_n = theta_n1 ** 2

    I = np.eye(len(D_x_prev))

    x_ext = F.dot(x_est_prev)
    Dx_ext = F.dot(D_x_prev).dot(F.T) + G.dot(D_ksi).dot(G.T)
    K = Dx_ext.dot(H.T) / (H.dot(Dx_ext).dot(H.T) + D_n)
    D_x = (I - K.dot(H)).dot(Dx_ext)
    x_est = x_ext + K.dot(y_meas - H.dot(x_ext))

    return x_est, D_x


# фильтрация Калманом измерений дальности и скорости
def kalman_filter_xV(x_est_prev, D_x_prev, y_meas, T, ksi_Vr, Vr_n1, Vr_n2):
    F = np.array([[1, T], [0, 1]])
    G = np.array([[0, 0], [0, T]])
    H = np.array([[1, 0], [0, 1]])

    D_ksi = ksi_Vr ** 2
    D_n = np.array([[Vr_n1 ** 2, 0], [0, Vr_n2 ** 2]])

    I = np.eye(len(D_x_prev))

    x_ext = F.dot(x_est_prev)
    Dx_ext = F.dot(D_x_prev).dot(F.T) + G.dot(D_ksi).dot(G.T)
    K = Dx_ext.dot(H.T).dot(np.linalg.inv(H.dot(Dx_ext).dot(H.T) + D_n))
    D_x = (I - K.dot(H)).dot(Dx_ext)
    x_est = x_ext + K.dot(y_meas - H.dot(x_ext))

    return x_est, D_x


# формирование фильтрованных массивов измерений
def func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr, n1, n2, ksi_theta, theta_n1):
    x_est = np.zeros([len(t_meas), 2])
    x_est_theta = np.zeros([len(t_meas), 2])

    D_x_est = 0
    D_x_est_theta = 0

    for k in range(len(t_meas)):

        if k == 0:
            x_est[k] = [R_meas[k], Vr_meas[k]]
            x_est_theta[k] = [theta_meas[k], 0.0001]
            D_x_est = np.array([[1, 0], [0, 1]])
            D_x_est_theta = np.array([[1, 0], [0, 1]])

        else:
            x_est[k], D_x_est = kalman_filter_xV(x_est[k - 1], D_x_est, np.array([R_meas[k], Vr_meas[k]]),
                                                 t_meas[k] - t_meas[k - 1], ksi_Vr, n1,
                                                 n2)

            x_est_theta[k], D_x_est_theta = kalman_filter_theta(x_est_theta[k - 1], D_x_est_theta,
                                                                theta_meas[k], t_meas[k] - t_meas[k - 1],
                                                                ksi_theta,
                                                                theta_n1)

    R_meas = x_est[:, 0]
    Vr_meas = x_est[:, 1]
    theta_meas = x_est_theta[:, 0]

    return R_meas, Vr_meas, theta_meas


# разбиение на участки для активно-реактивного снаряда - type_bullet = 6
def func_active_reactive(t_meas, R_meas, Vr_meas):
    Thres_dRdt = 2000
    Thres_dVrdt = 500

    dRdt_set = np.diff(R_meas) / np.diff(t_meas)
    dVrdt_set = np.diff(Vr_meas) / np.diff(t_meas)

    flag = 0

    t_ind_end_1part = 0
    t_ind_start_2part = 0

    for k in range(len(t_meas) - 1):

        dRdt = dRdt_set[k]
        dVrdt = dVrdt_set[k]

        if (np.abs(dVrdt) > Thres_dVrdt) and (flag == 0):
            flag = 1
            t_ind_end_1part = k

        if np.abs(dRdt) > Thres_dRdt:
            t_ind_start_2part = k + 1

    return t_ind_end_1part, t_ind_start_2part


# линейная аппроксимация измерений
def func_linear_piece_app(x_L, y_L, h_L, y_0, m, g, SKO_R, SKO_Vr, SKO_theta, k0, dR, t_meas_full,
                          R_meas_full, Vr_meas_full, theta_meas_full, winlen, step_sld, parameters_bounds):
    if winlen > 29:
        Nkol = 15
    else:
        Nkol = 5

    h_0_1 = R_meas_full[0] * np.sin(theta_meas_full[0]) + h_L
    x_0_1 = np.sqrt((R_meas_full[0] * np.cos(theta_meas_full[0])) ** 2 - y_L ** 2) + x_L

    h_0_2 = R_meas_full[1] * np.sin(theta_meas_full[1]) + h_L
    x_0_2 = np.sqrt((R_meas_full[1] * np.cos(theta_meas_full[1])) ** 2 - y_L ** 2) + x_L

    Vx0 = (x_0_2 - x_0_1) / (t_meas_full[1] - t_meas_full[0])
    Vh0 = (h_0_2 - h_0_1) / (t_meas_full[1] - t_meas_full[0])
    absV0 = np.sqrt(Vx0 ** 2 + Vh0 ** 2)

    x_est_init = [k0, absV0, dR, np.arctan((h_0_2 - h_0_1) / (x_0_2 - x_0_1))]

    if winlen > len(t_meas_full):
        WindowSet = [[1, len(t_meas_full)]]
    else:
        WindowSet = [[1, winlen]]
        u = 1

    while 1:

        lb = WindowSet[u - 1][0] + step_sld
        rb = WindowSet[u - 1][1] + step_sld
        if rb > len(t_meas_full):
            break
        else:
            WindowSet.append([lb, rb])
            u = u + 1

    if len(WindowSet) > 3:
        WindowSet[3:] = []

    meas_t_ind = []
    x_est_top = []
    xhy_0_set = []
    window_set = []
    meas_t = []

    NoW = np.fix(len(t_meas_full) / winlen)
    if (len(t_meas_full) - NoW * winlen) > Nkol:
        NoW = NoW + 1
    NoW = int(NoW)

    for w in range(NoW):

        if w == 0:

            meas_t = [i for i in range(len(t_meas_full))]
            meas_t_ind.append(meas_t)
            t_meas_t = t_meas_full
            R_meas_t = R_meas_full
            theta_meas_t = theta_meas_full
            Vr_meas_t = Vr_meas_full

        else:

            meas_t_ind.append(meas_t[window_set[w - 1][1] - 1 + meas_t_ind[w - 1][0]:])
            t_meas_t = t_meas_full[meas_t_ind[w][0]:]
            R_meas_t = R_meas_full[meas_t_ind[w][0]:]
            theta_meas_t = theta_meas_full[meas_t_ind[w][0]:]
            Vr_meas_t = Vr_meas_full[meas_t_ind[w][0]:]

        for q in range(len(WindowSet)):

            t_meas = t_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]

            if q != (len(WindowSet) - 1):
                if len(t_meas) < (WindowSet[q + 1][0] - 1):
                    break

            R_meas = R_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            theta_meas = theta_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            Vr_meas = Vr_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]

            t_meas = t_meas - t_meas[0]

            h_0 = R_meas[0] * np.sin(theta_meas[0]) + h_L
            x_0 = np.sqrt((R_meas[0] * np.cos(theta_meas[0])) ** 2 - y_L ** 2) + x_L

            xhy_0 = [x_0, h_0, y_0]

            h_0_2 = R_meas[1] * np.sin(theta_meas[1]) + h_L
            x_0_2 = np.sqrt((R_meas[1] * np.cos(theta_meas[1])) ** 2 - y_L ** 2) + x_L

            Vx0 = (x_0_2 - x_0) / (t_meas[1] - t_meas[0])
            Vh0 = (h_0_2 - h_0) / (t_meas[1] - t_meas[0])
            absV0 = np.sqrt(Vx0 ** 2 + Vh0 ** 2)

            if w == 0 and q == 0:
                x_est = x_est_init
            else:
                x_est = [k0, absV0, dR, np.arctan((h_0_2 - h_0) / (x_0_2 - x_0))]

            for p in range(30):

                d = np.zeros(4)
                dd = np.zeros((4, 4))

                for k in range(len(R_meas)):
                    t_k = t_meas[k]

                    R = np.sqrt(
                        (x_L - ((m / x_est[0]) * x_est[1] * np.cos(x_est[3]) * (
                                1 - np.exp(-x_est[0] * t_k / m)) + x_0)) ** 2 +
                        y_L ** 2 + (h_L - ((m / x_est[0]) * (x_est[1] * np.sin(x_est[3]) + (m * g) / x_est[0]) *
                                           (1 - np.exp(-x_est[0] * t_k / m)) - (m * g * t_k) / x_est[0] + h_0)) ** 2) + \
                        x_est[2]

                    theta = np.arctan((((m / x_est[0]) * (x_est[1] * np.sin(x_est[3]) + (m * g) / x_est[0]) * (
                            1 - np.exp(-x_est[0] * t_k / m)) - (m * g * t_k) / x_est[0] + h_0) - h_L) / np.sqrt(
                        (((m / x_est[0]) * x_est[1] * np.cos(x_est[3]) * (
                                1 - np.exp(-x_est[0] * t_k / m)) + x_0) - x_L) ** 2 + y_L ** 2))

                    Vr = ((x_est[1] * np.exp(-x_est[0] * t_k / m) * np.cos(x_est[3])) *
                          (((m / x_est[0]) * x_est[1] * np.cos(x_est[3]) * (
                                  1 - np.exp(-x_est[0] * t_k / m)) + x_0) - x_L) +
                          (x_est[1] * np.sin(x_est[3]) * np.exp(-x_est[0] * t_k / m) - (m * g / x_est[0]) *
                           (1 - np.exp(-x_est[0] * t_k / m))) * (((m / x_est[0]) *
                                                                  (x_est[1] * np.sin(x_est[3]) + (m * g) / x_est[0]) *
                                                                  (1 - np.exp(-x_est[0] * t_k / m)) - (m * g * t_k) /
                                                                  x_est[0] + h_0) - h_L)) / \
                         np.sqrt((x_L - ((m / x_est[0]) * x_est[1] * np.cos(x_est[3]) *
                                         (1 - np.exp(-x_est[0] * t_k / m)) + x_0)) ** 2 + y_L ** 2 + (
                                         h_L - ((m / x_est[0]) *
                                                (x_est[1] * np.sin(x_est[3]) + (m * g) / x_est[0]) * (
                                                        1 - np.exp(-x_est[0] * t_k / m)) -
                                                (m * g * t_k) / x_est[0] + h_0)) ** 2)

                    DRDk = dRdk_lin.dRdk_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g, x_est[2])
                    DRDv0 = dRdv0_lin.dRdv0_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                x_est[2])
                    DRDdeltaR = 1
                    DRDalpha = dRdalpha_lin.dRdalpha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                         g, x_est[2])

                    D2RDk2 = d2Rdk2_lin.dRdalpha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                     x_est[2])
                    D2RDv02 = d2Rdv02_lin.d2Rdv02_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                      x_est[2])
                    D2RDdeltaR2 = 0
                    D2RDalpha2 = d2Rdalpha2_lin.d2Rdaplpa2_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                               h_0, m, g, x_est[2])

                    D2RDkDv0 = d2Rdkdv0_lin.d2rdkdv0_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                         g, x_est[2])
                    D2RDkDdeltaR = 0
                    D2RDkDalpha = d2Rdkdalpha_lin.d2Rdkdalpha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                                  h_0, m, g, x_est[2])
                    D2RDv0DdeltaR = 0
                    D2RDv0Dalpha = d2Rdv0dalpha_lin.d2Rdv0dalpha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                                     x_0, h_0, m, g, x_est[2])
                    D2RDdeltaRDalpha = 0

                    DVrDk = dVrdk_lin.dVrdk_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                x_est[2])
                    DVrDv0 = dVrdv0_lin.dVrdv0_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                   x_est[2])
                    DVrDdeltaR = 0
                    DVrDalpha = dVrdalpha_lin.dVrdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                        g, x_est[2])

                    D2VrDk2 = d2Vrdk2_lin.d2Vrdk2_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                      x_est[2])
                    D2VrDv02 = d2Vrdv02_lin.d2Vrdv02_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                         g, x_est[2])
                    D2VrDdeltaR2 = 0
                    D2VrDalpha2 = d2Vrdalpha2_lin.d2Vrdalpha2_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                                  h_0, m, g, x_est[2])

                    D2VrDkDv0 = d2Vrdkdv0_lin.d2Vrdkdv0_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                            m, g, x_est[2])
                    D2VrDkDdeltaR = 0
                    D2VrDkDalpha = d2Vrdkdalpha_lin.d2Vrdkdalha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                                    x_0, h_0, m, g, x_est[2])
                    D2VrDv0DdeltaR = 0
                    D2VrDv0Dalpha = d2Vrdv0dalpha_lin.d2Vrdv0alpha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                                       x_0, h_0, m, g, x_est[2])
                    D2VrDdeltaRDalpha = 0

                    DthetaDk = dthetadk_lin.dthetadk_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                         m, g,
                                                         x_est[2])

                    DthetaDv0 = dthetadv0_lin.dthetadv0_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                            h_0,
                                                            m, g, x_est[2])
                    DthetaDdeltaR = 0
                    DthetaDalpha = dthetadalpha_lin.dthetadalpha_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3],
                                                                     x_est[0], x_0, h_0,
                                                                     m, g, x_est[2])

                    D2thetaDk2 = d2thetadk2_lin.d2thetadk2_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                               x_0, h_0,
                                                               m, g, x_est[2])
                    D2thetaDv02 = d2thetadv02_lin.d2thetadv02_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                                  x_0, h_0,
                                                                  m, g, x_est[2])
                    D2thetaDdeltaR2 = 0
                    D2thetaDalpha2 = d2thetadalpha2_lin.d2thetadalpha2_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3],
                                                                           x_est[0], x_0, h_0,
                                                                           m, g, x_est[2])

                    D2thetaDkDv0 = d2thetadkdv0_lin.d2thetadkdv0_lin(x_L, y_L, h_L, t_k, x_est[1], x_est[3],
                                                                     x_est[0], x_0, h_0,
                                                                     m, g, x_est[2])
                    D2thetaDkDdeltaR = 0
                    D2thetaDkDalpha = d2thetadkdalpha_lin.d2thetadkdalpha_lin(x_L, y_L, h_L, t_k, x_est[1],
                                                                              x_est[3], x_est[0], x_0,
                                                                              h_0,
                                                                              m, g, x_est[2])
                    D2thetaDv0DdeltaR = 0
                    D2thetaDv0Dalpha = d2thetadv0dalpha_lin.d2thetadv0dalpha_lin(x_L, y_L, h_L, t_k, x_est[1],
                                                                                 x_est[3], x_est[0],
                                                                                 x_0, h_0, m, g, x_est[2])
                    D2thetaDdeltaRDalpha = 0

                    d[0] = d[0] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDk + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDk + (1 / SKO_theta ** 2) * (
                                   theta_meas[k] - theta) * DthetaDk

                    d[1] = d[1] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDv0 + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDv0 + (1 / SKO_theta ** 2) * (
                                   theta_meas[k] - theta) * DthetaDv0
                    d[2] = d[2] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDdeltaR + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDdeltaR + (1 / SKO_theta ** 2) * (
                                   theta_meas[k] - theta) * DthetaDdeltaR
                    d[3] = d[3] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDalpha + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDalpha + (1 / SKO_theta ** 2) * (
                                   theta_meas[k] - theta) * DthetaDalpha

                    dd[0, 0] = dd[0, 0] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDk2 - DRDk ** 2) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDk2 - DVrDk ** 2) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDk2 - DthetaDk ** 2)
                    dd[1, 1] = dd[1, 1] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDv02 - DRDv0 ** 2) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDv02 - DVrDv0 ** 2) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDv02 - DthetaDv0 ** 2)
                    dd[2, 2] = dd[2, 2] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDdeltaR2 - DRDdeltaR ** 2) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDdeltaR2 - DVrDdeltaR ** 2) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDdeltaR2 - DthetaDdeltaR ** 2)
                    dd[3, 3] = dd[3, 3] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDalpha2 - DRDalpha ** 2) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDalpha2 - DVrDalpha ** 2) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDalpha2 - DthetaDalpha ** 2)

                    dd[0, 1] = dd[0, 1] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDkDv0 - DRDk * DRDv0) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDkDv0 - DVrDk * DVrDv0) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDkDv0 - DthetaDk * DthetaDv0)
                    dd[0, 2] = dd[0, 2] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDkDdeltaR - DRDk * DRDdeltaR) + (1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDkDdeltaR - DVrDk * DVrDdeltaR) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDkDdeltaR - DthetaDk * DthetaDdeltaR)
                    dd[0, 3] = dd[0, 3] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDkDalpha - DRDk * DRDalpha) + (1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDkDalpha - DVrDk * DVrDalpha) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDkDalpha - DthetaDk * DthetaDalpha)

                    dd[1, 2] = dd[1, 2] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDv0DdeltaR - DRDv0 * DRDdeltaR) + (1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDv0DdeltaR - DVrDv0 * DVrDdeltaR) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDv0DdeltaR - DthetaDv0 * DthetaDdeltaR)
                    dd[1, 3] = dd[1, 3] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDv0Dalpha - DRDv0 * DRDalpha) + (1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDv0Dalpha - DVrDv0 * DVrDalpha) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDv0Dalpha - DthetaDv0 * DthetaDalpha)

                    dd[2, 3] = dd[2, 3] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDdeltaRDalpha - DRDdeltaR * DRDalpha) + (
                                       1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDdeltaRDalpha - DVrDdeltaR * DVrDalpha) + (
                                       1 / SKO_theta ** 2) * ((theta_meas[k] - theta)
                                                              * D2thetaDdeltaRDalpha - DthetaDdeltaR * DthetaDalpha)

                    dd[1, 0] = dd[0, 1]
                    dd[2, 0] = dd[0, 2]
                    dd[3, 0] = dd[0, 3]
                    dd[2, 1] = dd[1, 2]
                    dd[3, 1] = dd[1, 3]
                    dd[3, 2] = dd[2, 3]

                dd_dd = np.dot(np.linalg.inv(dd), d)
                x_est = x_est - dd_dd

            if np.isreal(x_est[0]) and np.isreal(x_est[1]) and np.isreal(x_est[2]) and np.isreal(x_est[3]):
                if ((x_est[0] > parameters_bounds[0][0] and x_est[0] < parameters_bounds[0][1]) and
                        (x_est[1] > parameters_bounds[1][0] and x_est[1] < parameters_bounds[1][1]) and
                        (x_est[2] > parameters_bounds[2][0] and x_est[2] < parameters_bounds[2][1]) and
                        (x_est[3] > parameters_bounds[3][0] and x_est[3] < parameters_bounds[3][1])):
                    xhy_0_set.append(xhy_0)
                    x_est_top.append(x_est)
                    window_set.append(WindowSet[q])
                    print(window_set[w], x_est_top[w], xhy_0_set[w], meas_t_ind[w])
                    break

    return xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas_full, R_meas_full, Vr_meas_full, theta_meas_full


# квадратичная аппроксимация
def func_quad_piece_app(x_L, y_L, h_L, y_0, m, g, SKO_R, SKO_Vr, SKO_theta, k0, dR, t_meas_full,
                        R_meas_full, Vr_meas_full, theta_meas_full, winlen, step_sld, parameters_bounds):
    if winlen > 29:
        Nkol = 15
    else:
        Nkol = 5

    h_0_1 = R_meas_full[0] * np.sin(theta_meas_full[0]) + h_L
    x_0_1 = np.sqrt((R_meas_full[0] * np.cos(theta_meas_full[0])) ** 2 - y_L ** 2) + x_L

    h_0_2 = R_meas_full[1] * np.sin(theta_meas_full[1]) + h_L
    x_0_2 = np.sqrt((R_meas_full[1] * np.cos(theta_meas_full[1])) ** 2 - y_L ** 2) + x_L

    Vx0 = (x_0_2 - x_0_1) / (t_meas_full[1] - t_meas_full[0])
    Vh0 = (h_0_2 - h_0_1) / (t_meas_full[1] - t_meas_full[0])
    absV0 = np.sqrt(Vx0 ** 2 + Vh0 ** 2)

    x_est_init = [k0, absV0, dR, np.arctan((h_0_2 - h_0_1) / (x_0_2 - x_0_1))]

    u = 0

    if winlen > len(t_meas_full):
        WindowSet = [[1, len(t_meas_full)]]
    else:
        WindowSet = [[1, winlen]]
        u = 1

    while 1:

        lb = WindowSet[u - 1][0] + step_sld
        rb = WindowSet[u - 1][1] + step_sld
        if rb > len(t_meas_full):
            break
        else:
            WindowSet.append([lb, rb])
            u = u + 1

    if len(WindowSet) > 3:
        WindowSet[3:] = []

    meas_t_ind = []
    x_est_top = []
    xhy_0_set = []
    window_set = []
    meas_t = []

    NoW = np.fix(len(t_meas_full) / winlen)

    if (len(t_meas_full) - NoW * winlen) > Nkol:
        NoW = NoW + 1
    NoW = int(NoW)

    for w in range(NoW):

        if w == 0:

            meas_t = [i for i in range(len(t_meas_full))]
            meas_t_ind.append(meas_t)
            t_meas_t = t_meas_full
            R_meas_t = R_meas_full
            theta_meas_t = theta_meas_full
            Vr_meas_t = Vr_meas_full

        else:

            meas_t_ind.append(meas_t[window_set[w - 1][1] - 1 + meas_t_ind[w - 1][0]:])
            t_meas_t = t_meas_full[meas_t_ind[w][0]:]
            R_meas_t = R_meas_full[meas_t_ind[w][0]:]
            theta_meas_t = theta_meas_full[meas_t_ind[w][0]:]
            Vr_meas_t = Vr_meas_full[meas_t_ind[w][0]:]

        for q in range(len(WindowSet)):

            t_meas = t_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]

            if q != (len(WindowSet) - 1):
                if len(t_meas) < (WindowSet[q + 1][0] - 1):
                    break

            R_meas = R_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            theta_meas = theta_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            Vr_meas = Vr_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]

            t_meas = t_meas - t_meas[0]

            h_0 = R_meas[0] * np.sin(theta_meas[0]) + h_L
            x_0 = np.sqrt((R_meas[0] * np.cos(theta_meas[0])) ** 2 - y_L ** 2) + x_L

            xhy_0 = [x_0, h_0, y_0]

            h_0_2 = R_meas[1] * np.sin(theta_meas[1]) + h_L
            x_0_2 = np.sqrt((R_meas[1] * np.cos(theta_meas[1])) ** 2 - y_L ** 2) + x_L

            Vx0 = (x_0_2.real - x_0.real) / (t_meas[1] - t_meas[0])
            Vh0 = (h_0_2 - h_0) / (t_meas[1] - t_meas[0])
            absV0 = np.sqrt(Vx0 ** 2 + Vh0 ** 2)

            if w == 0 and q == 0:
                x_est = x_est_init
            else:
                x_est = [k0, absV0, dR, np.arctan((h_0_2 - h_0) / (x_0_2 - x_0))]

            for p in range(30):

                d = np.zeros(4)
                dd = np.zeros((4, 4))

                for k in range(len(R_meas)):
                    t_k = t_meas[k]

                    R = np.sqrt(
                        (x_L - (x_0 + (m / x_est[0]) * np.log(
                            1 + (x_est[0] * x_est[1] * t_k * np.cos(x_est[3])) / m))) ** 2 + y_L ** 2 + (
                                h_L - (h_0 + (m / x_est[0]) * np.log(
                            np.cos(t_k * np.sqrt(x_est[0] * g / m)) + np.sqrt(x_est[0] / (m * g)) * x_est[1] * np.sin(
                                x_est[3]) * np.sin(
                                t_k * np.sqrt(x_est[0] * g / m))))) ** 2) + x_est[2]

                    Vr = ((x_est[1] * np.cos(x_est[3]) * (
                            (x_0 + (m / x_est[0]) * np.log(
                                1 + (x_est[0] * x_est[1] * t_k * np.cos(x_est[3])) / m)) - x_L)) / (
                                  1 + (x_est[0] * t_k * x_est[1] * np.cos(x_est[3])) / m) + (
                                  (np.sqrt(m * g * x_est[0]) *
                                   x_est[1] * np.sin(
                                              x_est[3]) - m * g * np.tan(np.sqrt(x_est[0] * g / m) * t_k)) / (
                                          np.sqrt(m * g * x_est[0]) +
                                          x_est[0]
                                          * x_est[1] * np.sin(
                                      x_est[3]) * np.tan(np.sqrt(x_est[0] * g / m) * t_k))) * (
                                  (h_0 + (m / x_est[0]) * np.log(
                                      np.cos(np.sqrt(x_est[0] * g / m) * t_k) + np.sqrt(x_est[0] / (m * g)) * x_est[
                                          1] * np.sin(
                                          x_est[3]) * np.sin(
                                          np.sqrt(x_est[0] * g / m) * t_k))) - h_L)) / (
                             np.sqrt((x_L - (x_0 + (m / x_est[0]) * np.log(
                                 1 + (x_est[0] * x_est[1] * t_k * np.cos(x_est[3])) / m))) ** 2 + y_L ** 2 + (
                                             h_L - (
                                             h_0 + (m / x_est[0]) * np.log(
                                         np.cos(np.sqrt(x_est[
                                                            0] * g / m) * t_k) + np.sqrt(
                                             x_est[0] / (m * g)) * x_est[
                                             1] * np.sin(
                                             x_est[3]) * np.sin(
                                             np.sqrt(x_est[
                                                         0] * g / m) * t_k)))) ** 2))

                    theta = np.arctan(((h_0 + (m / x_est[0]) * np.log(
                        np.cos(t_k * np.sqrt(x_est[0] * g / m)) + np.sqrt(x_est[0] / (m * g)) * x_est[1] * np.sin(
                            x_est[3]) * np.sin(
                            t_k * np.sqrt(x_est[0] * g / m)))) - h_L) / np.sqrt(
                        (x_L - (x_0 + (m / x_est[0]) * np.log(
                            1 + (x_est[0] * x_est[1] * t_k * np.cos(x_est[3])) / m))) ** 2 + y_L ** 2))

                    DRDk = dRdk.dRdk(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g, x_est[2])

                    DRDv0 = dRdv0.dRdv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g, x_est[2])

                    DRDdeltaR = 1

                    DRDalpha = dRdalpha.dRdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                 x_est[2])

                    D2RDk2 = d2Rdk2.d2Rdk2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g, x_est[2])
                    D2RDv02 = d2Rdv02.d2Rdv02(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                              x_est[2])
                    D2RDdeltaR2 = 0
                    D2RDalpha2 = d2Rdalpha2.d2Rdalpha2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                       x_est[2])

                    D2RDkDv0 = d2Rdkdv0.d2Rdkdv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                 x_est[2])
                    D2RDkDdeltaR = 0
                    D2RDkDalpha = d2Rdkdalpha.d2Rdkdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                          g,
                                                          x_est[2])
                    D2RDv0DdeltaR = 0
                    D2RDv0Dalpha = d2Rdv0dalpha.d2Rdv0dalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                             m,
                                                             g, x_est[2])
                    D2RDdeltaRDalpha = 0

                    DVrDk = dVrdk.dVrdk(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g, x_est[2])
                    DVrDv0 = dVrdv0.dVrdv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g, x_est[2])
                    DVrDdeltaR = 0
                    DVrDalpha = dVrdalpha.dVrdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                    x_est[2])

                    D2VrDk2 = d2Vrdk2.d2Vrdk2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                              x_est[2])
                    D2VrDv02 = d2Vrdv02.d2Vrdv02(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                 x_est[2])
                    D2VrDdeltaR2 = 0
                    D2VrDalpha2 = d2Vrdalpha2.d2Vrdalpha2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                          g,
                                                          x_est[2])

                    D2VrDkDv0 = d2Vrdkdv0.d2Vrdkdv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                    x_est[2])
                    D2VrDkDdeltaR = 0
                    D2VrDkDalpha = d2Vrdkdalpha.d2Vrdkdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                             m,
                                                             g, x_est[2])
                    D2VrDv0DdeltaR = 0
                    D2VrDv0Dalpha = d2Vrdv0dalpha.d2Vrdv0dalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                                h_0,
                                                                m, g, x_est[2])
                    D2VrDdeltaRDalpha = 0

                    DthetaDdeltaR = 0

                    DthetaDalpha = dthetadalpha.dthetadalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                             m,
                                                             g, x_est[2])

                    DthetaDk = dthetadk.dthetadk(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                 x_est[2])
                    DthetaDv0 = dthetadv0.dthetadv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                    g,
                                                    x_est[2])

                    D2thetaDk2 = d2thetadk2.d2thetadk2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                       m, g,
                                                       x_est[2])
                    D2thetaDv02 = d2thetadv02.d2thetadv02(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                          h_0, m,
                                                          g,
                                                          x_est[2])
                    D2thetaDdeltaR2 = 0
                    D2thetaDalpha2 = d2thetadalpha2.d2theradalpha2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                                   x_0,
                                                                   h_0, m, g, x_est[2])

                    D2thetaDkDv0 = d2thetadkdv0.d2thetadkdv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0,
                                                             h_0,
                                                             m,
                                                             g, x_est[2])
                    D2thetaDkDdeltaR = 0
                    D2thetaDkDalpha = d2thetadkdalpha.d2thetadkdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3],
                                                                      x_est[0],
                                                                      x_0,
                                                                      h_0, m, g, x_est[2])
                    D2thetaDv0DdeltaR = 0
                    D2thetaDv0Dalpha = d2thetadv0dalpha.d2thetadv0dalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3],
                                                                         x_est[0],
                                                                         x_0, h_0, m, g, x_est[2])
                    D2thetaDdeltaRDalpha = 0

                    d[0] = d[0] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDk + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDk + (1 / SKO_theta ** 2) * (theta_meas[k] - theta) * DthetaDk
                    d[1] = d[1] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDv0 + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDv0 + (1 / SKO_theta ** 2) * (theta_meas[k] - theta) * DthetaDv0
                    d[2] = d[2] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDdeltaR + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDdeltaR + (1 / SKO_theta ** 2) * (
                                   theta_meas[k] - theta) * DthetaDdeltaR
                    d[3] = d[3] + (1 / SKO_R ** 2) * (R_meas[k] - R) * DRDalpha + (1 / SKO_Vr ** 2) * (
                            Vr_meas[k] - Vr) * DVrDalpha + (1 / SKO_theta ** 2) * (
                                   theta_meas[k] - theta) * DthetaDalpha

                    dd[0, 0] = dd[0, 0] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDk2 - DRDk ** 2) + (
                            1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDk2 - DVrDk ** 2) + (1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDk2 - DthetaDk ** 2)
                    dd[1, 1] = dd[1, 1] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDv02 - DRDv0 ** 2) + (
                            1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDv02 - DVrDv0 ** 2) + (1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDv02 - DthetaDv0 ** 2)
                    dd[2, 2] = dd[2, 2] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDdeltaR2 - DRDdeltaR ** 2) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDdeltaR2 - DVrDdeltaR ** 2) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDdeltaR2 - DthetaDdeltaR ** 2)
                    dd[3, 3] = dd[3, 3] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDalpha2 - DRDalpha ** 2) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDalpha2 - DVrDalpha ** 2) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDalpha2 - DthetaDalpha ** 2)

                    dd[0, 1] = dd[0, 1] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDkDv0 - DRDk * DRDv0) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDkDv0 - DVrDk * DVrDv0) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDkDv0 - DthetaDk * DthetaDv0)
                    dd[0, 2] = dd[0, 2] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDkDdeltaR - DRDk * DRDdeltaR) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDkDdeltaR - DVrDk * DVrDdeltaR) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDkDdeltaR - DthetaDk * DthetaDdeltaR)
                    dd[0, 3] = dd[0, 3] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDkDalpha - DRDk * DRDalpha) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDkDalpha - DVrDk * DVrDalpha) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDkDalpha - DthetaDk * DthetaDalpha)

                    dd[1, 2] = dd[1, 2] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDv0DdeltaR - DRDv0 * DRDdeltaR) + (
                                       1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDv0DdeltaR - DVrDv0 * DVrDdeltaR) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDv0DdeltaR - DthetaDv0 * DthetaDdeltaR)
                    dd[1, 3] = dd[1, 3] + (1 / SKO_R ** 2) * ((R_meas[k] - R) * D2RDv0Dalpha - DRDv0 * DRDalpha) + (
                            1 / SKO_Vr ** 2) * ((Vr_meas[k] - Vr) * D2VrDv0Dalpha - DVrDv0 * DVrDalpha) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[k] - theta) * D2thetaDv0Dalpha - DthetaDv0 * DthetaDalpha)

                    dd[2, 3] = dd[2, 3] + (1 / SKO_R ** 2) * (
                            (R_meas[k] - R) * D2RDdeltaRDalpha - DRDdeltaR * DRDalpha) + (
                                       1 / SKO_Vr ** 2) * (
                                       (Vr_meas[k] - Vr) * D2VrDdeltaRDalpha - DVrDdeltaR * DVrDalpha) + (
                                       1 / SKO_theta ** 2) * (
                                       (theta_meas[
                                            k] - theta) * D2thetaDdeltaRDalpha - DthetaDdeltaR * DthetaDalpha)

                    dd[1, 0] = dd[0, 1]
                    dd[2, 0] = dd[0, 2]
                    dd[3, 0] = dd[0, 3]
                    dd[2, 1] = dd[1, 2]
                    dd[3, 1] = dd[1, 3]
                    dd[3, 2] = dd[2, 3]

                dd_dd = np.dot(np.linalg.inv(dd), d)
                x_est = x_est - dd_dd

            if np.isreal(x_est[0]) and np.isreal(x_est[1]) and np.isreal(x_est[2]) and np.isreal(x_est[3]):
                if ((x_est[0] > parameters_bounds[0][0] and x_est[0] < parameters_bounds[0][1]) and
                        (x_est[1] > parameters_bounds[1][0] and x_est[1] < parameters_bounds[1][1]) and
                        (x_est[2] > parameters_bounds[2][0] and x_est[2] < parameters_bounds[2][1]) and
                        (x_est[3] > parameters_bounds[3][0] and x_est[3] < parameters_bounds[3][1])):
                    xhy_0_set.append(xhy_0)
                    x_est_top.append(x_est)
                    window_set.append(WindowSet[q])
                    print(window_set[w], x_est_top[w], xhy_0_set[w], meas_t_ind[w])
                    break

    return xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas_full, R_meas_full, Vr_meas_full, theta_meas_full


# оценка измерений для линейной аппроксимации
def func_linear_piece_estimation(xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, m, g, x_L, y_L, h_L):
    N = 300
    t_meas_plot = []
    x_tr_er_plot = []
    h_tr_er_plot = []
    R_est_full_plot = []
    Vr_est_full_plot = []
    theta_est_full_plot = []
    Vx_true_er_plot = []
    Vh_true_er_plot = []
    V_abs_est_plot = []

    alpha_tr_er_plot = []
    A_abs_est_plot = []
    Ax_true_er_plot = []
    Ah_true_er_plot = []

    for s in range(len(x_est_top)):

        x_est_fin = x_est_top[s]

        if s == (len(x_est_top) - 1):
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]
            tmax = t_meas[meas_t_ind[s][-1]]
            t = []
            n = 0
            for i in range(N):
                if i == 0:
                    n = 0
                else:
                    n += (tmax - tmin) / (N - 1)
                t.append(n)
            t = np.array(t)

        else:
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]
            tmax = t_meas[meas_t_ind[s][window_set[s][1] - 1]]
            t = []
            n = 0
            for i in range(N):
                if i == 0:
                    n = 0
                else:
                    n += (tmax - tmin) / (N - 1)
                t.append(n)
            t = np.array(t)

        t_meas_plot.append(t + tmin)

        x_0 = xhy_0_set[s][0]
        h_0 = xhy_0_set[s][1]

        x_tr_er = np.zeros(len(t))
        h_tr_er = np.zeros(len(t))
        R_est_full = np.zeros(len(t))
        theta_est_full = np.zeros(len(t))
        Vr_est_full = np.zeros(len(t))
        V_abs_est = np.zeros(len(t))
        Vx_true_er = np.zeros(len(t))
        Vh_true_er = np.zeros(len(t))

        alpha_tr_er = np.zeros(len(t))
        A_abs_est = np.zeros(len(t))
        Ax_true_er = np.zeros(len(t))
        Ah_true_er = np.zeros(len(t))

        k0 = x_est_fin[0]
        v0 = x_est_fin[1]
        dR = x_est_fin[2]
        alpha = x_est_fin[3]

        for k in range(len(t)):

            x_tr_er[k] = x_0 + (m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m))

            h_tr_er[k] = ((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) * (
                    1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0)

            Vx_true_er[k] = v0 * np.exp(-k0 * t[k] / m) * np.cos(alpha)
            Vh_true_er[k] = v0 * np.sin(alpha) * np.exp(-k0 * t[k] / m) - (m * g / k0) * (1 - np.exp(-k0 * t[k] / m))

            R_est_full[k] = np.sqrt(
                (x_L - ((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0)) ** 2 +
                y_L ** 2 + (h_L - ((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                   (1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0)) ** 2) + dR

            theta_est_full[k] = np.arctan((((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                            (1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0) - h_L) / np.sqrt(
                (((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0) - x_L) ** 2 + y_L ** 2))

            Vr_est_full[k] = ((v0 * np.exp(-k0 * t[k] / m) * np.cos(alpha)) *
                              (((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0) - x_L) +
                              (v0 * np.sin(alpha) * np.exp(-k0 * t[k] / m) - (m * g / k0) *
                               (1 - np.exp(-k0 * t[k] / m))) * (((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                                                 (1 - np.exp(-k0 * t[k] / m)) - (
                                                                         m * g * t[k]) / k0 + h_0) - h_L)) / \
                             np.sqrt((x_L - ((m / k0) * v0 * np.cos(alpha) *
                                             (1 - np.exp(-k0 * t[k] / m)) + x_0)) ** 2 + y_L ** 2 + (h_L - ((m / k0) *
                                                                                                            (
                                                                                                                    v0 * np.sin(
                                                                                                                alpha) + (
                                                                                                                            m * g) / k0) * (
                                                                                                                    1 - np.exp(
                                                                                                                -k0 *
                                                                                                                t[
                                                                                                                    k] / m)) - (
                                                                                                                    m * g *
                                                                                                                    t[
                                                                                                                        k]) / k0 + h_0)) ** 2)

            V_abs_est[k] = np.sqrt(Vx_true_er[k] ** 2 + Vh_true_er[k] ** 2)

            alpha_tr_er[k] = np.arctan(Vh_true_er[k] / Vx_true_er[k])

            if k != 0:
                A_abs_est[k] = (V_abs_est[k] - V_abs_est[k - 1]) / (t[k] - t[k - 1])
                Ax_true_er[k] = (Vx_true_er[k] - Vx_true_er[k - 1]) / (t[k] - t[k - 1])
                Ah_true_er[k] = (Vh_true_er[k] - Vh_true_er[k - 1]) / (t[k] - t[k - 1])

        A_abs_est[0] = A_abs_est[1]
        Ax_true_er[0] = Ax_true_er[1]
        Ah_true_er[0] = Ah_true_er[1]

        x_tr_er_plot.append(x_tr_er)
        h_tr_er_plot.append(h_tr_er)
        R_est_full_plot.append(R_est_full)
        Vr_est_full_plot.append(Vr_est_full)
        theta_est_full_plot.append(theta_est_full)

        Vx_true_er_plot.append(Vx_true_er)
        Vh_true_er_plot.append(Vh_true_er)
        V_abs_est_plot.append(V_abs_est)

        alpha_tr_er_plot.append(alpha_tr_er)
        A_abs_est_plot.append(A_abs_est)
        Ax_true_er_plot.append(Ax_true_er)
        Ah_true_er_plot.append(Ah_true_er)

    return t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
           Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
           Ah_true_er_plot


# оценка измерений для квадратичной аппроксимации
def func_quad_piece_estimation(xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, m, g, x_L, y_L, h_L):
    N = 300
    t_meas_plot = []
    x_tr_er_plot = []
    h_tr_er_plot = []
    R_est_full_plot = []
    Vr_est_full_plot = []
    Vx_true_er_plot = []
    Vh_true_er_plot = []
    theta_est_full_plot = []
    V_abs_est_plot = []

    alpha_tr_er_plot = []
    A_abs_est_plot = []
    Ax_true_er_plot = []
    Ah_true_er_plot = []

    for s in range(len(x_est_top)):
        x_est_fin = x_est_top[s]

        if s == (len(x_est_top) - 1):
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]
            tmax = t_meas[meas_t_ind[s][-1]]
            t = []
            n = 0
            for i in range(N):
                if i == 0:
                    n = 0
                else:
                    n += (tmax - tmin) / (N - 1)
                t.append(n)
            t = np.array(t)

        else:
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]
            tmax = t_meas[meas_t_ind[s][window_set[s][1] - 1]]
            t = []
            n = 0
            for i in range(N):
                if i == 0:
                    n = 0
                else:
                    n += (tmax - tmin) / (N - 1)
                t.append(n)
            t = np.array(t)

        t_meas_plot.append(t + tmin)

        x_0 = xhy_0_set[s][0].real
        h_0 = xhy_0_set[s][1].real

        x_tr_er = np.zeros(len(t))
        h_tr_er = np.zeros(len(t))
        R_est_full = np.zeros(len(t))
        theta_est_full = np.zeros(len(t))
        Vr_est_full = np.zeros(len(t))
        V_abs_est = np.zeros(len(t))
        Vx_true_er = np.zeros(len(t))
        Vh_true_er = np.zeros(len(t))

        alpha_tr_er = np.zeros(len(t))
        A_abs_est = np.zeros(len(t))
        Ax_true_er = np.zeros(len(t))
        Ah_true_er = np.zeros(len(t))

        k0 = x_est_fin[0].real
        v0 = x_est_fin[1].real
        dR = x_est_fin[2].real
        alpha = x_est_fin[3].real

        for k in range(len(t)):

            x_tr_er[k] = (m / k0) * np.log(
                1 + (k0 / m) * v0 * t[k] * np.cos(alpha)) + x_0

            h_tr_er[k] = (m / k0) * np.log(
                np.cos(np.sqrt((k0 * g) / m) * t[k]) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                    alpha) * np.sin(np.sqrt((k0 * g) / m) * t[k])) + h_0

            R_est_full[k] = np.sqrt((x_L - (x_0 + (m / k0) * np.log(
                1 + (k0 * v0 * t[k] * np.cos(alpha)) / m))) ** 2 + y_L + (
                                            h_L - (
                                            h_0 + (m / k0) * np.log(
                                        np.cos(t[k] * np.sqrt(k0 * g / m)) + np.sqrt(
                                            k0 / (m * g)) * v0 * np.sin(
                                            alpha) * np.sin(
                                            t[k] * np.sqrt(k0 * g / m))))) ** 2) + dR

            theta_est_full[k] = np.arctan(((h_0 + (m / k0) * np.log(
                np.cos(t[k] * np.sqrt(k0 * g / m)) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                    alpha) * np.sin(t[k] * np.sqrt(k0 * g / m)))) - h_L) / np.sqrt((x_L - (
                    x_0 + (m / k0) * np.log(
                1 + (k0 * v0 * t[k] * np.cos(alpha)) / m))) ** 2 + y_L ** 2))

            Vr_est_full[k] = ((v0 * np.cos(alpha) * ((x_0 + (m / k0) * np.log(
                1 + (k0 * v0 * t[k] * np.cos(alpha)) / m)) - x_L)) / (1 + (
                    k0 * t[k] * v0 * np.cos(alpha)) / m) + ((np.sqrt(
                m * g * k0) * v0 * np.sin(alpha) - m * g * np.tan(
                np.sqrt(k0 * g / m) * t[k])) / (np.sqrt(m * g * k0) + k0 *
                                                v0 * np.sin(
                        alpha) * np.tan(np.sqrt(k0 * g / m) * t[k]))) * (
                                      (h_0 + (m / k0) * np.log(
                                          np.cos(np.sqrt(k0 * g / m) * t[k]) + np.sqrt(
                                              k0 / (m * g)) * v0 * np.sin(
                                              alpha) * np.sin(
                                              np.sqrt(k0 * g / m) * t[k]))) - h_L)) / (
                                 np.sqrt((x_L - (
                                         x_0 + (m / k0) * np.log(
                                     1 + (k0 * v0 * t[k] * np.cos(
                                         alpha)) / m))) ** 2 + y_L ** 2 + (h_L - (
                                         h_0 + (m / k0) * np.log(
                                     np.cos(np.sqrt(k0 * g / m) * t[k]) + np.sqrt(
                                         k0 / (m * g)) * v0 * np.sin(
                                         alpha) * np.sin(
                                         np.sqrt(k0 * g / m) * t[k])))) ** 2))

            Vx_true_er[k] = (v0 * np.cos(alpha)) / (
                    1 + k0 * v0 * t[k] * np.cos(alpha) / m)

            Vh_true_er[k] = (np.sqrt(m * g * k0) * v0 * np.sin(
                alpha) - m * g * np.tan(
                np.sqrt((k0 * g) / m) * t[k])) / (
                                    np.sqrt(m * g * k0) + k0 * v0 * np.sin(
                                alpha) * np.tan(np.sqrt((k0 * g) / m) * t[k]))

            V_abs_est[k] = np.sqrt(Vx_true_er[k] ** 2 + Vh_true_er[k] ** 2)

            alpha_tr_er[k] = np.arctan(Vh_true_er[k] / Vx_true_er[k])

            if k != 0:
                A_abs_est[k] = (V_abs_est[k] - V_abs_est[k - 1]) / (t[k] - t[k - 1])
                Ax_true_er[k] = (Vx_true_er[k] - Vx_true_er[k - 1]) / (t[k] - t[k - 1])
                Ah_true_er[k] = (Vh_true_er[k] - Vh_true_er[k - 1]) / (t[k] - t[k - 1])

        A_abs_est[0] = A_abs_est[1]
        Ax_true_er[0] = Ax_true_er[1]
        Ah_true_er[0] = Ah_true_er[1]

        x_tr_er_plot.append(x_tr_er)
        h_tr_er_plot.append(h_tr_er)
        R_est_full_plot.append(R_est_full)
        Vr_est_full_plot.append(Vr_est_full)
        Vx_true_er_plot.append(Vx_true_er)
        Vh_true_er_plot.append(Vh_true_er)
        theta_est_full_plot.append(theta_est_full)
        V_abs_est_plot.append(V_abs_est)

        alpha_tr_er_plot.append(alpha_tr_er)
        A_abs_est_plot.append(A_abs_est)
        Ax_true_er_plot.append(Ax_true_er)
        Ah_true_er_plot.append(Ah_true_er)

    return t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
           Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
           Ah_true_er_plot


# оцененные значения для линейной аппроксимации в момент измерений
def func_linear_piece_estimation_error(xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, R_meas, Vr_meas,
                                       theta_meas, m, g, x_L, y_L, h_L):
    t_err_plot = []
    R_est_full_plot = []
    Vr_est_full_plot = []
    theta_est_full_plot = []
    R_est_err_plot = []
    Vr_est_err_plot = []
    theta_est_err_plot = []

    for s in range(len(x_est_top)):
        x_est_fin = x_est_top[s]

        if s == (len(x_est_top) - 1):
            # время измерений - tmin для значений
            t = t_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            R_er = R_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            Vr_er = Vr_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            theta_er = theta_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]

        else:
            t = t_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            R_er = R_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            Vr_er = Vr_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            theta_er = theta_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]

        t_err_plot.append(t)
        t = t - tmin

        x_0 = xhy_0_set[s][0].real
        h_0 = xhy_0_set[s][1].real

        R_est_full = np.zeros(len(t))
        theta_est_full = np.zeros(len(t))
        Vr_est_full = np.zeros(len(t))

        R_est_err = np.zeros(len(t))
        theta_est_err = np.zeros(len(t))
        Vr_est_err = np.zeros(len(t))

        k0 = x_est_fin[0].real
        v0 = x_est_fin[1].real
        dR = x_est_fin[2].real
        alpha = x_est_fin[3].real

        for k in range(len(t)):
            R_est_full[k] = np.sqrt(
                (x_L - ((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0)) ** 2 +
                y_L ** 2 + (h_L - ((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                   (1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0)) ** 2) + dR

            theta_est_full[k] = np.arctan((((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                            (1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0) - h_L) / np.sqrt(
                (((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0) - x_L) ** 2 + y_L ** 2))

            Vr_est_full[k] = ((v0 * np.exp(-k0 * t[k] / m) * np.cos(alpha)) *
                              (((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0) - x_L) +
                              (v0 * np.sin(alpha) * np.exp(-k0 * t[k] / m) - (m * g / k0) *
                               (1 - np.exp(-k0 * t[k] / m))) * (((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                                                 (1 - np.exp(-k0 * t[k] / m)) - (
                                                                         m * g * t[k]) / k0 + h_0) - h_L)) / \
                             np.sqrt((x_L - ((m / k0) * v0 * np.cos(alpha) *
                                             (1 - np.exp(-k0 * t[k] / m)) + x_0)) ** 2 + y_L ** 2 + (h_L - ((m / k0) *
                                                                                                            (
                                                                                                                    v0 * np.sin(
                                                                                                                alpha) + (
                                                                                                                            m * g) / k0) * (
                                                                                                                    1 - np.exp(
                                                                                                                -k0 *
                                                                                                                t[
                                                                                                                    k] / m)) - (
                                                                                                                    m * g *
                                                                                                                    t[
                                                                                                                        k]) / k0 + h_0)) ** 2)

            R_est_err[k] = R_est_full[k] - R_er[k]
            Vr_est_err[k] = Vr_est_full[k] - Vr_er[k]
            theta_est_err[k] = theta_est_full[k] - theta_er[k]

        R_est_full_plot.append(R_est_full)
        Vr_est_full_plot.append(Vr_est_full)
        theta_est_full_plot.append(theta_est_full)
        R_est_err_plot.append(R_est_err)
        Vr_est_err_plot.append(Vr_est_err)
        theta_est_err_plot.append(theta_est_err)

    return R_est_err_plot, Vr_est_err_plot, theta_est_err_plot


# оцененные значения для квадратичной аппроксимации в момент измерений
def func_quad_piece_estimation_error(xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, R_meas, Vr_meas, theta_meas,
                                     m, g, x_L, y_L, h_L):
    t_err_plot = []
    R_est_full_plot = []
    Vr_est_full_plot = []
    theta_est_full_plot = []
    R_est_err_plot = []
    Vr_est_err_plot = []
    theta_est_err_plot = []

    for s in range(len(x_est_top)):
        x_est_fin = x_est_top[s]

        if s == (len(x_est_top) - 1):
            # время измерений - tmin для значений
            t = t_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            R_er = R_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            Vr_er = Vr_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            theta_er = theta_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][-1]]
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]

        else:
            t = t_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            R_er = R_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            Vr_er = Vr_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            theta_er = theta_meas[meas_t_ind[s][window_set[s][0] - 1]:meas_t_ind[s][window_set[s][1]]]
            tmin = t_meas[meas_t_ind[s][window_set[s][0] - 1]]

        t_err_plot.append(t)
        t = t - tmin

        x_0 = xhy_0_set[s][0].real
        h_0 = xhy_0_set[s][1].real

        R_est_full = np.zeros(len(t))
        theta_est_full = np.zeros(len(t))
        Vr_est_full = np.zeros(len(t))

        R_est_err = np.zeros(len(t))
        theta_est_err = np.zeros(len(t))
        Vr_est_err = np.zeros(len(t))

        k0 = x_est_fin[0].real
        v0 = x_est_fin[1].real
        dR = x_est_fin[2].real
        alpha = x_est_fin[3].real

        for k in range(len(t)):
            R_est_full[k] = np.sqrt((x_L - (x_0 + (m / k0) * np.log(
                1 + (k0 * v0 * t[k] * np.cos(alpha)) / m))) ** 2 + y_L + (
                                            h_L - (
                                            h_0 + (m / k0) * np.log(
                                        np.cos(t[k] * np.sqrt(k0 * g / m)) + np.sqrt(
                                            k0 / (m * g)) * v0 * np.sin(
                                            alpha) * np.sin(
                                            t[k] * np.sqrt(k0 * g / m))))) ** 2) + dR

            theta_est_full[k] = np.arctan(((h_0 + (m / k0) * np.log(
                np.cos(t[k] * np.sqrt(k0 * g / m)) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                    alpha) * np.sin(t[k] * np.sqrt(k0 * g / m)))) - h_L) / np.sqrt((x_L - (
                    x_0 + (m / k0) * np.log(
                1 + (k0 * v0 * t[k] * np.cos(alpha)) / m))) ** 2 + y_L ** 2))

            Vr_est_full[k] = ((v0 * np.cos(alpha) * ((x_0 + (m / k0) * np.log(
                1 + (k0 * v0 * t[k] * np.cos(alpha)) / m)) - x_L)) / (1 + (
                    k0 * t[k] * v0 * np.cos(alpha)) / m) + ((np.sqrt(
                m * g * k0) * v0 * np.sin(alpha) - m * g * np.tan(
                np.sqrt(k0 * g / m) * t[k])) / (np.sqrt(m * g * k0) + k0 *
                                                v0 * np.sin(
                        alpha) * np.tan(np.sqrt(k0 * g / m) * t[k]))) * (
                                      (h_0 + (m / k0) * np.log(
                                          np.cos(np.sqrt(k0 * g / m) * t[k]) + np.sqrt(
                                              k0 / (m * g)) * v0 * np.sin(
                                              alpha) * np.sin(
                                              np.sqrt(k0 * g / m) * t[k]))) - h_L)) / (
                                 np.sqrt((x_L - (
                                         x_0 + (m / k0) * np.log(
                                     1 + (k0 * v0 * t[k] * np.cos(
                                         alpha)) / m))) ** 2 + y_L ** 2 + (h_L - (
                                         h_0 + (m / k0) * np.log(
                                     np.cos(np.sqrt(k0 * g / m) * t[k]) + np.sqrt(
                                         k0 / (m * g)) * v0 * np.sin(
                                         alpha) * np.sin(
                                         np.sqrt(k0 * g / m) * t[k])))) ** 2))

            R_est_err[k] = R_est_full[k] - R_er[k]
            Vr_est_err[k] = Vr_est_full[k] - Vr_er[k]
            theta_est_err[k] = theta_est_full[k] - theta_er[k]

        R_est_full_plot.append(R_est_full)
        Vr_est_full_plot.append(Vr_est_full)
        theta_est_full_plot.append(theta_est_full)
        R_est_err_plot.append(R_est_err)
        Vr_est_err_plot.append(Vr_est_err)
        theta_est_err_plot.append(theta_est_err)


    return R_est_err_plot, Vr_est_err_plot, theta_est_err_plot

def func_std_error_meas(t_meas, R_meas, Vr_meas, theta_meas, R_est_err_plot, Vr_est_err_plot, theta_est_err_plot,
                        sko_R_tz, sko_Vr_tz, sko_theta_tz):
    # выводить std
    track_meas = {}
    points = []

    # валидные точки - 0, если не валидные 1 ( СКО больше ТЗ)
    valid_R = []
    valid_Vr = []
    valid_theta = []

    R_true = []
    Vr_true = []
    theta_true = []

    Nlen = 0

    for k in range(len(R_est_err_plot)):
        for j in range(len(R_est_err_plot[k])):
            if (-3 * sko_R_tz < R_est_err_plot[k][j]) and (R_est_err_plot[k][j] < 3 * sko_R_tz):
                R_true.append(R_est_err_plot[k][j])
                valid_R.append(0)
            else:
                valid_R.append(1)

            if (-3 * sko_Vr_tz < Vr_est_err_plot[k][j]) and (Vr_est_err_plot[k][j] < 3 * sko_Vr_tz):
                Vr_true.append(Vr_est_err_plot[k][j])
                valid_Vr.append(0)
            else:
                valid_Vr.append(1)

            if (-3 * sko_theta_tz < theta_est_err_plot[k][j]) and (theta_est_err_plot[k][j] < 3 * sko_theta_tz):
                theta_true.append(theta_est_err_plot[k][j])
                valid_theta.append(0)
            else:
                valid_theta.append(1)

            Nlen += 1

    SKO_R_true = np.std(np.array(R_true))
    SKO_V_true = np.std(np.array(Vr_true))
    SKO_theta_true = np.std(np.array(theta_true))

    for i in range(Nlen):
        points.append({"t": t_meas[i], "R": R_meas[i], "Vr": Vr_meas[i], "theta": theta_meas[i], "valid_R": valid_R[i],
                       "valid_Vr": valid_Vr[i], "valid_theta": valid_theta[i]})

    track_meas["points"] = points

    print(Nlen, 'число измерений без отсева')

    print(len(R_true), 'число измерений дальности после отсева')

    print(len(Vr_true), 'число измерений скорости после отсева')

    print(len(theta_true), 'число измерений угла после отсева')

    return track_meas, SKO_R_true, SKO_V_true, SKO_theta_true


# оценка измерений до точки падения для линейной аппроксимации
def func_trajectory_end_linear(m, g, xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, x_L, y_L, h_L):
    N = 10000

    xhy_0_fin = xhy_0_set[len(xhy_0_set) - 1]
    x_est_fin = x_est_top[len(xhy_0_set) - 1]

    k0 = x_est_fin[0]
    v0 = x_est_fin[1]
    dR = x_est_fin[2]
    alpha = x_est_fin[3]

    x_0 = xhy_0_fin[0]
    h_0 = xhy_0_fin[1]

    meas_t_ind = meas_t_ind[len(xhy_0_set) - 1]
    window_set = window_set[len(xhy_0_set) - 1]

    t_meas_t = t_meas[meas_t_ind[0]:]
    tmin = t_meas_t[window_set[0] - 1]
    tmax = 30

    t = []
    n = 0
    for i in range(N):
        if i == 0:
            n = 0
        else:
            n += (tmax - tmin) / (N - 1)
        t.append(n)

    t = np.array(t)

    x_true_fin = np.zeros(N)
    h_true_fin = np.zeros(N)

    Vx_true_fin = np.zeros(N)
    Vh_true_fin = np.zeros(N)

    R_true_fin = np.zeros(N)
    Vr_true_fin = np.zeros(N)

    theta_true_fin = np.zeros(N)
    V_abs_true_fin = np.zeros(N)

    alpha_true_fin = np.zeros(N)
    A_abs_true_fin = np.zeros(N)
    Ax_true_fin = np.zeros(N)
    Ah_true_fin = np.zeros(N)

    last_k = 0

    for k in range(N):

        x_true_fin[k] = x_0 + (m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m))

        h_true_fin[k] = ((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) * (
                1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0)

        Vx_true_fin[k] = v0 * np.exp(-k0 * t[k] / m) * np.cos(alpha)
        Vh_true_fin[k] = v0 * np.sin(alpha) * np.exp(-k0 * t[k] / m) - (m * g / k0) * (1 - np.exp(-k0 * t[k] / m))

        R_true_fin[k] = np.sqrt(
            (x_L - ((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0)) ** 2 +
            y_L ** 2 + (h_L - ((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                               (1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0)) ** 2) + dR

        theta_true_fin[k] = np.arctan((((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                        (1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0) - h_L) / np.sqrt(
            (((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0) - x_L) ** 2 + y_L ** 2))

        Vr_true_fin[k] = ((v0 * np.exp(-k0 * t[k] / m) * np.cos(alpha)) *
                          (((m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m)) + x_0) - x_L) +
                          (v0 * np.sin(alpha) * np.exp(-k0 * t[k] / m) - (m * g / k0) *
                           (1 - np.exp(-k0 * t[k] / m))) * (((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) *
                                                             (1 - np.exp(-k0 * t[k] / m)) - (
                                                                     m * g * t[k]) / k0 + h_0) - h_L)) / \
                         np.sqrt((x_L - ((m / k0) * v0 * np.cos(alpha) *
                                         (1 - np.exp(-k0 * t[k] / m)) + x_0)) ** 2 + y_L ** 2 + (h_L - ((m / k0) *
                                                                                                        (
                                                                                                                v0 * np.sin(
                                                                                                            alpha) + (
                                                                                                                        m * g) / k0) * (
                                                                                                                1 - np.exp(
                                                                                                            -k0 *
                                                                                                            t[
                                                                                                                k] / m)) - (
                                                                                                                m * g *
                                                                                                                t[
                                                                                                                    k]) / k0 + h_0)) ** 2)

        V_abs_true_fin[k] = np.sqrt(Vx_true_fin[k] ** 2 + Vh_true_fin[k] ** 2)

        alpha_true_fin[k] = np.arctan(Vh_true_fin[k] / Vx_true_fin[k])

        if k > 0:
            A_abs_true_fin[k] = (V_abs_true_fin[k] - V_abs_true_fin[k - 1]) / (t[k] - t[k - 1])
            Ax_true_fin[k] = (Vx_true_fin[k] - Vx_true_fin[k - 1]) / (t[k] - t[k - 1])
            Ah_true_fin[k] = (Vh_true_fin[k] - Vh_true_fin[k - 1]) / (t[k] - t[k - 1])

        if h_true_fin[k] < 0:
            last_k = k
            break

    A_abs_true_fin[0] = A_abs_true_fin[1]
    Ax_true_fin[0] = Ax_true_fin[1]
    Ah_true_fin[0] = Ah_true_fin[1]

    return t[:last_k] + tmin, x_true_fin[:last_k], h_true_fin[:last_k], R_true_fin[:last_k], \
           Vr_true_fin[:last_k], theta_true_fin[:last_k], Vx_true_fin[:last_k], \
           Vh_true_fin[:last_k], V_abs_true_fin[:last_k], alpha_true_fin[:last_k], A_abs_true_fin[:last_k], \
           Ax_true_fin[:last_k], Ah_true_fin[:last_k]


# оценка измерений до точки падения для квадратичной аппроксимации
def func_trajectory_end_quad(m, g, xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, x_L, y_L, h_L):
    N = 10000

    xhy_0_fin = xhy_0_set[len(xhy_0_set) - 1]
    x_est_fin = x_est_top[len(xhy_0_set) - 1]

    k0 = x_est_fin[0]
    v0 = x_est_fin[1]
    dR = x_est_fin[2]
    alpha = x_est_fin[3]

    x_0 = xhy_0_fin[0]
    h_0 = xhy_0_fin[1]

    meas_t_ind = meas_t_ind[len(xhy_0_set) - 1]
    window_set = window_set[len(xhy_0_set) - 1]

    t_meas_t = t_meas[meas_t_ind[0]:]
    tmin = t_meas_t[window_set[0] - 1]
    tmax = 100

    t = []
    n = 0
    for i in range(N):
        if i == 0:
            n = 0
        else:
            n += (tmax - tmin) / (N - 1)
        t.append(n)

    t = np.array(t)

    x_true_fin = np.zeros(N)
    h_true_fin = np.zeros(N)

    Vx_true_fin = np.zeros(N)
    Vh_true_fin = np.zeros(N)

    R_true_fin = np.zeros(N)
    Vr_true_fin = np.zeros(N)

    theta_true_fin = np.zeros(N)
    V_abs_true_fin = np.zeros(N)

    alpha_true_fin = np.zeros(N)
    A_abs_true_fin = np.zeros(N)
    Ax_true_fin = np.zeros(N)
    Ah_true_fin = np.zeros(N)

    last_k = 0

    for k in range(N):

        x_true_fin[k] = (m / k0) * np.log(
            1 + (k0 / m) * v0 * t[k] * np.cos(alpha)) + x_0

        h_true_fin[k] = (m / k0) * np.log(
            np.cos(np.sqrt((k0 * g) / m) * t[k]) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                alpha) * np.sin(np.sqrt((k0 * g) / m) * t[k])) + h_0

        R_true_fin[k] = np.sqrt((x_L - (x_0 + (m / k0) * np.log(
            1 + (k0 * v0 * t[k] * np.cos(alpha)) / m))) ** 2 + y_L + (
                                        h_L - (
                                        h_0 + (m / k0) * np.log(
                                    np.cos(t[k] * np.sqrt(k0 * g / m)) + np.sqrt(
                                        k0 / (m * g)) * v0 * np.sin(
                                        alpha) * np.sin(
                                        t[k] * np.sqrt(k0 * g / m))))) ** 2) + dR

        theta_true_fin[k] = np.arctan(((h_0 + (m / k0) * np.log(
            np.cos(t[k] * np.sqrt(k0 * g / m)) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                alpha) * np.sin(t[k] * np.sqrt(k0 * g / m)))) - h_L) / np.sqrt((x_L - (
                x_0 + (m / k0) * np.log(
            1 + (k0 * v0 * t[k] * np.cos(alpha)) / m))) ** 2 + y_L ** 2))

        Vr_true_fin[k] = ((v0 * np.cos(alpha) * ((x_0 + (m / k0) * np.log(
            1 + (k0 * v0 * t[k] * np.cos(alpha)) / m)) - x_L)) / (1 + (
                k0 * t[k] * v0 * np.cos(alpha)) / m) + ((np.sqrt(
            m * g * k0) * v0 * np.sin(alpha) - m * g * np.tan(
            np.sqrt(k0 * g / m) * t[k])) / (np.sqrt(m * g * k0) + k0 *
                                            v0 * np.sin(
                    alpha) * np.tan(np.sqrt(k0 * g / m) * t[k]))) * (
                                  (h_0 + (m / k0) * np.log(
                                      np.cos(np.sqrt(k0 * g / m) * t[k]) + np.sqrt(
                                          k0 / (m * g)) * v0 * np.sin(
                                          alpha) * np.sin(
                                          np.sqrt(k0 * g / m) * t[k]))) - h_L)) / (
                             np.sqrt((x_L - (
                                     x_0 + (m / k0) * np.log(
                                 1 + (k0 * v0 * t[k] * np.cos(
                                     alpha)) / m))) ** 2 + y_L ** 2 + (h_L - (
                                     h_0 + (m / k0) * np.log(
                                 np.cos(np.sqrt(k0 * g / m) * t[k]) + np.sqrt(
                                     k0 / (m * g)) * v0 * np.sin(
                                     alpha) * np.sin(
                                     np.sqrt(k0 * g / m) * t[k])))) ** 2))

        Vx_true_fin[k] = (v0 * np.cos(alpha)) / (
                1 + k0 * v0 * t[k] * np.cos(alpha) / m)

        Vh_true_fin[k] = (np.sqrt(m * g * k0) * v0 * np.sin(
            alpha) - m * g * np.tan(
            np.sqrt((k0 * g) / m) * t[k])) / (
                                 np.sqrt(m * g * k0) + k0 * v0 * np.sin(
                             alpha) * np.tan(np.sqrt((k0 * g) / m) * t[k]))

        V_abs_true_fin[k] = np.sqrt(Vx_true_fin[k] ** 2 + Vh_true_fin[k] ** 2)

        alpha_true_fin[k] = np.arctan(Vh_true_fin[k] / Vx_true_fin[k])

        if k > 0:
            A_abs_true_fin[k] = (V_abs_true_fin[k] - V_abs_true_fin[k - 1]) / (t[k] - t[k - 1])
            Ax_true_fin[k] = (Vx_true_fin[k] - Vx_true_fin[k - 1]) / (t[k] - t[k - 1])
            Ah_true_fin[k] = (Vh_true_fin[k] - Vh_true_fin[k - 1]) / (t[k] - t[k - 1])

        if h_true_fin[k] < 0:
            last_k = k
            break

    A_abs_true_fin[0] = A_abs_true_fin[1]
    Ax_true_fin[0] = Ax_true_fin[1]
    Ah_true_fin[0] = Ah_true_fin[1]

    return t[:last_k] + tmin, x_true_fin[:last_k], h_true_fin[:last_k], R_true_fin[:last_k], \
           Vr_true_fin[:last_k], theta_true_fin[:last_k], Vx_true_fin[:last_k], \
           Vh_true_fin[:last_k], V_abs_true_fin[:last_k], alpha_true_fin[:last_k], A_abs_true_fin[:last_k], \
           Ax_true_fin[:last_k], Ah_true_fin[:last_k]


# функция для определения коэффициентов - для промежутка активно-реактивного снаряда
def func_lsm_linear(X, H):
    N = len(X)
    sum_X = sum(X)
    sum_X2 = sum(X ** 2)
    sum_X3 = sum(X ** 3)
    sum_X4 = sum(X ** 4)
    sum_H = sum(H)
    sum_HX = sum(X * H)
    sum_HX2 = sum(X ** 2 * H)

    d = N * (sum_X2 * sum_X4 - sum_X3 * sum_X3) - sum_X * (sum_X * sum_X4 - sum_X3 * sum_X2) + sum_X2 * (
            sum_X * sum_X3 - sum_X2 * sum_X2)
    d1 = sum_H * (sum_X2 * sum_X4 - sum_X3 * sum_X3) - sum_X * (sum_HX * sum_X4 - sum_X3 * sum_HX2) + sum_X2 * (
            sum_HX * sum_X3 - sum_X2 * sum_HX2)
    d2 = N * (sum_HX * sum_X4 - sum_X3 * sum_HX2) - sum_H * (sum_X * sum_X4 - sum_X3 * sum_X2) + sum_X2 * (
            sum_X * sum_HX2 - sum_HX * sum_X2)
    d3 = N * (sum_X2 * sum_HX2 - sum_HX * sum_X3) - sum_X * (sum_X * sum_HX2 - sum_HX * sum_X2) + sum_H * (
            sum_X * sum_X3 - sum_X2 * sum_X2)

    out = [d1 / d, d2 / d, d3 / d]

    return out


# заполнение промеждутка для активно-реактивного снаряда оцененными измерениями
def func_active_reactive_trajectory(x_tr_er_1, h_tr_er_1, t_meas_1, x_tr_er_2, h_tr_er_2, t_meas_2, x_L, y_L, h_L):
    N = 300
    NoW = N / 20
    NoW = int(NoW)
    Nkol = 10

    tmin = t_meas_1[-1][-1]
    tmax = t_meas_2[0][0]

    x_tr_act = np.zeros(2 * NoW)
    h_tr_act = np.zeros(2 * NoW)

    for i in range(NoW):
        x_tr_act[i] = x_tr_er_1[-1][0:len(x_tr_er_1[-1]):20][i]
        h_tr_act[i] = h_tr_er_1[-1][0:len(h_tr_er_1[-1]):20][i]

    for i in range(NoW, 2 * NoW):
        x_tr_act[i] = x_tr_er_2[0][0:len(x_tr_er_2[0]):20][i - NoW]
        h_tr_act[i] = h_tr_er_2[0][0:len(h_tr_er_2[0]):20][i - NoW]

    kf = func_lsm_linear(x_tr_act, h_tr_act)

    x_tr_gap = np.linspace(x_tr_er_1[-1][-1] + 10, x_tr_er_2[0][0] - 10, num=Nkol)
    t_tr_act_est = np.linspace(tmin, tmax, num=Nkol)

    x_tr_act_est = np.zeros(2 * NoW + Nkol)
    h_tr_act_est = np.zeros(2 * NoW + Nkol)

    for k in range(NoW):
        x_tr_act_est[k] = x_tr_er_1[-1][0:len(x_tr_er_1[-1]):20][k]

    for k in range(NoW, NoW + Nkol):
        x_tr_act_est[k] = x_tr_gap[k - NoW]

    for k in range(NoW + Nkol, 2 * NoW + Nkol):
        x_tr_act_est[k] = x_tr_er_2[0][0:len(x_tr_er_2[0]):20][k - NoW - Nkol]

    for i in range(len(x_tr_act_est)):
        h_tr_act_est[i] = kf[2] * x_tr_act_est[i] ** 2 + kf[1] * x_tr_act_est[i] + kf[0]

    x_tr_act_est = x_tr_act_est[NoW:NoW + Nkol]
    h_tr_act_est = h_tr_act_est[NoW:NoW + Nkol]

    Nlen = len(x_tr_act_est)

    R_tr_act_est = np.zeros(Nlen)
    Vr_tr_act_est = np.zeros(Nlen)
    theta_tr_act_est = np.zeros(Nlen)

    Vx_tr_act_est = np.zeros(Nlen)
    Vh_tr_act_est = np.zeros(Nlen)
    V_abs_tr_act_est = np.zeros(Nlen)

    alpha_tr_act_est = np.zeros(Nlen)

    Ax_tr_act_est = np.zeros(Nlen)
    Ah_tr_act_est = np.zeros(Nlen)
    A_abs_tr_act_est = np.zeros(Nlen)

    for k in range(Nlen):
        if k != 0:
            Vx_tr_act_est[k] = (x_tr_act_est[k] - x_tr_act_est[k - 1]) / (t_tr_act_est[k] - t_tr_act_est[k - 1])
            Vh_tr_act_est[k] = (h_tr_act_est[k] - h_tr_act_est[k - 1]) / (t_tr_act_est[k] - t_tr_act_est[k - 1])
            V_abs_tr_act_est[k] = np.sqrt(Vx_tr_act_est[k] ** 2 + Vh_tr_act_est[k] ** 2)

    Vx_tr_act_est[0] = Vx_tr_act_est[1]
    Vh_tr_act_est[0] = Vh_tr_act_est[1]
    V_abs_tr_act_est[0] = V_abs_tr_act_est[1]

    for k in range(Nlen):
        if k != 0:
            Ax_tr_act_est[k] = (Vx_tr_act_est[k] - Vx_tr_act_est[k - 1]) / (t_tr_act_est[k] - t_tr_act_est[k - 1])
            Ah_tr_act_est[k] = (Vh_tr_act_est[k] - Vh_tr_act_est[k - 1]) / (t_tr_act_est[k] - t_tr_act_est[k - 1])
            A_abs_tr_act_est[k] = (V_abs_tr_act_est[k] - V_abs_tr_act_est[k - 1]) / (
                    t_tr_act_est[k] - t_tr_act_est[k - 1])

    Ax_tr_act_est[0] = Ax_tr_act_est[1]
    Ah_tr_act_est[0] = Ah_tr_act_est[1]
    A_abs_tr_act_est[0] = A_abs_tr_act_est[1]

    for k in range(Nlen):
        R_tr_act_est[k] = np.sqrt((x_L - x_tr_act_est[k]) ** 2 + y_L ** 2 + (h_L - h_tr_act_est[k]) ** 2)

        Vr_tr_act_est[k] = (Vx_tr_act_est[k] * (x_tr_act_est[k] - x_L) + Vh_tr_act_est[k] * (
                h_tr_act_est[k] - h_L)) / (np.sqrt(
            (x_L - x_tr_act_est[k]) ** 2 + y_L ** 2 + (h_L - h_tr_act_est[k]) ** 2))

        theta_tr_act_est[k] = np.arctan((h_tr_act_est[k] - h_L) / (np.sqrt((x_tr_act_est[k] - x_L) ** 2 + y_L ** 2)))
        alpha_tr_act_est[k] = np.arctan(Vh_tr_act_est[k] / Vx_tr_act_est[k])

    return t_tr_act_est, x_tr_act_est, h_tr_act_est, R_tr_act_est, Vr_tr_act_est, theta_tr_act_est, Vx_tr_act_est, \
           Vh_tr_act_est, V_abs_tr_act_est, alpha_tr_act_est, A_abs_tr_act_est, Ax_tr_act_est, Ah_tr_act_est


# деривация для снарядов
def func_derivation(K1, K2, x_fin, v0, alpha):
    z_deriv = (K1 + K2 * x_fin) * v0 ** 2 * np.sin(alpha) ** 2
    return z_deriv


# функция деривации для пуль
def func_derivation_bullet(m, d, l, eta, K_inch, K_gran, K_fut, v0, t_pol):
    eta = eta / d
    l_cal = l / d
    d_inch = d * K_inch
    m_gran = m * K_gran
    Sg = (30 * m_gran) / (eta ** 2 * d_inch ** 3 * l_cal * (1 + l_cal ** 2))
    Sg_corr = Sg * ((v0 * K_fut) / 2800) ** (1 / 3)
    z_deriv_corr = (1.25 * (Sg_corr + 1.2) * t_pol ** 1.83) / K_inch
    return z_deriv_corr


# функция для учета ветра
def func_wind(t_fin, x_fin, v0, alpha, wind_module, wind_direction, az):
    Aw = np.deg2rad(az) - (np.deg2rad(wind_direction) + np.pi)
    Wz = wind_module * np.sin(Aw)
    z_wind = Wz * (t_fin - x_fin / (v0 / np.cos(alpha)))
    return z_wind


# функция для перевода точки падения в координаты
def func_tochka_fall(z, x_fin, can_B, can_L, az):
    x_fall = x_fin
    z_fall = z
    x_sp_gk, y_sp_gk = BLH2XY_GK(can_B, can_L)
    print(x_sp_gk, y_sp_gk, 'зоны падения sp_gk')
    print(int(x_sp_gk / 10e5), int(y_sp_gk / 10e5), 'зоны')
    RM = np.array([[np.cos(az), np.sin(az)], [-np.sin(az), np.cos(az)]])
    deltaXY_gk = RM.dot(np.array([[z_fall], [x_fall]]))
    x_fall_gk = x_sp_gk - 10e5 * int(x_sp_gk / 10e5) + deltaXY_gk[1]
    z_fall_gk = y_sp_gk - 10e5 * int(y_sp_gk / 10e5) + deltaXY_gk[0]

    return x_fall_gk, z_fall_gk


# перевод координат
def BLH2XY_GK(B, L):
    B_rad = np.deg2rad(B)
    n = np.fix((6 + L) / 6)
    l = (L - (3 + 6 * (n - 1))) / 57.29577951

    x = 6367558.4968 * B_rad - np.sin(2 * B_rad) * (
            16002.8900 + 66.9607 * np.sin(B_rad) ** 2 + 0.3515 * np.sin(B_rad) ** 4
            - l ** 2 * (1594561.25 + 5336.535 * np.sin(B_rad) ** 2 + 26.790 * np.sin(
        B_rad) ** 4 + 0.149 * np.sin(B_rad) ** 6 + l ** 2 * (672483.4 - 811219.9 * np.sin(
        B_rad) ** 2 + 5420.0 * np.sin(B_rad) ** 4 - 10.6 * np.sin(B_rad) ** 6
                                                             + l ** 2 * (278194 - 830174 * np.sin(
                B_rad) ** 2 + 572434 * np.sin(B_rad) ** 4 - 16010 * np.sin(B_rad) ** 6
                                                                         + l ** 2 * (109500 - 574700 * np.sin(
                        B_rad) ** 2 + 863700 * np.sin(B_rad) ** 4 - 398600 * np.sin(B_rad) ** 6)))))

    y = (5 + 10 * n) * 10 ** 5 + l * np.cos(B_rad) * (
            6378245 + 21346.1415 * np.sin(B_rad) ** 2 + 107.1590 * np.sin(B_rad) ** 4
            + 0.5977 * np.sin(B_rad) ** 6 + l ** 2 * (
                    1070204.16 - 2136826.66 * np.sin(B_rad) ** 2 + 17.98 * np.sin(
                B_rad) ** 4 - 11.99 * np.sin(B_rad) ** 6
                    + l ** 2 * (270806 - 1523417 * np.sin(
                B_rad) ** 2 + 1327645 * np.sin(B_rad) ** 4 - 21701 * np.sin(
                B_rad) ** 6
                                + l ** 2 * (79690 - 866190 * np.sin(
                        B_rad) ** 2 + 1730360 * np.sin(B_rad) ** 4 - 945460 * np.sin(
                        B_rad) ** 6))))

    return x, y


def calculate_ellipse(x, y, a, b, angle, steps):
    beta = np.deg2rad(-angle)
    sinbeta = np.sin(beta)
    cosbeta = np.cos(beta)

    alpha = np.linspace(0, 360, steps).T
    alpha = np.deg2rad(alpha)
    sinalpha = np.sin(alpha)
    cosalpha = np.cos(alpha)

    X = x + (a * cosalpha * cosbeta - b * sinalpha * sinbeta)
    Y = y + (a * cosalpha * sinbeta + b * sinalpha * cosbeta)

    return X, Y
