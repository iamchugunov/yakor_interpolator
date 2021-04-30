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


def func_linear_piece_app(x_L, y_L, h_L, y_0, m, g, SKO_R, SKO_Vr, SKO_theta, k0, dR, t_meas_full,
                          R_meas_full, Vr_meas_full, theta_meas_full, winlen, step_sld, parameters_bounds):

    Nkol = 0

    if winlen == 30:
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

    #NoW = int(len(R_meas_full) / (winlen - 1))

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
                    print(window_set[w], x_est_top[w], xhy_0_set[w])
                    break

    return xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas_full, R_meas_full, Vr_meas_full, theta_meas_full


def func_quad_piece_app(x_L, y_L, h_L, y_0, m, g, SKO_R, SKO_Vr, SKO_theta, k0, dR, t_meas_full,
                        R_meas_full, Vr_meas_full, theta_meas_full, winlen, step_sld, parameters_bounds):

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
    if (len(t_meas_full) - NoW * winlen) > 2:
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


def func_linear_piece_estimation(xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, N, m, g, x_L, y_L, h_L):

    Nlen = len(x_est_top)
    t_meas_plot = []
    x_tr_er_plot = []
    h_tr_er_plot = []
    R_est_full_plot = []
    Vr_est_full_plot = []
    theta_est_full_plot = []
    index_plot = []
    Vx_true_er_plot = []
    Vh_true_er_plot = []
    V_abs_full_plot = []

    for s in range(Nlen):

        x_est_fin = x_est_top[s]

        if s == (Nlen - 1):

            t_meas_t = t_meas[meas_t_ind[s][0]:]
            tmin = t_meas_t[window_set[s][0] - 1]
            tmax = t_meas_t[-1]
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

            t_meas_t = t_meas[meas_t_ind[s][0]:]
            tmin = t_meas_t[window_set[s][0] - 1]
            tmax = t_meas_t[window_set[s][1] - 1]
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

        x_tr_er = np.zeros(N)
        h_tr_er = np.zeros(N)
        R_est_full = np.zeros(N)
        theta_est_full = np.zeros(N)
        Vr_est_full = np.zeros(N)
        Vx_true_er = np.zeros(N)
        Vh_true_er = np.zeros(N)
        V_abs_full = np.zeros(N)
        index = np.zeros(N)

        k0 = x_est_fin[0]
        v0 = x_est_fin[1]
        dR = x_est_fin[2]
        alpha = x_est_fin[3]

        for k in range(N):
            index[k] = k

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

            V_abs_full[k] = np.sqrt(Vx_true_er[k] ** 2 + Vh_true_er[k] ** 2)

        x_tr_er_plot.append(x_tr_er)
        h_tr_er_plot.append(h_tr_er)
        R_est_full_plot.append(R_est_full)
        Vr_est_full_plot.append(Vr_est_full)
        theta_est_full_plot.append(theta_est_full)
        index_plot.append(index)

        Vx_true_er_plot.append(Vx_true_er)
        Vh_true_er_plot.append(Vh_true_er)
        V_abs_full_plot.append(V_abs_full)

    return t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
           Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot


def func_quad_piece_estimation(xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas, N, m, g, x_L, y_L, h_L):
    Nlen = len(x_est_top)
    t_meas_plot = []
    x_tr_er_plot = []
    h_tr_er_plot = []
    R_est_full_plot = []
    Vr_est_full_plot = []
    Vx_true_er_plot = []
    Vh_true_er_plot = []
    theta_true_2_plot = []
    theta_est_full_plot = []
    V_abs_est_plot = []

    for s in range(Nlen):
        x_est_fin = x_est_top[s]

        if s == (Nlen - 1):
            t_meas_t = t_meas[meas_t_ind[s][0]:]
            tmin = t_meas_t[window_set[s][0] - 1]
            tmax = t_meas_t[-1]
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

            t_meas_t = t_meas[meas_t_ind[s][0]:]
            tmin = t_meas_t[window_set[s][0] - 1]
            tmax = t_meas_t[window_set[s][1] - 1]
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

        x_tr_er = np.zeros(N)
        h_tr_er = np.zeros(N)
        R_est_full = np.zeros(N)
        theta_est_full = np.zeros(N)
        Vr_est_full = np.zeros(N)
        V_abs_est = np.zeros(N)
        Vx_true_er = np.zeros(N)
        Vh_true_er = np.zeros(N)
        theta_true_2 = np.zeros(N)

        k0 = x_est_fin[0].real
        v0 = x_est_fin[1].real
        dR = x_est_fin[2].real
        alpha = x_est_fin[3].real

        for k in range(N):
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

            x_true = (m / k0) * np.log(
                1 + (k0 / m) * v0 * t[k] * np.cos(alpha)) + x_0

            h_true = (m / k0) * np.log(
                np.cos(np.sqrt((k0 * g) / m) * t[k]) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                    alpha) * np.sin(np.sqrt((k0 * g) / m) * t[k])) + h_0

            theta_true_2[k] = np.arctan((h_true - h_L) / np.sqrt((x_true - x_L) ** 2 + y_L ** 2))

        x_tr_er_plot.append(x_tr_er)
        h_tr_er_plot.append(h_tr_er)
        R_est_full_plot.append(R_est_full)
        Vr_est_full_plot.append(Vr_est_full)
        Vx_true_er_plot.append(Vx_true_er)
        Vh_true_er_plot.append(Vh_true_er)
        theta_true_2_plot.append(theta_true_2)
        theta_est_full_plot.append(theta_est_full)
        V_abs_est_plot.append(V_abs_est)

    return t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
           Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot


def func_trajectory_end_linear(m, g, xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas):

    N = 10000
    print(len(xhy_0_set))

    NoW = len(xhy_0_set) - 1

    xhy_0_fin = xhy_0_set[NoW]
    x_est_fin = x_est_top[NoW]

    k0 = x_est_fin[0]
    v0 = x_est_fin[1]
    dR = x_est_fin[2]
    alpha = x_est_fin[3]

    x_0 = xhy_0_fin[0]
    h_0 = xhy_0_fin[1]

    meas_t_ind = meas_t_ind[NoW]
    window_set = window_set[NoW]

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

    last_k = 0

    for k in range(N):

        x_true_fin[k] = x_0 + (m / k0) * v0 * np.cos(alpha) * (1 - np.exp(-k0 * t[k] / m))

        h_true_fin[k] = ((m / k0) * (v0 * np.sin(alpha) + (m * g) / k0) * (
                1 - np.exp(-k0 * t[k] / m)) - (m * g * t[k]) / k0 + h_0)

        if h_true_fin[k] < 0:
            last_k = k
            # тут
            break

    print(x_true_fin[last_k - 1])
    print(x_true_fin[last_k])

    return t[:last_k], x_true_fin[:last_k], h_true_fin[:last_k]

def func_trajectory_end_quad(m, g, xhy_0_set, x_est_top, meas_t_ind, window_set, t_meas):

    N = 10000

    NoW = len(xhy_0_set) - 1

    xhy_0_fin = xhy_0_set[NoW]
    x_est_fin = x_est_top[NoW]

    k0 = x_est_fin[0]
    v0 = x_est_fin[1]
    dR = x_est_fin[2]
    alpha = x_est_fin[3]

    x_0 = xhy_0_fin[0]
    h_0 = xhy_0_fin[1]

    meas_t_ind = meas_t_ind[NoW]
    window_set = window_set[NoW]

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

    last_k = 0

    for k in range(N):

        x_true_fin[k] = (m / k0) * np.log(
            1 + (k0 / m) * v0 * t[k] * np.cos(alpha)) + x_0

        h_true_fin[k] = (m / k0) * np.log(
            np.cos(np.sqrt((k0 * g) / m) * t[k]) + np.sqrt(k0 / (m * g)) * v0 * np.sin(
                alpha) * np.sin(np.sqrt((k0 * g) / m) * t[k])) + h_0

        if h_true_fin[k] < 0:
            last_k = k
            # тут
            break

    print(x_true_fin[last_k - 1])
    print(x_true_fin[last_k])

    return t[:last_k], x_true_fin[:last_k], h_true_fin[:last_k]

def func_active_reactive_trajectory(x_tr_er_1, h_tr_er_1, t_meas_1, x_tr_er_2, h_tr_er_2, t_meas_2, N):
    NoW = N / 20
    NoW = int(NoW)
    Nkol = 10

    t_act = np.zeros(2 * NoW)
    x_tr_act = np.zeros(2 * NoW)
    h_tr_act = np.zeros(2 * NoW)

    for i in range(NoW):
        t_act[i] = t_meas_1[-1][0:len(t_meas_1[-1]):20][i]
        x_tr_act[i] = x_tr_er_1[-1][0:len(x_tr_er_1[-1]):20][i]
        h_tr_act[i] = h_tr_er_1[-1][0:len(h_tr_er_1[-1]):20][i]

    for i in range(NoW, 2 * NoW):
        t_act[i] = t_meas_2[0][0:len(t_meas_2[0]):20][i - NoW]
        x_tr_act[i] = x_tr_er_2[0][0:len(x_tr_er_2[0]):20][i - NoW]
        h_tr_act[i] = h_tr_er_2[0][0:len(h_tr_er_2[0]):20][i - NoW]

    kf = func_lsm_linear(x_tr_act, h_tr_act)

    x_tr_gap = np.linspace(x_tr_er_1[-1][-1] + 10, x_tr_er_2[0][0] - 10, num=Nkol)
    t_act_gap = np.linspace(t_meas_1[-1][-1] + 10, t_meas_2[0][0] - 10, num=Nkol)

    t_tr_act_est = np.zeros(2 * NoW + Nkol)
    x_tr_act_est = np.zeros(2 * NoW + Nkol)
    h_tr_act_est = np.zeros(2 * NoW + Nkol)

    for k in range(NoW):
        t_tr_act_est[k] = t_meas_1[-1][0:len(t_meas_1[-1]):20][k]
        x_tr_act_est[k] = x_tr_er_1[-1][0:len(x_tr_er_1[-1]):20][k]

    for k in range(NoW, NoW + Nkol):
        t_tr_act_est[k] = t_act_gap[k - NoW]
        x_tr_act_est[k] = x_tr_gap[k - NoW]

    for k in range(NoW + Nkol, 2 * NoW + Nkol):
        t_tr_act_est[k] = t_meas_2[0][0:len(t_meas_2[0]):20][k - NoW - Nkol]
        x_tr_act_est[k] = x_tr_er_2[0][0:len(x_tr_er_2[0]):20][k - NoW - Nkol]

    for i in range(len(x_tr_act_est)):
        h_tr_act_est[i] = kf[2] * x_tr_act_est[i] ** 2 + kf[1] * x_tr_act_est[i] + kf[0]

    return t_tr_act_est[NoW:NoW + Nkol], x_tr_act_est[NoW:NoW + Nkol], h_tr_act_est[NoW:NoW + Nkol]


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


def func_wind(t_meas_2, t_tr_fin, x_tr_fin, abs_V_0, theta_0):

    W = 4
    Az = np.deg2rad(0)
    alp_w = np.deg2rad(90)
    Aw = Az - (alp_w + np.pi)
    Wx = W * np.cos(Aw)
    Wz = W * np.sin(Aw)
    t_pol = t_meas_2[0] + t_tr_fin
    x_pol = x_tr_fin
    # здесь добавить время + 0.2 секунды - мб из-за начала?

    Zw = Wz * (t_pol - x_pol / (abs_V_0 * np.cos(theta_0)))


def func_meas_smooth_2(R_in, Vr_in, t_in):
    thres_1 = 25
    N = 15

    outlier_set = np.zeros(N)
    R_in_diff = np.zeros(N)
    for q in range(N):
        R_in_diff[q] = R_in[q + 1] - R_in[q]
        if abs(R_in_diff[q]) > thres_1:
            outlier_set[q] = 1
        else:
            outlier_set[q] = 0

    flag = 0
    for q in range(N):
        if outlier_set[q] == 0 and outlier_set[q + 1] == 0:
            start_ind = q
            flag = 1
            break

    if flag != 1:
        start_ind = 1

    R_in = R_in[start_ind:]
    t_in = t_in[start_ind:]
    Vr_in = Vr_in[start_ind:]
    thres_2 = 500
    R_out = np.zeros(len(R_in))
    R_out[0] = R_in[0]

    flag_bad_prev = np.zeros(len(R_in))
    flag_bad_prev[0] = 0
    for k in range(1, len(R_in)):

        dRdt = (R_in[k] - R_in[k - 1]) / (t_in[k] - t_in[k - 1])
        if abs(dRdt - Vr_in[k]) > thres_2:
            if flag_bad_prev[k - 1] == 0:
                R_out[k] = R_out[k - 1] + Vr_in[k - 1] * (t_in[k] - t_in[k - 1])
                flag_bad_prev[k] = 1
            else:
                R_out[k] = R_in[k]
                flag_bad_prev[k] = 0

        else:
            if flag_bad_prev[k - 1] == 0:
                R_out[k] = R_in[k]
                flag_bad_prev[k] = 0
            else:
                R_out[k] = R_out[k - 1] + Vr_in[k - 1] * (t_in[k] - t_in[k - 1])
                flag_bad_prev[k] = 1

    return R_out, start_ind


def func_derivation(x_0, h_0, m, g, d, l, h, mu, eta, i_f, kf, alpha, v0, N, Tmax):
    C_q = m / (1000 * d ** 3)
    C = 1000 * mu * d ** 5 * (C_q / (4 * g))
    m_z = 0.075 * l * d ** 3 / C
    c_b = i_f / (C_q * d)

    V_sr_set = []
    for i in range(200, 1400, 50):
        V_sr_set.append(i)
    V_sr_set = np.array(V_sr_set)

    Vsr_new_set = []
    for i in np.arange(200, 1350.1, 0.1):
        Vsr_new_set.append(i)
    Vsr_new_set = np.array(Vsr_new_set)

    # данные для сета
    KNKM_set = [0.2000, 0.16, 0.2, 0.246, 0.148, 0.162, 0.184, 0.199, 0.205, 0.209, 0.211, 0.216,
                0.220, 0.244, 0.299, 0.233, 0.236, 0.239, 0.241, 0.243, 0.245, 0.247, 0.248, 0.248]
    KNKM_set = np.array(KNKM_set)

    KNKM_set_new = np.zeros(KNKM_set.shape)

    for i in range(KNKM_set.shape[0]):
        KNKM_set_new[i] = KNKM_set[i] * (4.5 / (l / d))

    KnKm_finset = pchip_interpolate(V_sr_set, KNKM_set_new, Vsr_new_set)

    T_s = Tmax / (N - 1)

    t = []

    for i in np.arange(0, Tmax + T_s, T_s):
        t.append(i)
    t = np.array(t)

    B = (np.pi * mu * l * v0) / (2 * eta * h)
    B_KnKm = 6 * 10 ** 2 * c_b ** 1.7 * (C * v0) / (eta * m * d)

    x_true_full = np.zeros(N)
    h_true_full = np.zeros(N)
    Vx_true = np.zeros(N)
    Vh_true = np.zeros(N)
    z_true = np.zeros(N)
    int_2 = np.zeros(N)
    Vx_int = np.zeros(N)
    psi_true = np.zeros(N)
    psi_true_new = np.zeros(N)
    z_true_new = np.zeros(N)
    psi_true_new_2 = np.zeros(N)
    z_true_new_2 = np.zeros(N)
    h_true = np.zeros(N)
    x_true = np.zeros(N)

    KnKm_ratio = 0
    last_k = 0

    flag = 0
    for p in range(N):

        x_true_full[p] = (m / kf) * np.log(1 + (kf / m) * v0 * t[p] * np.cos(alpha)) + x_0
        h_true_full[p] = (m / kf) * np.log(
            np.cos(np.sqrt((kf * g) / m) * t[p]) + np.sqrt(kf / (m * g)) * v0 * np.sin(alpha) * np.sin(
                np.sqrt((kf * g) / m) * t[p])) + h_0

        if h_true_full[p] < 0 and flag == 0:
            max_dist = x_true_full[p - 1]
            max_h = max(h_true_full[0: p - 1])
            max_ind = h_true_full[0: p - 1].argmax()
            dist_to_peak = x_true_full[max_ind]
            t_full = t[p - 1]
            S_full = np.sqrt(dist_to_peak ** 2 + (81 / 64) * max_h ** 2) + np.sqrt(
                (max_dist - dist_to_peak) ** 2 + (81 / 64) * max_h ** 2)
            V_sr = S_full / t_full
            KnKm_ratio = KnKm_finset[(abs(Vsr_new_set - V_sr)) == min(abs(Vsr_new_set - V_sr))]
            flag = 1

    for k in range(N):

        Vx_true[k] = (v0 * np.cos(alpha)) / (1 + kf * v0 * t[k] * np.cos(alpha) / m)
        Vh_true[k] = (np.sqrt(m * g * kf) * v0 * np.sin(alpha) - m * g * np.tan(np.sqrt((kf * g) / m) * t[k])) / (
                np.sqrt(m * g * kf) + kf * v0 * np.sin(alpha) * np.tan(np.sqrt((kf * g) / m) * t[k]))
        x_true[k] = (m / kf) * np.log(1 + (kf / m) * v0 * t[k] * np.cos(alpha)) + x_0
        h_true[k] = (m / kf) * np.log(
            np.cos(np.sqrt((kf * g) / m) * t[k]) + np.sqrt(kf / (m * g)) * v0 * np.sin(alpha) * np.sin(
                np.sqrt((kf * g) / m) * t[k])) + h_0

        if k != 0:

            Vx_int[k] = Vx_int[k - 1] + Vx_true[k] * T_s
            int_2[k] = int_2[k - 1] + (KnKm_ratio * np.exp(-m_z * t[k]) / (Vx_true[k] ** 2 + Vh_true[k] ** 2)) * T_s
        else:

            Vx_int[k] = Vx_true[k] * T_s
            int_2[k] = (KnKm_ratio * np.exp(-m_z * t[k]) / (Vx_true[k] ** 2 + Vh_true[k] ** 2)) * T_s

        z_true[k] = (np.pi * g * mu * l * v0) / (2 * eta * h) * Vx_int[k] * int_2[k]
        abs_V_k = np.sqrt(Vx_true[k] ** 2 + Vh_true[k] ** 2)
        abs_V_0 = np.sqrt(Vx_true[0] ** 2 + Vh_true[0] ** 2)
        theta_k = np.arctan(Vh_true[k] / Vx_true[k])
        theta_0 = np.arctan(Vh_true[0] / Vx_true[0])
        Br = (theta_k ** 2 - theta_0 ** 2) / 2 + (theta_k ** 4 - theta_0 ** 4) / 8 + theta_0 * np.log(
            np.tan(theta_0 / 2 + np.pi / 4) / np.tan(theta_k / 2 + np.pi / 4))

        if k != 0:
            psi_true[k] = np.arctan(z_true[k] / x_true[k])
            psi_true_new[k] = np.arctan(((2 * B * KnKm_ratio) / (
                    g * x_true[k] * (1 / abs_V_k + np.log(abs_V_0 / abs_V_k) / (abs_V_0 - abs_V_k)))) * Br)
            z_true_new[k] = x_true[k] * np.tan(psi_true_new[k])
            psi_true_new_2[k] = np.arctan(
                ((2 * B_KnKm) / (g * x_true[k] * (1 / abs_V_k + np.log(abs_V_0 / abs_V_k) / (abs_V_0 - abs_V_k)))) * Br)
            z_true_new_2[k] = x_true[k] * np.tan(psi_true_new_2[k])

        else:
            psi_true[k] = 0
            psi_true_new[k] = 0
            z_true_new[k] = 0
            psi_true_new_2[k] = 0
            z_true_new_2[k] = 0

        if h_true[k] < 0:
            last_k = k + 1
            break

    return t, last_k, z_true, z_true_new, z_true_new_2


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
