import numpy as np
from cmath import sqrt

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

def func_quad_piece_app(x_L, y_L, h_L, x_0, y_0, h_0, m, g, SKO_R, SKO_Vr, SKO_theta, k0, dR, alpha, t_meas_full,
                        R_meas_full, Vr_meas_full, theta_meas_full, winlen, step_sld, parameters_bounds):
    x_0 = 0
    y_0 = 0
    h_0 = 0

    h_0_1 = R_meas_full[0] * np.sin(theta_meas_full[0]) + h_L
    x_0_1 = sqrt((R_meas_full[0] * np.cos(theta_meas_full[0])) ** 2 - y_L ** 2) + x_L

    h_0_2 = R_meas_full[1] * np.sin(theta_meas_full[1]) + h_L
    x_0_2 = sqrt((R_meas_full[1] * np.cos(theta_meas_full[1])) ** 2 - y_L ** 2) + x_L

    Vx0 = (x_0_2 - x_0_1) / (t_meas_full[1] - t_meas_full[0])
    Vh0 = (h_0_2 - h_0_1) / (t_meas_full[1] - t_meas_full[0])
    absV0 = sqrt(Vx0 ** 2 + Vh0 ** 2)


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

    # создаем пустые массивы для данных
    meas_t_ind = []
    x_est_top = []
    xhy_0_set = []

    Nlen = int(len(R_meas_full) / (winlen - 1))

    for w in range(Nlen):

        if w == 0:
            meas_t = [i for i in range(len(t_meas_full))]
            meas_t_ind.append(meas_t)
            t_meas_t = t_meas_full
            R_meas_t = R_meas_full
            theta_meas_t = theta_meas_full
            Vr_meas_t = Vr_meas_full

        else:
            meas_t_ind.append(meas_t[(winlen - 1) * w:])
            t_meas_t = t_meas_full[meas_t_ind[w][0]:]
            R_meas_t = R_meas_full[meas_t_ind[w][0]:]
            theta_meas_t = theta_meas_full[meas_t_ind[w][0]:]
            Vr_meas_t = Vr_meas_full[meas_t_ind[w][0]:]

        for q in range(len(WindowSet)):

            t_meas = t_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            R_meas = R_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            theta_meas = theta_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]
            Vr_meas = Vr_meas_t[WindowSet[q][0] - 1: WindowSet[q][1]]

            t_meas = t_meas - t_meas[0]

            h_0 = R_meas[0] * np.sin(theta_meas[0]) + h_L
            x_0 = sqrt((R_meas[0] * np.cos(theta_meas[0])) ** 2 - y_L ** 2) + x_L

            xhy_0 = [x_0, h_0, y_0]

            h_0_2 = R_meas[1] * np.sin(theta_meas[1]) + h_L
            x_0_2 = sqrt((R_meas[1] * np.cos(theta_meas[1])) ** 2 - y_L ** 2) + x_L

            Vx0 = (x_0_2 - x_0) / (t_meas[1] - t_meas[0])
            Vh0 = (h_0_2 - h_0) / (t_meas[1] - t_meas[0])
            absV0 = sqrt(Vx0 ** 2 + Vh0 ** 2)

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

                    DthetaDk = dthetadk.dthetadk(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                 x_est[2])
                    DthetaDv0 = dthetadv0.dthetadv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                    x_est[2])
                    DthetaDdeltaR = 0
                    DthetaDalpha = dthetadalpha.dthetadalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                             m,
                                                             g, x_est[2])

                    D2thetaDk2 = d2thetadk2.d2thetadk2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m, g,
                                                       x_est[2])
                    D2thetaDv02 = d2thetadv02.d2thetadv02(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0, m,
                                                          g,
                                                          x_est[2])
                    D2thetaDdeltaR2 = 0
                    D2thetaDalpha2 = d2thetadalpha2.d2theradalpha2(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
                                                                   x_0,
                                                                   h_0, m, g, x_est[2])

                    D2thetaDkDv0 = d2thetadkdv0.d2thetadkdv0(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0], x_0, h_0,
                                                             m,
                                                             g, x_est[2])
                    D2thetaDkDdeltaR = 0
                    D2thetaDkDalpha = d2thetadkdalpha.d2thetadkdalpha(x_L, y_L, h_L, t_k, x_est[1], x_est[3], x_est[0],
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
                                       (theta_meas[k] - theta) * D2thetaDdeltaRDalpha - DthetaDdeltaR * DthetaDalpha)

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
                    break



    return xhy_0_set, x_est_top, meas_t_ind, t_meas_full, R_meas_full, Vr_meas_full, theta_meas_full

def func_quad_piece_estimation(xhy_0_set, x_est_top, meas_t_ind, t_meas_full, N, winlen, m, g, x_L, y_L, h_L):

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

        if s == (Nlen-1):
            t_meas_t = t_meas_full[meas_t_ind[s][0]:]
            tmin = t_meas_t[0]
            tmax = t_meas_full[-1]
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
            t_meas_t = t_meas_full[meas_t_ind[s][0]:meas_t_ind[s][winlen]]
            tmin = t_meas_t[0]
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

        t_meas_plot.append(t + tmin)

        # размерность [[x,y,z]]
        x_0 = xhy_0_set[s][0].real
        h_0 = xhy_0_set[s][1]

        # массивы для данных
        x_tr_er = np.zeros(N)
        h_tr_er = np.zeros(N)
        R_est_full = np.zeros(N)
        theta_est_full = np.zeros(N)
        Vr_est_full = np.zeros(N)
        V_abs_est = np.zeros(N)
        Vx_true_er = np.zeros(N)
        Vh_true_er = np.zeros(N)
        theta_true_2 = np.zeros(N)

        for k in range(N):

            x_tr_er[k] = (m / x_est_fin[0].real) * np.log(
                1 + (x_est_fin[0].real / m) * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real)) + x_0

            h_tr_er[k] = (m / x_est_fin[0].real) * np.log(
                np.cos(sqrt((x_est_fin[0].real * g) / m) * t[k]) + sqrt(x_est_fin[0].real / (m * g)) * x_est_fin[1].real * np.sin(
                    x_est_fin[3].real) * np.sin(sqrt((x_est_fin[0].real * g) / m) * t[k])) + h_0

            R_est_full[k] = np.sqrt((x_L - (x_0 + (m / x_est_fin[0].real) * np.log(
                1 + (x_est_fin[0].real * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real)) / m))) ** 2 + y_L + (h_L - (
                        h_0 + (m / x_est_fin[0].real) * np.log(
                    np.cos(t[k] * sqrt(x_est_fin[0].real * g / m)) + np.sqrt(x_est_fin[0].real / (m * g)) * x_est_fin[1].real * np.sin(
                        x_est_fin[3].real) * np.sin(t[k] * np.sqrt(x_est_fin[0].real * g / m))))) ** 2) + x_est_fin[2].real

            theta_est_full[k] = np.arctan(((h_0 + (m / x_est_fin[0].real) * np.log(
                np.cos(t[k] * sqrt(x_est_fin[0].real * g / m)) + sqrt(x_est_fin[0].real / (m * g)) * x_est_fin[1].real * np.sin(
                    x_est_fin[3].real) * np.sin(t[k] * sqrt(x_est_fin[0].real * g / m)))) - h_L) / sqrt((x_L - (
                        x_0 + (m / x_est_fin[0].real) * np.log(
                    1 + (x_est_fin[0].real * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real)) / m))) ** 2 + y_L ** 2))

            Vr_est_full[k] = ((x_est_fin[1].real * np.cos(x_est_fin[3].real) * ((x_0 + (m / x_est_fin[0].real) * np.log(
                1 + (x_est_fin[0].real * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real)) / m)) - x_L)) / (1 + (
                        x_est_fin[0].real * t[k] * x_est_fin[1].real * np.cos(x_est_fin[3].real)) / m) + ((sqrt(
                m * g * x_est_fin[0].real) * x_est_fin[1].real * np.sin(x_est_fin[3].real) - m * g * np.tan(
                sqrt(x_est_fin[0].real * g / m) * t[k])) / (sqrt(m * g * x_est_fin[0].real) + x_est_fin[0].real * x_est_fin[1].real * np.sin(
                x_est_fin[3].real) * np.tan(sqrt(x_est_fin[0].real * g / m) * t[k]))) * ((h_0 + (m / x_est_fin[0].real) * np.log(
                np.cos(sqrt(x_est_fin[0].real * g / m) * t[k]) + sqrt(x_est_fin[0].real / (m * g)) * x_est_fin[1].real * np.sin(
                    x_est_fin[3].real) * np.sin(sqrt(x_est_fin[0].real * g / m) * t[k]))) - h_L)) / (sqrt((x_L - (
                        x_0 + (m / x_est_fin[0].real) * np.log(
                    1 + (x_est_fin[0].real * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real)) / m))) ** 2 + y_L ** 2 + (h_L - (
                        h_0 + (m / x_est_fin[0].real) * np.log(
                    np.cos(sqrt(x_est_fin[0].real * g / m) * t[k]) + sqrt(x_est_fin[0].real / (m * g)) * x_est_fin[1].real * np.sin(
                        x_est_fin[3].real) * np.sin(sqrt(x_est_fin[0].real* g / m) * t[k])))) ** 2))

            Vx_true_er[k] = (x_est_fin[1].real * np.cos(x_est_fin[3].real)) / (
                        1 + x_est_fin[0].real * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real) / m)

            Vh_true_er[k] = (sqrt(m * g * x_est_fin[0].real) * x_est_fin[1].real * np.sin(x_est_fin[3].real) - m * g * np.tan(
                sqrt((x_est_fin[0].real * g) / m) * t[k])) / (sqrt(m * g * x_est_fin[0].real) + x_est_fin[0].real * x_est_fin[1].real * np.sin(
                x_est_fin[3].real) * np.tan(sqrt((x_est_fin[0].real * g) / m) * t[k]))

            V_abs_est[k] = np.sqrt(Vx_true_er[k] ** 2 + Vh_true_er[k] ** 2)

            x_true = (m / x_est_fin[0].real) * np.log(1 + (x_est_fin[0].real / m) * x_est_fin[1].real * t[k] * np.cos(x_est_fin[3].real)) + x_0

            h_true = (m / x_est_fin[0].real) * np.log(
                np.cos(sqrt((x_est_fin[0].real * g) / m) * t[k]) + sqrt(x_est_fin[0].real / (m * g)) * x_est_fin[1].real * np.sin(
                    x_est_fin[3].real) * np.sin(sqrt((x_est_fin[0].real * g) / m) * t[k])) + h_0

            theta_true_2[k] = np.arctan((h_true - h_L) / sqrt((x_true - x_L) ** 2 + y_L ** 2))


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