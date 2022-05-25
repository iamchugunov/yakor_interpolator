import numpy as np
import pandas as pd
from scipy import interpolate


def active_reactive(time_meas_full_one_part, time_meas_full_two_part, x_est_stor_one_part, x_est_stor_two_part,
                    i_f_estimation, cannon_h, m, r, x_l, y_l, h_l, m_e=2.2, s_e=0.038, k_p=1.6446, delta_pi_y=0,
                    time_step=0.05):
    '''
    :param time_meas_full_one_part: ndarray
    :param time_meas_full_two_part: ndarray
    :param x_est_stor_one_part: ndarray
    :param x_est_stor_two_part: ndarray
    :param i_f_estimation: float
    :param cannon_h: float
    :param m: float
    :param r: float
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :param m_e: float
    :param s_e: float
    :param k_p: float
    :param delta_pi_y: float
    :param time_step: float
    :return: x_est_stor_active: ndarray
             y_ext_stor_active: ndarray
             time_meas: ndarray
             cx_set_active: ndarray
    '''

    data_43gost = pd.read_csv('43gost.csv')
    ballistic_coefficient, machs = ballistic_coefficient_43gost()

    time_meas_start = time_meas_full_one_part[-1]
    time_meas_end = time_meas_full_two_part[0]
    step_meas = round((time_meas_end - time_meas_start) / time_step) + 1
    time_meas = np.linspace(time_meas_start, time_meas_end, num=step_meas)

    time_meas_otn = (time_meas - time_meas_start) / (time_meas_end - time_meas_start)
    mu_p_graph = [0, 0.25, 0.5625, 0.82, 0.98, 0.98, 0.82, 0.77, 0.73, 0.72, 0.71, 0.70, 0.6875, 0.625, 0.375, 0.1, 0]
    time_graph = np.linspace(time_meas_start, time_meas_end, num=len(mu_p_graph))
    time_graph_otn = (time_graph - time_meas_start) / (time_meas_end - time_meas_start)

    mu_p_interp = interpolate.interp1d(time_graph_otn, mu_p_graph)(time_meas_otn)

    length_mu_p = len(mu_p_interp)

    length_i_l = 31
    length_b_y = 291
    i_l = np.linspace(1650, 1950, num=length_i_l)
    b_y = np.linspace(0.01, 0.3, num=length_b_y)

    mu_p = k_p * m_e * mu_p_interp / (time_meas_end - time_meas_start)

    norm_nev = np.zeros((length_b_y, length_i_l))
    norm_nev_x = np.zeros((length_b_y, length_i_l))
    norm_nev_h = np.zeros((length_b_y, length_i_l))

    velocity_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    alpha_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    velocity_x_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    velocity_h_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    x_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    h_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    m_ost = np.zeros((length_b_y, length_i_l, length_mu_p))
    as_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    as_tan_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    as_x_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    as_h_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))
    p_sum = np.zeros((length_b_y, length_i_l, length_mu_p))
    cx_set_active = np.zeros((length_b_y, length_i_l, length_mu_p))

    for i in range(length_b_y):
        for j in range(length_i_l):
            velocity_set_active[i, j, 0] = np.sqrt(x_est_stor_one_part[-1, 1] ** 2 + x_est_stor_one_part[-1, 4] ** 2)
            alpha_set_active[i, j, 0] = np.arctan(x_est_stor_one_part[-1, 4] / x_est_stor_one_part[-1, 1])
            # alpha
            velocity_x_set_active[i, j, 0] = x_est_stor_one_part[-1, 1]
            velocity_h_set_active[i, j, 0] = x_est_stor_one_part[-1, 4]
            x_set_active[i, j, 0] = x_est_stor_one_part[-1, 0]
            h_set_active[i, j, 0] = x_est_stor_one_part[-1, 3]

            velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
                h_set_active[i, j, 0] + cannon_h)
            rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(
                h_set_active[i, j, 0] + cannon_h)
            acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
                h_set_active[i, j, 0] + cannon_h)
            cx_int_tab = interpolate.interp1d(machs, ballistic_coefficient)(
                velocity_set_active[i, j, 0] / velocity_sound_gost)
            cx_set_active[i, j, 0] = cx_int_tab

            m_ost[i, j, 0] = m
            as_set_active[i, j, 0] = (rho_gost * (np.pi * r ** 2 / 4) * (
                    velocity_set_active[i, j, 0] ** 2 / 2) * cx_int_tab * i_f_estimation) / m_ost[i, j, 0]
            as_tan_active[i, j, 0] = - as_set_active[i, j, 0] - acc_gravity_gost * np.sin(alpha_set_active[i, j, 0])

            as_x_set_active[i, j, 0] = x_est_stor_one_part[-1, 2]
            as_h_set_active[i, j, 0] = x_est_stor_one_part[-1, 5]

            pi_y = interpolate.interp1d(data_43gost.height, data_43gost.pi)(h_set_active[i, j, 0])

            p_sum[i, j, 0] = mu_p[0] * i_l[j] - s_e * data_43gost.p[0] * (pi_y + delta_pi_y)

            for k in range(1, length_mu_p):
                velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
                    h_set_active[i, j, k - 1] + cannon_h)
                rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(
                    h_set_active[i, j, k - 1] + cannon_h)
                acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
                    h_set_active[i, j, k - 1] + cannon_h)
                cx_int_tab = interpolate.interp1d(machs, ballistic_coefficient)(
                    velocity_set_active[i, j, k - 1] / velocity_sound_gost)
                cx_set_active[i, j, k] = cx_int_tab
                pi_y = interpolate.interp1d(data_43gost.height, data_43gost.pi)(h_set_active[i, j, k - 1])

                m_ost[i, j, k] = m_ost[i, j, k - 1] - mu_p[k] * time_step
                p_sum[i, j, k] = mu_p[k] * i_l[j] - s_e * data_43gost.p[0] * (pi_y + delta_pi_y)

                as_set_active[i, j, k] = (rho_gost * (np.pi * r ** 2 / 4) * (
                        velocity_set_active[i, j, k - 1] ** 2 / 2) * cx_int_tab * i_f_estimation) / m_ost[i, j, k]
                as_tan_active[i, j, k] = - as_set_active[i, j, k] - acc_gravity_gost * np.sin(
                    alpha_set_active[i, j, k - 1]) + p_sum[i, j, k] / m_ost[i, j, k]

                velocity_set_active[i, j, k] = velocity_set_active[i, j, k - 1] + as_tan_active[i, j, k] * time_step
                alpha_set_active[i, j, k] = alpha_set_active[i, j, k - 1] - (
                        acc_gravity_gost * np.cos(alpha_set_active[i, j, k - 1]) / velocity_set_active[
                    i, j, k]) * time_step + (b_y[i] * p_sum[i, j, k]) / (
                                                    m_ost[i, j, k] * velocity_set_active[i, j, k]) * time_step
                as_x_set_active[i, j, k] = - as_set_active[i, j, k] * np.cos(alpha_set_active[i, j, k]) + p_sum[
                    i, j, k] / m_ost[i, j, k] * np.cos(alpha_set_active[i, j, k])
                as_h_set_active[i, j, k] = - as_set_active[i, j, k] * np.sin(alpha_set_active[i, j, k]) + p_sum[
                    i, j, k] / m_ost[i, j, k] * np.sin(alpha_set_active[i, j, k]) - acc_gravity_gost
                velocity_x_set_active[i, j, k] = velocity_set_active[i, j, k] * np.cos(alpha_set_active[i, j, k])
                velocity_h_set_active[i, j, k] = velocity_set_active[i, j, k] * np.sin(alpha_set_active[i, j, k])
                x_set_active[i, j, k] = x_set_active[i, j, k - 1] + velocity_x_set_active[i, j, k] * time_step
                h_set_active[i, j, k] = h_set_active[i, j, k - 1] + velocity_h_set_active[i, j, k] * time_step

            norm_nev_x[i, j] = abs(velocity_x_set_active[i, j, -1] - x_est_stor_two_part[0, 1])
            norm_nev_h[i, j] = abs(velocity_h_set_active[i, j, -1] - x_est_stor_two_part[0, 4])
            norm_nev[i, j] = norm_nev_x[i, j] + norm_nev_h[i, j]

    row, col = np.where(norm_nev == norm_nev.min())

    x_set_active = x_set_active[row, col, :].reshape(len(time_meas))
    h_set_active = h_set_active[row, col, :].reshape(len(time_meas))
    velocity_x_set_active = velocity_x_set_active[row, col, :].reshape(len(time_meas))
    velocity_h_set_active = velocity_h_set_active[row, col, :].reshape(len(time_meas))
    as_x_set_active = as_x_set_active[row, col, :].reshape(len(time_meas))
    as_h_set_active = as_h_set_active[row, col, :].reshape(len(time_meas))
    cx_set_active = cx_set_active[row, col, :].reshape(len(time_meas))

    y_set_active = np.zeros(len(time_meas))
    y_set_active[...] = x_est_stor_one_part[-1, 6]
    velocity_y_set_active = np.zeros(len(time_meas))
    velocity_y_set_active[...] = x_est_stor_one_part[-1, 7]
    as_y_set_active = np.zeros(len(time_meas))
    as_y_set_active[...] = x_est_stor_one_part[-1, 8]

    x_est_stor_active = np.column_stack((x_set_active, velocity_x_set_active, as_x_set_active, h_set_active,
                                         velocity_h_set_active, as_h_set_active, y_set_active, velocity_y_set_active,
                                         as_y_set_active))

    y_ext_stor_active = np.array([np.sqrt(
        (x_est_stor_active[:, 0] - x_l) ** 2 + (x_est_stor_active[:, 6] - y_l) ** 2 + (
                x_est_stor_active[:, 3] - h_l) ** 2),
        (x_est_stor_active[:, 1] * (x_est_stor_active[:, 0] - x_l) + x_est_stor_active[:, 4] * (
                x_est_stor_active[:, 3] - h_l) + x_est_stor_active[:, 7] * (
                 x_est_stor_active[:, 6] - y_l)) / np.sqrt(
            (x_est_stor_active[:, 0] - x_l) ** 2 + (x_est_stor_active[:, 6] - y_l) ** 2 + (
                    x_est_stor_active[:, 3] - h_l) ** 2),
        np.arcsin((x_est_stor_active[:, 3] - h_l) / np.sqrt(
            (x_est_stor_active[:, 0] - x_l) ** 2 + (x_est_stor_active[:, 6] - y_l) ** 2 + (
                    x_est_stor_active[:, 3] - h_l) ** 2))])

    return x_est_stor_active, y_ext_stor_active.T, cx_set_active, time_meas


def emissions_theta(theta_meas, thres_theta=0.015):
    '''
    exclusion of single emissions from measurements of angle (theta)
    :param theta_meas: ndarray
    :param thres_theta: float
    :return: float
    '''
    bad_ind = []
    for i in range(1, len(theta_meas) - 1):
        theta_diff_prev = theta_meas[i] - theta_meas[i - 1]
        theta_diff_next = theta_meas[i] - theta_meas[i + 1]
        if abs(theta_diff_prev) > thres_theta and abs(theta_diff_next) > thres_theta:
            bad_ind.append(i)
    return bad_ind


def act_react_partition(time_meas, radial_velocity_meas, time_act_dur=1.7,
                        diff_radial_velocity_drange_dtime=30, window_length=5):
    '''
    partitioning of active-reactive
    :param window_length: int
    :param diff_radial_velocity_drange_dtime: int
    :param time_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param time_act_dur: float
    :return: act_start_index: float
             act_end_index: float
    '''

    drange_meas_dtime = np.append(
        np.diff(radial_velocity_meas) / np.diff(time_meas),
        (radial_velocity_meas[-1] - radial_velocity_meas[-2]) / (time_meas[-1] - time_meas[-2]))

    act_start_index = 0
    mean_window_drange_prev = 0

    try:

        for i in range(len(time_meas) - window_length):
            drange_meas_dtime_window = drange_meas_dtime[i:window_length + i]
            mean_window_drange_dtime = np.max(drange_meas_dtime_window)
            if i > 0 and abs(mean_window_drange_dtime - mean_window_drange_prev) > diff_radial_velocity_drange_dtime:
                act_start_index = i + window_length
                break
            mean_window_drange_prev = mean_window_drange_dtime

    except NameError:
        act_start_index = 0

    times_act_start = time_meas[act_start_index]
    times_act_end_exp = times_act_start + time_act_dur

    act_end_index, times_act_start_value = min(enumerate(
        abs(time_meas - times_act_end_exp)
    ), key=lambda x: x[1])

    return act_start_index, act_end_index


def rts_angle_smoother(time_meas, theta_meas, sigma_theta, sigma_ksi, sigma_n):
    '''
    Rauch-Thug-Striebel algorithm angle filtering
    :param sigma_theta: float
    :param time_meas: ndarray
    :param theta_meas: ndarray
    :param sigma_ksi: float
    :param sigma_n: float
    :return: theta_smoother: ndarray
    '''
    length = len(time_meas)
    x_est_prev = np.array([theta_meas[0], sigma_theta])
    dx_est_prev = np.eye(2)

    x_est_stor = []
    dx_est_stor = []

    x_ext_stor = []
    dx_ext_stor = []

    dt_stor = []

    d_ksi = sigma_ksi ** 2
    dn = sigma_n ** 2

    I = np.eye(2)
    H = np.array([[1, 0]])

    for i, t in enumerate(time_meas):
        dt = 0.05 if i == 0 else t - time_meas[i - 1]
        F = np.array([[1, dt], [0, 1]])
        G = np.array([[0, 0], [0, dt]])

        x_ext = F.dot(x_est_prev)
        dx_ext = F.dot(dx_est_prev).dot(F.T) + (G * d_ksi).dot(G.T)
        s = H.dot(dx_ext).dot(H.T) + dn
        k = dx_ext.dot(H.T) * s ** (-1)
        x_est_prev = x_ext + k * (theta_meas[i] - H.dot(x_ext))
        dx_est_prev = (I - k.dot(H)).dot(dx_ext)
        x_est_stor.append(x_est_prev)
        dx_est_stor.append(dx_est_prev)
        x_ext_stor.append(x_ext)
        dx_ext_stor.append(dx_ext)
        dt_stor.append(dt)

    x_est_sm_prev = x_est_stor[-1]

    x_est_sm_stor = [x_est_sm_prev]
    dx_est_sm_prev = dx_est_stor[-1]

    theta_smoother = np.zeros(length)
    theta_smoother[0] = x_est_sm_prev[0][0]

    for i in range(length - 1):
        F = np.array([[1, dt_stor[length - i - 1]], [0, 1]])
        k_sm = dx_est_stor[length - i - 2].dot(F.T).dot(np.linalg.inv(dx_ext_stor[length - i - 1]))
        x_est_sm = x_est_stor[length - i - 2] + k_sm.dot((x_est_sm_prev - x_ext_stor[length - i - 1]))
        dx_est_sm = dx_est_stor[length - i - 2] + k_sm.dot(
            dx_est_sm_prev - dx_ext_stor[length - i - 1]).dot(k_sm.T)
        x_est_sm_stor.append(x_est_sm)
        x_est_sm_prev = x_est_sm
        dx_est_sm_prev = dx_est_sm

        theta_smoother[i + 1] = x_est_sm[0][0]

    return theta_smoother[::-1]


def rts_coord_smoother(time_meas, range_meas, radial_velocity_meas, sigma_coord, sigma_ksi, sigma_n1,
                       sigma_n2):
    '''
    Rauch-Thug-Striebel algorithm distance and speed filtering
    :param sigma_coord: float
    :param time_meas: ndarray
    :param range_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param sigma_ksi: float
    :param sigma_n1: float
    :param sigma_n2: float
    :return: range_smoother: ndarray
             radial_velocity_smoother: ndarray
    '''

    length = len(time_meas)
    x_est_prev = np.array([range_meas[0], radial_velocity_meas[0], sigma_coord])
    dx_est_prev = np.eye(3)

    x_est_stor = []
    x_ext_stor = []

    dx_est_stor = []
    dx_ext_stor = []
    dt_stor = []

    d_ksi = sigma_ksi ** 2
    dn = np.array([[sigma_n1 ** 2, 0], [0, sigma_n2 ** 2]])

    I = np.eye(3)
    H = np.array([[1, 0, 0], [0, 1, 0]])

    for i, t in enumerate(time_meas):
        dt = 0.05 if i == 0 else t - time_meas[i - 1]
        F = np.array([[1, dt, 0], [0, 1, dt], [0, 0, 1]])
        G = np.array([[0, 0, 0], [0, 0, 0], [0, 0, dt]])

        x_ext = F.dot(x_est_prev)
        dx_ext = F.dot(dx_est_prev).dot(F.T) + G.dot(d_ksi).dot(G.T)
        s = H.dot(dx_ext).dot(H.T) + dn
        k = dx_ext.dot(H.T).dot(np.linalg.inv(s))
        x_est_prev = x_ext + k.dot(np.array([range_meas[i], radial_velocity_meas[i]]) - H.dot(x_ext))
        dx_est_prev = (I - k.dot(H)).dot(dx_ext)
        x_est_stor.append(x_est_prev)
        dx_est_stor.append(dx_est_prev)
        x_ext_stor.append(x_ext)
        dx_ext_stor.append(dx_ext)
        dt_stor.append(dt)

    x_est_sm_prev = x_est_stor[-1]
    x_est_sm_stor = [x_est_sm_prev]
    dx_est_sm_prev = dx_est_stor[-1]

    range_smoother = np.zeros(length)
    radial_velocity_smoother = np.zeros(length)

    range_smoother[0] = x_est_sm_prev[0]
    radial_velocity_smoother[0] = x_est_sm_prev[1]

    for i in range(length - 1):
        F = np.array([[1, dt_stor[length - i - 1], 0], [0, 1, dt_stor[length - i - 1]], [0, 0, 1]])
        K_sm = dx_est_stor[length - i - 2].dot(F.T).dot(np.linalg.inv(dx_ext_stor[length - i - 1]))
        x_est_sm = x_est_stor[length - i - 2] + K_sm.dot((x_est_sm_prev - x_ext_stor[length - i - 1]))
        dx_est_sm = dx_est_stor[length - i - 2] + K_sm.dot(
            dx_est_sm_prev - dx_ext_stor[length - i - 1]).dot(K_sm.T)
        x_est_sm_stor.append(x_est_sm)
        x_est_sm_prev = x_est_sm
        dx_est_sm_prev = dx_est_sm

        range_smoother[i + 1] = x_est_sm[0]
        radial_velocity_smoother[i + 1] = x_est_sm[1]

    return range_smoother[::-1], radial_velocity_smoother[::-1]


def time_step_filling_data(time_meas, range_meas, radial_velocity_meas, theta_meas, time_step=0.05):
    '''
    timekeeping
    :param time_meas: ndarray
    :param range_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param theta_meas: ndarray
    :param time_step: float
    :return: time_meas_full: ndarray
             range_meas_full: ndarray
             radial_velocity_meas_full: ndarray
             theta_meas_full: ndarray
    '''

    time_meas_start = (np.fix(time_meas[0] / time_step) + 1) * time_step
    time_meas_end = np.fix(time_meas[-1] / time_step) * time_step
    step_meas = round((time_meas_end - time_meas_start) / time_step) + 1
    time_meas_full = np.linspace(time_meas_start, time_meas_end, num=step_meas)

    range_meas_full = np.zeros(step_meas)
    radial_velocity_meas_full = np.zeros(step_meas)
    theta_meas_full = np.zeros(step_meas)

    for i, time in enumerate(time_meas_full):
        range_meas_full[i] = interpolate.interp1d(time_meas, range_meas)(time)
        radial_velocity_meas_full[i] = interpolate.interp1d(time_meas, radial_velocity_meas)(time)
        theta_meas_full[i] = interpolate.interp1d(time_meas, theta_meas)(time)

    return time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full


def lsm_processing_window_rh(time_meas, range_meas, radial_velocity_meas, theta_meas,
                             alpha_meas, x_l, y_l, h_l, sigma_n_range=2, sigma_n_range_velocity=1):
    '''
    window approximation
    :param time_meas: ndarray
    :param range_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param theta_meas: ndarray
    :param alpha_meas: ndarray
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :param sigma_n_range: int
    :param sigma_n_range_velocity: int
    :return: x_estimation: ndarray
    '''

    length = len(time_meas)
    range_h_apr = range_meas * np.sin(theta_meas) + h_l
    range_x_apr = np.sqrt((range_meas * np.cos(theta_meas)) ** 2 - y_l ** 2) + x_l

    range_h_l = range_meas * np.sin(theta_meas)
    range_x_l = np.sqrt(range_meas ** 2 - range_h_l ** 2)

    radial_velocity_x_apr = (radial_velocity_meas / np.cos(theta_meas - alpha_meas)) * np.cos(alpha_meas)
    radial_velocity_h_apr = (radial_velocity_meas * range_meas - range_x_l * radial_velocity_x_apr) / range_h_l

    dn_1_3 = sigma_n_range ** 2 * np.eye(length)
    dn_2_4 = sigma_n_range_velocity ** 2 * np.eye(length)
    zero_matrix = np.zeros((length, length))

    dn = np.concatenate(
        [np.concatenate([dn_1_3, zero_matrix, zero_matrix, zero_matrix]),
         np.concatenate([zero_matrix, dn_2_4, zero_matrix, zero_matrix]),
         np.concatenate([zero_matrix, zero_matrix, dn_1_3, zero_matrix]),
         np.concatenate([zero_matrix, zero_matrix, zero_matrix, dn_2_4])], axis=1
    )

    z_matrix = np.concatenate([
        range_x_apr.T,
        radial_velocity_x_apr.T,
        range_h_apr.T,
        radial_velocity_h_apr.T
    ])
    z_matrix = z_matrix.reshape(20, 1)

    tau = np.zeros(length)
    f_range_x = np.zeros((length, 6))
    f_radial_velocity_x = np.zeros((length, 6))
    f_range_h = np.zeros((length, 6))
    f_radial_velocity_h = np.zeros((length, 6))

    for i, time in enumerate(time_meas):
        tau[i] = - (time_meas[-1] - time)
        f_range_x[i, :] = [1, tau[i], tau[i] ** 2 / 2, 0, 0, 0]
        f_radial_velocity_x[i, :] = [0, 1, tau[i], 0, 0, 0]
        f_range_h[i, :] = [0, 0, 0, 1, tau[i], tau[i] ** 2 / 2]
        f_radial_velocity_h[i, :] = [0, 0, 0, 0, 1, tau[i]]

    f_matrix = np.concatenate([
        f_range_x,
        f_radial_velocity_x,
        f_range_h,
        f_radial_velocity_h
    ])

    x_estimation = np.linalg.inv(f_matrix.T.dot(np.linalg.inv(dn)).dot(f_matrix)).dot(f_matrix.T).dot(
        np.linalg.inv(dn)).dot(
        z_matrix)

    return x_estimation


def lsm_processing_window_rh_new(time_meas, range_x_apr, range_h_apr, radial_velocity_x_apr,
                                 radial_velocity_h_apr, sigma_n_range=2, sigma_n_range_velocity=1):
    '''
    window approximation
    :param time_meas: ndarray
    :param range_x_apr: ndarray
    :param range_h_apr: ndarray
    :param radial_velocity_x_apr: ndarray
    :param radial_velocity_h_apr: ndarray
    :param sigma_n_range: int
    :param sigma_n_range_velocity: int
    :return: x_estimation: ndarray
    '''

    length = len(time_meas)

    dn_1_3 = sigma_n_range ** 2 * np.eye(length)
    dn_2_4 = sigma_n_range_velocity ** 2 * np.eye(length)

    zero_matrix = np.zeros((length, length))

    dn = np.concatenate(
        [np.concatenate([dn_1_3, zero_matrix, zero_matrix, zero_matrix]),
         np.concatenate([zero_matrix, dn_2_4, zero_matrix, zero_matrix]),
         np.concatenate([zero_matrix, zero_matrix, dn_1_3, zero_matrix]),
         np.concatenate([zero_matrix, zero_matrix, zero_matrix, dn_2_4])], axis=1
    )

    z_matrix = np.concatenate([
        range_x_apr.T,
        radial_velocity_x_apr.T,
        range_h_apr.T,
        radial_velocity_h_apr.T
    ])
    z_matrix = z_matrix.reshape(20, 1)

    tau = np.zeros(length)
    f_range_x = np.zeros((length, 6))
    f_radial_velocity_x = np.zeros((length, 6))
    f_range_h = np.zeros((length, 6))
    f_radial_velocity_h = np.zeros((length, 6))

    for i, time in enumerate(time_meas):
        tau[i] = - (time_meas[-1] - time)
        f_range_x[i, :] = [1, tau[i], tau[i] ** 2 / 2, 0, 0, 0]
        f_radial_velocity_x[i, :] = [0, 1, tau[i], 0, 0, 0]
        f_range_h[i, :] = [0, 0, 0, 1, tau[i], tau[i] ** 2 / 2]
        f_radial_velocity_h[i, :] = [0, 0, 0, 0, 1, tau[i]]

    f_matrix = np.concatenate([
        f_range_x,
        f_radial_velocity_x,
        f_range_h,
        f_radial_velocity_h
    ])

    x_estimation = np.linalg.inv(f_matrix.T.dot(np.linalg.inv(dn)).dot(f_matrix)).dot(f_matrix.T).dot(
        np.linalg.inv(dn)).dot(
        z_matrix)

    return x_estimation


def formation_estimation_on_alpha(time_meas, range_meas, radial_velocity_meas, theta_meas, x_l, y_l, h_l):
    '''
    estimation alpha
    :param time_meas: ndarray
    :param range_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param theta_meas: ndarray
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :return: x_estimation: ndarray
    '''

    estimation_int = np.array([0, 1, 2, 3, 4])

    alpha_meas = lsm_alpha(time_meas, range_meas, theta_meas)

    length = len(time_meas) - len(estimation_int) + 1
    x_estimation_stor = np.zeros((length, 6))

    for i in range(length):
        estimation_int_i = estimation_int + i
        range_meas_estimation = range_meas[estimation_int_i]
        radial_velocity_meas_estimation = radial_velocity_meas[estimation_int_i]
        theta_meas_estimation = theta_meas[estimation_int_i]
        time_meas_estimation = time_meas[estimation_int_i]
        alpha_meas_estimation = alpha_meas[estimation_int_i]

        x_estimation_tmp = lsm_processing_window_rh(time_meas_estimation, range_meas_estimation,
                                                    radial_velocity_meas_estimation, theta_meas_estimation,
                                                    alpha_meas_estimation, x_l, y_l, h_l)

        tau_shift_2_start = time_meas_estimation[0] - time_meas_estimation[-1]
        f_shift_2_start = np.array([
            [1, tau_shift_2_start, tau_shift_2_start ** 2 / 2, 0, 0, 0],
            [0, 1, tau_shift_2_start, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, tau_shift_2_start, tau_shift_2_start ** 2 / 2],
            [0, 0, 0, 0, 1, tau_shift_2_start],
            [0, 0, 0, 0, 0, 1]
        ])

        x_estimation_init = f_shift_2_start.dot(x_estimation_tmp)
        x_estimation_stor[i] = x_estimation_init.reshape(6)

    return x_estimation_stor


def formation_estimation_on_alpha_mina(time_meas, range_meas, radial_velocity_meas, theta_meas, x_l, y_l, h_l):
    '''
    estimation alpha
    :param time_meas: ndarray
    :param range_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param theta_meas: ndarray
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :return: x_estimation: ndarray
    '''

    estimation_int = np.array([0, 1, 2, 3, 4])

    alpha_test = np.arctan(
        np.diff(range_meas * np.sin(theta_meas)) / np.diff(range_meas * np.cos(theta_meas)))
    alpha_test = np.append(alpha_test, alpha_test[-1])
    alpha_meas = rts_angle_smoother(time_meas, alpha_test, sigma_theta=0.4, sigma_ksi=1e-2,
                                    sigma_n=5e-3)

    range_h_apr, range_x_apr, radial_velocity_x_apr, radial_velocity_h_apr = velocity_aprox(range_meas,
                                                                                            radial_velocity_meas,
                                                                                            theta_meas, alpha_meas, x_l,
                                                                                            y_l,
                                                                                            h_l)

    flag_cos_min = 0
    cos_min_thres = 0.25
    i_time_cos_min = []

    for i in range(len(time_meas)):
        if abs(np.cos(theta_meas[i] - alpha_meas[i])) < cos_min_thres:
            i_time_cos_min.append(i)
            flag_cos_min = 1

    time_meas_for_apr_ind = np.concatenate(
        [i_time_cos_min[0] - np.array([5, 4, 3, 2, 1]), i_time_cos_min[-1] + np.array([1, 2, 3, 4, 5])])
    time_meas_for_apr = time_meas[time_meas_for_apr_ind]
    time_meas_after_apr = time_meas[time_meas_for_apr_ind[0]:time_meas_for_apr_ind[-1] + 1]

    range_h_l_for_apr_test = range_meas[time_meas_for_apr_ind] * np.sin(theta_meas[time_meas_for_apr_ind])
    range_x_l_for_apr_test = np.sqrt(range_meas[time_meas_for_apr_ind] ** 2 - range_h_l_for_apr_test ** 2)

    radial_velocity_x_for_apr_test = (radial_velocity_meas[time_meas_for_apr_ind] / np.cos(
        theta_meas[time_meas_for_apr_ind] - alpha_meas[time_meas_for_apr_ind])) * np.cos(
        alpha_meas[time_meas_for_apr_ind])
    radial_velocity_h_for_apr_test = (radial_velocity_meas[time_meas_for_apr_ind] * range_meas[
        time_meas_for_apr_ind] - range_x_l_for_apr_test * radial_velocity_x_for_apr_test) / range_h_l_for_apr_test

    out_lsm_radial_velocity_x = lsm_cubic(time_meas_for_apr, radial_velocity_x_for_apr_test)
    radial_velocity_x_apr_lsm = np.zeros(len(time_meas_after_apr))
    for i, time in enumerate(time_meas_after_apr):
        radial_velocity_x_apr_lsm[i] = (out_lsm_radial_velocity_x[0] + time * out_lsm_radial_velocity_x[1] + time ** 2 *
                                        out_lsm_radial_velocity_x[
                                            2] + time ** 3 * out_lsm_radial_velocity_x[3])

    out_lsm_velocity_h = lsm_cubic(time_meas_for_apr, radial_velocity_h_for_apr_test)
    radial_velocity_h_apr_lsm = np.zeros(len(time_meas_after_apr))
    for i, time in enumerate(time_meas_after_apr):
        radial_velocity_h_apr_lsm[i] = (
                    out_lsm_velocity_h[0] + time * out_lsm_velocity_h[1] + time ** 2 * out_lsm_velocity_h[
                2] + time ** 3 * out_lsm_velocity_h[3])

    radial_velocity_x_apr[i_time_cos_min[0] - 5: i_time_cos_min[-1] + 6] = radial_velocity_x_apr_lsm
    radial_velocity_h_apr[i_time_cos_min[0] - 5: i_time_cos_min[-1] + 6] = radial_velocity_h_apr_lsm

    length = len(time_meas) - len(estimation_int) + 1
    x_estimation_stor = np.zeros((length, 6))

    if not flag_cos_min:
        for i in range(length):
            estimation_int_i = estimation_int + i
            range_meas_estimation = range_meas[estimation_int_i]
            radial_velocity_meas_estimation = radial_velocity_meas[estimation_int_i]
            theta_meas_estimation = theta_meas[estimation_int_i]
            time_meas_estimation = time_meas[estimation_int_i]
            alpha_meas_estimation = alpha_meas[estimation_int_i]

            x_estimation_tmp = lsm_processing_window_rh(time_meas_estimation, range_meas_estimation,
                                                        radial_velocity_meas_estimation, theta_meas_estimation,
                                                        alpha_meas_estimation, x_l, y_l, h_l)

            tau_shift_2_start = time_meas_estimation[0] - time_meas_estimation[-1]
            f_shift_2_start = np.array([
                [1, tau_shift_2_start, tau_shift_2_start ** 2 / 2, 0, 0, 0],
                [0, 1, tau_shift_2_start, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, tau_shift_2_start, tau_shift_2_start ** 2 / 2],
                [0, 0, 0, 0, 1, tau_shift_2_start],
                [0, 0, 0, 0, 0, 1]
            ])

            x_estimation_init = f_shift_2_start.dot(x_estimation_tmp)
            x_estimation_stor[i] = x_estimation_init.reshape(6)
    else:

        for i in range(length):
            estimation_int_i = estimation_int + i
            time_meas_estimation = time_meas[estimation_int_i]

            range_h_apr_k = range_h_apr[estimation_int_i]
            range_x_apr_k = range_x_apr[estimation_int_i]
            radial_velocity_x_apr_k = radial_velocity_x_apr[estimation_int_i]
            radial_velocity_h_apr_k = radial_velocity_h_apr[estimation_int_i]

            x_estimation_tmp = lsm_processing_window_rh_new(time_meas_estimation, range_x_apr_k, range_h_apr_k,
                                                            radial_velocity_x_apr_k, radial_velocity_h_apr_k)

            tau_shift_2_start = time_meas_estimation[0] - time_meas_estimation[-1]
            f_shift_2_start = np.array([
                [1, tau_shift_2_start, tau_shift_2_start ** 2 / 2, 0, 0, 0],
                [0, 1, tau_shift_2_start, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, 0, 0, 1, tau_shift_2_start, tau_shift_2_start ** 2 / 2],
                [0, 0, 0, 0, 1, tau_shift_2_start],
                [0, 0, 0, 0, 0, 1]
            ])

            x_estimation_init = f_shift_2_start.dot(x_estimation_tmp)
            x_estimation_stor[i] = x_estimation_init.reshape(6)

    return x_estimation_stor


def shape_factor_from_velocity(x_estimation_stor, time_meas, range_meas, theta_meas, r, m, cannon_h,
                               window_length):
    '''
    estimate to shape factor
    :param m: float
    :param r: float
    :param theta_meas: ndarray
    :param range_meas: ndarray
    :param time_meas: ndarray
    :param x_estimation_stor: ndarray
    :param window_length: float
    :param cannon_h: float
    :return: i_f_from_acceleration_x: ndarray
             velocity_abs_poly_estimation: float
    '''

    data_43gost = pd.read_csv('43gost.csv')

    velocity_abs_poly_estimation = np.sqrt(x_estimation_stor[:, 1] ** 2 + x_estimation_stor[:, 4] ** 2)
    acceleration_t_from_velocity_abs = np.diff(velocity_abs_poly_estimation) / np.diff(
        time_meas[:len(time_meas) - window_length + 1])
    acceleration_t_from_velocity_abs = np.append(acceleration_t_from_velocity_abs, acceleration_t_from_velocity_abs[-1])

    length = len(velocity_abs_poly_estimation)

    as_from_velocity_abs = np.zeros(length)
    cx_from_acceleration_x = np.zeros(length)
    i_f_from_acceleration_x = np.zeros(length)

    ballistic_coefficient, machs = ballistic_coefficient_43gost()
    alpha_meas = lsm_alpha(time_meas, range_meas, theta_meas)

    for i, velocity_abs in enumerate(velocity_abs_poly_estimation):
        velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
            x_estimation_stor[i, 3] + cannon_h)
        rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(
            x_estimation_stor[i, 3] + cannon_h)
        acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
            x_estimation_stor[i, 3] + cannon_h)

        as_from_velocity_abs[i] = acceleration_t_from_velocity_abs[i] + acc_gravity_gost * np.sin(
            alpha_meas[i])
        cx_from_acceleration_x[i] = interpolate.interp1d(machs, ballistic_coefficient)(
            velocity_abs_poly_estimation[i] / velocity_sound_gost).tolist()
        i_f_from_acceleration_x[i] = m * as_from_velocity_abs[i] / (
                rho_gost * cx_from_acceleration_x[i] * (np.pi * r ** 2 / 4) * velocity_abs_poly_estimation[
            i] ** 2 / 2)

    return i_f_from_acceleration_x, velocity_abs_poly_estimation


def approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length, std_window_length):
    '''
    estimate to shape factor total to trajector
    :param i_f_from_acceleration_x: ndarray
    :param std_shift_length: int
    :param std_window_length: int
    :return: i_f_estimation: float
    '''

    cur_index_full = []

    cur_index = np.arange(0, std_window_length)
    cur_index_full.append(cur_index)

    std_i_f = []

    for i in range(int(np.fix(len(i_f_from_acceleration_x) / std_shift_length) + 1)):
        if i != 0:
            cur_index_full.append(cur_index_full[-1] + std_shift_length)

        if cur_index_full[i][-1] >= len(i_f_from_acceleration_x):
            break
        std_i_f.append(np.std(i_f_from_acceleration_x[cur_index_full[i]]))

    min_index, min_value = min(enumerate(std_i_f), key=lambda x: x[1])
    i_f_estimation = abs(np.mean(i_f_from_acceleration_x[cur_index_full[:][min_index]]))

    return i_f_estimation


def initial_velocity_estimation_calculation_norm(time_meas, i_f_estimation, velocity_abs_poly_estimation,
                                                 alpha_0, r, m, cannon_h, velocity_abs_0_tab,
                                                 velocity_abs_0_max_dex=40, velocity_abs_0_step=0.2, time_step=0.05):
    '''
    initial velocity estimate
    :param time_meas: ndarray
    :param i_f_estimation: float
    :param velocity_abs_poly_estimation: ndarray
    :param alpha_0: float
    :param r: float
    :param m: float
    :param cannon_h: float
    :param velocity_abs_0_tab: int
    :param velocity_abs_0_max_dex: int
    :param velocity_abs_0_step: float
    :param time_step: float
    :return: velocity_0_estimation: float
    '''

    data_43gost = pd.read_csv('43gost.csv')

    velocity_abs_0_full = np.linspace(velocity_abs_0_tab - velocity_abs_0_max_dex,
                                      velocity_abs_0_tab + velocity_abs_0_max_dex,
                                      num=int(np.fix((2 * velocity_abs_0_max_dex) / velocity_abs_0_step + 1)))

    time_meas_tab = np.linspace(0, time_meas[9], num=int(np.fix(time_meas[9] / time_step) + 1))

    velocity_set_tab = np.zeros((len(velocity_abs_0_full), len(time_meas_tab)))
    alpha_set_tab = np.zeros((len(velocity_abs_0_full), len(time_meas_tab)))
    velocity_x_set_tab = np.zeros((len(velocity_abs_0_full), len(time_meas_tab)))
    velocity_h_set_tab = np.zeros((len(velocity_abs_0_full), len(time_meas_tab)))
    x_set_tab = np.zeros((len(velocity_abs_0_full), len(time_meas_tab)))
    h_set_tab = np.zeros((len(velocity_abs_0_full), len(time_meas_tab)))

    ballistic_coefficient, machs = ballistic_coefficient_43gost()

    norm_nev = np.zeros(len(velocity_abs_0_full))

    for i, velocity_abs_0 in enumerate(velocity_abs_0_full):
        velocity_set_tab[i, 0] = velocity_abs_0
        alpha_set_tab[i, 0] = alpha_0
        velocity_x_set_tab[i, 0] = velocity_abs_0 * np.cos(alpha_0)
        velocity_h_set_tab[i, 0] = velocity_abs_0 * np.sin(alpha_0)
        x_set_tab[i, 0] = 0
        h_set_tab[i, 0] = 0

        for j in range(1, len(time_meas_tab)):
            velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
                h_set_tab[i, j - 1] + cannon_h)
            rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(
                h_set_tab[i, j - 1] + cannon_h)
            acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
                h_set_tab[i, j - 1] + cannon_h)
            cx_int_tab = interpolate.interp1d(machs, ballistic_coefficient)(
                velocity_set_tab[i, j - 1] / velocity_sound_gost)
            as_tab = (rho_gost * (np.pi * r ** 2 / 4) * (
                    velocity_set_tab[i, j - 1] ** 2 / 2) * cx_int_tab * i_f_estimation) / m

            velocity_set_tab[i, j] = velocity_set_tab[i, j - 1] + (
                    - as_tab - acc_gravity_gost * np.sin(alpha_set_tab[i, j - 1])) * time_step
            alpha_set_tab[i, j] = alpha_set_tab[i, j - 1] - (
                    acc_gravity_gost * np.cos(alpha_set_tab[i, j - 1]) / velocity_set_tab[i, j - 1]) * time_step
            velocity_x_set_tab[i, j] = velocity_abs_0 * np.cos(alpha_0)
            velocity_h_set_tab[i, j] = velocity_abs_0 * np.sin(alpha_0)
            x_set_tab[i, j] = x_set_tab[i, j - 1] + velocity_x_set_tab[i, j - 1] * time_step
            h_set_tab[i, j] = h_set_tab[i, j - 1] + velocity_h_set_tab[i, j - 1] * time_step

        norm_nev[i] = np.linalg.norm(
            velocity_set_tab[i][len(velocity_set_tab[i]) - 10:] - velocity_abs_poly_estimation[0:10])

    min_index, min_value = min(enumerate(norm_nev), key=lambda x: x[1])
    velocity_0_estimation = velocity_abs_0_full[min_index]

    return velocity_0_estimation


def structuring_approximate_values(time_meas_start, x_set_0, h_set_0, velocity_x_set_0, velocity_h_set_0,
                                   i_f_estimation,
                                   alpha_0, time_meas, r, m, x_l, y_l, h_l, cannon_h, time_step=0.05):
    '''
    control action for the filter
    :param time_meas_start: float
    :param x_set_0: float
    :param h_set_0: float
    :param velocity_x_set_0: float
    :param velocity_h_set_0: float
    :param i_f_estimation: float
    :param alpha_0: float
    :param time_meas: ndarray
    :param r: float
    :param m: float
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :param cannon_h: float
    :param time_step: float
    :return: time_meas: ndarray
             x_set: ndarray
             h_set: ndarray
             velocity_x_set: ndarray
             velocity_h_set: ndarray
             as_x_set: ndarray
             as_h_set: ndarray
    '''

    data_43gost = pd.read_csv('43gost.csv')

    time_meas_end = np.fix(time_meas[-1] / time_step) * time_step
    step_meas = round((time_meas_end - time_meas_start) / time_step) + 1
    time_meas = np.linspace(time_meas_start, time_meas_end, num=step_meas)

    length = len(time_meas)

    ballistic_coefficient, machs = ballistic_coefficient_43gost()

    x_set = np.zeros(length)
    h_set = np.zeros(length)
    velocity_x_set = np.zeros(length)
    velocity_h_set = np.zeros(length)
    velocity_set = np.zeros(length)
    as_x_set = np.zeros(length)
    as_h_set = np.zeros(length)
    alpha_set = np.zeros(length)

    range_set = np.zeros(length)
    radial_velocity_set = np.zeros(length)
    theta_set = np.zeros(length)

    x_set[0] = x_set_0
    h_set[0] = h_set_0
    velocity_x_set[0] = velocity_x_set_0
    velocity_h_set[0] = velocity_h_set_0
    velocity_set[0] = np.sqrt(velocity_x_set[0] ** 2 + velocity_h_set[0] ** 2)
    alpha_set[0] = alpha_0

    velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
        h_set[0] + cannon_h)
    rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(h_set[0] + cannon_h)
    acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
        h_set[0] + cannon_h)
    cx_int = interpolate.interp1d(machs, ballistic_coefficient)(
        velocity_set[0] / velocity_sound_gost)
    as_set = (rho_gost * (np.pi * r ** 2 / 4) * (
            velocity_set[0] ** 2 / 2) * cx_int * i_f_estimation) / m

    as_x_set[0] = - as_set * np.cos(alpha_0)
    as_h_set[0] = - (as_set * np.sin(alpha_0) + acc_gravity_gost)

    range_set[0] = np.sqrt((x_set[0] - x_l) ** 2 + y_l ** 2 + (h_set[0] - h_l) ** 2)
    radial_velocity_set[0] = (velocity_x_set[0] * (x_set[0] - x_l) + velocity_h_set[
        0] * (h_set[0] - h_l)) / range_set[0]
    theta_set[0] = np.arcsin((h_set[0] - h_l) / range_set[0])

    for i in range(1, length):
        velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
            h_set[i - 1] + cannon_h)
        rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(h_set[i - 1] + cannon_h)
        acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
            h_set[i - 1] + cannon_h)
        cx_int = interpolate.interp1d(machs, ballistic_coefficient)(
            velocity_set[i - 1] / velocity_sound_gost)

        as_set = (rho_gost * (np.pi * r ** 2 / 4) * (
                velocity_set[i - 1] ** 2 / 2) * cx_int * i_f_estimation) / m
        velocity_set[i] = velocity_set[i - 1] + (
                - as_set - acc_gravity_gost * np.sin(alpha_set[i - 1])) * time_step
        alpha_set[i] = alpha_set[i - 1] + (
                - acc_gravity_gost * np.cos(alpha_set[i - 1]) / velocity_set[i - 1]) * time_step
        x_set[i] = x_set[i - 1] + velocity_x_set[i - 1] * time_step
        h_set[i] = h_set[i - 1] + velocity_h_set[i - 1] * time_step
        velocity_x_set[i] = velocity_set[i] * np.cos(alpha_set[i])
        velocity_h_set[i] = velocity_set[i] * np.sin(alpha_set[i])

        velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
            h_set[i] + cannon_h)
        rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(h_set[i] + cannon_h)
        acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
            h_set[i] + cannon_h)
        cx_int = interpolate.interp1d(machs, ballistic_coefficient)(
            velocity_set[i] / velocity_sound_gost)
        as_set = (rho_gost * (np.pi * r ** 2 / 4) * (
                velocity_set[i] ** 2 / 2) * cx_int * i_f_estimation) / m
        as_x_set[i] = - as_set * np.cos(alpha_set[i])
        as_h_set[i] = - (as_set * np.sin(alpha_set[i]) + acc_gravity_gost)

        range_set[i] = np.sqrt((x_set[i] - x_l) ** 2 + y_l ** 2 + (h_set[i] - h_l) ** 2)
        radial_velocity_set[i] = (velocity_x_set[i] * (x_set[i] - x_l) + velocity_h_set[
            i] * (h_set[i] - h_l)) / range_set[i]
        theta_set[i] = np.arcsin((h_set[i] - h_l) / range_set[i])

    return time_meas, x_set, h_set, velocity_x_set, velocity_h_set, as_x_set, as_h_set


def trajectory_points_approximation(y_meas_set, x_est_init, time_meas_full, x_l, y_l, h_l, cannon_h, time_meas, x_set,
                                    h_set, velocity_x_set, velocity_h_set, as_x_set, as_h_set,
                                    sigma_ksi_x=0.05, sigma_ksi_h=0.05, sigma_ksi_y=0.001, sigma_n_R=1, sigma_n_Vr=1,
                                    sigma_n_theta=np.deg2rad(0.5), sigma_n_y=0.1, sigma_n_Ax=0.5, sigma_n_Ah=0.5,
                                    time_step=0.05):
    '''
    trajectory points for measurements
    :param y_meas_set: list
    :param x_est_init: list
    :param time_meas_full: ndarray
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :param cannon_h: float
    :param time_meas: ndarray
    :param x_set: ndarray
    :param h_set: ndarray
    :param velocity_x_set: ndarray
    :param velocity_h_set: ndarray
    :param as_x_set: ndarray
    :param as_h_set: ndarray
    :param sigma_ksi_x: float
    :param sigma_ksi_h: float
    :param sigma_ksi_y: float
    :param sigma_n_R: int
    :param sigma_n_Vr: int
    :param sigma_n_theta: float
    :param sigma_n_y: float
    :param sigma_n_Ax: float
    :param sigma_n_Ah: float
    :param time_step: float
    :return: x_est_stor: ndarray
             y_ext_stor: ndarray
             cx_est_stor: ndarray
             time_meas: ndarray
    '''

    data_43gost = pd.read_csv('43gost.csv')
    ballictic_coefficient, machs = ballistic_coefficient_43gost()

    x_est_prev = x_est_init
    Dx_est_prev = np.eye(9)

    x_est_stor = []

    D_ksi = np.array([[sigma_ksi_x ** 2, 0, 0], [0, sigma_ksi_h ** 2, 0], [0, 0, sigma_ksi_y ** 2]])
    I = np.eye(9)

    Dn = np.array([[sigma_n_R ** 2, 0, 0, 0, 0, 0],
                   [0, sigma_n_Vr ** 2, 0, 0, 0, 0],
                   [0, 0, sigma_n_theta ** 2, 0, 0, 0],
                   [0, 0, 0, sigma_n_y ** 2, 0, 0],
                   [0, 0, 0, 0, sigma_n_Ax ** 2, 0],
                   [0, 0, 0, 0, 0, sigma_n_Ah ** 2]])

    y_ext_stor = []

    G = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])

    for i, time in enumerate(time_meas):

        time_disp = abs(time - time_meas_full) < 0.0001

        if sum(time_disp) == 1:

            time_ind = time_disp.tolist().index(True)

            x_ext = np.zeros(9)
            x_ext[1] = x_est_prev[1] + x_est_prev[2] * time_step
            x_ext[4] = x_est_prev[4] + x_est_prev[5] * time_step
            x_ext[7] = x_est_prev[7] + x_est_prev[8] * time_step

            x_ext[0] = x_est_prev[0] + x_est_prev[1] * time_step
            x_ext[2] = x_est_prev[2] + (as_x_set[i] - as_x_set[i - 1])
            x_ext[3] = x_est_prev[3] + x_est_prev[4] * time_step
            x_ext[5] = x_est_prev[5] + (as_h_set[i] - as_h_set[i - 1])
            x_ext[6] = x_est_prev[6] + x_est_prev[7] * time_step
            x_ext[8] = x_est_prev[8]

            dfdt = np.array([[1, time_step, 0, 0, 0, 0, 0, 0, 0],
                             [0, 1, time_step, 0, 0, 0, 0, 0, 0],
                             [0, 0, 1, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 1, time_step, 0, 0, 0, 0],
                             [0, 0, 0, 0, 1, time_step, 0, 0, 0],
                             [0, 0, 0, 0, 0, 1, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 1, time_step, 0],
                             [0, 0, 0, 0, 0, 0, 0, 1, time_step],
                             [0, 0, 0, 0, 0, 0, 0, 0, 1]])

            Dx_ext = dfdt.dot(Dx_est_prev).dot(dfdt.T) + G.dot(D_ksi).dot(G.T)

            H = [[(x_ext[0] - x_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2), 0,
                  0, (x_ext[3] - h_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
                  0,
                  0, (x_ext[6] - y_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
                  0,
                  0]]

            H3 = np.zeros(9)

            H3[0] = x_ext[1] / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) - (
                    x_ext[0] - x_l) * (
                            x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (x_ext[6] - y_l)) / (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5
            H3[3] = x_ext[4] / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) - (
                    x_ext[3] - h_l) * (
                            x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (x_ext[6] - y_l)) / (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5
            H3[6] = x_ext[7] / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) - (
                    x_ext[6] - y_l) * (
                            x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (x_ext[6] - y_l)) / (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5
            H3[1] = (x_ext[0] - x_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)
            H3[4] = (x_ext[3] - h_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)
            H3[7] = (x_ext[6] - y_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)

            H.append(list(H3))

            H5 = np.zeros(9)

            H5[0] = (x_ext[3] - h_l) * (x_ext[0] - x_l) / (
                    np.sqrt(1 - (x_ext[3] - h_l) ** 2 / (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)) * (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5)
            H5[3] = (((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** -0.5 - (
                    x_ext[3] - h_l) ** 2 / (
                             (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5) / np.sqrt(
                1 - (x_ext[3] - h_l) ** 2 / ((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2))
            H5[6] = (x_ext[3] - h_l) * (x_ext[6] - y_l) / (
                    np.sqrt(1 - (x_ext[3] - h_l) ** 2 / (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)) * (
                            (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5)

            H.append(list(H5))
            H.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
            H.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
            H.append([0, 0, 0, 0, 0, 1, 0, 0, 0])

            H = np.array(H)

            S = H.dot(Dx_ext).dot(H.T) + Dn
            K = Dx_ext.dot(H.T).dot(np.linalg.inv(S))

            y_ext = [np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
                     (x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (
                             x_ext[6] - y_l)) / np.sqrt(
                         (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
                     np.arcsin(
                         (x_ext[3] - h_l) / np.sqrt(
                             (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)),
                     x_ext[6], x_ext[2], x_ext[5]]

            y_ext_stor.append(y_ext)

            x_est_prev = x_ext + K.dot(
                (np.array(
                    [y_meas_set[0][time_ind], y_meas_set[1][time_ind], y_meas_set[2][time_ind], y_meas_set[3][time_ind],
                     as_x_set[i], as_h_set[i]]) - np.array(y_ext)))

            Dx_est_prev = (I - K.dot(H)).dot(Dx_ext)
            x_est_stor.append(x_est_prev)

        else:
            x_est_prev = [x_set[i], velocity_x_set[i], as_x_set[i], h_set[i],
                          velocity_h_set[i], as_h_set[i], x_est_prev[6], x_est_prev[7], x_est_prev[8]]
            x_est_stor.append(x_est_prev)
            y_ext_init = [np.sqrt((x_est_prev[0] - x_l) ** 2 + (x_est_prev[6] - y_l) ** 2 + (x_est_prev[3] - h_l) ** 2),
                          (x_est_prev[1] * (x_est_prev[0] - x_l) + x_est_prev[4] * (x_est_prev[3] - h_l) + x_est_prev[
                              7] * (
                                   x_est_prev[6] - y_l)) / np.sqrt(
                              (x_est_prev[0] - x_l) ** 2 + (x_est_prev[6] - y_l) ** 2 + (x_est_prev[3] - h_l) ** 2),
                          np.arcsin((x_est_prev[3] - h_l) / np.sqrt(
                              (x_est_prev[0] - x_l) ** 2 + (x_est_prev[6] - y_l) ** 2 + (x_est_prev[3] - h_l) ** 2)),
                          x_est_prev[6], x_est_prev[2], x_est_prev[5]]

            y_ext_stor.append(y_ext_init)

    x_est_stor = np.array(x_est_stor)
    velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(x_est_stor[:, 3] + cannon_h)
    velosity_est_stor = np.sqrt(x_est_stor[:, 1] ** 2 + x_est_stor[:, 4] ** 2)
    cx_est_stor = interpolate.interp1d(machs, ballictic_coefficient)(velosity_est_stor / velocity_sound_gost)

    y_ext_stor = np.array(y_ext_stor)
    y_ext_stor = y_ext_stor[:, :3]

    return x_est_stor, y_ext_stor, cx_est_stor, time_meas


def trajectory_points_approximation_act_react(y_meas_set, x_est_init, x_l, y_l, h_l, cannon_h, time_meas, as_x_set,
                                              as_h_set,
                                              sigma_ksi_x=0.05, sigma_ksi_h=0.05, sigma_ksi_y=0.001, sigma_n_R=4,
                                              sigma_n_Vr=1,
                                              sigma_n_theta=np.deg2rad(1), sigma_n_y=0.1, sigma_n_Ax=0.5,
                                              sigma_n_Ah=0.5,
                                              time_step=0.05):
    '''
    trajectory points for measurements
    :param y_meas_set: list
    :param x_est_init: list
    :param x_l: float
    :param y_l: float
    :param h_l: float
    :param cannon_h: float
    :param time_meas: ndarray
    :param as_x_set: ndarray
    :param as_h_set: ndarray
    :param sigma_ksi_x: float
    :param sigma_ksi_h: float
    :param sigma_ksi_y: float
    :param sigma_n_R: int
    :param sigma_n_Vr: int
    :param sigma_n_theta: float
    :param sigma_n_y: float
    :param sigma_n_Ax: float
    :param sigma_n_Ah: float
    :param time_step: float
    :return: x_est_stor: ndarray
             y_ext_stor: ndarray
             cx_est_stor: ndarray
             time_meas: ndarray
    '''

    data_43gost = pd.read_csv('43gost.csv')
    ballictic_coefficient, machs = ballistic_coefficient_43gost()

    x_est_prev = x_est_init
    Dx_est_prev = np.eye(9)

    x_est_stor = []

    D_ksi = np.array([[sigma_ksi_x ** 2, 0, 0], [0, sigma_ksi_h ** 2, 0], [0, 0, sigma_ksi_y ** 2]])
    I = np.eye(9)

    Dn = np.array([[sigma_n_R ** 2, 0, 0, 0, 0, 0],
                   [0, sigma_n_Vr ** 2, 0, 0, 0, 0],
                   [0, 0, sigma_n_theta ** 2, 0, 0, 0],
                   [0, 0, 0, sigma_n_y ** 2, 0, 0],
                   [0, 0, 0, 0, sigma_n_Ax ** 2, 0],
                   [0, 0, 0, 0, 0, sigma_n_Ah ** 2]])

    y_ext_stor = []

    G = np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0], [0, 0, 0], [0, 0, 0],
                  [0, 1, 0], [0, 0, 0], [0, 0, 0], [0, 0, 1]])

    x_est_stor.append(x_est_prev)

    y_ext_init = [np.sqrt((x_est_prev[0] - x_l) ** 2 + (x_est_prev[6] - y_l) ** 2 + (x_est_prev[3] - h_l) ** 2),
                  (x_est_prev[1] * (x_est_prev[0] - x_l) + x_est_prev[4] * (x_est_prev[3] - h_l) + x_est_prev[
                      7] * (
                           x_est_prev[6] - y_l)) / np.sqrt(
                      (x_est_prev[0] - x_l) ** 2 + (x_est_prev[6] - y_l) ** 2 + (x_est_prev[3] - h_l) ** 2),
                  np.arcsin((x_est_prev[3] - h_l) / np.sqrt(
                      (x_est_prev[0] - x_l) ** 2 + (x_est_prev[6] - y_l) ** 2 + (x_est_prev[3] - h_l) ** 2)),
                  x_est_prev[6],
                  x_est_prev[2],
                  x_est_prev[5]]

    y_ext_stor.append(y_ext_init)

    for i in range(1, len(time_meas)):
        x_ext = np.zeros(9)
        x_ext[1] = x_est_prev[1] + x_est_prev[2] * time_step
        x_ext[4] = x_est_prev[4] + x_est_prev[5] * time_step
        x_ext[7] = x_est_prev[7] + x_est_prev[8] * time_step

        x_ext[0] = x_est_prev[0] + x_est_prev[1] * time_step
        x_ext[2] = x_est_prev[2] + (as_x_set[i] - as_x_set[i - 1])
        x_ext[3] = x_est_prev[3] + x_est_prev[4] * time_step
        x_ext[5] = x_est_prev[5] + (as_h_set[i] - as_h_set[i - 1])
        x_ext[6] = x_est_prev[6] + x_est_prev[7] * time_step
        x_ext[8] = x_est_prev[8]

        dfdt = np.array([[1, time_step, 0, 0, 0, 0, 0, 0, 0],
                         [0, 1, time_step, 0, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 1, time_step, 0, 0, 0, 0],
                         [0, 0, 0, 0, 1, time_step, 0, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 1, time_step, 0],
                         [0, 0, 0, 0, 0, 0, 0, 1, time_step],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1]])

        Dx_ext = dfdt.dot(Dx_est_prev).dot(dfdt.T) + G.dot(D_ksi).dot(G.T)

        H = [[(x_ext[0] - x_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2), 0,
              0, (x_ext[3] - h_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
              0,
              0, (x_ext[6] - y_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
              0,
              0]]

        H3 = np.zeros(9)

        H3[0] = x_ext[1] / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) - (
                x_ext[0] - x_l) * (
                        x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (x_ext[6] - y_l)) / (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5
        H3[3] = x_ext[4] / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) - (
                x_ext[3] - h_l) * (
                        x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (x_ext[6] - y_l)) / (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5
        H3[6] = x_ext[7] / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) - (
                x_ext[6] - y_l) * (
                        x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (x_ext[6] - y_l)) / (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5
        H3[1] = (x_ext[0] - x_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)
        H3[4] = (x_ext[3] - h_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)
        H3[7] = (x_ext[6] - y_l) / np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)

        H.append(list(H3))

        H5 = np.zeros(9)

        H5[0] = (x_ext[3] - h_l) * (x_ext[0] - x_l) / (
                np.sqrt(1 - (x_ext[3] - h_l) ** 2 / (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)) * (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5)
        H5[3] = (((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** -0.5 - (
                x_ext[3] - h_l) ** 2 / (
                         (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5) / np.sqrt(
            1 - (x_ext[3] - h_l) ** 2 / ((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2))
        H5[6] = (x_ext[3] - h_l) * (x_ext[6] - y_l) / (
                np.sqrt(1 - (x_ext[3] - h_l) ** 2 / (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)) * (
                        (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2) ** 1.5)

        H.append(list(H5))
        H.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
        H.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
        H.append([0, 0, 0, 0, 0, 1, 0, 0, 0])

        H = np.array(H)

        S = H.dot(Dx_ext).dot(H.T) + Dn
        K = Dx_ext.dot(H.T).dot(np.linalg.inv(S))

        y_ext = [np.sqrt((x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
                 (x_ext[1] * (x_ext[0] - x_l) + x_ext[4] * (x_ext[3] - h_l) + x_ext[7] * (
                         x_ext[6] - y_l)) / np.sqrt(
                     (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2),
                 np.arcsin(
                     (x_ext[3] - h_l) / np.sqrt(
                         (x_ext[0] - x_l) ** 2 + (x_ext[6] - y_l) ** 2 + (x_ext[3] - h_l) ** 2)),
                 x_ext[6], x_ext[2], x_ext[5]]

        y_ext_stor.append(y_ext)

        x_est_prev = x_ext + K.dot(
            (np.array(
                [y_meas_set[0][i], y_meas_set[1][i], y_meas_set[2][i], y_meas_set[3][i],
                 as_x_set[i], as_h_set[i]]) - np.array(y_ext)))

        Dx_est_prev = (I - K.dot(H)).dot(Dx_ext)
        x_est_stor.append(x_est_prev)

    x_est_stor = np.array(x_est_stor)
    velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(x_est_stor[:, 3] + cannon_h)
    velosity_est_stor = np.sqrt(x_est_stor[:, 1] ** 2 + x_est_stor[:, 4] ** 2)
    cx_est_stor = interpolate.interp1d(machs, ballictic_coefficient)(velosity_est_stor / velocity_sound_gost)

    y_ext_stor = np.array(y_ext_stor)
    y_ext_stor = y_ext_stor[:, :3]

    return x_est_stor, y_ext_stor, cx_est_stor, time_meas


def extrapolation_to_point_fall(x_est_stor, cx_est_stor, time_meas, i_f_estimation, r, m, x_l, y_l, h_l, cannon_h,
                                time_step=0.05):
    '''
    extrapolation of the trajectory to the point of fall
    :param cx_est_stor: ndarray
    :param h_l: float
    :param y_l: float
    :param x_l: float
    :param x_est_stor: ndarray
    :param time_meas: ndarray
    :param i_f_estimation: float
    :param r: float
    :param m: float
    :param cannon_h: float
    :param time_step: float
    :return: x_est_fin_stor: ndarray
             y_ext_fin_stor: ndarray
             cx_est_fin_stor: ndarray
             time_meas_fin_stor: ndarray
    '''

    data_43gost = pd.read_csv('43gost.csv')

    x_est_fin_stor = []
    time_meas_fin_stor = []
    cx_est_fin_stor = []

    x_est_prev = x_est_stor[-1]
    cx_est_prev = cx_est_stor[-1]

    x_est_fin_stor.append(x_est_prev)
    cx_est_fin_stor.append(cx_est_prev)

    time_meas_fin_stor.append(time_meas[-1])

    ballistic_coefficient, machs = ballistic_coefficient_43gost()

    i = 0

    while x_est_fin_stor[i][3] > 0:
        i += 1

        time_meas_fin_stor.append(time_meas_fin_stor[i - 1] + time_step)

        x_ext = np.zeros(9)
        x_ext[1] = x_est_prev[1] + x_est_prev[2] * time_step
        x_ext[4] = x_est_prev[4] + x_est_prev[5] * time_step
        x_ext[7] = x_est_prev[7] + x_est_prev[8] * time_step
        x_ext[0] = x_est_prev[0] + x_est_prev[1] * time_step
        x_ext[3] = x_est_prev[3] + x_est_prev[4] * time_step
        x_ext[6] = x_est_prev[6] + x_est_prev[7] * time_step
        x_ext[8] = x_est_prev[8]

        velocity_cur = np.sqrt(x_ext[1] ** 2 + x_ext[4] ** 2)
        velocity_sound_gost = interpolate.interp1d(data_43gost.height, data_43gost.a)(
            x_ext[3] + cannon_h)
        rho_gost = interpolate.interp1d(data_43gost.height, data_43gost.rho)(x_ext[3] + cannon_h)
        acc_gravity_gost = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
            x_ext[3] + cannon_h)
        cx_int = interpolate.interp1d(machs, ballistic_coefficient)(
            velocity_cur / velocity_sound_gost)
        as_cur = (rho_gost * (np.pi * r ** 2 / 4) * (
                velocity_cur ** 2 / 2) * cx_int * i_f_estimation) / m
        alpha_cur = np.arctan(x_ext[4] / x_ext[1])
        as_x_cur = - as_cur * np.cos(alpha_cur)
        as_h_cur = - (as_cur * np.sin(alpha_cur) + acc_gravity_gost)

        velocity_cur_prev = np.sqrt(x_est_prev[1] ** 2 + x_est_prev[4] ** 2)
        velocity_sound_gost_prev = interpolate.interp1d(data_43gost.height, data_43gost.a)(
            x_est_prev[3] + cannon_h)
        rho_gost_prev = interpolate.interp1d(data_43gost.height, data_43gost.rho)(x_est_prev[3] + cannon_h)
        acc_gravity_gost_prev = interpolate.interp1d(data_43gost.height, data_43gost.acc_gravity)(
            x_est_prev[3] + cannon_h)
        cx_int_prev = interpolate.interp1d(machs, ballistic_coefficient)(
            velocity_cur_prev / velocity_sound_gost_prev)
        as_cur_prev = (rho_gost_prev * (np.pi * r ** 2 / 4) * (
                velocity_cur_prev ** 2 / 2) * cx_int_prev * i_f_estimation) / m
        alpha_cur_prev = np.arctan(x_est_prev[4] / x_est_prev[1])
        as_x_cur_prev = - as_cur_prev * np.cos(alpha_cur_prev)
        as_h_cur_prev = - (as_cur_prev * np.sin(alpha_cur_prev) + acc_gravity_gost_prev)

        x_ext[2] = x_est_prev[2] + (as_x_cur - as_x_cur_prev)
        x_ext[5] = x_est_prev[5] + (as_h_cur - as_h_cur_prev)

        x_est_fin_stor.append(x_ext)
        cx_est_fin_stor.append(cx_int_prev)
        x_est_prev = x_ext

    x_est_fin_stor = np.array(x_est_fin_stor)
    cx_est_fin_stor = np.array(cx_est_fin_stor)
    y_ext_init_stor = np.array([np.sqrt(
        (x_est_fin_stor[:, 0] - x_l) ** 2 + (x_est_fin_stor[:, 6] - y_l) ** 2 + (x_est_fin_stor[:, 3] - h_l) ** 2),
        (x_est_fin_stor[:, 1] * (x_est_fin_stor[:, 0] - x_l) + x_est_fin_stor[:, 4] * (
                x_est_fin_stor[:, 3] - h_l) + x_est_fin_stor[:, 7] * (
                 x_est_fin_stor[:, 6] - y_l)) / np.sqrt(
            (x_est_fin_stor[:, 0] - x_l) ** 2 + (x_est_fin_stor[:, 6] - y_l) ** 2 + (
                    x_est_fin_stor[:, 3] - h_l) ** 2),
        np.arcsin((x_est_fin_stor[:, 3] - h_l) / np.sqrt(
            (x_est_fin_stor[:, 0] - x_l) ** 2 + (x_est_fin_stor[:, 6] - y_l) ** 2 + (
                    x_est_fin_stor[:, 3] - h_l) ** 2)),
        x_est_fin_stor[:, 6], x_est_fin_stor[:, 2], x_est_fin_stor[:, 5]])

    y_ext_init_stor = y_ext_init_stor.T
    y_ext_init_stor = y_ext_init_stor[:, :3]

    return x_est_fin_stor, y_ext_init_stor, cx_est_fin_stor, np.array(time_meas_fin_stor)


def merging_to_date_trajectory(time_meas_stor, x_est_stor, y_ext_stor):
    '''
    :param time_meas_stor: ndarray
    :param x_est_stor: ndarray
    :param y_ext_stor: ndarray
    :return: data_stor: DataFrame
    '''
    theta_est_stor = np.arctan(x_est_stor[:, 4] / x_est_stor[:, 1])
    x_est_stor_full = np.column_stack((x_est_stor, np.rad2deg(theta_est_stor)))
    data_stor_x = pd.DataFrame(data=x_est_stor_full,
                               columns=['x', 'Vx', 'Ax', 'y', 'Vy', 'Ay', 'z', 'Vz', 'Az', 'C', 'theta'])
    data_stor_x.insert(0, 't', time_meas_stor)
    # theta - градусы
    # evr - градусы
    data_stor_y = pd.DataFrame(data=y_ext_stor, columns=['DistanceR', 'VrR', 'EvR'])
    data_stor = pd.concat([data_stor_x, data_stor_y], axis=1)

    return data_stor


def ballistic_coefficient_43gost():
    '''
    ballistic coefficient of velocity shell mach
    :return: cx: ndarray
             machs: ndarray
    '''
    machs = np.linspace(0, 10, num=1001)
    cx = np.zeros(1001)
    for i, mach in enumerate(machs):
        if mach < 0.73357:
            cx[i] = 0.157
        elif 0.73357 <= mach < 0.90962:
            cx[i] = -3.871879 + 15.734575 * mach - 20.511918 * mach ** 2 + 8.928144 * mach ** 3
        elif 0.90962 <= mach < 0.99765:
            cx[i] = 122.720358 - 390.742644 * mach + 413.613130 * mach ** 2 - 145.266282 * mach ** 3
        elif 0.99765 <= mach < 1.17371:
            cx[i] = -19.848947 + 52.409513 * mach - 45.299813 * mach ** 2 + 13.064840 * mach ** 3
        elif 1.17371 <= mach < 1.58451:
            cx[i] = -0.639686 + 2.250136 * mach - 1.600055 * mach ** 2 + 0.363206 * mach ** 3
        elif 1.58451 <= mach < 2.64084:
            cx[i] = 0.643812 - 0.278701 * mach + 0.069619 * mach ** 2 - 0.006051 * mach ** 3
        elif 2.64084 <= mach < 3.72652:
            cx[i] = 0.621061 - 0.242875 * mach + 0.053243 * mach ** 2 - 0.003765 * mach ** 3
        else:
            cx[i] = 0.260
    return cx, machs


def lsm_data(time_meas, data):
    '''
    approximation date
    :param time_meas: ndarray
    :param data: ndarray
    :return: data_lsm: ndarray
    '''
    out_lsm = lsm_cubic(np.array(time_meas[:-1]), data)
    data_lsm = np.zeros(len(time_meas))
    for i, time in enumerate(time_meas):
        data_lsm[i] = (out_lsm[0] + time * out_lsm[1] + time ** 2 * out_lsm[2] + time ** 3 * out_lsm[3])
    return data_lsm


def lsm_alpha(time_meas, range_meas, theta_meas):
    '''
    approximation alpha
    :param time_meas: ndarray
    :param range_meas: ndarray
    :param theta_meas: ndarray
    :return: alpha_meas: ndarray
    '''
    alpha_test = np.arctan(
        np.diff(range_meas * np.sin(theta_meas)) / np.diff(range_meas * np.cos(theta_meas)))
    alpha_meas = lsm_data(time_meas, alpha_test)

    return alpha_meas


def velocity_aprox(range_meas, radial_velocity_meas, theta_meas, alpha_meas, x_l, y_l, h_l):
    '''
    range and range_velociry aprox
    :param h_l: float
    :param y_l: float
    :param x_l: float
    :param range_meas: ndarray
    :param radial_velocity_meas: ndarray
    :param theta_meas: ndarray
    :param alpha_meas: ndarray
    :return: range_h_apr,
             range_x_apr,
             velocity_x_apr,
             velocity_h_apr
    '''

    range_h_apr = range_meas * np.sin(theta_meas) + h_l
    range_x_apr = np.sqrt((range_meas * np.cos(theta_meas)) ** 2 - y_l ** 2) + x_l
    range_h_l = range_meas * np.sin(theta_meas)
    range_x_l = np.sqrt(range_meas ** 2 - range_h_l ** 2)
    velocity_x_apr = (radial_velocity_meas / np.cos(theta_meas - alpha_meas)) * np.cos(alpha_meas)
    velocity_h_apr = (radial_velocity_meas * range_meas - range_x_l * velocity_x_apr) / range_h_l

    return range_h_apr, range_x_apr, velocity_x_apr, velocity_h_apr


def lsm_cubic(X, H):
    '''
    cubic approximation
    :param X: ndarray
    :param H: ndarray
    :return: out: ndarray
    '''
    N = len(X)
    sum_X = sum(X)
    sum_X2 = sum(X ** 2)
    sum_X3 = sum(X ** 3)
    sum_X4 = sum(X ** 4)
    sum_X5 = sum(X ** 5)
    sum_X6 = sum(X ** 6)
    sum_H = sum(H)
    sum_HX = sum(X * H)
    sum_HX2 = sum(X ** 2 * H)
    sum_HX3 = sum(X ** 3 * H)

    A = np.array([[N, sum_X, sum_X2, sum_X3], [sum_X, sum_X2, sum_X3, sum_X4], [sum_X2, sum_X3, sum_X4, sum_X5],
                  [sum_X3, sum_X4, sum_X5, sum_X6]])
    b = np.array([[sum_H], [sum_HX], [sum_HX2], [sum_HX3]])

    out = np.linalg.solve(A, b)

    return out


def derivation_calculation(x_fall, velocity_0, alpha_0, K1, K2):
    '''
    :param x_fall: float
    :param velocity_0: float
    :param alpha_0: float
    :param K1: float
    :param K2: float
    :return: z_derivation: float
    '''
    z_derivation = (K1 + K2 * x_fall) * velocity_0 ** 2 * np.sin(alpha_0) ** 2
    return z_derivation


def derivation_calculation_bullet(m, d, l, eta, velocity_0, time_fall, K_inch=39.3701, K_gran=15432.4, K_fut=3.28084):
    '''
    :param m: float
    :param d: float
    :param l: float
    :param eta: float
    :param velocity_0: float
    :param time_fall: float
    :param K_inch: float
    :param K_gran: float
    :param K_fut: float
    :return: z_derivation_corr: float
    '''
    eta = eta / d
    l_cal = l / d
    d_inch = d * K_inch
    m_gran = m * K_gran
    sg = (30 * m_gran) / (eta ** 2 * d_inch ** 3 * l_cal * (1 + l_cal ** 2))
    sg_corr = sg * ((velocity_0 * K_fut) / 2800) ** (1 / 3)
    z_derivation_corr = (1.25 * (sg_corr + 1.2) * time_fall ** 1.83) / K_inch
    return z_derivation_corr


def wind_displacement(time_fall, x_fall, velocity_0, alpha_0, wind_module, wind_direction, azimuth):
    '''
    :param time_fall: float
    :param x_fall: float
    :param velocity_0: float
    :param alpha_0: float
    :param wind_module: float
    :param wind_direction: float
    :param azimuth: float
    :return: z_wind: float
    '''
    Aw = np.deg2rad(azimuth) - (np.deg2rad(wind_direction) + np.pi)
    Wz = wind_module * np.sin(Aw)
    z_wind = Wz * (time_fall - x_fall / (velocity_0 / np.cos(alpha_0)))
    return z_wind


def point_of_fall(z, x_fall, cannon_b, cannon_l, azimuth):
    '''
    :param z: float
    :param x_fall: float
    :param cannon_b: float
    :param cannon_l: float
    :param azimuth: float
    :return: x_fall_gk: float
             z_fall_gk: float
    '''
    z_fall = z
    x_sp_gk, y_sp_gk = BLH2XY_GK(cannon_b, cannon_l)
    RM = np.array([[np.cos(azimuth), np.sin(azimuth)], [-np.sin(azimuth), np.cos(azimuth)]])
    deltaXY_gk = RM.dot(np.array([[z_fall], [x_fall]]))
    x_fall_gk = x_sp_gk - 10e5 * int(x_sp_gk / 10e5) + deltaXY_gk[1]
    z_fall_gk = y_sp_gk - 10e5 * int(y_sp_gk / 10e5) + deltaXY_gk[0]

    return x_fall_gk, z_fall_gk


def sko_error_meas(y_ext_stor, time_meas_stor, time_meas, range_smoother, radial_velocity_smoother, theta_smoother):
    '''
    SKO measurement (do not take into account the last 2 points)
    :param y_ext_stor: ndarray
    :param time_meas_stor: ndarray
    :param time_meas: ndarray
    :param range_smoother: ndarray
    :param radial_velocity_smoother: ndarray
    :param theta_smoother: ndarray
    :return: sko_range: float
             sko_radial_velocity: float
             sko_theta: float grad
    '''
    range_stor = interpolate.interp1d(time_meas_stor, y_ext_stor[:, 0])(time_meas[:-2])
    radial_velocity_stor = interpolate.interp1d(time_meas_stor, y_ext_stor[:, 1])(time_meas[:-2])
    theta_stor = interpolate.interp1d(time_meas_stor, y_ext_stor[:, 2])(time_meas[:-2])

    sko_range = np.std(range_stor - range_smoother[:-2], ddof=1)
    sko_radial_velocity = np.std(radial_velocity_stor - radial_velocity_smoother[:-2], ddof=1)
    sko_theta = np.std(theta_stor - theta_smoother[:-2], ddof=1)

    return sko_range, sko_radial_velocity, np.rad2deg(sko_theta)


def BLH2XY_GK(B, L):
    '''
    B - cannon_b, L - cannon_L
    :param B: float
    :param L: float
    :return: x: float
             y: float
    '''
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
