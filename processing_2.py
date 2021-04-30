import numpy as np
import json
import numpy as np
import pymap3d as pm
import os

from function import kalman_filter_xV, kalman_filter_theta, func_linear_piece_app, func_linear_piece_estimation, \
    func_quad_piece_app, func_quad_piece_estimation, func_derivation, func_meas_smooth_2, BLH2XY_GK, calculate_ellipse, \
    func_trajectory_end, func_filter_data, func_active_reactive, func_active_reactive_trajectory, func_lsm_linear


def process_initial_data(mes, config):
    # blh2ENU для локатора
    # blh2ENU для снаряда

    config.loc_B = mes["loc_B"]
    config.loc_L = mes["loc_L"]
    config.loc_H = mes["loc_H"]

    config.can_B = mes["can_B"]
    config.can_L = mes["can_L"]
    config.can_H = mes["can_H"]

    config.loc_Y, config.loc_X, config.loc_Z = pm.geodetic2enu(config.loc_B, config.loc_L, config.loc_H,
                                                               config.can_B, config.can_L, config.can_H)

    config.can_X = 0
    config.can_L = 0
    config.can_H = 0

    config.alpha = mes["alpha"]
    config.az = mes["az"]
    config.hei = mes["hei"]

    config.wind_module = mes["wind_module"]
    config.wind_direction = mes["wind_direction"]

    config.temperature = mes["temperature"]
    config.pressure = mes["pressure"]

    config.bullet_type = mes["bullet_type"]

    bullet = config.bullets[config.bullet_type - 1]

    config.lin_kv = bullet["lin_kv"]
    config.v0 = bullet["v0"]
    config.m = bullet["m"]
    config.k0 = bullet["k0"]
    config.dR = bullet["dR"]
    config.SKO_R = bullet["SKO_R"]
    config.SKO_Vr = bullet["SKO_Vr"]
    config.SKO_theta = bullet["SKO_theta"]

    config.l = bullet["l"]
    config.d = bullet["d"]
    config.h = bullet["h"]
    config.mu = bullet["mu"]
    config.i = bullet["i"]
    config.eta = bullet["eta"]

    config.k_bounds = bullet["k_bounds"]
    config.v0_bounds = bullet["v0_bounds"]
    config.dR_bounds = bullet["dR_bounds"]
    config.angle_bounds = bullet["angle_bounds"]

    config.ksi_Vr = bullet["ksi_Vr"]
    config.n1 = bullet["n1"]
    config.n2 = bullet["n2"]
    config.ksi_theta = bullet["ksi_theta"]
    config.theta_n1 = bullet["theta_n1"]

    config.ini_data_flag = 1
    config.flag_return = 0


def process_measurements(data, config):
    if config.ini_data_flag:

        N = 300
        g = 9.8155

        Ndlen = len(data["meas"])

        t_meas = np.zeros(Ndlen)
        R_meas = np.zeros(Ndlen)
        Vr_meas = np.zeros(Ndlen)
        theta_meas = np.zeros(Ndlen)

        for i in range(Ndlen):
            # считываем пришедшие данные
            t_meas[i] = data["meas"][i]["execTime_sec"]
            R_meas[i] = data["meas"][i]["R"]
            Vr_meas[i] = abs(data["meas"][i]["Vr"])
            theta_meas[i] = np.deg2rad(data["meas"][i]["Epsilon"])

        if config.bullet_type == 1 or config.bullet_type == 2:  # 5.45 bullet or 7.65 bullet

            winlen = 10
            step_sld = 2

            R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, config.ksi_Vr,
                                                           config.n1, config.n2,
                                                           config.ksi_theta,
                                                           config.theta_n1)

            xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, R_meas_tr_quad, \
            Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                      config.can_Y,
                                                                      config.m, g, config.SKO_R,
                                                                      config.SKO_Vr, config.SKO_theta, config.k0,
                                                                      config.dR, t_meas,
                                                                      R_meas, Vr_meas, theta_meas, winlen,
                                                                      step_sld, config.parameters_bounds)

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
                xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_fin, x_true_fin, h_true_fin = func_trajectory_end(config.m, g, xhy_0_set_quad, x_est_fin_quad,
                                                                meas_t_ind_quad,
                                                                window_set_quad, t_meas_tr_quad)

            config.flag_return = 1

        if config.bullet_type == 3:  # 82 mina

            winlen = 30
            step_sld = 10

            xhy_0_set_linear, x_est_fin_linear, meas_t_ind_linear, window_set_linear, t_meas_tr_linear, R_meas_tr_linear, \
            Vr_meas_tr_linear, theta_meas_tr_linear = func_linear_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                            config.can_Y,
                                                                            config.m, g, config.SKO_R,
                                                                            config.SKO_Vr, config.SKO_theta, config.k0,
                                                                            config.dR, t_meas,
                                                                            R_meas, Vr_meas, theta_meas, winlen,
                                                                            step_sld, config.parameters_bounds)

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_linear_piece_estimation(
                xhy_0_set_linear, x_est_fin_linear, meas_t_ind_linear, window_set_linear, t_meas_tr_linear, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_fin, x_true_fin, h_true_fin = func_trajectory_end(config.m, g, xhy_0_set_linear, x_est_fin_linear,
                                                                meas_t_ind_linear,
                                                                window_set_linear, t_meas_tr_linear)

            config.flag_return = 1

        if config.bullet_type == 4:  # 122 reactive

            time_in = 0

            for i in range(len(t_meas)):
                if t_meas[i] > 3:
                    time_in = i
                    break

            t_meas = t_meas[time_in:]
            R_meas = R_meas[time_in:]
            Vr_meas = Vr_meas[time_in:]
            theta_meas = theta_meas[time_in:]

            winlen = 30
            step_sld = 10

            R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, config.ksi_Vr,
                                                           config.n1, config.n2,
                                                           config.ksi_theta,
                                                           config.theta_n1)

            xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, R_meas_tr_quad, \
            Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                      config.can_Y,
                                                                      config.m, g, config.SKO_R,
                                                                      config.SKO_Vr, config.SKO_theta, config.k0,
                                                                      config.dR, t_meas,
                                                                      R_meas, Vr_meas, theta_meas, winlen,
                                                                      step_sld, config.parameters_bounds)

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
                xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_fin, x_true_fin, h_true_fin = func_trajectory_end(config.m, g, xhy_0_set_quad, x_est_fin_quad,
                                                                meas_t_ind_quad,
                                                                window_set_quad, t_meas_tr_quad)

            config.flag_return = 1

        if config.bullet_type == 5:  # 122 - art

            winlen = 30
            step_sld = 10

            R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, config.ksi_Vr,
                                                           config.n1, config.n2,
                                                           config.ksi_theta,
                                                           config.theta_n1)

            xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, R_meas_tr_quad, \
            Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                      config.can_Y,
                                                                      config.m, g, config.SKO_R,
                                                                      config.SKO_Vr, config.SKO_theta, config.k0,
                                                                      config.dR, t_meas,
                                                                      R_meas, Vr_meas, theta_meas, winlen,
                                                                      step_sld, config.parameters_bounds)

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
                xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_fin, x_true_fin, h_true_fin = func_trajectory_end(config.m, g, xhy_0_set_quad, x_est_fin_quad,
                                                                meas_t_ind_quad,
                                                                window_set_quad, t_meas_tr_quad)

            config.flag_return = 1

        if config.bullet_type == 6:  # 152 - act-react

            winlen = 30
            step_sld = 10

            t_ind_end_1part, t_ind_start_2part = func_active_reactive(t_meas, R_meas, Vr_meas)

            t_meas_1 = t_meas[:t_ind_end_1part]
            R_meas_1 = R_meas[:t_ind_end_1part]
            Vr_meas_1 = Vr_meas[:t_ind_end_1part]
            theta_meas_1 = theta_meas[:t_ind_end_1part]

            t_meas_2 = t_meas[t_ind_start_2part:]
            R_meas_2 = R_meas[t_ind_start_2part:]
            Vr_meas_2 = Vr_meas[t_ind_start_2part:]
            theta_meas_2 = theta_meas[t_ind_start_2part:]

            R_meas_1, Vr_meas_1, theta_meas_1 = func_filter_data(t_meas_1, R_meas_1, Vr_meas_1, theta_meas_1,
                                                                 config.ksi_Vr,
                                                                 config.n1, config.n2,
                                                                 config.ksi_theta,
                                                                 config.theta_n1)

            R_meas_2, Vr_meas_2, theta_meas_2 = func_filter_data(t_meas_2, R_meas_2, Vr_meas_2, theta_meas_2,
                                                                 config.ksi_Vr,
                                                                 config.n1, config.n2,
                                                                 config.ksi_theta,
                                                                 config.theta_n1)

            parameters_bounds_1 = [config.k_bounds[0], config.v0_bounds[0], config.dR_bounds[0], config.angle_bounds[0]]
            parameters_bounds_2 = [config.k_bounds[1], config.v0_bounds[1], config.dR_bounds[1], config.angle_bounds[1]]

            xhy_0_set_quad_1, x_est_fin_quad_1, meas_t_ind_quad_1, window_set_quad_1, t_meas_tr_quad_1, R_meas_tr_quad_1, \
            Vr_meas_tr_quad_1, theta_meas_tr_quad_1 = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                          config.can_Y,
                                                                          config.m, g, config.SKO_R,
                                                                          config.SKO_Vr, config.SKO_theta, config.k0,
                                                                          config.dR, t_meas_1,
                                                                          R_meas_1, Vr_meas_1, theta_meas_1, winlen,
                                                                          step_sld, parameters_bounds_1)

            xhy_0_set_quad_2, x_est_fin_quad_2, meas_t_ind_quad_2, window_set_quad_2, t_meas_tr_quad_2, R_meas_tr_quad_2, \
            Vr_meas_tr_quad_2, theta_meas_tr_quad_2 = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                          config.can_Y,
                                                                          config.m, g, config.SKO_R,
                                                                          config.SKO_Vr, config.SKO_theta, config.k0,
                                                                          config.dR, t_meas_2,
                                                                          R_meas_2, Vr_meas_2, theta_meas_2, winlen,
                                                                          step_sld, parameters_bounds_2)

            t_meas_plot_1, x_tr_er_plot_1, h_tr_er_plot_1, R_est_full_plot_1, Vr_est_full_plot_1, \
            theta_est_full_plot_1, Vx_true_er_plot_1, Vh_true_er_plot_1, V_abs_full_plot_1 = func_quad_piece_estimation(
                xhy_0_set_quad_1, x_est_fin_quad_1, meas_t_ind_quad_1, window_set_quad_1, t_meas_tr_quad_1, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_meas_plot_2, x_tr_er_plot_2, h_tr_er_plot_2, R_est_full_plot_2, Vr_est_full_plot_2, \
            theta_est_full_plot_2, Vx_true_er_plot_2, Vh_true_er_plot_2, V_abs_full_plot_2 = func_quad_piece_estimation(
                xhy_0_set_quad_2, x_est_fin_quad_2, meas_t_ind_quad_2, window_set_quad_2, t_meas_tr_quad_2, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_fin, x_true_fin, h_true_fin = func_trajectory_end(config.m, g, xhy_0_set_quad_2, x_est_fin_quad_2,
                                                                meas_t_ind_quad_2,
                                                                window_set_quad_2, t_meas_tr_quad_2)

            t_tr_act_est, x_tr_act_est, h_tr_act_est = func_active_reactive_trajectory(x_tr_er_plot_1, h_tr_er_plot_1,
                                                                                       t_meas_plot_1,
                                                                                       x_tr_er_plot_2, h_tr_er_plot_2,
                                                                                       t_meas_plot_2,
                                                                                       N)

            config.flag_return = 1

        if config.bullet_type == 7:  # 152 art

            winlen = 30
            step_sld = 10

            R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, config.ksi_Vr,
                                                           config.n1, config.n2,
                                                           config.ksi_theta,
                                                           config.theta_n1)

            xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, R_meas_tr_quad, \
            Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                      config.can_Y,
                                                                      config.m, g, config.SKO_R,
                                                                      config.SKO_Vr, config.SKO_theta, config.k0,
                                                                      config.dR, t_meas,
                                                                      R_meas, Vr_meas, theta_meas, winlen,
                                                                      step_sld, config.parameters_bounds)

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
                xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, N,
                config.m, g, config.loc_X, config.loc_Y, config.loc_H)

            t_fin, x_true_fin, h_true_fin = func_trajectory_end(m, g, xhy_0_set_quad, x_est_fin_quad,
                                                                meas_t_ind_quad,
                                                                window_set_quad, t_meas_tr_quad)

            config.flag_return = 1

        # как-то определить оптимальный размер - 30 весь файл?

        # парсим измерения, вычисляем траекторию, поднимаем флажок если все посчиталось
        if config.flag_return == 1:
            # linear ...

            track_points = {}
            points = []
            for i in range(len(t_meas_plot)):
                for j in range(len(t_meas_plot[i]) - 1):
                    points.append({"t": t_meas_plot[i][j], "x_tr": x_tr_er_plot[i][j], "h_tr": h_tr_er_plot[i][j],
                                   "R": R_est_full_plot[i][j],
                                   "Vr": Vr_est_full_plot[i][j], "theta": theta_est_full_plot[i][j]})

            track_points["points"] = points
            track_points["valid"] = True

            config.track = track_points

            # track что вывести туда и посчитать

        flag = 1

        if flag:
            return True
        else:
            return False
    else:
        return False
