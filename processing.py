import numpy as np
import pymap3d as pm

global g
g = 9.8155

from function import kalman_filter_xV, kalman_filter_theta, func_linear_piece_app, func_linear_piece_estimation, \
    func_quad_piece_app, func_quad_piece_estimation, func_derivation, func_meas_smooth_2, \
    BLH2XY_GK, calculate_ellipse


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

    # заполняем параметры пули в конфиг из списка
    # поднимаем флаг
    config.ini_data_flag = 1

    config.flag_return = 0


def process_measurements(data, config):
    if config.ini_data_flag:

        # приходит дата - измерения
        # размер пришедшей даты - точек
        alpha = np.deg2rad(45)

        Ndlen = len(data["meas"])

        print(Ndlen, "Ndlen")



        t_meas = np.zeros(Ndlen)
        R_meas = np.zeros(Ndlen)
        Vr_meas = np.zeros(Ndlen)
        theta_meas = np.zeros(Ndlen)

        for i in range(Ndlen):
            # считываем пришедшие даныне
            t_meas[i] = data["meas"][i]["execTime_sec"]
            R_meas[i] = data["meas"][i]["R"]
            Vr_meas[i] = data["meas"][i]["Vr"]
            theta_meas[i] = data["meas"][i]["Epsilon"]

        Vr_meas = abs(Vr_meas)
        theta_meas = np.deg2rad(theta_meas)
        # подгрузили измерения

        x_est = np.zeros([Ndlen, 2])
        x_est_theta = np.zeros([Ndlen, 2])

        # по умолчанию фильтруем данные
        for k in range(Ndlen):

            if k == 0:
                x_est[k] = [R_meas[k], Vr_meas[k]]
                x_est_theta[k] = [theta_meas[k], 0.0001]
                D_x_est = np.array([[1, 0], [0, 1]])
                D_x_est_theta = np.array([[1, 0], [0, 1]])
            else:
                x_est[k], D_x_est = kalman_filter_xV(x_est[k - 1], D_x_est, np.array([R_meas[k], Vr_meas[k]]),
                                                     t_meas[k] - t_meas[k - 1], config.ksi_Vr, config.n1,
                                                     config.n2)

                x_est_theta[k], D_x_est_theta = kalman_filter_theta(x_est_theta[k - 1], D_x_est_theta,
                                                                    theta_meas[k], t_meas[k] - t_meas[k - 1],
                                                                    config.ksi_theta,
                                                                    config.theta_n1)

        # подобрать определенные средние параметры
        winlen = 30
        step_sld = 10
        N = 300
        # подобрать два данных параметра

        # подбор вот этих параметров

        # берем данные отфильтрованные
        R_meas = x_est[:, 0]
        theta_meas = x_est_theta[:, 0]
        Vr_meas = x_est[:, 1]

        if config.bullet_type == 3:  # если мина - линейное, другие снаряды - квадратичная
            xhy_0_set_linear, x_est_fin_linear_piece, meas_t_ind_linear, t_meas_tr_linear, R_meas_tr_linear, \
            Vr_meas_tr_linear, theta_meas_tr_linear = func_linear_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                            config.can_X, config.can_Y, config.can_Z,
                                                                            config.m, g, config.SKO_R,
                                                                            config.SKO_Vr, config.SKO_theta, config.k0,
                                                                            config.dR, alpha, t_meas,
                                                                            R_meas, Vr_meas, theta_meas, winlen,
                                                                            step_sld,
                                                                            [config.k_bounds, config.v0_bounds,
                                                                             config.dR_bounds, config.angle_bounds],
                                                                            )

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_linear_piece_estimation(
                xhy_0_set_linear, x_est_fin_linear_piece, meas_t_ind_linear, t_meas_tr_linear, N,
                winlen, config.m, g, config.loc_X, config.loc_Y, config.loc_H)
            print(t_meas_plot)

            config.flag_return = 1

        else:

            xhy_0_set_quad, x_est_fin_quad_piece, meas_t_ind_quad, t_meas_tr_quad, R_meas_tr_quad, \
            Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_H,
                                                                      config.can_X, config.can_Y, config.can_Z,
                                                                      config.m, g, config.SKO_R,
                                                                      config.SKO_Vr, config.SKO_theta, config.k0,
                                                                      config.dR, alpha, t_meas,
                                                                      R_meas, Vr_meas, theta_meas, winlen,
                                                                      step_sld, [config.k_bounds, config.v0_bounds,
                                                                             config.dR_bounds, config.angle_bounds],
                                                                      )

            t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
            theta_est_full_plot,  Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
                xhy_0_set_quad, x_est_fin_quad_piece, meas_t_ind_quad, t_meas_tr_quad, N,
                winlen, config.m, g, config.loc_X, config.loc_Y, config.loc_H)

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
