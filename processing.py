import time
import sys
import numpy as np
import pymap3d as pm
import ctypes
import traceback

from function import length_winlen, func_linear_piece_app, func_linear_piece_estimation, \
    func_quad_piece_app, func_quad_piece_estimation, func_derivation, func_filter_data, func_active_reactive, \
    func_wind, func_point_fall, func_derivation_bullet, func_linear_piece_estimation_error, \
    func_quad_piece_estimation_error, func_std_error_meas, sampling_points, \
    func_trajectory_start, func_quad_piece_estimation_start, func_trajectory_end, \
    func_linear_piece_estimation_start, func_linear_piece_app_start, func_quad_piece_app_start, \
    func_active_reactive_trajectory, func_emissions_theta, func_trajectory_start_react, func_angle_smoother, \
    func_coord_smoother


def process_initial_data(mes, config):
    # blh2ENU for locator
    # blh2ENU for bullet

    # mes - message
    try:

        config.loc_B = mes["loc_B"]
        config.loc_L = mes["loc_L"]
        config.loc_H = mes["loc_H"]

        config.can_B = mes["can_B"]
        config.can_L = mes["can_L"]
        config.can_H = mes["can_H"]

        config.loc_Y, config.loc_X, config.loc_Z = pm.geodetic2enu(config.loc_B, config.loc_L, config.loc_H,
                                                                   config.can_B, config.can_L, config.can_H)

        config.can_X = 0
        config.can_Y = 0
        config.can_Z = 0

        config.alpha = np.deg2rad(mes["alpha"])
        config.az = np.deg2rad(mes["az"])
        config.hei = mes["hei"]

        config.wind_module = mes["wind_module"]
        config.wind_direction = mes["wind_direction"]

        config.temperature = mes["temperature"]
        config.atm_pressure = mes["atm_pressure"]

        config.bullet_type = mes["bullet_type"]

        # type bullet
        bullet = config.bullets[config.bullet_type - 1]

        config.lin_kv = bullet["lin_kv"]
        config.v0 = bullet["v0"]
        config.m = bullet["m"]
        config.k0 = bullet["k0"]
        config.dR = bullet["dR"]
        config.SKO_R = bullet["SKO_R"]
        config.SKO_Vr = bullet["SKO_Vr"]
        config.SKO_theta = bullet["SKO_theta"]
        # config - rad

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
        # config - rad

        config.sigma_theta = bullet["sigma_theta"]
        config.sigma_RVr = bullet["sigma_RVr"]

        # flag = 1 - message
        config.ini_data_flag = 1
        #flag = 0 - measurements
        config.ini_meas_flag = 0
        # flag = 0 - result
        config.data_points = 0
        # flag = 0 - return
        config.flag_return = 0

    except KeyError:

        # received message ini data with error
        config.ini_data_flag = 0
        config.ini_meas_flag = 0
        config.data_points = 0
        config.flag_return = 1

        track_meas = {}
        track_meas["valid"] = False
        track_meas["error"] = "received message inital data with error"

        config.track = track_meas
        config.track_meas = track_meas


def process_measurements(data, config):
    # добавить время
    if config.ini_data_flag:

        start_time = time.process_time()

        # N - number of points for threading trajectory
        N = 300
        Nend = 10000

        g = 9.8155

        sko_R_tz = 5
        sko_Vr_tz = 0.5
        sko_theta_tz = np.deg2rad(0.1)  # grad - > rad

        K_inch = 39.3701
        K_gran = 15432.4
        K_fut = 3.28084

        rho_0 = 1.225
        # rho_0 = 1.29
        T = 288
        M = 0.0289644
        R = 8.31447

        Ndlen = len(data["points"])
        winlen, step_sld = length_winlen(Ndlen)

        t_meas = np.zeros(Ndlen)
        R_meas = np.zeros(Ndlen)
        Vr_meas = np.zeros(Ndlen)
        theta_meas = np.zeros(Ndlen)
        az_meas = np.zeros(Ndlen)

        sR = 0
        sVr = 0
        stheta = 0
        TD = 0

        try:
            for i in range(Ndlen):
                t_meas[i] = data["points"][i]["execTime"]
                R_meas[i] = data["points"][i]["R"]
                Vr_meas[i] = abs(data["points"][i]["Vr"])
                # angle in grad -> rad
                theta_meas[i] = np.deg2rad(data["points"][i]["Epsilon"])
                az_meas[i] = data["points"][i]["Beta"]

            # SKO measurement
            sR = data["points"][0]["sR"]
            sVr = abs(data["points"][0]["sVr"])
            # SKO angle grad - > rad
            stheta = np.deg2rad(data["points"][0]["sEpsilon"])
            saz = np.deg2rad(data["points"][0]["sBeta"])

            # sampling frequency
            TD = (t_meas[1] - t_meas[0]) / 5

            config.ini_meas_flag = 1

        except KeyError:

            config.flag_return = 1

            track_meas = {}
            track_meas["valid"] = False
            track_meas["error"] = "received message measurements with error"

            config.track = track_meas
            config.track_meas = track_meas

        if config.bullet_type == 1 or config.bullet_type == 2:  # 5.45 bullet or 7.65 bullet

            try:

                Cx = 0
                r = 0
                N = 100
                Nend = 5000

                if config.bullet_type == 1:
                    Cx = 0.38
                    r = 0.00545 / 2

                if config.bullet_type == 2:
                    Cx = 0.44 #0.32
                    r = 0.00762 / 2 #0.00762


                parameters_bounds = [config.k_bounds, config.v0_bounds, config.dR_bounds, config.angle_bounds]

                # measurements filtering
                theta_meas_filter = func_angle_smoother(theta_meas, t_meas, config.sigma_theta)
                R_meas_filter,  Vr_meas_filter = func_coord_smoother(R_meas, Vr_meas, t_meas, config.sigma_RVr)

                xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, R_meas_tr, \
                Vr_meas_tr, theta_meas_tr = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                config.can_Y,
                                                                config.m, g, config.SKO_R, config.SKO_Vr,
                                                                config.SKO_theta,
                                                                config.k0, config.dR, t_meas,
                                                                R_meas_filter, Vr_meas_filter, theta_meas_filter,
                                                                winlen,
                                                                step_sld, parameters_bounds, types=1)
                print(x_est_fin,'fin')

                x_est_start = func_trajectory_start(Cx, r, rho_0, M, R, T, config.m, g, xhy_0_set,
                                                    x_est_fin, t_meas, N)
                print(x_est_start, 'start')

                x_est_app_start = func_quad_piece_app_start(config.loc_X, config.loc_Y, config.loc_Z,
                                                            config.can_Y,
                                                            config.m, g, config.SKO_R,
                                                            config.SKO_Vr, config.SKO_theta, x_est_start, t_meas,
                                                            R_meas_filter, Vr_meas_filter,
                                                            theta_meas_filter,
                                                            window_set, parameters_bounds)

                print(x_est_app_start, 'app')

                t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
                Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
                Ah_true_er_plot = func_quad_piece_estimation(
                    xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_start, x_true_start, h_true_start, R_true_start, Vr_true_start, theta_true_start, Vx_true_start, Vh_true_start, \
                V_abs_true_start, alpha_true_start, A_abs_true_start, Ax_true_start, Ah_true_start = func_quad_piece_estimation_start(
                    x_est_app_start, t_meas_plot, config.m, g, config.loc_X, config.loc_Y, config.loc_Z, N)


                t_fin, x_true_fin, h_true_fin, R_true_fin, Vr_true_fin, theta_true_fin, Vx_true_fin, Vh_true_fin, V_abs_true_fin, \
                alpha_true_fin, A_abs_true_fin, Ax_true_fin, Ah_true_fin = func_trajectory_end(Cx, r, rho_0, M, R, T,
                                                                                               config.m, g,
                                                                                               x_tr_er_plot,
                                                                                               h_tr_er_plot,
                                                                                               Vx_true_er_plot,
                                                                                               Vh_true_er_plot,
                                                                                               V_abs_est_plot,
                                                                                               Ax_true_er_plot,
                                                                                               Ah_true_er_plot,
                                                                                               A_abs_est_plot,
                                                                                               alpha_tr_er_plot,
                                                                                               t_meas_plot,
                                                                                               R_est_full_plot,
                                                                                               Vr_est_full_plot,
                                                                                               theta_est_full_plot,
                                                                                               config.loc_X,
                                                                                               config.loc_Y,
                                                                                               config.loc_Z, config.hei, Nend)

                R_est_err, Vr_est_err, theta_est_err, t_err_plot, R_er_plot, Vr_er_plot, theta_er_plot = func_quad_piece_estimation_error(
                    xhy_0_set, x_est_fin,
                    meas_t_ind, window_set, t_meas,
                    R_meas_filter,
                    Vr_meas_filter,
                    theta_meas_filter, config.m, g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                track_meas, sko_R_meas, sko_Vr_meas, sko_theta_meas = func_std_error_meas(t_err_plot, R_er_plot,
                                                                                          Vr_er_plot,
                                                                                          theta_er_plot,
                                                                                          R_est_err,
                                                                                          Vr_est_err,
                                                                                          theta_est_err, sko_R_tz,
                                                                                          sko_Vr_tz,
                                                                                          sko_theta_tz)

                z_derivation = func_derivation_bullet(config.m, config.d, config.l, config.eta, K_inch, K_gran, K_fut,
                                                      config.v0,
                                                      t_fin[-1])
                z_wind = func_wind(t_fin[-1], x_true_fin[-1], config.v0, config.alpha, config.wind_module,
                                   config.wind_direction, config.az)
                z = z_wind + z_derivation

                x_fall_gk, z_fall_gk = func_point_fall(z, x_true_fin[-1], config.can_B, config.can_L, config.az)

                Vb = x_true_fin[-1] * np.sin(3 * sko_theta_tz)
                Vd = x_true_fin[-1] * np.sin(3 * sko_theta_tz)

                track_meas = {}
                meas = []

                for i in range(len(t_start)):
                    meas.append({"t": t_start[i], "x": x_true_start[i], "y": h_true_start[i],
                                 "z": 0, "V": V_abs_true_start[i], "Vx": Vx_true_start[i],
                                 "Vy": Vh_true_start[i], "Vz": 0, "A": A_abs_true_start[i],
                                 "Ax": Ax_true_start[i],
                                 "Ay": Ah_true_start[i], "Az": 0, "C": x_est_start[0],
                                 "alpha": np.rad2deg(alpha_true_start[i]),
                                 "DistanceR": R_true_start[i], "AzR": 0,
                                 "VrR": Vr_true_start[i], "EvR": np.rad2deg(theta_true_start[i])})

                for i in range(len(t_meas_plot)):
                    for j in range(len(t_meas_plot[i]) - 1):
                        meas.append({"t": t_meas_plot[i][j], "x": x_tr_er_plot[i][j], "y": h_tr_er_plot[i][j],
                                     "z": 0, "V": V_abs_est_plot[i][j], "Vx": Vx_true_er_plot[i][j],
                                     "Vy": Vh_true_er_plot[i][j], "Vz": 0, "A": A_abs_est_plot[i][j],
                                     "Ax": Ax_true_er_plot[i][j],
                                     "Ay": Ah_true_er_plot[i][j], "Az": 0, "C": x_est_fin[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot[i][j]),
                                     "DistanceR": R_est_full_plot[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot[i][j], "EvR": np.rad2deg(theta_est_full_plot[i][j])})

                for i in range(len(t_fin)):
                    meas.append({"t": t_fin[i], "x": x_true_fin[i], "y": h_true_fin[i],
                                 "z": 0, "V": V_abs_true_fin[i], "Vx": Vx_true_fin[i],
                                 "Vy": Vh_true_fin[i], "Vz": 0, "A": A_abs_true_fin[i],
                                 "Ax": Ax_true_fin[i],
                                 "Ay": Ah_true_fin[i], "Az": 0, "C": x_est_fin[-1][0],
                                 "alpha": np.rad2deg(alpha_true_fin[i]),
                                 "DistanceR": R_true_fin[i], "AzR": 0,
                                 "VrR": Vr_true_fin[i], "EvR": np.rad2deg(theta_true_fin[i])})

                #meas_sampling = sampling_points(meas, TD)
                meas_sampling = meas

                track_meas["points"] = meas_sampling
                track_meas["endpoint_x"] = x_true_fin[-1]
                track_meas["endpoint_y"] = h_true_fin[-1]
                track_meas["endpoint_z"] = z
                track_meas["endpoint_GK_x"] = x_fall_gk[0]
                track_meas["endpoint_GK_z"] = z_fall_gk[0]
                track_meas["Vb"] = Vb
                track_meas["Vd"] = Vd
                track_meas["SKO_R"] = sko_R_meas
                track_meas["SKO_V"] = sko_Vr_meas
                track_meas["SKO_theta"] = sko_theta_meas
                track_meas["valid"] = True

                for i in range(len(az_meas) - 1):
                    for j in range(len(track_meas["points"])):
                        if t_meas[i] <= track_meas["points"][j]["t"] < t_meas[i + 1]:
                            track_meas["points"][j]["AzR"] = az_meas[i]

                print('')
                print(x_true_fin[-1], 'х - точки падения')
                print(h_true_fin[-1], 'h - точки падения')

                print(z, 'z - точки падения')
                print(x_fall_gk[0], 'х_fall_gk - точки падения')
                print(z_fall_gk[0], 'z_fall_gk - точки падения')

                print(sko_R_meas, sko_Vr_meas, np.rad2deg(sko_theta_meas), 'значение СКО после отсева измерений')
                print(sR, sVr, np.rad2deg(stheta), "значение СКО измеренное - из файла")
                print(sko_R_tz, sko_Vr_tz, np.rad2deg(sko_theta_tz), 'СКО по ТЗ')


                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1

                track_meas = {}
                track_meas["valid"] = False
                track_meas["error"] = "calculation error 5.45 bullet or 7.65 bullet"

                config.track = track_meas
                config.track_meas = track_meas

        if config.bullet_type == 3:  # 82 mina

            try:

                Cx = 0.185
                r = 0.082 / 2

                parameters_bounds = [config.k_bounds, config.v0_bounds, config.dR_bounds, config.angle_bounds]

                theta_meas_filter = func_angle_smoother(theta_meas, t_meas, config.sigma_theta)
                R_meas_filter,  Vr_meas_filter = func_coord_smoother(R_meas, Vr_meas, t_meas, config.sigma_RVr)

                xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, R_meas_tr, \
                Vr_meas_tr, theta_meas_tr = func_linear_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                  config.can_Y,
                                                                  config.m, g, config.SKO_R,
                                                                  config.SKO_Vr, config.SKO_theta, config.k0,
                                                                  config.dR, t_meas,
                                                                  R_meas_filter, Vr_meas_filter, theta_meas_filter,
                                                                  winlen,
                                                                  step_sld, parameters_bounds)

                x_est_start = func_trajectory_start(Cx, r, rho_0, M, R, T, config.m, g, xhy_0_set,
                                                    x_est_fin, t_meas, N)

                x_est_app_start = func_linear_piece_app_start(config.loc_X, config.loc_Y, config.loc_Z,
                                                              config.can_Y,
                                                              config.m, g, config.SKO_R,
                                                              config.SKO_Vr, config.SKO_theta, x_est_start, t_meas,
                                                              R_meas_filter, Vr_meas_filter,
                                                              theta_meas_filter,
                                                              window_set, parameters_bounds)

                t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
                Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
                Ah_true_er_plot = func_linear_piece_estimation(
                    xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_start, x_true_start, h_true_start, R_true_start, Vr_true_start, theta_true_start, Vx_true_start, Vh_true_start, \
                V_abs_true_start, alpha_true_start, A_abs_true_start, Ax_true_start, Ah_true_start = func_linear_piece_estimation_start(
                    x_est_app_start, t_meas_plot, config.m, g, config.loc_X, config.loc_Y, config.loc_Z, N)

                t_fin, x_true_fin, h_true_fin, R_true_fin, Vr_true_fin, theta_true_fin, Vx_true_fin, Vh_true_fin, V_abs_true_fin, \
                alpha_true_fin, A_abs_true_fin, Ax_true_fin, Ah_true_fin = func_trajectory_end(Cx, r, rho_0, M, R, T,
                                                                                               config.m, g,
                                                                                               x_tr_er_plot,
                                                                                               h_tr_er_plot,
                                                                                               Vx_true_er_plot,
                                                                                               Vh_true_er_plot,
                                                                                               V_abs_est_plot,
                                                                                               Ax_true_er_plot,
                                                                                               Ah_true_er_plot,
                                                                                               A_abs_est_plot,
                                                                                               alpha_tr_er_plot,
                                                                                               t_meas_plot,
                                                                                               R_est_full_plot,
                                                                                               Vr_est_full_plot,
                                                                                               theta_est_full_plot,
                                                                                               config.loc_X,
                                                                                               config.loc_Y,
                                                                                               config.loc_Z, config.hei, Nend)

                R_est_err, Vr_est_err, theta_est_err, t_err_plot, R_er_plot, Vr_er_plot, theta_er_plot = func_linear_piece_estimation_error(
                    xhy_0_set, x_est_fin,
                    meas_t_ind, window_set, t_meas,
                    R_meas_filter,
                    Vr_meas_filter,
                    theta_meas_filter, config.m, g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                track_meas, sko_R_meas, sko_Vr_meas, sko_theta_meas = func_std_error_meas(t_err_plot, R_er_plot,
                                                                                          Vr_er_plot,
                                                                                          theta_er_plot,
                                                                                          R_est_err,
                                                                                          Vr_est_err,
                                                                                          theta_est_err, sko_R_tz,
                                                                                          sko_Vr_tz,
                                                                                          sko_theta_tz)

                z_wind = func_wind(t_fin[-1], x_true_fin[-1], config.v0, config.alpha, config.wind_module,
                                   config.wind_direction, config.az)

                z = z_wind

                x_fall_gk, z_fall_gk = func_point_fall(z, x_true_fin[-1], config.can_B, config.can_L,
                                                       config.az)

                Vb = x_true_fin[-1] * np.sin(3 * sko_theta_tz)
                Vd = 3 * sko_R_tz

                track_meas = {}
                meas = []

                for i in range(len(t_start)):
                    meas.append({"t": t_start[i], "x": x_true_start[i], "y": h_true_start[i],
                                 "z": 0, "V": V_abs_true_start[i], "Vx": Vx_true_start[i],
                                 "Vy": Vh_true_start[i], "Vz": 0, "A": A_abs_true_start[i],
                                 "Ax": Ax_true_start[i],
                                 "Ay": Ah_true_start[i], "Az": 0, "C": x_est_start[0],
                                 "alpha": np.rad2deg(alpha_true_start[i]),
                                 "DistanceR": R_true_start[i], "AzR": 0,
                                 "VrR": Vr_true_start[i], "EvR": np.rad2deg(theta_true_start[i])})

                for i in range(len(t_meas_plot)):
                    for j in range(len(t_meas_plot[i]) - 1):
                        meas.append({"t": t_meas_plot[i][j], "x": x_tr_er_plot[i][j], "y": h_tr_er_plot[i][j],
                                     "z": 0, "V": V_abs_est_plot[i][j], "Vx": Vx_true_er_plot[i][j],
                                     "Vy": Vh_true_er_plot[i][j], "Vz": 0, "A": A_abs_est_plot[i][j],
                                     "Ax": Ax_true_er_plot[i][j],
                                     "Ay": Ah_true_er_plot[i][j], "Az": 0, "C": x_est_fin[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot[i][j]),
                                     "DistanceR": R_est_full_plot[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot[i][j], "EvR": np.rad2deg(theta_est_full_plot[i][j])})

                for i in range(len(t_fin)):
                    meas.append({"t": t_fin[i], "x": x_true_fin[i], "y": h_true_fin[i],
                                 "z": 0, "V": V_abs_true_fin[i], "Vx": Vx_true_fin[i],
                                 "Vy": Vh_true_fin[i], "Vz": 0, "A": A_abs_true_fin[i],
                                 "Ax": Ax_true_fin[i],
                                 "Ay": Ah_true_fin[i], "Az": 0, "C": x_est_fin[-1][0],
                                 "alpha": np.rad2deg(alpha_true_fin[i]),
                                 "DistanceR": R_true_fin[i], "AzR": 0,
                                 "VrR": Vr_true_fin[i], "EvR": np.rad2deg(theta_true_fin[i])})

                #meas_sampling = sampling_points(meas, TD)
                meas_sampling = meas

                track_meas["points"] = meas_sampling
                track_meas["endpoint_x"] = x_true_fin[-1]
                track_meas["endpoint_y"] = h_true_fin[-1]
                track_meas["endpoint_z"] = z
                track_meas["endpoint_GK_x"] = x_fall_gk[0]
                track_meas["endpoint_GK_z"] = z_fall_gk[0]
                track_meas["Vb"] = Vb
                track_meas["Vd"] = Vd
                track_meas["SKO_R"] = sko_R_meas
                track_meas["SKO_V"] = sko_Vr_meas
                track_meas["SKO_theta"] = sko_theta_meas
                track_meas["valid"] = True

                for i in range(len(az_meas) - 1):
                    for j in range(len(track_meas["points"])):
                        if t_meas[i] <= track_meas["points"][j]["t"] < t_meas[i + 1]:
                            track_meas["points"][j]["AzR"] = az_meas[i]

                print('')
                print(x_true_fin[-1], 'х - точки падения')
                print(h_true_fin[-1], 'h - точки падения')

                print(z, 'z - точки падения')
                print(x_fall_gk[0], 'х_fall_gk - точки падения')
                print(z_fall_gk[0], 'z_fall_gk - точки падения')

                print(sko_R_meas, sko_Vr_meas, np.rad2deg(sko_theta_meas), 'значение СКО после отсева измерений')
                print(sR, sVr, np.rad2deg(stheta), "значение СКО измеренное - из файла")
                print(sko_R_tz, sko_Vr_tz, np.rad2deg(sko_theta_tz), 'СКО по ТЗ')

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1

                track_meas = {}
                track_meas["valid"] = False
                track_meas["error"] = "calculation error 82 mina"

                config.track = track_meas
                config.track_meas = track_meas

        if config.bullet_type == 4:  # 122 reactive

            try:

                Cx = 0.535  # 0.295; 0.54
                r = 0.122 / 2

                dv_dt = np.zeros(len(Vr_meas) - 1)
                st_passive_ind = 0
                for i in range(1, len(Vr_meas)):
                    dv_dt[i - 1] = (Vr_meas[i] - Vr_meas[i - 1]) / (t_meas[i] - t_meas[i - 1])
                    if (i > 3) and (dv_dt[i - 1] < 0) and (dv_dt[i - 2]) < 0:
                        st_passive_ind = i - 2
                        break

                t_meas_start = t_meas[st_passive_ind:]
                R_meas_start = R_meas[st_passive_ind:]
                Vr_meas_start = Vr_meas[st_passive_ind:]
                theta_meas_start = theta_meas[st_passive_ind:]

                bad_ind = func_emissions_theta(theta_meas_start, thres_theta=0.015)

                t_meas_start = np.delete(t_meas_start, bad_ind)
                R_meas_start = np.delete(R_meas_start, bad_ind)
                Vr_meas_start = np.delete(Vr_meas_start, bad_ind)
                theta_meas_start = np.delete(theta_meas_start, bad_ind)

                parameters_bounds = [config.k_bounds, config.v0_bounds, config.dR_bounds, config.angle_bounds]

                theta_meas_filter = func_angle_smoother(theta_meas_start, t_meas_start, config.sigma_theta)
                R_meas_filter,  Vr_meas_filter = func_coord_smoother(R_meas_start, Vr_meas_start, t_meas_start, config.sigma_RVr)

                xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, R_meas_tr, \
                Vr_meas_tr, theta_meas_tr = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                config.can_Y,
                                                                config.m, g, config.SKO_R,
                                                                config.SKO_Vr, config.SKO_theta, config.k0,
                                                                config.dR, t_meas_start,
                                                                R_meas_filter, Vr_meas_filter, theta_meas_filter,
                                                                winlen,
                                                                step_sld, parameters_bounds,types=1)

                t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
                Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
                Ah_true_er_plot = func_quad_piece_estimation(
                    xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_start, x_true_start, h_true_start, R_true_start, Vr_true_start, theta_true_start, Vx_true_start, Vh_true_start, \
                V_abs_true_start, alpha_true_start, A_abs_true_start, Ax_true_start, Ah_true_start = func_trajectory_start_react(
                    xhy_0_set, x_est_fin, t_meas_start, config.loc_X, config.loc_Y, config.loc_Z, N)

                t_fin, x_true_fin, h_true_fin, R_true_fin, Vr_true_fin, theta_true_fin, Vx_true_fin, Vh_true_fin, V_abs_true_fin, \
                alpha_true_fin, A_abs_true_fin, Ax_true_fin, Ah_true_fin = func_trajectory_end(Cx, r, rho_0, M, R, T,
                                                                                               config.m, g,
                                                                                               x_tr_er_plot,
                                                                                               h_tr_er_plot,
                                                                                               Vx_true_er_plot,
                                                                                               Vh_true_er_plot,
                                                                                               V_abs_est_plot,
                                                                                               Ax_true_er_plot,
                                                                                               Ah_true_er_plot,
                                                                                               A_abs_est_plot,
                                                                                               alpha_tr_er_plot,
                                                                                               t_meas_plot,
                                                                                               R_est_full_plot,
                                                                                               Vr_est_full_plot,
                                                                                               theta_est_full_plot,
                                                                                               config.loc_X,
                                                                                               config.loc_Y,
                                                                                               config.loc_Z, config.hei, Nend)

                R_est_err, Vr_est_err, theta_est_err, t_err_plot, R_er_plot, Vr_er_plot, theta_er_plot = func_quad_piece_estimation_error(
                    xhy_0_set, x_est_fin,
                    meas_t_ind, window_set, t_meas,
                    R_meas_filter,
                    Vr_meas_filter,
                    theta_meas_filter, config.m, g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                track_meas, sko_R_meas, sko_Vr_meas, sko_theta_meas = func_std_error_meas(t_err_plot, R_er_plot,
                                                                                          Vr_er_plot,
                                                                                          theta_er_plot,
                                                                                          R_est_err,
                                                                                          Vr_est_err,
                                                                                          theta_est_err, sko_R_tz,
                                                                                          sko_Vr_tz,
                                                                                          sko_theta_tz)

                z_wind = func_wind(t_fin[-1], x_true_fin[-1], config.v0, config.alpha, config.wind_module,
                                   config.wind_direction, config.az)

                z = z_wind

                x_fall_gk, z_fall_gk = func_point_fall(z, x_true_fin[-1], config.can_B, config.can_L,
                                                       config.az)

                Vb = x_true_fin[-1] * np.sin(3 * sko_theta_tz)
                Vd = 3 * sko_R_tz

                track_meas = {}
                meas = []

                for i in range(len(t_start)):
                    meas.append({"t": t_start[i], "x": x_true_start[i], "y": h_true_start[i],
                                 "z": 0, "V": V_abs_true_start[i], "Vx": Vx_true_start[i],
                                 "Vy": Vh_true_start[i], "Vz": 0, "A": A_abs_true_start[i],
                                 "Ax": Ax_true_start[i],
                                 "Ay": Ah_true_start[i], "Az": 0, "C": x_est_fin[0][0],
                                 "alpha": np.rad2deg(alpha_true_start[i]),
                                 "DistanceR": R_true_start[i], "AzR": 0,
                                 "VrR": Vr_true_start[i], "EvR": np.rad2deg(theta_true_start[i])})

                for i in range(len(t_meas_plot)):
                    for j in range(len(t_meas_plot[i]) - 1):
                        meas.append({"t": t_meas_plot[i][j], "x": x_tr_er_plot[i][j], "y": h_tr_er_plot[i][j],
                                     "z": 0, "V": V_abs_est_plot[i][j], "Vx": Vx_true_er_plot[i][j],
                                     "Vy": Vh_true_er_plot[i][j], "Vz": 0, "A": A_abs_est_plot[i][j],
                                     "Ax": Ax_true_er_plot[i][j],
                                     "Ay": Ah_true_er_plot[i][j], "Az": 0, "C": x_est_fin[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot[i][j]),
                                     "DistanceR": R_est_full_plot[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot[i][j], "EvR": np.rad2deg(theta_est_full_plot[i][j])})

                for i in range(len(t_fin)):
                    meas.append({"t": t_fin[i], "x": x_true_fin[i], "y": h_true_fin[i],
                                 "z": 0, "V": V_abs_true_fin[i], "Vx": Vx_true_fin[i],
                                 "Vy": Vh_true_fin[i], "Vz": 0, "A": A_abs_true_fin[i],
                                 "Ax": Ax_true_fin[i],
                                 "Ay": Ah_true_fin[i], "Az": 0, "C": x_est_fin[-1][0],
                                 "alpha": np.rad2deg(alpha_true_fin[i]),
                                 "DistanceR": R_true_fin[i], "AzR": 0,
                                 "VrR": Vr_true_fin[i], "EvR": np.rad2deg(theta_true_fin[i])})

                #meas_sampling = sampling_points(meas, TD)
                meas_sampling = meas

                track_meas["points"] = meas_sampling
                track_meas["endpoint_x"] = x_true_fin[-1]
                track_meas["endpoint_y"] = h_true_fin[-1]
                track_meas["endpoint_z"] = z
                track_meas["endpoint_GK_x"] = x_fall_gk[0]
                track_meas["endpoint_GK_z"] = z_fall_gk[0]
                track_meas["Vb"] = Vb
                track_meas["Vd"] = Vd
                track_meas["SKO_R"] = sko_R_meas
                track_meas["SKO_V"] = sko_Vr_meas
                track_meas["SKO_theta"] = sko_theta_meas
                track_meas["valid"] = True

                for i in range(len(az_meas) - 1):
                    for j in range(len(track_meas["points"])):
                        if t_meas[i] <= track_meas["points"][j]["t"] < t_meas[i + 1]:
                            track_meas["points"][j]["AzR"] = az_meas[i]

                print('')
                print(x_true_fin[-1], 'х - точки падения')
                print(h_true_fin[-1], 'h - точки падения')

                print(z, 'z - точки падения')
                print(x_fall_gk[0], 'х_fall_gk - точки падения')
                print(z_fall_gk[0], 'z_fall_gk - точки падения')

                print(sko_R_meas, sko_Vr_meas, np.rad2deg(sko_theta_meas), 'значение СКО после отсева измерений')
                print(sR, sVr, np.rad2deg(stheta), "значение СКО измеренное - из файла")
                print(sko_R_tz, sko_Vr_tz, np.rad2deg(sko_theta_tz), 'СКО по ТЗ')

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1

                track_meas = {}
                track_meas["valid"] = False
                track_meas["error"] = "calculation error 122 reactive"

                config.track = track_meas
                config.track_meas = track_meas

        if config.bullet_type == 5:  # 122 - art

            try:

                Cx = 0.295
                r = 0.122 / 2

                parameters_bounds = [config.k_bounds, config.v0_bounds, config.dR_bounds, config.angle_bounds]

                K1 = 0.00461217647718868
                K2 = -2.04678100654676e-07

                theta_meas_filter = func_angle_smoother(theta_meas, t_meas, config.sigma_theta)
                R_meas_filter,  Vr_meas_filter = func_coord_smoother(R_meas, Vr_meas, t_meas, config.sigma_RVr)

                xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, R_meas_tr, \
                Vr_meas_tr, theta_meas_tr = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                config.can_Y,
                                                                config.m, g, config.SKO_R,
                                                                config.SKO_Vr, config.SKO_theta, config.k0,
                                                                config.dR, t_meas,
                                                                R_meas_filter, Vr_meas_filter, theta_meas_filter,
                                                                winlen,
                                                                step_sld, parameters_bounds, types=1)

                x_est_start = func_trajectory_start(Cx, r, rho_0, M, R, T, config.m, g, xhy_0_set,
                                                    x_est_fin, t_meas, N)

                x_est_app_start = func_linear_piece_app_start(config.loc_X, config.loc_Y, config.loc_Z,
                                                              config.can_Y,
                                                              config.m, g, config.SKO_R,
                                                              config.SKO_Vr, config.SKO_theta, x_est_start, t_meas,
                                                              R_meas_filter, Vr_meas_filter,
                                                              theta_meas_filter,
                                                              window_set, parameters_bounds)

                t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
                Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
                Ah_true_er_plot = func_linear_piece_estimation(
                    xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_start, x_true_start, h_true_start, R_true_start, Vr_true_start, theta_true_start, Vx_true_start, Vh_true_start, \
                V_abs_true_start, alpha_true_start, A_abs_true_start, Ax_true_start, Ah_true_start = func_linear_piece_estimation_start(
                    x_est_app_start, t_meas_plot, config.m, g, config.loc_X, config.loc_Y, config.loc_Z, N)

                t_fin, x_true_fin, h_true_fin, R_true_fin, Vr_true_fin, theta_true_fin, Vx_true_fin, Vh_true_fin, V_abs_true_fin, \
                alpha_true_fin, A_abs_true_fin, Ax_true_fin, Ah_true_fin = func_trajectory_end(Cx, r, rho_0, M, R, T,
                                                                                               config.m, g,
                                                                                               x_tr_er_plot,
                                                                                               h_tr_er_plot,
                                                                                               Vx_true_er_plot,
                                                                                               Vh_true_er_plot,
                                                                                               V_abs_est_plot,
                                                                                               Ax_true_er_plot,
                                                                                               Ah_true_er_plot,
                                                                                               A_abs_est_plot,
                                                                                               alpha_tr_er_plot,
                                                                                               t_meas_plot,
                                                                                               R_est_full_plot,
                                                                                               Vr_est_full_plot,
                                                                                               theta_est_full_plot,
                                                                                               config.loc_X,
                                                                                               config.loc_Y,
                                                                                               config.loc_Z, config.hei, Nend)

                R_est_err, Vr_est_err, theta_est_err, t_err_plot, R_er_plot, Vr_er_plot, theta_er_plot = func_quad_piece_estimation_error(
                    xhy_0_set, x_est_fin,
                    meas_t_ind, window_set, t_meas,
                    R_meas_filter,
                    Vr_meas_filter,
                    theta_meas_filter, config.m, g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                track_meas, sko_R_meas, sko_Vr_meas, sko_theta_meas = func_std_error_meas(t_err_plot, R_er_plot,
                                                                                          Vr_er_plot,
                                                                                          theta_er_plot,
                                                                                          R_est_err,
                                                                                          Vr_est_err,
                                                                                          theta_est_err, sko_R_tz,
                                                                                          sko_Vr_tz,
                                                                                          sko_theta_tz)

                z_derivation = func_derivation(K1, K2, x_true_fin[-1], config.v0, config.alpha)

                z_wind = func_wind(t_fin[-1], x_true_fin[-1], config.v0, config.alpha, config.wind_module,
                                   config.wind_direction, config.az)

                z = z_wind + z_derivation

                x_fall_gk, z_fall_gk = func_point_fall(z, x_true_fin[-1], config.can_B, config.can_L,
                                                       config.az)

                Vb = x_true_fin[-1] * np.sin(3 * sko_theta_tz)
                Vd = 3 * sko_R_tz

                track_meas = {}
                meas = []

                for i in range(len(t_start)):
                    meas.append({"t": t_start[i], "x": x_true_start[i], "y": h_true_start[i],
                                 "z": 0, "V": V_abs_true_start[i], "Vx": Vx_true_start[i],
                                 "Vy": Vh_true_start[i], "Vz": 0, "A": A_abs_true_start[i],
                                 "Ax": Ax_true_start[i],
                                 "Ay": Ah_true_start[i], "Az": 0, "C": x_est_start[0],
                                 "alpha": np.rad2deg(alpha_true_start[i]),
                                 "DistanceR": R_true_start[i], "AzR": 0,
                                 "VrR": Vr_true_start[i], "EvR": np.rad2deg(theta_true_start[i])})

                for i in range(len(t_meas_plot)):
                    for j in range(len(t_meas_plot[i]) - 1):
                        meas.append({"t": t_meas_plot[i][j], "x": x_tr_er_plot[i][j], "y": h_tr_er_plot[i][j],
                                     "z": 0, "V": V_abs_est_plot[i][j], "Vx": Vx_true_er_plot[i][j],
                                     "Vy": Vh_true_er_plot[i][j], "Vz": 0, "A": A_abs_est_plot[i][j],
                                     "Ax": Ax_true_er_plot[i][j],
                                     "Ay": Ah_true_er_plot[i][j], "Az": 0, "C": x_est_fin[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot[i][j]),
                                     "DistanceR": R_est_full_plot[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot[i][j], "EvR": np.rad2deg(theta_est_full_plot[i][j])})

                for i in range(len(t_fin)):
                    meas.append({"t": t_fin[i], "x": x_true_fin[i], "y": h_true_fin[i],
                                 "z": 0, "V": V_abs_true_fin[i], "Vx": Vx_true_fin[i],
                                 "Vy": Vh_true_fin[i], "Vz": 0, "A": A_abs_true_fin[i],
                                 "Ax": Ax_true_fin[i],
                                 "Ay": Ah_true_fin[i], "Az": 0, "C": x_est_fin[-1][0],
                                 "alpha": np.rad2deg(alpha_true_fin[i]),
                                 "DistanceR": R_true_fin[i], "AzR": 0,
                                 "VrR": Vr_true_fin[i], "EvR": np.rad2deg(theta_true_fin[i])})

                #meas_sampling = sampling_points(meas, TD)
                meas_sampling = meas

                track_meas["points"] = meas_sampling
                track_meas["endpoint_x"] = x_true_fin[-1]
                track_meas["endpoint_y"] = h_true_fin[-1]
                track_meas["endpoint_z"] = z
                track_meas["endpoint_GK_x"] = x_fall_gk[0]
                track_meas["endpoint_GK_z"] = z_fall_gk[0]
                track_meas["Vb"] = Vb
                track_meas["Vd"] = Vd
                track_meas["SKO_R"] = sko_R_meas
                track_meas["SKO_V"] = sko_Vr_meas
                track_meas["SKO_theta"] = sko_theta_meas
                track_meas["valid"] = True

                for i in range(len(az_meas) - 1):
                    for j in range(len(track_meas["points"])):
                        if t_meas[i] <= track_meas["points"][j]["t"] < t_meas[i + 1]:
                            track_meas["points"][j]["AzR"] = az_meas[i]

                print('')
                print(x_true_fin[-1], 'х - точки падения')
                print(h_true_fin[-1], 'h - точки падения')

                print(z, 'z - точки падения')
                print(x_fall_gk[0], 'х_fall_gk - точки падения')
                print(z_fall_gk[0], 'z_fall_gk - точки падения')

                print(sko_R_meas, sko_Vr_meas, np.rad2deg(sko_theta_meas), 'значение СКО после отсева измерений')
                print(sR, sVr, np.rad2deg(stheta), "значение СКО измеренное - из файла")
                print(sko_R_tz, sko_Vr_tz, np.rad2deg(sko_theta_tz), 'СКО по ТЗ')

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1

                track_meas = {}
                track_meas["valid"] = False
                track_meas["error"] = "calculation error 122 art"

                config.track = track_meas
                config.track_meas = track_meas

        if config.bullet_type == 6:  # 152 act-react

            try:

                Cx = 0.39  # 0.59
                r = 0.152 / 2

                K1 = 0.00324881940048771
                K2 = -2.37948707596265e-08

                parameters_bounds_1 = [config.k_bounds[0], config.v0_bounds[0], config.dR_bounds[0],
                                       config.angle_bounds[0]]
                parameters_bounds_2 = [config.k_bounds[1], config.v0_bounds[1], config.dR_bounds[1],
                                       config.angle_bounds[1]]

                t_ind_end_1part, t_ind_start_2part = func_active_reactive(t_meas, R_meas, Vr_meas)

                if t_ind_end_1part == 0 and t_ind_start_2part == 0:
                    config.flag_return = 1

                    track_meas = {}
                    track_meas["valid"] = False
                    track_meas["error"] = "no two parts active-reactive error"

                    config.track = track_meas
                    config.track_meas = track_meas
                    return

                t_meas_1 = t_meas[:t_ind_end_1part]
                R_meas_1 = R_meas[:t_ind_end_1part]
                Vr_meas_1 = Vr_meas[:t_ind_end_1part]
                theta_meas_1 = theta_meas[:t_ind_end_1part]

                t_meas_2 = t_meas[t_ind_start_2part:]
                R_meas_2 = R_meas[t_ind_start_2part:]
                Vr_meas_2 = Vr_meas[t_ind_start_2part:]
                theta_meas_2 = theta_meas[t_ind_start_2part:]

                Ndlen1 = len(t_meas_1)
                Ndlen2 = len(t_meas_2)

                winlen1, step_sld1 = length_winlen(Ndlen1)
                winlen2, step_sld2 = length_winlen(Ndlen2)

                theta_meas_1_filter = func_angle_smoother(theta_meas_1, t_meas_1, config.sigma_theta)
                R_meas_1_filter,  Vr_meas_1_filter = func_coord_smoother(R_meas_1, Vr_meas_1, t_meas_1, config.sigma_RVr)

                theta_meas_2_filter = func_angle_smoother(theta_meas_2, t_meas_2, config.sigma_theta)
                R_meas_2_filter,  Vr_meas_2_filter = func_coord_smoother(R_meas_2, Vr_meas_2, t_meas_2, config.sigma_RVr)

                xhy_0_set_1, x_est_fin_1, meas_t_ind_1, window_set_1, t_meas_tr_1, R_meas_tr_1, \
                Vr_meas_tr_1, theta_meas_tr_1 = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                    config.can_Y,
                                                                    config.m, g, config.SKO_R,
                                                                    config.SKO_Vr, config.SKO_theta, config.k0,
                                                                    config.dR, t_meas_1,
                                                                    R_meas_1_filter, Vr_meas_1_filter,
                                                                    theta_meas_1_filter,
                                                                    winlen1,
                                                                    step_sld1, parameters_bounds_1, types=2)

                x_est_start = func_trajectory_start(Cx, r, rho_0, M, R, T, config.m, g, xhy_0_set_1,
                                                    x_est_fin_1, t_meas_1, N)

                x_est_app_start = func_quad_piece_app_start(config.loc_X, config.loc_Y, config.loc_Z,
                                                            config.can_Y,
                                                            config.m, g, config.SKO_R,
                                                            config.SKO_Vr, config.SKO_theta, x_est_start, t_meas_1,
                                                            R_meas_1_filter, Vr_meas_1_filter,
                                                            theta_meas_1_filter,
                                                            window_set_1, parameters_bounds_1)

                t_meas_plot_1, x_tr_er_plot_1, h_tr_er_plot_1, R_est_full_plot_1, Vr_est_full_plot_1, \
                theta_est_full_plot_1, Vx_true_er_plot_1, Vh_true_er_plot_1, V_abs_full_plot_1, alpha_tr_er_plot_1, \
                A_abs_est_plot_1, Ax_true_er_plot_1, Ah_true_er_plot_1 = func_quad_piece_estimation(
                    xhy_0_set_1, x_est_fin_1, meas_t_ind_1, window_set_1, t_meas_tr_1, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_start, x_true_start, h_true_start, R_true_start, Vr_true_start, theta_true_start, Vx_true_start, Vh_true_start, \
                V_abs_true_start, alpha_true_start, A_abs_true_start, Ax_true_start, Ah_true_start = func_quad_piece_estimation_start(
                    x_est_app_start, t_meas_plot_1, config.m, g, config.loc_X, config.loc_Y, config.loc_Z, N)

                print('')

                xhy_0_set_2, x_est_fin_2, meas_t_ind_2, window_set_2, t_meas_tr_2, R_meas_tr_2, \
                Vr_meas_tr_2, theta_meas_tr_2 = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                    config.can_Y,
                                                                    config.m, g, config.SKO_R,
                                                                    config.SKO_Vr, config.SKO_theta, config.k0,
                                                                    config.dR, t_meas_2,
                                                                    R_meas_2_filter, Vr_meas_2_filter,
                                                                    theta_meas_2_filter,
                                                                    winlen2,
                                                                    step_sld2, parameters_bounds_2, types=3)

                t_meas_plot_2, x_tr_er_plot_2, h_tr_er_plot_2, R_est_full_plot_2, Vr_est_full_plot_2, \
                theta_est_full_plot_2, Vx_true_er_plot_2, Vh_true_er_plot_2, V_abs_full_plot_2, alpha_tr_er_plot_2, \
                A_abs_est_plot_2, Ax_true_er_plot_2, Ah_true_er_plot_2 = func_quad_piece_estimation(
                    xhy_0_set_2, x_est_fin_2, meas_t_ind_2, window_set_2, t_meas_tr_2, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_tr_act_est, x_tr_act_est, h_tr_act_est, R_tr_act_est, Vr_tr_act_est, theta_tr_act_est, Vx_tr_act_est, \
                Vh_tr_act_est, V_abs_tr_act_est, alpha_tr_act_est, A_abs_tr_act_est, Ax_tr_act_est, Ah_tr_act_est \
                    = func_active_reactive_trajectory(x_tr_er_plot_1, h_tr_er_plot_1,
                                                      t_meas_plot_1, Vx_true_er_plot_1, Vh_true_er_plot_1,
                                                      Ax_true_er_plot_1, Ah_true_er_plot_1,
                                                      x_tr_er_plot_2, h_tr_er_plot_2,
                                                      t_meas_plot_2, Vx_true_er_plot_2, Vh_true_er_plot_2,
                                                      Ax_true_er_plot_2, Ah_true_er_plot_2, N,
                                                      config.loc_X, config.loc_Y, config.loc_Z)

                t_fin, x_true_fin, h_true_fin, R_true_fin, Vr_true_fin, theta_true_fin, Vx_true_fin, Vh_true_fin, V_abs_true_fin, \
                alpha_true_fin, A_abs_true_fin, Ax_true_fin, Ah_true_fin = func_trajectory_end(Cx, r, rho_0, M, R, T,
                                                                                               config.m, g,
                                                                                               x_tr_er_plot_2,
                                                                                               h_tr_er_plot_2,
                                                                                               Vx_true_er_plot_2,
                                                                                               Vh_true_er_plot_2,
                                                                                               V_abs_full_plot_2,
                                                                                               Ax_true_er_plot_2,
                                                                                               Ah_true_er_plot_2,
                                                                                               A_abs_est_plot_2,
                                                                                               alpha_tr_er_plot_2,
                                                                                               t_meas_plot_2,
                                                                                               R_est_full_plot_2,
                                                                                               Vr_est_full_plot_2,
                                                                                               theta_est_full_plot_2,
                                                                                               config.loc_X,
                                                                                               config.loc_Y,
                                                                                               config.loc_Z, config.hei, Nend)

                R_est_err_1, Vr_est_err_1, theta_est_err_1, t_err_plot_1, R_er_plot_1, Vr_er_plot_1, theta_er_plot_1 = func_quad_piece_estimation_error(
                    xhy_0_set_1, x_est_fin_1,
                    meas_t_ind_1, window_set_1,
                    t_meas_1,
                    R_meas_1_filter,
                    Vr_meas_1_filter,
                    theta_meas_1_filter, config.m,
                    g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                R_est_err_2, Vr_est_err_2, theta_est_err_2, t_err_plot_2, R_er_plot_2, Vr_er_plot_2, theta_er_plot_2 = func_quad_piece_estimation_error(
                    xhy_0_set_2, x_est_fin_2,
                    meas_t_ind_2, window_set_2,
                    t_meas_2,
                    R_meas_2_filter,
                    Vr_meas_2_filter,
                    theta_meas_2_filter, config.m,
                    g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                for i in range(len(R_est_err_2)):
                    R_est_err_1.append(R_est_err_2[i])
                    Vr_est_err_1.append(Vr_est_err_2[i])
                    theta_est_err_1.append(theta_est_err_2[i])
                    t_err_plot_1.append(t_err_plot_2[i])
                    R_er_plot_1.append(R_er_plot_2[i])
                    Vr_er_plot_1.append(Vr_er_plot_2[i])
                    theta_er_plot_1.append(theta_er_plot_2[i])

                track_meas, sko_R_meas, sko_Vr_meas, sko_theta_meas = func_std_error_meas(t_err_plot_1, R_er_plot_1,
                                                                                          Vr_er_plot_1,
                                                                                          theta_er_plot_1,
                                                                                          R_est_err_1,
                                                                                          Vr_est_err_1,
                                                                                          theta_est_err_1, sko_R_tz,
                                                                                          sko_Vr_tz,
                                                                                          sko_theta_tz)

                z_derivation = func_derivation(K1, K2, x_true_fin[-1], config.v0, config.alpha)

                z_wind = func_wind(t_fin[-1], x_true_fin[-1], config.v0, config.alpha, config.wind_module,
                                   config.wind_direction, config.az)

                z = z_wind + z_derivation

                x_fall_gk, z_fall_gk = func_point_fall(z, x_true_fin[-1], config.can_B, config.can_L,
                                                       config.az)

                Vb = x_true_fin[-1] * np.sin(3 * sko_theta_tz)
                Vd = 3 * sko_R_tz

                track_meas = {}
                meas = []

                for i in range(len(t_start)):
                    meas.append({"t": t_start[i], "x": x_true_start[i], "y": h_true_start[i],
                                 "z": 0, "V": V_abs_true_start[i], "Vx": Vx_true_start[i],
                                 "Vy": Vh_true_start[i], "Vz": 0, "A": A_abs_true_start[i],
                                 "Ax": Ax_true_start[i],
                                 "Ay": Ah_true_start[i], "Az": 0, "C": x_est_start[0],
                                 "alpha": np.rad2deg(alpha_true_start[i]),
                                 "DistanceR": R_true_start[i], "AzR": 0,
                                 "VrR": Vr_true_start[i], "EvR": np.rad2deg(theta_true_start[i])})

                for i in range(len(t_meas_plot_1)):
                    for j in range(len(t_meas_plot_1[i]) - 1):
                        meas.append({"t": t_meas_plot_1[i][j], "x": x_tr_er_plot_1[i][j], "y": h_tr_er_plot_1[i][j],
                                     "z": 0, "V": V_abs_full_plot_1[i][j], "Vx": Vx_true_er_plot_1[i][j],
                                     "Vy": Vh_true_er_plot_1[i][j], "Vz": 0, "A": A_abs_est_plot_1[i][j],
                                     "Ax": Ax_true_er_plot_1[i][j],
                                     "Ay": Ah_true_er_plot_1[i][j], "Az": 0, "C": x_est_fin_1[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot_1[i][j]),
                                     "DistanceR": R_est_full_plot_1[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot_1[i][j], "EvR": np.rad2deg(theta_est_full_plot_1[i][j])})

                for i in range(len(t_tr_act_est)):
                    meas.append({"t": t_tr_act_est[i], "x": x_tr_act_est[i], "y": h_tr_act_est[i],
                                 "z": 0, "V": V_abs_tr_act_est[i], "Vx": Vx_tr_act_est[i],
                                 "Vy": Vh_tr_act_est[i], "Vz": 0, "A": A_abs_tr_act_est[i],
                                 "Ax": Ax_tr_act_est[i],
                                 "Ay": Ah_tr_act_est[i], "Az": 0, "C": x_est_fin_1[-1][0],
                                 "alpha": np.rad2deg(alpha_tr_act_est[i]),
                                 "DistanceR": R_tr_act_est[i], "AzR": 0,
                                 "VrR": Vr_tr_act_est[i], "EvR": np.rad2deg(theta_tr_act_est[i])})

                for i in range(len(t_meas_plot_2)):
                    for j in range(len(t_meas_plot_2[i]) - 1):
                        meas.append({"t": t_meas_plot_2[i][j], "x": x_tr_er_plot_2[i][j], "y": h_tr_er_plot_2[i][j],
                                     "z": 0, "V": V_abs_full_plot_2[i][j], "Vx": Vx_true_er_plot_2[i][j],
                                     "Vy": Vh_true_er_plot_2[i][j], "Vz": 0, "A": A_abs_est_plot_2[i][j],
                                     "Ax": Ax_true_er_plot_2[i][j],
                                     "Ay": Ah_true_er_plot_2[i][j], "Az": 0, "C": x_est_fin_2[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot_2[i][j]),
                                     "DistanceR": R_est_full_plot_2[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot_2[i][j], "EvR": np.rad2deg(theta_est_full_plot_2[i][j])})

                for i in range(len(t_fin)):
                    meas.append({"t": t_fin[i], "x": x_true_fin[i], "y": h_true_fin[i],
                                 "z": 0, "V": V_abs_true_fin[i], "Vx": Vx_true_fin[i],
                                 "Vy": Vh_true_fin[i], "Vz": 0, "A": A_abs_true_fin[i],
                                 "Ax": Ax_true_fin[i],
                                 "Ay": Ah_true_fin[i], "Az": 0, "C": x_est_fin_2[-1][0],
                                 "alpha": np.rad2deg(alpha_true_fin[i]),
                                 "DistanceR": R_true_fin[i], "AzR": 0,
                                 "VrR": Vr_true_fin[i], "EvR": np.rad2deg(theta_true_fin[i])})

               # meas_sampling = sampling_points(meas, TD)
                meas_sampling = meas

                track_meas["points"] = meas_sampling
                track_meas["endpoint_x"] = x_true_fin[-1]
                track_meas["endpoint_y"] = h_true_fin[-1]
                track_meas["endpoint_z"] = z
                track_meas["endpoint_GK_x"] = x_fall_gk[0]
                track_meas["endpoint_GK_z"] = z_fall_gk[0]
                track_meas["Vb"] = Vb
                track_meas["Vd"] = Vd
                track_meas["SKO_R"] = sko_R_meas
                track_meas["SKO_V"] = sko_Vr_meas
                track_meas["SKO_theta"] = sko_theta_meas
                track_meas["valid"] = True

                for i in range(len(az_meas) - 1):
                    for j in range(len(track_meas["points"])):
                        if t_meas[i] <= track_meas["points"][j]["t"] < t_meas[i + 1]:
                            track_meas["points"][j]["AzR"] = az_meas[i]

                print('')
                print(x_true_fin[-1], 'х - точки падения')
                print(h_true_fin[-1], 'h - точки падения')

                print(z, 'z - точки падения')
                print(x_fall_gk[0], 'х_fall_gk - точки падения')
                print(z_fall_gk[0], 'z_fall_gk - точки падения')

                print(sko_R_meas, sko_Vr_meas, np.rad2deg(sko_theta_meas), 'значение СКО после отсева измерений')
                print(sR, sVr, np.rad2deg(stheta), "значение СКО измеренное - из файла")
                print(sko_R_tz, sko_Vr_tz, np.rad2deg(sko_theta_tz), 'СКО по ТЗ')

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1

                track_meas = {}
                track_meas["valid"] = False
                track_meas["error"] = "calculation error 152 act-react"

                config.track = track_meas
                config.track_meas = track_meas

        if config.bullet_type == 7:  # 152 art

            try:

                Cx = 0.26
                r = 0.152 / 2

                K1 = 0.00469403894621853
                K2 = -1.48037192545477e-07

                parameters_bounds = [config.k_bounds, config.v0_bounds, config.dR_bounds, config.angle_bounds]

                t_ind_end_1part, t_ind_start_2part = func_active_reactive(t_meas, R_meas, Vr_meas)

                if t_ind_end_1part != 0 and t_ind_start_2part != 0:
                    config.flag_return = 1

                    track_meas = {}
                    track_meas["valid"] = False
                    track_meas["error"] = "two part 152 art error"

                    config.track = track_meas
                    config.track_meas = track_meas
                    return

                theta_meas_filter = func_angle_smoother(theta_meas, t_meas, config.sigma_theta)
                R_meas_filter,  Vr_meas_filter = func_coord_smoother(R_meas, Vr_meas, t_meas, config.sigma_RVr)


                xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, R_meas_tr, \
                Vr_meas_tr, theta_meas_tr = func_quad_piece_app(config.loc_X, config.loc_Y, config.loc_Z,
                                                                config.can_Y,
                                                                config.m, g, config.SKO_R,
                                                                config.SKO_Vr, config.SKO_theta, config.k0,
                                                                config.dR, t_meas,
                                                                R_meas_filter, Vr_meas_filter, theta_meas_filter,
                                                                winlen,
                                                                step_sld, parameters_bounds, types=1)

                x_est_start = func_trajectory_start(Cx, r, rho_0, M, R, T, config.m, g, xhy_0_set,
                                                    x_est_fin, t_meas, N)

                x_est_app_start = func_quad_piece_app_start(config.loc_X, config.loc_Y, config.loc_Z,
                                                            config.can_Y,
                                                            config.m, g, config.SKO_R,
                                                            config.SKO_Vr, config.SKO_theta, x_est_start, t_meas,
                                                            R_meas_filter, Vr_meas_filter,
                                                            theta_meas_filter,
                                                            window_set, parameters_bounds)

                t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, theta_est_full_plot, \
                Vx_true_er_plot, Vh_true_er_plot, V_abs_est_plot, alpha_tr_er_plot, A_abs_est_plot, Ax_true_er_plot, \
                Ah_true_er_plot = func_quad_piece_estimation(
                    xhy_0_set, x_est_fin, meas_t_ind, window_set, t_meas_tr, N,
                    config.m, g, config.loc_X, config.loc_Y, config.loc_Z)

                t_start, x_true_start, h_true_start, R_true_start, Vr_true_start, theta_true_start, Vx_true_start, Vh_true_start, \
                V_abs_true_start, alpha_true_start, A_abs_true_start, Ax_true_start, Ah_true_start = func_quad_piece_estimation_start(
                    x_est_app_start, t_meas_plot, config.m, g, config.loc_X, config.loc_Y, config.loc_Z, N)

                t_fin, x_true_fin, h_true_fin, R_true_fin, Vr_true_fin, theta_true_fin, Vx_true_fin, Vh_true_fin, V_abs_true_fin, \
                alpha_true_fin, A_abs_true_fin, Ax_true_fin, Ah_true_fin = func_trajectory_end(Cx, r, rho_0, M, R, T,
                                                                                               config.m, g,
                                                                                               x_tr_er_plot,
                                                                                               h_tr_er_plot,
                                                                                               Vx_true_er_plot,
                                                                                               Vh_true_er_plot,
                                                                                               V_abs_est_plot,
                                                                                               Ax_true_er_plot,
                                                                                               Ah_true_er_plot,
                                                                                               A_abs_est_plot,
                                                                                               alpha_tr_er_plot,
                                                                                               t_meas_plot,
                                                                                               R_est_full_plot,
                                                                                               Vr_est_full_plot,
                                                                                               theta_est_full_plot,
                                                                                               config.loc_X,
                                                                                               config.loc_Y,
                                                                                               config.loc_Z, config.hei, Nend)

                R_est_err, Vr_est_err, theta_est_err, t_err_plot, R_er_plot, Vr_er_plot, theta_er_plot = func_quad_piece_estimation_error(
                    xhy_0_set, x_est_fin,
                    meas_t_ind, window_set, t_meas,
                    R_meas_filter,
                    Vr_meas_filter,
                    theta_meas_filter, config.m, g,
                    config.loc_X,
                    config.loc_Y, config.loc_Z)

                track_meas, sko_R_meas, sko_Vr_meas, sko_theta_meas = func_std_error_meas(t_err_plot, R_er_plot,
                                                                                          Vr_er_plot,
                                                                                          theta_er_plot,
                                                                                          R_est_err,
                                                                                          Vr_est_err,
                                                                                          theta_est_err, sko_R_tz,
                                                                                          sko_Vr_tz,
                                                                                          sko_theta_tz)

                z_derivation = func_derivation(K1, K2, x_true_fin[-1], config.v0, config.alpha)

                z_wind = func_wind(t_fin[-1], x_true_fin[-1], config.v0, config.alpha, config.wind_module,
                                   config.wind_direction, config.az)

                z = z_wind + z_derivation

                x_fall_gk, z_fall_gk = func_point_fall(z, x_true_fin[-1], config.can_B, config.can_L,
                                                       config.az)

                Vb = x_true_fin[-1] * np.sin(3 * sko_theta_tz)
                Vd = 3 * sko_R_tz

                track_meas = {}
                meas = []

                for i in range(len(t_start)):
                    meas.append({"t": t_start[i], "x": x_true_start[i], "y": h_true_start[i],
                                 "z": 0, "V": V_abs_true_start[i], "Vx": Vx_true_start[i],
                                 "Vy": Vh_true_start[i], "Vz": 0, "A": A_abs_true_start[i],
                                 "Ax": Ax_true_start[i],
                                 "Ay": Ah_true_start[i], "Az": 0, "C": x_est_start[0],
                                 "alpha": np.rad2deg(alpha_true_start[i]),
                                 "DistanceR": R_true_start[i], "AzR": 0,
                                 "VrR": Vr_true_start[i], "EvR": np.rad2deg(theta_true_start[i])})

                for i in range(len(t_meas_plot)):
                    for j in range(len(t_meas_plot[i]) - 1):
                        meas.append({"t": t_meas_plot[i][j], "x": x_tr_er_plot[i][j], "y": h_tr_er_plot[i][j],
                                     "z": 0, "V": V_abs_est_plot[i][j], "Vx": Vx_true_er_plot[i][j],
                                     "Vy": Vh_true_er_plot[i][j], "Vz": 0, "A": A_abs_est_plot[i][j],
                                     "Ax": Ax_true_er_plot[i][j],
                                     "Ay": Ah_true_er_plot[i][j], "Az": 0, "C": x_est_fin[i][0],
                                     "alpha": np.rad2deg(alpha_tr_er_plot[i][j]),
                                     "DistanceR": R_est_full_plot[i][j], "AzR": 0,
                                     "VrR": Vr_est_full_plot[i][j], "EvR": np.rad2deg(theta_est_full_plot[i][j])})

                for i in range(len(t_fin)):
                    meas.append({"t": t_fin[i], "x": x_true_fin[i], "y": h_true_fin[i],
                                 "z": 0, "V": V_abs_true_fin[i], "Vx": Vx_true_fin[i],
                                 "Vy": Vh_true_fin[i], "Vz": 0, "A": A_abs_true_fin[i],
                                 "Ax": Ax_true_fin[i],
                                 "Ay": Ah_true_fin[i], "Az": 0, "C": x_est_fin[-1][0],
                                 "alpha": np.rad2deg(alpha_true_fin[i]),
                                 "DistanceR": R_true_fin[i], "AzR": 0,
                                 "VrR": Vr_true_fin[i], "EvR": np.rad2deg(theta_true_fin[i])})

                #meas_sampling = sampling_points(meas, TD)
                meas_sampling = meas

                track_meas["points"] = meas_sampling
                track_meas["endpoint_x"] = x_true_fin[-1]
                track_meas["endpoint_y"] = h_true_fin[-1]
                track_meas["endpoint_z"] = z
                track_meas["endpoint_GK_x"] = x_fall_gk[0]
                track_meas["endpoint_GK_z"] = z_fall_gk[0]
                track_meas["Vb"] = Vb
                track_meas["Vd"] = Vd
                track_meas["SKO_R"] = sko_R_meas
                track_meas["SKO_V"] = sko_Vr_meas
                track_meas["SKO_theta"] = sko_theta_meas
                track_meas["valid"] = True

                print('')
                print(x_true_fin[-1], 'х - точки падения')
                print(h_true_fin[-1], 'h - точки падения')

                print(z, 'z - точки падения')
                print(x_fall_gk[0], 'х_fall_gk - точки падения')
                print(z_fall_gk[0], 'z_fall_gk - точки падения')

                print(sko_R_meas, sko_Vr_meas, np.rad2deg(sko_theta_meas), 'значение СКО после отсева измерений')
                print(sR, sVr, np.rad2deg(stheta), "значение СКО измеренное - из файла")
                print(sko_R_tz, sko_Vr_tz, np.rad2deg(sko_theta_tz), 'СКО по ТЗ')

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1

                track_meas = {}
                track_meas["valid"] = False
                track_meas["error"] = "calculation error 152 art"

                config.track = track_meas
                config.track_meas = track_meas

        if config.data_points == 1:

            hashes = '#' * int(round(20))
            spaces = ' ' * (20 - len(hashes))
            sys.stdout.write("\rCalculating %: [{0}] {1}% {2} seconds".format(hashes + spaces, int(round(100)),
                                                                              (time.process_time() - start_time)))
            sys.stdout.flush()

            config.track = track_meas
            config.track_meas = track_meas

        flag = 1

        if flag:
            return True
        else:
            return False
    else:
        return False
