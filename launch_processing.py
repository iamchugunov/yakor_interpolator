import time
import sys
import json
import pymap3d as pm

from app_functions import *


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

        config.alpha_0 = np.deg2rad(mes["alpha"])
        config.az = np.deg2rad(mes["az"])
        config.hei = mes["hei"]

        config.wind_module = mes["wind_module"]
        config.wind_direction = mes["wind_direction"]

        config.temperature = mes["temperature"]
        config.atm_pressure = mes["atm_pressure"]

        config.bullet_type = mes["bullet_type"]

        # type bullet
        bullet = config.bullets[config.bullet_type - 1]

        config.m = bullet["m"]

        config.SKO_R = bullet["SKO_R"]
        config.SKO_Vr = bullet["SKO_Vr"]
        config.SKO_theta = np.deg2rad(bullet["SKO_theta"])

        # config - rad
        config.v0 = bullet["v0"]
        config.r = bullet["r"]
        config.l = bullet["l"]
        config.d = bullet["d"]
        config.eta = bullet["eta"]

        # flag = 1 - message
        config.ini_data_flag = 1
        # flag = 0 - measurements
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

        track = {"valid": False, "error": "received message initial data with error"}
        config.track = track


def process_measurements(data, config):
    if config.ini_data_flag:

        start_time = time.process_time()
        track = {}

        points_length = len(data["points"])
        time_meas = np.zeros(points_length)
        range_meas = np.zeros(points_length)
        radial_velocity_meas = np.zeros(points_length)
        theta_meas = np.zeros(points_length)

        sko_R_tz = 5
        sko_Vr_tz = 0.5
        sko_theta_tz = np.deg2rad(0.1)  # grad - > rad

        try:

            for i, point in enumerate(data["points"]):
                time_meas[i] = point["execTime"]
                range_meas[i] = point["R"]
                radial_velocity_meas[i] = abs(point["Vr"])
                theta_meas[i] = np.deg2rad(point["Epsilon"])

            # SKO measurement
            sR = data["points"][0]["sR"]
            sVr = abs(data["points"][0]["sVr"])
            # SKO angle grad - > rad
            stheta = np.deg2rad(data["points"][0]["sEpsilon"])
            saz = np.deg2rad(data["points"][0]["sBeta"])

            config.ini_meas_flag = 1

        except KeyError:

            config.flag_return = 1
            track = {"valid": False, "error": "received message measurements with error"}
            config.track = track

        if config.bullet_type == 1:  # 5.45 bullet

            try:

                theta_smoother = rts_angle_smoother(time_meas, theta_meas, sigma_theta=0.4, sigma_ksi=0.04,
                                                    sigma_n=5e-3)
                range_smoother, radial_velocity_smoother = rts_coord_smoother(time_meas, range_meas,
                                                                              radial_velocity_meas, sigma_coord=-40,
                                                                              sigma_ksi=10, sigma_n1=1,
                                                                              sigma_n2=0.03)

                time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full = time_step_filling_data(
                    time_meas, range_smoother, radial_velocity_smoother, theta_smoother)

                x_estimation_stor = formation_estimation_on_alpha(time_meas_full, range_meas_full,
                                                                  radial_velocity_meas_full, theta_meas_full,
                                                                  config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(x_estimation_stor,
                                                                                                   time_meas_full,
                                                                                                   range_meas_full,
                                                                                                   theta_meas_full,
                                                                                                   config.r, config.m,
                                                                                                   config.can_H,
                                                                                                   window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=3,
                                                                     std_window_length=10)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full, i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control, x_set_control, h_set_control, velocity_x_set_control, velocity_h_set_control, \
                as_x_set_control, as_h_set_control = structuring_approximate_values(time_start, x_set_0, h_set_0,
                                                                                    velocity_x_set_0,
                                                                                    velocity_h_set_0, i_f_estimation,
                                                                                    config.alpha_0, time_meas, config.r,
                                                                                    config.m, config.loc_X,
                                                                                    config.loc_Y, config.loc_Z,
                                                                                    config.can_H)

                y_meas_set = [range_meas_full,
                              radial_velocity_meas_full,
                              theta_meas_full,
                              0.01 * np.ones(len(time_meas_full))]

                x_est_init = [0, velocity_x_set_control[0], as_x_set_control[0], 0,
                              velocity_h_set_control[0], as_h_set_control[0], 0, 0, 0]

                x_est_stor, y_ext_stor, time_meas_stor = trajectory_points_approximation(y_meas_set, x_est_init,
                                                                                         time_meas_full, config.loc_X,
                                                                                         config.loc_Y, config.loc_Z,
                                                                                         time_meas_control,
                                                                                         x_set_control, h_set_control,
                                                                                         velocity_x_set_control,
                                                                                         velocity_h_set_control,
                                                                                         as_x_set_control,
                                                                                         as_h_set_control)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(x_est_stor,
                                                                                                 time_meas_stor,
                                                                                                 i_f_estimation,
                                                                                                 config.r, config.m,
                                                                                                 config.loc_X,
                                                                                                 config.loc_Y,
                                                                                                 config.loc_Z,
                                                                                                 config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(y_ext_stor, time_meas_stor, time_meas,
                                                                           range_smoother, radial_velocity_smoother,
                                                                           theta_smoother)

                time_meas_full = np.concatenate((time_meas_stor, time_meas_fin_stor))
                x_est_stor_full = np.concatenate((x_est_stor, x_est_fin_stor))
                y_ext_stor_full = np.concatenate((y_ext_stor, y_ext_fin_stor))

                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_derivation = derivation_calculation_bullet(config.m, config.d, config.l, config.eta,
                                                             velocity_0_estimation,
                                                             time_fall)

                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind + z_derivation

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz),
                         "Vd": x_fall * np.sin(3 * sko_theta_tz),
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 5.45 bullet or 7.65 bullet"}
                config.track = track

        if config.bullet_type == 2:  # 7.65 bullet

            try:

                theta_smoother = rts_angle_smoother(time_meas, theta_meas, sigma_theta=0.4, sigma_ksi=0.04,
                                                    sigma_n=5e-3)
                range_smoother, radial_velocity_smoother = rts_coord_smoother(time_meas, range_meas,
                                                                              radial_velocity_meas, sigma_coord=-40,
                                                                              sigma_ksi=10, sigma_n1=1,
                                                                              sigma_n2=0.03)

                time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full = time_step_filling_data(
                    time_meas, range_smoother, radial_velocity_smoother, theta_smoother)

                x_estimation_stor = formation_estimation_on_alpha(time_meas_full, range_meas_full,
                                                                  radial_velocity_meas_full, theta_meas_full,
                                                                  config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(x_estimation_stor,
                                                                                                   time_meas_full,
                                                                                                   range_meas_full,
                                                                                                   theta_meas_full,
                                                                                                   config.r, config.m,
                                                                                                   config.can_H,
                                                                                                   window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=2,
                                                                     std_window_length=8)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full, i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control, x_set_control, h_set_control, velocity_x_set_control, velocity_h_set_control, \
                as_x_set_control, as_h_set_control = structuring_approximate_values(time_start, x_set_0, h_set_0,
                                                                                    velocity_x_set_0,
                                                                                    velocity_h_set_0, i_f_estimation,
                                                                                    config.alpha_0, time_meas, config.r,
                                                                                    config.m, config.loc_X,
                                                                                    config.loc_Y, config.loc_Z,
                                                                                    config.can_H)

                y_meas_set = [range_meas_full,
                              radial_velocity_meas_full,
                              theta_meas_full,
                              0.01 * np.ones(len(time_meas_full))]

                x_est_init = [0, velocity_x_set_control[0], as_x_set_control[0], 0,
                              velocity_h_set_control[0], as_h_set_control[0], 0, 0, 0]

                x_est_stor, y_ext_stor, time_meas_stor = trajectory_points_approximation(y_meas_set, x_est_init,
                                                                                         time_meas_full, config.loc_X,
                                                                                         config.loc_Y, config.loc_Z,
                                                                                         time_meas_control,
                                                                                         x_set_control, h_set_control,
                                                                                         velocity_x_set_control,
                                                                                         velocity_h_set_control,
                                                                                         as_x_set_control,
                                                                                         as_h_set_control)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(x_est_stor,
                                                                                                 time_meas_stor,
                                                                                                 i_f_estimation,
                                                                                                 config.r, config.m,
                                                                                                 config.loc_X,
                                                                                                 config.loc_Y,
                                                                                                 config.loc_Z,
                                                                                                 config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(y_ext_stor, time_meas_stor, time_meas,
                                                                           range_smoother, radial_velocity_smoother,
                                                                           theta_smoother)

                time_meas_full = np.concatenate((time_meas_stor, time_meas_fin_stor))
                x_est_stor_full = np.concatenate((x_est_stor, x_est_fin_stor))
                y_ext_stor_full = np.concatenate((y_ext_stor, y_ext_fin_stor))

                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_derivation = derivation_calculation_bullet(config.m, config.d, config.l, config.eta,
                                                             velocity_0_estimation,
                                                             time_fall)

                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind + z_derivation

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz),
                         "Vd": x_fall * np.sin(3 * sko_theta_tz),
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 5.45 bullet or 7.65 bullet"}
                config.track = track

        if config.bullet_type == 3:  # 82 mina

            try:

                theta_smoother = rts_angle_smoother(time_meas, theta_meas, sigma_theta=0.4, sigma_ksi=0.1, sigma_n=5e-3)
                range_smoother, radial_velocity_smoother = rts_coord_smoother(time_meas, range_meas,
                                                                              radial_velocity_meas, sigma_coord=-40,
                                                                              sigma_ksi=10, sigma_n1=1,
                                                                              sigma_n2=0.3)

                time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full = time_step_filling_data(
                    time_meas, range_smoother, radial_velocity_smoother, theta_smoother)

                x_estimation_stor = formation_estimation_on_alpha(time_meas_full, range_meas_full,
                                                                  radial_velocity_meas_full, theta_meas_full,
                                                                  config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(x_estimation_stor,
                                                                                                   time_meas_full,
                                                                                                   range_meas_full,
                                                                                                   theta_meas_full,
                                                                                                   config.r, config.m,
                                                                                                   config.can_H,
                                                                                                   window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=5,
                                                                     std_window_length=30)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full, i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control, x_set_control, h_set_control, velocity_x_set_control, velocity_h_set_control, \
                as_x_set_control, as_h_set_control = structuring_approximate_values(time_start, x_set_0, h_set_0,
                                                                                    velocity_x_set_0,
                                                                                    velocity_h_set_0, i_f_estimation,
                                                                                    config.alpha_0, time_meas, config.r,
                                                                                    config.m, config.loc_X,
                                                                                    config.loc_Y, config.loc_Z,
                                                                                    config.can_H)

                y_meas_set = [range_meas_full,
                              radial_velocity_meas_full,
                              theta_meas_full,
                              0.01 * np.ones(len(time_meas_full))]

                x_est_init = [0, velocity_x_set_control[0], as_x_set_control[0], 0,
                              velocity_h_set_control[0], as_h_set_control[0], 0, 0, 0]

                x_est_stor, y_ext_stor, time_meas_stor = trajectory_points_approximation(y_meas_set, x_est_init,
                                                                                         time_meas_full, config.loc_X,
                                                                                         config.loc_Y, config.loc_Z,
                                                                                         time_meas_control,
                                                                                         x_set_control, h_set_control,
                                                                                         velocity_x_set_control,
                                                                                         velocity_h_set_control,
                                                                                         as_x_set_control,
                                                                                         as_h_set_control)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(x_est_stor,
                                                                                                 time_meas_stor,
                                                                                                 i_f_estimation,
                                                                                                 config.r, config.m,
                                                                                                 config.loc_X,
                                                                                                 config.loc_Y,
                                                                                                 config.loc_Z,
                                                                                                 config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(y_ext_stor, time_meas_stor, time_meas,
                                                                           range_smoother, radial_velocity_smoother,
                                                                           theta_smoother)

                time_meas_full = np.concatenate((time_meas_stor, time_meas_fin_stor))
                x_est_stor_full = np.concatenate((x_est_stor, x_est_fin_stor))
                y_ext_stor_full = np.concatenate((y_ext_stor, y_ext_fin_stor))

                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz), "Vd": 3 * sko_R_tz,
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 82 mina"}
                config.track = track

        if config.bullet_type == 4:  # 122 reactive

            try:

                dv_dt = np.zeros(len(radial_velocity_meas) - 1)
                st_passive_ind = 0
                for i in range(1, len(radial_velocity_meas)):
                    dv_dt[i - 1] = (radial_velocity_meas[i] - radial_velocity_meas[i - 1]) / (
                            time_meas[i] - time_meas[i - 1])
                    if (i > 3) and (dv_dt[i - 1] < 0) and (dv_dt[i - 2]) < 0:
                        st_passive_ind = i - 2
                        break

                time_meas = time_meas[st_passive_ind:]
                range_meas = range_meas[st_passive_ind:]
                radial_velocity_meas = radial_velocity_meas[st_passive_ind:]
                theta_meas = theta_meas[st_passive_ind:]

                bad_ind = emissions_theta(theta_meas)

                time_meas = np.delete(time_meas, bad_ind)
                range_meas = np.delete(range_meas, bad_ind)
                radial_velocity_meas = np.delete(radial_velocity_meas, bad_ind)
                theta_meas = np.delete(theta_meas, bad_ind)

                theta_smoother = rts_angle_smoother(time_meas, theta_meas, sigma_theta=0.4, sigma_ksi=0.1, sigma_n=5e-3)
                range_smoother, radial_velocity_smoother = rts_coord_smoother(time_meas, range_meas,
                                                                              radial_velocity_meas, sigma_coord=-40,
                                                                              sigma_ksi=10, sigma_n1=1,
                                                                              sigma_n2=0.3)

                time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full = time_step_filling_data(
                    time_meas, range_smoother, radial_velocity_smoother, theta_smoother)

                x_estimation_stor = formation_estimation_on_alpha(time_meas_full, range_meas_full,
                                                                  radial_velocity_meas_full, theta_meas_full,
                                                                  config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(x_estimation_stor,
                                                                                                   time_meas_full,
                                                                                                   range_meas_full,
                                                                                                   theta_meas_full,
                                                                                                   config.r, config.m,
                                                                                                   config.can_H,
                                                                                                   window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=5,
                                                                     std_window_length=30)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full, i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control, x_set_control, h_set_control, velocity_x_set_control, velocity_h_set_control, \
                as_x_set_control, as_h_set_control = structuring_approximate_values(time_start, x_set_0, h_set_0,
                                                                                    velocity_x_set_0,
                                                                                    velocity_h_set_0, i_f_estimation,
                                                                                    config.alpha_0, time_meas, config.r,
                                                                                    config.m, config.loc_X,
                                                                                    config.loc_Y, config.loc_Z,
                                                                                    config.can_H)

                y_meas_set = [range_meas_full,
                              radial_velocity_meas_full,
                              theta_meas_full,
                              0.01 * np.ones(len(time_meas_full))]

                x_est_init = [0, velocity_x_set_control[0], as_x_set_control[0], 0,
                              velocity_h_set_control[0], as_h_set_control[0], 0, 0, 0]

                x_est_stor, y_ext_stor, time_meas_stor = trajectory_points_approximation(y_meas_set, x_est_init,
                                                                                         time_meas_full, config.loc_X,
                                                                                         config.loc_Y, config.loc_Z,
                                                                                         time_meas_control,
                                                                                         x_set_control, h_set_control,
                                                                                         velocity_x_set_control,
                                                                                         velocity_h_set_control,
                                                                                         as_x_set_control,
                                                                                         as_h_set_control)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(x_est_stor,
                                                                                                 time_meas_stor,
                                                                                                 i_f_estimation,
                                                                                                 config.r, config.m,
                                                                                                 config.loc_X,
                                                                                                 config.loc_Y,
                                                                                                 config.loc_Z,
                                                                                                 config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(y_ext_stor, time_meas_stor, time_meas,
                                                                           range_smoother, radial_velocity_smoother,
                                                                           theta_smoother)

                time_meas_full = np.concatenate((time_meas_stor, time_meas_fin_stor))
                x_est_stor_full = np.concatenate((x_est_stor, x_est_fin_stor))
                y_ext_stor_full = np.concatenate((y_ext_stor, y_ext_fin_stor))

                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz), "Vd": 3 * sko_R_tz,
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 122 reactive"}
                config.track = track

        if config.bullet_type == 5:  # 122 - art

            try:

                K1 = 0.00461217647718868
                K2 = -2.04678100654676e-07

                theta_smoother = rts_angle_smoother(time_meas, theta_meas, sigma_theta=0.4, sigma_ksi=0.1, sigma_n=5e-3)
                range_smoother, radial_velocity_smoother = rts_coord_smoother(time_meas, range_meas,
                                                                              radial_velocity_meas, sigma_coord=-40,
                                                                              sigma_ksi=10, sigma_n1=1,
                                                                              sigma_n2=0.3)

                time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full = time_step_filling_data(
                    time_meas, range_smoother, radial_velocity_smoother, theta_smoother)

                x_estimation_stor = formation_estimation_on_alpha(time_meas_full, range_meas_full,
                                                                  radial_velocity_meas_full, theta_meas_full,
                                                                  config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(x_estimation_stor,
                                                                                                   time_meas_full,
                                                                                                   range_meas_full,
                                                                                                   theta_meas_full,
                                                                                                   config.r, config.m,
                                                                                                   config.can_H,
                                                                                                   window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=5,
                                                                     std_window_length=30)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full, i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control, x_set_control, h_set_control, velocity_x_set_control, velocity_h_set_control, \
                as_x_set_control, as_h_set_control = structuring_approximate_values(time_start, x_set_0, h_set_0,
                                                                                    velocity_x_set_0,
                                                                                    velocity_h_set_0, i_f_estimation,
                                                                                    config.alpha_0, time_meas, config.r,
                                                                                    config.m, config.loc_X,
                                                                                    config.loc_Y, config.loc_Z,
                                                                                    config.can_H)

                y_meas_set = [range_meas_full,
                              radial_velocity_meas_full,
                              theta_meas_full,
                              0.01 * np.ones(len(time_meas_full))]

                x_est_init = [0, velocity_x_set_control[0], as_x_set_control[0], 0,
                              velocity_h_set_control[0], as_h_set_control[0], 0, 0, 0]

                x_est_stor, y_ext_stor, time_meas_stor = trajectory_points_approximation(y_meas_set, x_est_init,
                                                                                         time_meas_full, config.loc_X,
                                                                                         config.loc_Y, config.loc_Z,
                                                                                         time_meas_control,
                                                                                         x_set_control, h_set_control,
                                                                                         velocity_x_set_control,
                                                                                         velocity_h_set_control,
                                                                                         as_x_set_control,
                                                                                         as_h_set_control)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(x_est_stor,
                                                                                                 time_meas_stor,
                                                                                                 i_f_estimation,
                                                                                                 config.r, config.m,
                                                                                                 config.loc_X,
                                                                                                 config.loc_Y,
                                                                                                 config.loc_Z,
                                                                                                 config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(y_ext_stor, time_meas_stor, time_meas,
                                                                           range_smoother, radial_velocity_smoother,
                                                                           theta_smoother)

                time_meas_full = np.concatenate((time_meas_stor, time_meas_fin_stor))
                x_est_stor_full = np.concatenate((x_est_stor, x_est_fin_stor))
                y_ext_stor_full = np.concatenate((y_ext_stor, y_ext_fin_stor))

                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_derivation = derivation_calculation(x_fall, velocity_0_estimation, config.alpha_0, K1, K2)
                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind + z_derivation

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz), "Vd": 3 * sko_R_tz,
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 122 art"}
                config.track = track

        if config.bullet_type == 6:  # 152 act-react
            try:

                K1 = 0.00469403894621853
                K2 = -1.48037192545477e-07

                act_start_index, act_end_index = act_react_partition(time_meas, range_meas, radial_velocity_meas)

                time_meas_one_part = time_meas[:act_start_index - 1]
                range_meas_one_part = range_meas[:act_start_index - 1]
                radial_velocity_meas_one_part = radial_velocity_meas[:act_start_index - 1]
                theta_meas_one_part = theta_meas[:act_start_index - 1]

                theta_smoother_one_part = rts_angle_smoother(time_meas_one_part, theta_meas_one_part, sigma_theta=0.4,
                                                             sigma_ksi=1,
                                                             sigma_n=5e-3)
                range_smoother_one_part, radial_velocity_smoother_one_part = rts_coord_smoother(time_meas_one_part,
                                                                                                range_meas_one_part,
                                                                                                radial_velocity_meas_one_part,
                                                                                                sigma_coord=-40,
                                                                                                sigma_ksi=1e1,
                                                                                                sigma_n1=0.1e1,
                                                                                                sigma_n2=0.3e0)

                time_meas_full_one_part, range_meas_full_one_part, radial_velocity_meas_full_one_part, theta_meas_full_one_part = time_step_filling_data(
                    time_meas_one_part, range_smoother_one_part, radial_velocity_smoother_one_part,
                    theta_smoother_one_part)

                x_estimation_stor_one_part = formation_estimation_on_alpha(time_meas_full_one_part,
                                                                           range_meas_full_one_part,
                                                                           radial_velocity_meas_full_one_part,
                                                                           theta_meas_full_one_part,
                                                                           config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(
                    x_estimation_stor_one_part,
                    time_meas_full_one_part,
                    range_meas_full_one_part,
                    theta_meas_full_one_part,
                    config.r, config.m,
                    config.can_H, window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=5,
                                                                     std_window_length=30)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full_one_part,
                                                                                     i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control_one_part, x_set_control_one_part, h_set_control_one_part, velocity_x_set_control_one_part, velocity_h_set_control_one_part, \
                as_x_set_control_one_part, as_h_set_control_one_part = structuring_approximate_values(time_start,
                                                                                                      x_set_0, h_set_0,
                                                                                                      velocity_x_set_0,
                                                                                                      velocity_h_set_0,
                                                                                                      i_f_estimation,
                                                                                                      config.alpha_0,
                                                                                                      time_meas_full_one_part,
                                                                                                      config.r,
                                                                                                      config.m,
                                                                                                      config.loc_X,
                                                                                                      config.loc_Y,
                                                                                                      config.loc_Z,
                                                                                                      config.can_H)

                y_meas_set_one_part = [range_meas_full_one_part,
                                       radial_velocity_meas_full_one_part,
                                       theta_meas_full_one_part,
                                       0.01 * np.ones(len(time_meas_full_one_part))]

                x_est_init_one_part = [0, velocity_x_set_control_one_part[0], as_x_set_control_one_part[0], 0,
                                       velocity_h_set_control_one_part[0], as_h_set_control_one_part[0], 0, 0, 0]

                x_est_stor_one_part, y_ext_stor_one_part, time_meas_stor_one_part = trajectory_points_approximation(
                    y_meas_set_one_part,
                    x_est_init_one_part,
                    time_meas_full_one_part,
                    config.loc_X,
                    config.loc_Y, config.loc_Z,
                    time_meas_control_one_part,
                    x_set_control_one_part,
                    h_set_control_one_part,
                    velocity_x_set_control_one_part,
                    velocity_h_set_control_one_part,
                    as_x_set_control_one_part,
                    as_h_set_control_one_part)

                config.m = config.m - 2.2

                time_meas_two_part = time_meas[act_end_index:]
                range_meas_two_part = range_meas[act_end_index:]
                radial_velocity_meas_two_part = radial_velocity_meas[act_end_index:]
                theta_meas_two_part = theta_meas[act_end_index:]

                theta_smoother_two_part = rts_angle_smoother(time_meas_two_part, theta_meas_two_part, sigma_theta=0.4,
                                                             sigma_ksi=1,
                                                             sigma_n=5e-3)
                range_smoother_two_part, radial_velocity_smoother_two_part = rts_coord_smoother(time_meas_two_part,
                                                                                                range_meas_two_part,
                                                                                                radial_velocity_meas_two_part,
                                                                                                sigma_coord=-40,
                                                                                                sigma_ksi=1e1,
                                                                                                sigma_n1=0.1e1,
                                                                                                sigma_n2=0.3e0)

                time_meas_full_two_part, range_meas_full_two_part, radial_velocity_meas_full_two_part, theta_meas_full_two_part = time_step_filling_data(
                    time_meas_two_part, range_smoother_two_part, radial_velocity_smoother_two_part,
                    theta_smoother_two_part)

                x_estimation_stor_two_part = formation_estimation_on_alpha(time_meas_full_two_part,
                                                                           range_meas_full_two_part,
                                                                           radial_velocity_meas_full_two_part,
                                                                           theta_meas_full_two_part,
                                                                           config.loc_X, config.loc_Y, config.loc_Z)

                x_set_0_two_part = x_estimation_stor_two_part[0, 0]
                h_set_0_two_part = x_estimation_stor_two_part[0, 3]
                time_meas_0_two_part = time_meas_full_two_part[0]
                velocity_x_set_0_two_part = x_estimation_stor_two_part[0, 1]
                velocity_h_set_0_two_part = x_estimation_stor_two_part[0, 4]
                alpha_0_two_part = np.arctan(x_estimation_stor_two_part[0, 4] / x_estimation_stor_two_part[0, 1])

                time_meas_control_two_part, x_set_control_two_part, h_set_control_two_part, velocity_x_set_control_two_part, velocity_h_set_control_two_part, \
                as_x_set_control_two_part, as_h_set_control_two_part = structuring_approximate_values(
                    time_meas_0_two_part,
                    x_set_0_two_part,
                    h_set_0_two_part,
                    velocity_x_set_0_two_part,
                    velocity_h_set_0_two_part,
                    i_f_estimation,
                    alpha_0_two_part,
                    time_meas,
                    config.r,
                    config.m,
                    config.loc_X,
                    config.loc_Y,
                    config.loc_Z,
                    config.can_H)

                y_meas_set_two_part = [range_meas_full_two_part,
                                       radial_velocity_meas_full_two_part,
                                       theta_meas_full_two_part,
                                       0.01 * np.ones(len(time_meas_full_two_part))]

                x_est_init_two_part = [x_set_control_two_part[0], velocity_x_set_control_two_part[0],
                                       as_x_set_control_two_part[0],
                                       h_set_control_two_part[0],
                                       velocity_h_set_control_two_part[0], as_h_set_control_two_part[0], 0, 0, 0]

                x_est_stor_two_part, y_ext_stor_two_part, time_meas_stor_two_part = trajectory_points_approximation_act_react(
                    y_meas_set_two_part,
                    x_est_init_two_part,
                    config.loc_X,
                    config.loc_Y, config.loc_Z,
                    time_meas_control_two_part,
                    as_x_set_control_two_part,
                    as_h_set_control_two_part)

                x_est_stor_active, y_ext_stor_active, time_meas_active = active_reactive(time_meas_full_one_part,
                                                                                         time_meas_full_two_part,
                                                                                         x_est_stor_one_part,
                                                                                         x_est_stor_two_part,
                                                                                         i_f_estimation, config.can_H,
                                                                                         config.m, config.r,
                                                                                         config.loc_X,
                                                                                         config.loc_Y, config.loc_Z)

                x_est_init_two_part_active = [x_est_stor_active[-1, 0], x_est_stor_active[-1, 1],
                                              x_est_stor_active[-1, 2],
                                              x_est_stor_active[-1, 3], x_est_stor_active[-1, 4],
                                              x_est_stor_active[-1, 5],
                                              x_est_stor_active[-1, 6], x_est_stor_active[-1, 7],
                                              x_est_stor_active[-1, 8]]

                x_est_stor_two_part_active, y_ext_stor_two_part_active, time_meas_stor_two_part_active = trajectory_points_approximation_act_react(
                    y_meas_set_two_part,
                    x_est_init_two_part_active, config.loc_X,
                    config.loc_Y, config.loc_Z,
                    time_meas_control_two_part,
                    as_x_set_control_two_part,
                    as_h_set_control_two_part)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(
                    x_est_stor_two_part_active,
                    time_meas_stor_two_part_active,
                    i_f_estimation,
                    config.r, config.m,
                    config.loc_X,
                    config.loc_Y, config.loc_Z,
                    config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(
                    np.concatenate((y_ext_stor_one_part, y_ext_stor_two_part_active)),
                    np.concatenate((time_meas_stor_one_part, time_meas_stor_two_part)),
                    np.concatenate((time_meas_one_part, time_meas_two_part)),
                    np.concatenate((range_smoother_one_part, range_smoother_two_part)),
                    np.concatenate((radial_velocity_smoother_one_part, radial_velocity_smoother_two_part)),
                    np.concatenate((theta_smoother_one_part, theta_smoother_two_part)))

                x_est_stor_full = np.concatenate(
                    (x_est_stor_one_part, x_est_stor_active, x_est_stor_two_part_active, x_est_fin_stor))
                y_ext_stor_full = np.concatenate(
                    (y_ext_stor_one_part, y_ext_stor_active, y_ext_stor_two_part_active, y_ext_fin_stor))
                time_meas_full = np.concatenate(
                    (time_meas_stor_one_part, time_meas_active, time_meas_stor_two_part_active, time_meas_fin_stor))
                
                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_derivation = derivation_calculation(x_fall, velocity_0_estimation, config.alpha_0, K1, K2)
                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind + z_derivation

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz), "Vd": 3 * sko_R_tz,
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 152 act-react"}
                config.track = track

        if config.bullet_type == 7:  # 152 art
            try:

                K1 = 0.00469403894621853
                K2 = -1.48037192545477e-07

                theta_smoother = rts_angle_smoother(time_meas, theta_meas, sigma_theta=0.4, sigma_ksi=0.1, sigma_n=5e-3)
                range_smoother, radial_velocity_smoother = rts_coord_smoother(time_meas, range_meas,
                                                                              radial_velocity_meas, sigma_coord=-40,
                                                                              sigma_ksi=10, sigma_n1=1,
                                                                              sigma_n2=0.3)

                time_meas_full, range_meas_full, radial_velocity_meas_full, theta_meas_full = time_step_filling_data(
                    time_meas, range_smoother, radial_velocity_smoother, theta_smoother)

                x_estimation_stor = formation_estimation_on_alpha(time_meas_full, range_meas_full,
                                                                  radial_velocity_meas_full, theta_meas_full,
                                                                  config.loc_X, config.loc_Y, config.loc_Z)

                i_f_from_acceleration_x, velocity_abs_poly_estimation = shape_factor_from_velocity(x_estimation_stor,
                                                                                                   time_meas_full,
                                                                                                   range_meas_full,
                                                                                                   theta_meas_full,
                                                                                                   config.r, config.m,
                                                                                                   config.can_H,
                                                                                                   window_length=5)

                i_f_estimation = approximate_evaluation_shape_factor(i_f_from_acceleration_x, std_shift_length=5,
                                                                     std_window_length=30)

                velocity_0_estimation = initial_velocity_estimation_calculation_norm(time_meas_full, i_f_estimation,
                                                                                     velocity_abs_poly_estimation,
                                                                                     config.alpha_0,
                                                                                     config.r, config.m, config.can_H,
                                                                                     config.v0)
                x_set_0 = 0
                h_set_0 = 0
                time_start = 0
                velocity_x_set_0 = velocity_0_estimation * np.cos(config.alpha_0)
                velocity_h_set_0 = velocity_0_estimation * np.sin(config.alpha_0)

                time_meas_control, x_set_control, h_set_control, velocity_x_set_control, velocity_h_set_control, \
                as_x_set_control, as_h_set_control = structuring_approximate_values(time_start, x_set_0, h_set_0,
                                                                                    velocity_x_set_0,
                                                                                    velocity_h_set_0, i_f_estimation,
                                                                                    config.alpha_0, time_meas, config.r,
                                                                                    config.m, config.loc_X,
                                                                                    config.loc_Y, config.loc_Z,
                                                                                    config.can_H)

                y_meas_set = [range_meas_full,
                              radial_velocity_meas_full,
                              theta_meas_full,
                              0.01 * np.ones(len(time_meas_full))]

                x_est_init = [0, velocity_x_set_control[0], as_x_set_control[0], 0,
                              velocity_h_set_control[0], as_h_set_control[0], 0, 0, 0]

                x_est_stor, y_ext_stor, time_meas_stor = trajectory_points_approximation(y_meas_set, x_est_init,
                                                                                         time_meas_full, config.loc_X,
                                                                                         config.loc_Y, config.loc_Z,
                                                                                         time_meas_control,
                                                                                         x_set_control, h_set_control,
                                                                                         velocity_x_set_control,
                                                                                         velocity_h_set_control,
                                                                                         as_x_set_control,
                                                                                         as_h_set_control)

                x_est_fin_stor, y_ext_fin_stor, time_meas_fin_stor = extrapolation_to_point_fall(x_est_stor,
                                                                                                 time_meas_stor,
                                                                                                 i_f_estimation,
                                                                                                 config.r, config.m,
                                                                                                 config.loc_X,
                                                                                                 config.loc_Y,
                                                                                                 config.loc_Z,
                                                                                                 config.can_H)

                sko_range, sko_radial_velocity, sko_theta = sko_error_meas(y_ext_stor, time_meas_stor, time_meas,
                                                                           range_smoother, radial_velocity_smoother,
                                                                           theta_smoother)

                time_meas_full = np.concatenate((time_meas_stor, time_meas_fin_stor))
                x_est_stor_full = np.concatenate((x_est_stor, x_est_fin_stor))
                y_ext_stor_full = np.concatenate((y_ext_stor, y_ext_fin_stor))

                data_stor = merging_to_date_trajectory(time_meas_full, x_est_stor_full, y_ext_stor_full)

                time_fall = data_stor.values[-1][0]
                x_fall = data_stor.values[-1][1]
                h_fall = data_stor.values[-1][4]

                z_derivation = derivation_calculation(x_fall, velocity_0_estimation, config.alpha_0, K1, K2)
                z_wind = wind_displacement(time_fall, x_fall, velocity_0_estimation, config.alpha_0, config.wind_module,
                                           config.wind_direction, config.az)
                z_fall = z_wind + z_derivation

                x_fall_gk, z_fall_gk = point_of_fall(z_fall, x_fall, config.can_B, config.can_L, config.az)

                track = {"points": json.loads(data_stor.to_json(orient='index')), "endpoint_x": x_fall,
                         "endpoint_y": h_fall, "endpoint_z": z_fall, "endpoint_GK_x": x_fall_gk[0],
                         "endpoint_GK_z": z_fall_gk[0], "Vb": x_fall * np.sin(3 * sko_theta_tz), "Vd": 3 * sko_R_tz,
                         "SKO_R": sko_range, "SKO_V": sko_radial_velocity, " SKO_theta": sko_theta,
                         "valid": True}

                config.data_points = 1
                config.flag_return = 1

            except TypeError:

                config.flag_return = 1
                track = {"valid": False, "error": "calculation error 152 art"}
                config.track = track

        if config.data_points == 1:
            hashes = '#' * int(round(20))
            spaces = ' ' * (20 - len(hashes))
            sys.stdout.write("\rCalculating %: [{0}] {1}% {2} seconds".format(hashes + spaces, int(round(100)),
                                                                              (time.process_time() - start_time)))
            sys.stdout.flush()

            config.track = track

        flag = 1

        if flag:
            return True
        else:
            return False
    else:
        return False
