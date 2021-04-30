import numpy as np
import json
import numpy as np
import pymap3d as pm
import os

global g
g = 9.8155

from function import kalman_filter_xV, kalman_filter_theta, func_linear_piece_app, func_linear_piece_estimation, \
    func_quad_piece_app, func_quad_piece_estimation, func_derivation, func_meas_smooth_2, BLH2XY_GK, calculate_ellipse, \
    func_trajectory_end, func_filter_data, func_active_reactive, func_active_reactive_trajectory, func_lsm_linear

with open('poits.json', 'r', encoding='utf-8') as fh:  # открываем файл на чтение
    data = json.load(fh)

print(np.deg2rad(0.3))
# Ndlen = len(data["meas"])

bullets = []

with open('bullets.json', 'r') as file:
    for line in file:
        bullets.append(json.loads(line))

bullet_type = 6

# with open('coordinate.json', 'r', encoding='utf-8') as fh: #открываем файл на чтение
#     coordinate = json.load(fh)

# coordinate = {"loc_B": 56.2891, "loc_L": 43.0838,"loc_H": 6,"can_B": 56.2893,"can_L": 43.0838,"can_H": 5}

coordinate = {"loc_B": 56.288611, "loc_L": 43.083825, "loc_H": 0, "can_B": 56.288847, "can_L": 43.083825, "can_H": 2}

loc_B = coordinate["loc_B"]
loc_L = coordinate["loc_L"]
loc_H = coordinate["loc_H"]

can_B = coordinate["can_B"]
can_L = coordinate["can_L"]
can_H = coordinate["can_H"]

loc_Y, loc_X, loc_H = pm.geodetic2enu(loc_B, loc_L, loc_H, can_B, can_L, can_H)
print(loc_X, loc_Y, loc_H)

# loc_H = 0

can_X = 0
can_Y = 0
can_H = 0

bullet = bullets[bullet_type - 1]

flag_lin_kv = bullet["lin_kv"]
v0 = bullet["v0"]
m = bullet["m"]
k0 = bullet["k0"]
dR = bullet["dR"]
SKO_R = bullet["SKO_R"]
SKO_Vr = bullet["SKO_Vr"]
SKO_theta = bullet["SKO_theta"]

l = bullet["l"]
d = bullet["d"]
h = bullet["h"]
mu = bullet["mu"]
i = bullet["i"]
eta = bullet["eta"]

k_bounds = bullet["k_bounds"]
v0_bounds = bullet["v0_bounds"]
dR_bounds = bullet["dR_bounds"]
angle_bounds = bullet["angle_bounds"]

parameters_bounds = [k_bounds, v0_bounds, dR_bounds, angle_bounds]

ksi_Vr = bullet["ksi_Vr"]
n1 = bullet["n1"]
n2 = bullet["n2"]
ksi_theta = bullet["ksi_theta"]
theta_n1 = bullet["theta_n1"]

# заполняем параметры пули в конфиг из списка

# поднимаем флаг
ini_data_flag = 1

#
# t_meas = np.zeros(Ndlen)
# R_meas = np.zeros(Ndlen)
# Vr_meas = np.zeros(Ndlen)
# theta_meas = np.zeros(Ndlen)
#
# for i in range(Ndlen):
#     t_meas[i] = data["meas"][i]["execTime_sec"]
#     R_meas[i] = data["meas"][i]["R"]
#     Vr_meas[i] = data["meas"][i]["Vr"]
#     theta_meas[i] = data["meas"][i]["Epsilon"]

directory_folder = os.getcwd()
raw_meas = np.loadtxt(directory_folder + '/152_14-46.txt')
# raw_meas = np.loadtxt(directory_folder + '/545_15-48.txt')


# alpha = np.deg2rad(45)

t_meas = raw_meas[:, 1]
R_meas = raw_meas[:, 3]
Vr_meas = abs(raw_meas[:, 4])
theta_meas = np.deg2rad(raw_meas[:, 5])

N = 300

if bullet_type == 1 or bullet_type == 2:
    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr, n1, n2, ksi_theta,
                                                   theta_n1)
    winlen = 10;
    step_sld = 2;

    xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, R_meas_tr_quad, \
    Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(loc_X, loc_Y, loc_H,
                                                              can_Y,
                                                              m, g, SKO_R,
                                                              SKO_Vr, SKO_theta, k0,
                                                              dR, t_meas,
                                                              R_meas, Vr_meas, theta_meas, winlen,
                                                              step_sld, parameters_bounds)

    t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
    theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
        xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, N,
        m, g, loc_X, loc_Y, loc_H)

if bullet_type == 3:  # если мина - линейное, другие снаряды - квадратичная

    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr, n1, n2, ksi_theta,
                                                   theta_n1)

    winlen = 30;
    step_sld = 10;

    xhy_0_set_linear, x_est_fin_linear, meas_t_ind_linear, window_set_linear, t_meas_tr_linear, R_meas_tr_linear, \
    Vr_meas_tr_linear, theta_meas_tr_linear = func_linear_piece_app(loc_X, loc_Y, loc_H,
                                                                    can_Y,
                                                                    m, g, SKO_R,
                                                                    SKO_Vr, SKO_theta, k0,
                                                                    dR, t_meas,
                                                                    R_meas, Vr_meas, theta_meas, winlen,
                                                                    step_sld, parameters_bounds)

    t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
    theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_linear_piece_estimation(
        xhy_0_set_linear, x_est_fin_linear, meas_t_ind_linear, window_set_linear, t_meas_tr_linear, N,
        m, g, loc_X, loc_Y, loc_H)

if bullet_type == 6:

    t_ind_end_1part, t_ind_start_2part = func_active_reactive(t_meas, R_meas, Vr_meas)

    t_meas_1 = t_meas[:t_ind_end_1part]
    R_meas_1 = R_meas[:t_ind_end_1part]
    Vr_meas_1 = Vr_meas[:t_ind_end_1part]
    theta_meas_1 = theta_meas[:t_ind_end_1part]

    t_meas_2 = t_meas[t_ind_start_2part:]
    R_meas_2 = R_meas[t_ind_start_2part:]
    Vr_meas_2 = Vr_meas[t_ind_start_2part:]
    theta_meas_2 = theta_meas[t_ind_start_2part:]

    R_meas_1, Vr_meas_1, theta_meas_1 = func_filter_data(t_meas_1, R_meas_1, Vr_meas_1, theta_meas_1, ksi_Vr, n1, n2,
                                                         ksi_theta,
                                                         theta_n1)

    R_meas_2, Vr_meas_2, theta_meas_2 = func_filter_data(t_meas_2, R_meas_2, Vr_meas_2, theta_meas_2, ksi_Vr, n1, n2,
                                                         ksi_theta,
                                                         theta_n1)

    winlen = 30;
    step_sld = 10;

    parameters_bounds_1 = [k_bounds[0], v0_bounds[0], dR_bounds[0], angle_bounds[0]]
    parameters_bounds_2 = [k_bounds[1], v0_bounds[1], dR_bounds[1], angle_bounds[1]]

    print("1")
    print(len(t_meas_1))

    xhy_0_set_quad_1, x_est_fin_quad_1, meas_t_ind_quad_1, window_set_quad_1, t_meas_tr_quad_1, R_meas_tr_quad_1, \
    Vr_meas_tr_quad_1, theta_meas_tr_quad_1 = func_quad_piece_app(loc_X, loc_Y, loc_H,
                                                                  can_Y,
                                                                  m, g, SKO_R,
                                                                  SKO_Vr, SKO_theta, k0,
                                                                  dR, t_meas_1,
                                                                  R_meas_1, Vr_meas_1, theta_meas_1, winlen,
                                                                  step_sld, parameters_bounds_1)
    print("2")
    print(len(t_meas_2))
    xhy_0_set_quad_2, x_est_fin_quad_2, meas_t_ind_quad_2, window_set_quad_2, t_meas_tr_quad_2, R_meas_tr_quad_2, \
    Vr_meas_tr_quad_2, theta_meas_tr_quad_2 = func_quad_piece_app(loc_X, loc_Y, loc_H,
                                                                  can_Y,
                                                                  m, g, SKO_R,
                                                                  SKO_Vr, SKO_theta, k0,
                                                                  dR, t_meas_2,
                                                                  R_meas_2, Vr_meas_2, theta_meas_2, winlen,
                                                                  step_sld, parameters_bounds_2)

    t_meas_plot_1, x_tr_er_plot_1, h_tr_er_plot_1, R_est_full_plot_1, Vr_est_full_plot_1, \
    theta_est_full_plot_1, Vx_true_er_plot_1, Vh_true_er_plot_1, V_abs_full_plot_1 = func_quad_piece_estimation(
        xhy_0_set_quad_1, x_est_fin_quad_1, meas_t_ind_quad_1, window_set_quad_1, t_meas_tr_quad_1, N,
        m, g, loc_X, loc_Y, loc_H)

    t_meas_plot_2, x_tr_er_plot_2, h_tr_er_plot_2, R_est_full_plot_2, Vr_est_full_plot_2, \
    theta_est_full_plot_2, Vx_true_er_plot_2, Vh_true_er_plot_2, V_abs_full_plot_2 = func_quad_piece_estimation(
        xhy_0_set_quad_2, x_est_fin_quad_2, meas_t_ind_quad_2, window_set_quad_2, t_meas_tr_quad_2, N,
        m, g, loc_X, loc_Y, loc_H)

    t_fin, x_true_fin, h_true_fin= func_trajectory_end(m, g, xhy_0_set_quad_2, x_est_fin_quad_2,
                                                                meas_t_ind_quad_2,
                                                                window_set_quad_2, t_meas_tr_quad_2)

t_tr_act_est, x_tr_act_est, h_tr_act_est = func_active_reactive_trajectory(x_tr_er_plot_1, h_tr_er_plot_1, t_meas_plot_1,
                                                             x_tr_er_plot_2, h_tr_er_plot_2, t_meas_plot_2, N)

print(x_true_fin, h_true_fin, t_fin)
# почему время не с самой первой точки?
print(t_fin[-1] + t_meas_plot_2[-1][0], "nj")
print(t_fin[-1] + t_meas_plot_1[0][0], "ff")
import plotly.graph_objs as go

fig = go.Figure()
for i in range(len(x_tr_er_plot_1)):
    fig.add_trace(go.Scatter(x=x_tr_er_plot_1[i], y=h_tr_er_plot_1[i], name=('Т' + str(i))))

for j in range(len(x_tr_er_plot_2)):
    fig.add_trace(go.Scatter(x=x_tr_er_plot_2[j], y=h_tr_er_plot_2[j], name=('Т' + str(j + 6))))

fig.add_trace(go.Scatter(x=x_true_fin, y=h_true_fin, name=('model')))
fig.add_trace(go.Scatter(x=x_tr_act_est, y=h_tr_act_est, name=('gap')))

fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  hovermode='x',
                  title_text='Траектория полёта снаряда',
                  xaxis_title="x, м",
                  yaxis_title="h, м")

fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
fig.show(renderer="browser")
# if bullet_type == 3:
#
#     xhy_0_set_quad, x_est_fin_quad_piece, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, R_meas_tr_quad, \
#     Vr_meas_tr_quad, theta_meas_tr_quad = func_quad_piece_app(loc_X, loc_Y, loc_H,
#                                                               can_X, can_Y, can_H,
#                                                               m, g, SKO_R,
#                                                               SKO_Vr, SKO_theta, k0,
#                                                               dR, alpha, t_meas,
#                                                               R_meas, Vr_meas, theta_meas, winlen,
#                                                               step_sld, parameters_bounds)
#
#
#
#
#     t_meas_plot, x_tr_er_plot, h_tr_er_plot, R_est_full_plot, Vr_est_full_plot, \
#     theta_est_full_plot, Vx_true_er_plot, Vh_true_er_plot, V_abs_full_plot = func_quad_piece_estimation(
#         xhy_0_set_quad, x_est_fin_quad_piece, meas_t_ind_quad, window_set_quad, t_meas_tr_quad, N,
#          m, g, loc_X, loc_Y, loc_H)


# t_fin, x_true_fin, h_true_fin, last_k = func_trajectory_end(m, g, xhy_0_set_quad, x_est_fin_quad, meas_t_ind_quad, window_set_quad, t_meas_tr_quad)
#
# track_points = {}
# points = []
# for i in range(len(t_meas_plot)):
#     for j in range(len(t_meas_plot[i]) - 1):
#         points.append({"t": t_meas_plot[i][j], "x_tr" : x_tr_er_plot[i][j], "h_tr": h_tr_er_plot[i][j], "R": R_est_full_plot[i][j],
#                        "Vr": Vr_est_full_plot[i][j], "theta": theta_est_full_plot[i][j]})
#
# track_points["points"] = points
#
# print(track_points)
#
#
# nlen = len(track_points["points"])
# t_meas = []
# for i in range(nlen):
#     t_meas.append(track_points["points"][i]["t"])
#
# print(t_meas)
#
# import plotly.graph_objs as go
#
# fig = go.Figure()
# for i in range(len(x_tr_er_plot)):
#     fig.add_trace(go.Scatter(x=x_tr_er_plot[i], y=h_tr_er_plot[i], name=('Т' + str(i))))
# fig.add_trace(go.Scatter(x=x_true_fin[:last_k], y=h_true_fin[:last_k], name=('model')))
# fig.update_layout(legend_orientation="h",
#                   legend=dict(x=.5, xanchor="center"),
#                   hovermode='x',
#                   title_text='Траектория полёта снаряда',
#                   xaxis_title="x, м",
#                   yaxis_title="h, м")
#
# fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
# fig.show(renderer="browser")
