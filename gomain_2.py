import plotly.graph_objs as go
import json
import numpy as np
import pymap3d as pm
import os
import time

global g
g = 9.8155
t0 = time.time()
from function import kalman_filter_xV, kalman_filter_theta, func_linear_piece_app, func_linear_piece_estimation, \
    func_quad_piece_app, func_quad_piece_estimation, func_derivation, func_meas_smooth_2, BLH2XY_GK, calculate_ellipse, \
    func_trajectory_end_quad, func_trajectory_end_linear, func_filter_data, \
    func_active_reactive, func_active_reactive_trajectory, func_lsm_linear

with open('poits.json', 'r', encoding='utf-8') as fh:  # открываем файл на чтение
    data = json.load(fh)

bullets = []

with open('bullets.json', 'r') as file:
    for line in file:
        bullets.append(json.loads(line))

bullet_type = 3
directory_folder = os.getcwd()
raw_meas = np.loadtxt(directory_folder + '/82_17-10.txt')

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


t_meas = raw_meas[:, 1]
R_meas = raw_meas[:, 3]
Vr_meas = abs(raw_meas[:, 4])
theta_meas = np.deg2rad(raw_meas[:, 5])

print(len(t_meas))

N = 300


if bullet_type == 1 or bullet_type == 2:  # 5.45 bullet or 7.65 bullet

    winlen = 10
    step_sld = 2

    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr,
                                                   n1, n2,
                                                   ksi_theta,
                                                   theta_n1)

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

    t_fin, x_true_fin, h_true_fin = func_trajectory_end_quad(m, g, xhy_0_set_quad, x_est_fin_quad,
                                                        meas_t_ind_quad,
                                                        window_set_quad, t_meas_tr_quad)



    flag_return = 1


if bullet_type == 3:  # 82 mina

    winlen = 30
    step_sld = 10

    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr,
                                                   n1, n2,
                                                   ksi_theta,
                                                   theta_n1)

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

    t_fin, x_true_fin, h_true_fin = func_trajectory_end_linear(m, g, xhy_0_set_linear, x_est_fin_linear,
                                                        meas_t_ind_linear,
                                                        window_set_linear, t_meas_tr_linear)

    flag_return = 1



if bullet_type == 4:  # 122 reactive

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

    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr,
                                                   n1, n2,
                                                   ksi_theta,
                                                   theta_n1)

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

    t_fin, x_true_fin, h_true_fin = func_trajectory_end_quad(m, g, xhy_0_set_quad, x_est_fin_quad,
                                                        meas_t_ind_quad,
                                                        window_set_quad, t_meas_tr_quad)

    flag_return = 1



if bullet_type == 5:  # 122 - art

    winlen = 30
    step_sld = 10

    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr,
                                                   n1, n2,
                                                   ksi_theta,
                                                   theta_n1)

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

    t_fin, x_true_fin, h_true_fin = func_trajectory_end_quad(m, g, xhy_0_set_quad, x_est_fin_quad,
                                                        meas_t_ind_quad,
                                                        window_set_quad, t_meas_tr_quad)

    flag_return = 1



if bullet_type == 6:  # 152 - act-react

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
                                                         ksi_Vr,
                                                         n1, n2,
                                                         ksi_theta,
                                                         theta_n1)

    R_meas_2, Vr_meas_2, theta_meas_2 = func_filter_data(t_meas_2, R_meas_2, Vr_meas_2, theta_meas_2,
                                                         ksi_Vr,
                                                         n1, n2,
                                                         ksi_theta,
                                                         theta_n1)

    parameters_bounds_1 = [k_bounds[0], v0_bounds[0], dR_bounds[0], angle_bounds[0]]
    parameters_bounds_2 = [k_bounds[1], v0_bounds[1], dR_bounds[1], angle_bounds[1]]

    xhy_0_set_quad_1, x_est_fin_quad_1, meas_t_ind_quad_1, window_set_quad_1, t_meas_tr_quad_1, R_meas_tr_quad_1, \
    Vr_meas_tr_quad_1, theta_meas_tr_quad_1 = func_quad_piece_app(loc_X, loc_Y, loc_H,
                                                                  can_Y,
                                                                  m, g, SKO_R,
                                                                  SKO_Vr, SKO_theta, k0,
                                                                  dR, t_meas_1,
                                                                  R_meas_1, Vr_meas_1, theta_meas_1, winlen,
                                                                  step_sld, parameters_bounds_1)

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

    t_fin, x_true_fin, h_true_fin = func_trajectory_end_quad(m, g, xhy_0_set_quad_2, x_est_fin_quad_2,
                                                        meas_t_ind_quad_2,
                                                        window_set_quad_2, t_meas_tr_quad_2)

    t_tr_act_est, x_tr_act_est, h_tr_act_est = func_active_reactive_trajectory(x_tr_er_plot_1, h_tr_er_plot_1,
                                                                               t_meas_plot_1,
                                                                               x_tr_er_plot_2, h_tr_er_plot_2,
                                                                               t_meas_plot_2,
                                                                               N)

    flag_return = 1

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

if bullet_type == 7:  # 152 art

    winlen = 30
    step_sld = 10

    R_meas, Vr_meas, theta_meas = func_filter_data(t_meas, R_meas, Vr_meas, theta_meas, ksi_Vr,
                                                   n1, n2,
                                                   ksi_theta,
                                                   theta_n1)

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

    t_fin, x_true_fin, h_true_fin = func_trajectory_end_quad(m, g, xhy_0_set_quad, x_est_fin_quad,
                                                        meas_t_ind_quad,
                                                        window_set_quad, t_meas_tr_quad)

    flag_return = 1

    fig = go.Figure()
    for i in range(len(x_tr_er_plot)):
        fig.add_trace(go.Scatter(x=x_tr_er_plot[i], y=h_tr_er_plot[i], name=('Т' + str(i))))

    fig.add_trace(go.Scatter(x=x_true_fin, y=h_true_fin, name=('fin')))
    
print(time.time() - t0, "время обработки в секундах")

fig = go.Figure()
for i in range(len(x_tr_er_plot)-1):
    fig.add_trace(go.Scatter(x=x_tr_er_plot[i], y=h_tr_er_plot[i], name=('Т' + str(i))))

fig.add_trace(go.Scatter(x=x_true_fin, y=h_true_fin, name=('fin')))


fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  hovermode='x',
                  title_text='Траектория полёта снаряда',
                  xaxis_title="x, м",
                  yaxis_title="h, м")

fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
fig.show(renderer="browser")

fig = go.Figure()
for i in range(len(t_meas_plot)):
    fig.add_trace(go.Scatter(x=t_meas_plot[i], y=R_est_full_plot[i], name=('Т' + str(i))))

fig.add_trace(go.Scatter(x=t_meas, y=R_meas, mode='markers', name='Среднее'))


fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  hovermode='x',
                  title_text='Зависимость измерений и оценки дальности снаряда от времени',
                  xaxis_title="t, с",
                  yaxis_title="R, м")

fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
fig.show(renderer="browser")

fig = go.Figure()
for i in range(len(t_meas_plot)):
    fig.add_trace(go.Scatter(x=t_meas_plot[i], y=Vr_est_full_plot[i], name=('Т' + str(i))))

fig.add_trace(go.Scatter(x=t_meas, y=Vr_meas, mode='markers', name='Среднее'))


fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  hovermode='x',
                  title_text='Vr',
                  xaxis_title="t, с",
                  yaxis_title="V, м/c")

fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
fig.show(renderer="browser")

fig = go.Figure()
for i in range(len(t_meas_plot)):
    fig.add_trace(go.Scatter(x=t_meas_plot[i], y=np.rad2deg(theta_est_full_plot[i]), name=('Т' + str(i))))

fig.add_trace(go.Scatter(x=t_meas, y=np.rad2deg(theta_meas), mode='markers', name='Среднее'))


fig.update_layout(legend_orientation="h",
                  legend=dict(x=.5, xanchor="center"),
                  hovermode='x',
                  title_text='Theta',
                  xaxis_title="t, с",
                  yaxis_title="theta, rad")

fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
fig.show(renderer="browser")



# fig = go.Figure()
# for i in range(len(x_tr_er_plot_1)):
#     fig.add_trace(go.Scatter(x=x_tr_er_plot_1[i], y=h_tr_er_plot_1[i], name=('Т' + str(i))))
#
# for j in range(len(x_tr_er_plot_2)):
#     fig.add_trace(go.Scatter(x=x_tr_er_plot_2[j], y=h_tr_er_plot_2[j], name=('Т' + str(j + 6))))
#
# fig.add_trace(go.Scatter(x=x_true_fin, y=h_true_fin, name=('model')))
# fig.add_trace(go.Scatter(x=x_tr_act_est, y=h_tr_act_est, name=('gap')))
#
# fig.update_layout(legend_orientation="h",
#                   legend=dict(x=.5, xanchor="center"),
#                   hovermode='x',
#                   title_text='Траектория полёта снаряда',
#                   xaxis_title="x, м",
#                   yaxis_title="h, м")
#
# fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
# fig.show(renderer="browser")