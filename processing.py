import pymap3d as pm

def process_initial_data(mes, config):

    # blh2ENU для локатора
    # blh2ENU для снаряда
    #
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


# def blh2ENU(фждцкпофдлопюфрп)

def process_measurements(mes, config):
    if config.iti_data_flag:
        # парсим измерения, вычисляем траекторию, поднимаем флажок если все посчиталось
        if config.lin_kv == 1:
            # linear ...
            config.track = []
        # config.v0*config.k0
        config.track = []
        flag = 1

        if flag:
            return True
        else:
            return False
    else:
        return False


