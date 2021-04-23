

def process_initial_data(mes, config):
    # blh2ENU для локатора
    # blh2ENU для снаряда
    #
    # config.loc_X =
    # config.loc_Y =
    # config.loc_Z =
    #
    # config.can_X =
    # config.can_Y =
    # config.can_Z =

    config.bullet_type = mes["bullet type"]
    bullet = config.bullets[config.bullet_type]
    config.v0 = bullet["v0"]
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


