class Config():

    def __init__(self):

        self.IP = '192.168.1.1'
        self.PORT = 5050
        self.ADDR = (self.IP, self.PORT)

        self.ini_data_flag = 0

        # inition data

        # locator coords
        # BLH
        self.loc_B = 0.
        self.loc_L = 0.
        self.loc_H = 0.
        # ENU
        self.loc_X = 0.
        self.loc_Y = 0.
        self.loc_Z = 0.

        # cannon coords
        # BLH
        self.can_B = 0.
        self.can_L = 0.
        self.can_H = 0.
        # ENU
        self.can_X = 0.
        self.can_Y = 0.
        self.can_Z = 0.

        # trowing angle
        self.alpha = 0.
        # azimuth
        self.az = 0.
        # heigth ??
        self.hei = 0.
        # wind params
        self.wind_module = 0.
        self.wind_direction = 0.

        # bullet type
        # 1 - 5.45
        # 2 - 7.62
        # 3 - 82
        # 4 - 122 reactive
        # 5 - 122 art
        # 6 - 152 reactive
        # 7 - 152 art
        self.bullet_type = 0

        # bullet params

        # model type: 1 - linear, 2 - quad
        self.lin_kv = 0
        # ini velo
        self.v0 = 0.
        # mass
        self.m = 0.
        # resistance koef
        self.k = 0.
        # dR
        self.dR = 0.
        # std
        self.SKO_R = 0.
        self.SKO_Vr = 0.
        self.SKO_theta = 0.
        # derivation parameters
        self.l = 0.
        self.d = 0.
        self.h = 0.
        self.mu = 0.
        self.i = 0.
        self.eta = 0.
        # bounds
        self.k_bounds = [0, 0]
        self.V0_bounds = [0, 0]
        self.dR_bounds = [0, 0]
        self.angle_bounds = [0, 0]
        # filter params
        self.ksi_Vr = 0.
        self.n1 = 0.
        self.n2 = 0.
        self.ksi_theta = 0.
        self.theta_n1 = 0.

        # читаем файл БД снарядов и запоминаем их в список всех снарядов
        self.bullets = []


        # track ??
        self.track = []






