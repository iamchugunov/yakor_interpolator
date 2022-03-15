import json

class Config():

    def __init__(self):

        with open('config.json', 'r') as file:
            connect_params = (json.loads(file.read()))
        self.IP = connect_params["IP"]
        self.PORT = connect_params["PORT"]
        self.ADDR = (self.IP, self.PORT)

        # ini data flag
        self.ini_data_flag = 0
        # measurement flag
        self.ini_meas_flag = 0
        # data points
        self.data_points = 0

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
        self.alpha_0 = 0.
        # azimuth
        self.az = 0.
        # heigth
        self.hei = 0.
        # wind params
        self.wind_module = 0.
        self.wind_direction = 0.
        # temp
        self.temperature = 0.
        self.atm_pressure = 0.

        # bullet type
        # 1 - 5.45; 2 - 7.62; 3 - 82; 4 - 122 reactive; 5 - 122 art; 6 - 152 act - reactive; 7 - 152 art
        self.bullet_type = 0

        # bullet params

        #velocity
        self.v0 = 0.

        # mass
        self.m = 0.

        # std
        self.SKO_R = 0.
        self.SKO_Vr = 0.
        self.SKO_theta = 0.

        # derivation parameters
        self.r = 0.
        self.l = 0.
        self.d = 0.
        self.eta = 0.

        self.bullets = []

        with open('bullets.json', 'r') as file:
            for line in file:
                self.bullets.append(json.loads(line))

        # flag_return
        self.flag_return = 0

        # track - result approx trajectory
        self.track = []






