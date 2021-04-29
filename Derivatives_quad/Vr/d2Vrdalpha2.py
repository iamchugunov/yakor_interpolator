import numpy as np
def  d2Vrdalpha2(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = ((((v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / (
                        (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) + (
                        v0 * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                        (k * t * v0 * np.cos(alpha)) / m + 1)) * (
                       (2 * t * v0 * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                           (k * t * v0 * np.cos(alpha)) / m + 1) - (2 * t ** 2 * v0 ** 2 * np.sin(alpha) ** 2) / (
                                   (k * t * v0 * np.cos(alpha)) / m + 1) ** 2 + (
                                   2 * v0 ** 2 * np.sin(t * ((g * k) / m) ** (1 / 2)) ** 2 * np.cos(alpha) ** 2 * (h0 - hL + (
                                       m * np.log(
                                   np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                       alpha) * (k / (g * m)) ** (1 / 2))) / k)) / (g * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2)) ** 2) - (
                                   2 * m * v0 ** 2 * np.sin(t * ((g * k) / m) ** (1 / 2)) ** 2 * np.cos(alpha) ** 2) / (g * k * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2)) ** 2) + (2 * k * t ** 2 * v0 ** 2 * np.sin(alpha) ** 2 * (
                           x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                   m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2) + (
                                   2 * m * v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                       1 / 2) * (h0 - hL + (m * np.log(
                               np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                           k / (g * m)) ** (1 / 2))) / k)) / (k * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2))))) / (2 * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (m * np.log(
            np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                        1 / 2))) / k) ** 2) ** (3 / 2)) - (
                      (t * v0 ** 2 * np.cos(alpha) ** 2) / ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2 - (
                          2 * t * v0 ** 2 * np.sin(alpha) ** 2) / ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2 + (
                                  v0 * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                  (k * t * v0 * np.cos(alpha)) / m + 1) + (v0 * np.sin(alpha) * (h0 - hL + (m * np.log(
                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                              k / (g * m)) ** (1 / 2))) / k) * (g * k * m) ** (1 / 2)) / (
                                  (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) - (
                                  2 * k ** 2 * v0 ** 2 * np.tan(t * ((g * k) / m) ** (1 / 2)) ** 2 * np.cos(alpha) ** 2 * (
                                      v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                                  t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
                              np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                          k / (g * m)) ** (1 / 2))) / k)) / (
                                  (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha)) ** 3 + (3 * k * t ** 2 * v0 ** 3 * np.cos(alpha) * np.sin(alpha) ** 2) / (
                                  m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 3) - (k * t * v0 ** 2 * np.cos(alpha) ** 2 * (
                          x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                  m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2) + (
                                  2 * k * v0 ** 2 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.cos(alpha) ** 2 * (h0 - hL + (
                                      m * np.log(
                                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                      alpha) * (k / (g * m)) ** (1 / 2))) / k) * (g * k * m) ** (1 / 2)) / (
                                  (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha)) ** 2 + (2 * k * t * v0 ** 2 * np.sin(alpha) ** 2 * (
                          x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                  m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2) - (
                                  k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                      v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                                  t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
                              np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                          k / (g * m)) ** (1 / 2))) / k)) / (
                                  (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha)) ** 2 + (v0 ** 2 * np.sin(t * ((g * k) / m) ** (1 / 2)) ** 2 * np.cos(alpha) ** 2 * (
                          v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(t * ((g * k) / m) ** (1 / 2)))) / (g * (
                          (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                                         1 / 2)) ** 2) - (
                                  2 * k ** 2 * t ** 2 * v0 ** 3 * np.cos(alpha) * np.sin(alpha) ** 2 * (
                                      x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                  m ** 2 * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 3) + (
                                  2 * m * v0 ** 2 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.tan(
                              t * ((g * k) / m) ** (1 / 2)) * np.cos(alpha) ** 2 * (
                                              v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                                          t * ((g * k) / m) ** (1 / 2))) * (k / (g * m)) ** (1 / 2)) / (((g * k * m) ** (
                          1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) ** 2 * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                              1 / 2))) - (
                                  2 * m * v0 ** 2 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.cos(alpha) ** 2 * (k / (g * m)) ** (
                                      1 / 2) * (g * k * m) ** (1 / 2)) / (k * (
                          (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                                         1 / 2))) + (
                                  m * v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                      v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                                  t * ((g * k) / m) ** (1 / 2))) * (k / (g * m)) ** (1 / 2)) / (k * (
                          (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                                         1 / 2)))) / (
                      (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (m * np.log(
                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                              k / (g * m)) ** (1 / 2))) / k) ** 2) ** (1 / 2) - (((2 * t * v0 * np.sin(alpha) * (
                x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / ((k * t * v0 * np.cos(alpha)) / m + 1) - (
                                                                                           2 * m * v0 * np.sin(
                                                                                       t * ((g * k) / m) ** (
                                                                                                   1 / 2)) * np.cos(
                                                                                       alpha) * (k / (g * m)) ** (
                                                                                                       1 / 2) * (
                                                                                                       h0 - hL + (
                                                                                                           m * np.log(np.cos(
                                                                                                       t * ((
                                                                                                                        g * k) / m) ** (
                                                                                                                   1 / 2)) + v0 * np.sin(
                                                                                                       t * ((
                                                                                                                        g * k) / m) ** (
                                                                                                                   1 / 2)) * np.sin(
                                                                                                       alpha) * (k / (
                                                                                                               g * m)) ** (
                                                                                                                               1 / 2))) / k)) / (
                                                                                           k * (np.cos(
                                                                                       t * ((g * k) / m) ** (
                                                                                                   1 / 2)) + v0 * np.sin(
                                                                                       t * ((g * k) / m) ** (
                                                                                                   1 / 2)) * np.sin(
                                                                                       alpha) * (k / (g * m)) ** (
                                                                                                            1 / 2)))) * (
                                                                                          (v0 * np.sin(alpha) * (
                                                                                                      x0 - xL + (
                                                                                                          m * np.log((
                                                                                                                              k * t * v0 * np.cos(
                                                                                                                          alpha)) / m + 1)) / k)) / (
                                                                                                      (k * t * v0 * np.cos(
                                                                                                          alpha)) / m + 1) + (
                                                                                                      t * v0 ** 2 * np.cos(
                                                                                                  alpha) * np.sin(
                                                                                                  alpha)) / ((
                                                                                                                         k * t * v0 * np.cos(
                                                                                                                     alpha)) / m + 1) ** 2 - (
                                                                                                      v0 * np.cos(
                                                                                                  alpha) * (h0 - hL + (
                                                                                                          m * np.log(np.cos(
                                                                                                      t * ((
                                                                                                                       g * k) / m) ** (
                                                                                                                  1 / 2)) + v0 * np.sin(
                                                                                                      t * ((
                                                                                                                       g * k) / m) ** (
                                                                                                                  1 / 2)) * np.sin(
                                                                                                      alpha) * (k / (
                                                                                                              g * m)) ** (
                                                                                                                              1 / 2))) / k) * (
                                                                                                                  g * k * m) ** (
                                                                                                                  1 / 2)) / (
                                                                                                      (g * k * m) ** (
                                                                                                          1 / 2) + k * v0 * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
                                                                                                  alpha)) + (
                                                                                                      k * v0 * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.cos(
                                                                                                  alpha) * (v0 * np.sin(
                                                                                                  alpha) * (
                                                                                                                        g * k * m) ** (
                                                                                                                        1 / 2) - g * m * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2))) * (
                                                                                                                  h0 - hL + (
                                                                                                                      m * np.log(
                                                                                                                  np.cos(
                                                                                                                      t * (
                                                                                                                                  (
                                                                                                                                              g * k) / m) ** (
                                                                                                                                  1 / 2)) + v0 * np.sin(
                                                                                                                      t * (
                                                                                                                                  (
                                                                                                                                              g * k) / m) ** (
                                                                                                                                  1 / 2)) * np.sin(
                                                                                                                      alpha) * (
                                                                                                                              k / (
                                                                                                                                  g * m)) ** (
                                                                                                                              1 / 2))) / k)) / (
                                                                                                      (g * k * m) ** (
                                                                                                          1 / 2) + k * v0 * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
                                                                                                  alpha)) ** 2 - (
                                                                                                      k * t * v0 ** 2 * np.cos(
                                                                                                  alpha) * np.sin(
                                                                                                  alpha) * (x0 - xL + (
                                                                                                          m * np.log((
                                                                                                                              k * t * v0 * np.cos(
                                                                                                                          alpha)) / m + 1)) / k)) / (
                                                                                                      m * ((
                                                                                                                       k * t * v0 * np.cos(
                                                                                                                   alpha)) / m + 1) ** 2) - (
                                                                                                      m * v0 * np.sin(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.cos(
                                                                                                  alpha) * (v0 * np.sin(
                                                                                                  alpha) * (
                                                                                                                        g * k * m) ** (
                                                                                                                        1 / 2) - g * m * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2))) * (
                                                                                                                  k / (
                                                                                                                      g * m)) ** (
                                                                                                                  1 / 2)) / (
                                                                                                      k * ((
                                                                                                                       g * k * m) ** (
                                                                                                                       1 / 2) + k * v0 * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
                                                                                                  alpha)) * (np.cos(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) + v0 * np.sin(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
                                                                                                  alpha) * (k / (
                                                                                                          g * m)) ** (
                                                                                                                         1 / 2))))) / (
                      (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (m * np.log(
                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                              k / (g * m)) ** (1 / 2))) / k) ** 2) ** (3 / 2) + (3 * (((v0 * np.sin(alpha) * (g * k * m) ** (
                1 / 2) - g * m * np.tan(t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / ((g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
        alpha)) + (v0 * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / ((k * t * v0 * np.cos(
        alpha)) / m + 1)) * ((2 * t * v0 * np.sin(alpha) * (
                x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / ((k * t * v0 * np.cos(alpha)) / m + 1) - (
                                         2 * m * v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.cos(alpha) * (k / (g * m)) ** (
                                             1 / 2) * (h0 - hL + (m * np.log(
                                     np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                         alpha) * (k / (g * m)) ** (1 / 2))) / k)) / (k * (
                np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                    k / (g * m)) ** (1 / 2)))) ** 2) / (4 * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (m * np.log(
            np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                        1 / 2))) / k) ** 2) ** (5 / 2))

    return out
