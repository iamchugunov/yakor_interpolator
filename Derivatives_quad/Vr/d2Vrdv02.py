import numpy as np
def  d2Vrdv02(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = (3 * ((2 * t * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                (k * t * v0 * np.cos(alpha)) / m + 1) + (
                            2 * m * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (1 / 2) * (
                                h0 - hL + (m * np.log(
                            np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                        k / (g * m)) ** (1 / 2))) / k)) / (k * (
                np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                    k / (g * m)) ** (1 / 2)))) ** 2 * (((v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
        t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / ((g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
        alpha)) + (v0 * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                                                 (k * t * v0 * np.cos(alpha)) / m + 1))) / (4 * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (m * np.log(
            np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                        1 / 2))) / k) ** 2) ** (5 / 2)) - ((((v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
        t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / ((g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
        alpha)) + (v0 * np.cos(alpha) * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                                                      (k * t * v0 * np.cos(alpha)) / m + 1)) * (
                                                                     (2 * t ** 2 * np.cos(alpha) ** 2) / (
                                                                         (k * t * v0 * np.cos(alpha)) / m + 1) ** 2 - (
                                                                                 2 * np.sin(
                                                                             t * ((g * k) / m) ** (1 / 2)) ** 2 * np.sin(
                                                                             alpha) ** 2 * (h0 - hL + (m * np.log(np.cos(
                                                                             t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(
                                                                             t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                                                             alpha) * (k / (g * m)) ** (
                                                                                                                          1 / 2))) / k)) / (
                                                                                 g * (np.cos(
                                                                             t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(
                                                                             t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                                                             alpha) * (k / (g * m)) ** (1 / 2)) ** 2) - (
                                                                                 2 * k * t ** 2 * np.cos(alpha) ** 2 * (
                                                                                     x0 - xL + (m * np.log((
                                                                                                                    k * t * v0 * np.cos(
                                                                                                                alpha)) / m + 1)) / k)) / (
                                                                                 m * ((k * t * v0 * np.cos(
                                                                             alpha)) / m + 1) ** 2) + (2 * m * np.sin(
                                                                 t * ((g * k) / m) ** (1 / 2)) ** 2 * np.sin(alpha) ** 2) / (
                                                                                 g * k * (np.cos(
                                                                             t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(
                                                                             t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                                                             alpha) * (k / (g * m)) ** (
                                                                                                      1 / 2)) ** 2))) / (
                      2 * ((x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (
                          m * np.log(
                      np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                  k / (g * m)) ** (1 / 2))) / k) ** 2) ** (3 / 2)) - (
                      (3 * k * t ** 2 * v0 * np.cos(alpha) ** 3) / (m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 3) - (
                          2 * t * np.cos(alpha) ** 2) / ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2 + (
                                  np.sin(t * ((g * k) / m) ** (1 / 2)) ** 2 * np.sin(alpha) ** 2 * (
                                      v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                                  t * ((g * k) / m) ** (1 / 2)))) / (g * (
                          (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                                         1 / 2)) ** 2) - (
                                  2 * k ** 2 * np.tan(t * ((g * k) / m) ** (1 / 2)) ** 2 * np.sin(alpha) ** 2 * (
                                      v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                                  t * ((g * k) / m) ** (1 / 2))) * (h0 - hL + (m * np.log(
                              np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                          k / (g * m)) ** (1 / 2))) / k)) / (
                                  (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha)) ** 3 + (2 * k * t * np.cos(alpha) ** 2 * (
                          x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                  m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2) + (
                                  2 * k * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) ** 2 * (h0 - hL + (m * np.log(
                              np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                          k / (g * m)) ** (1 / 2))) / k) * (g * k * m) ** (1 / 2)) / (
                                  (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha)) ** 2 - (2 * k ** 2 * t ** 2 * v0 * np.cos(alpha) ** 3 * (
                          x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                                  m ** 2 * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 3) + (
                                  2 * m * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha) ** 2 * (v0 * np.sin(alpha) * (g * k * m) ** (1 / 2) - g * m * np.tan(
                              t * ((g * k) / m) ** (1 / 2))) * (k / (g * m)) ** (1 / 2)) / (((g * k * m) ** (
                          1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) ** 2 * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                              1 / 2))) - (
                                  2 * m * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) ** 2 * (k / (g * m)) ** (
                                      1 / 2) * (g * k * m) ** (1 / 2)) / (k * (
                          (g * k * m) ** (1 / 2) + k * v0 * np.tan(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) * (np.cos(
                  t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                                                                                                         1 / 2)))) / (
                      (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2 + (h0 - hL + (m * np.log(
                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                              k / (g * m)) ** (1 / 2))) / k) ** 2) ** (1 / 2) - (((2 * t * np.cos(alpha) * (
                x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / ((k * t * v0 * np.cos(alpha)) / m + 1) + (
                                                                                           2 * m * np.sin(
                                                                                       t * ((g * k) / m) ** (
                                                                                                   1 / 2)) * np.sin(
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
                                                                                          (np.cos(alpha) * (x0 - xL + (
                                                                                                      m * np.log((
                                                                                                                          k * t * v0 * np.cos(
                                                                                                                      alpha)) / m + 1)) / k)) / (
                                                                                                      (k * t * v0 * np.cos(
                                                                                                          alpha)) / m + 1) + (
                                                                                                      np.sin(alpha) * (
                                                                                                          h0 - hL + (
                                                                                                              m * np.log(
                                                                                                          np.cos(t * ((
                                                                                                                               g * k) / m) ** (
                                                                                                                          1 / 2)) + v0 * np.sin(
                                                                                                              t * ((
                                                                                                                               g * k) / m) ** (
                                                                                                                          1 / 2)) * np.sin(
                                                                                                              alpha) * (
                                                                                                                      k / (
                                                                                                                          g * m)) ** (
                                                                                                                      1 / 2))) / k) * (
                                                                                                                  g * k * m) ** (
                                                                                                                  1 / 2)) / (
                                                                                                      (g * k * m) ** (
                                                                                                          1 / 2) + k * v0 * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
                                                                                                  alpha)) + (
                                                                                                      t * v0 * np.cos(
                                                                                                  alpha) ** 2) / ((
                                                                                                                             k * t * v0 * np.cos(
                                                                                                                         alpha)) / m + 1) ** 2 - (
                                                                                                      k * np.tan(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
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
                                                                                                      k * t * v0 * np.cos(
                                                                                                  alpha) ** 2 * (
                                                                                                                  x0 - xL + (
                                                                                                                      m * np.log(
                                                                                                                  (
                                                                                                                              k * t * v0 * np.cos(
                                                                                                                          alpha)) / m + 1)) / k)) / (
                                                                                                      m * ((
                                                                                                                       k * t * v0 * np.cos(
                                                                                                                   alpha)) / m + 1) ** 2) + (
                                                                                                      m * np.sin(
                                                                                                  t * ((g * k) / m) ** (
                                                                                                              1 / 2)) * np.sin(
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
                              k / (g * m)) ** (1 / 2))) / k) ** 2) ** (3 / 2)

    return out
