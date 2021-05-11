import numpy as np
def  d2thetadk2(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = - (((m * ((v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (2 * g * m * (k / (g * m)) ** (1 / 2)) - (
                g * t * np.sin(t * ((g * k) / m) ** (1 / 2))) / (2 * m * ((g * k) / m) ** (1 / 2)) + (
                                g * t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                    1 / 2)) / (2 * m * ((g * k) / m) ** (1 / 2))) ** 2) / (k * (
                np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                    k / (g * m)) ** (1 / 2)) ** 2) - (2 * m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k ** 3 + (m * ((g * t ** 2 * np.cos(t * ((g * k) / m) ** (1 / 2))) / (4 * k * m) - (
                g ** 2 * t * np.sin(t * ((g * k) / m) ** (1 / 2))) / (4 * m ** 2 * ((g * k) / m) ** (3 / 2)) + (
                                                         v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (
                                                         4 * g ** 2 * m ** 2 * (k / (g * m)) ** (3 / 2)) - (
                                                         t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (
                                                         2 * m ** 2 * (k / (g * m)) ** (1 / 2) * ((g * k) / m) ** (
                                                             1 / 2)) + (
                                                         g ** 2 * t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                                     alpha) * (k / (g * m)) ** (1 / 2)) / (
                                                         4 * m ** 2 * ((g * k) / m) ** (3 / 2)) + (
                                                         g * t ** 2 * v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                                                     alpha) * (k / (g * m)) ** (1 / 2)) / (4 * k * m))) / (k * (
                np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                    k / (g * m)) ** (1 / 2))) + (2 * m * (
                (v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (2 * g * m * (k / (g * m)) ** (1 / 2)) - (
                    g * t * np.sin(t * ((g * k) / m) ** (1 / 2))) / (2 * m * ((g * k) / m) ** (1 / 2)) + (
                            g * t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (1 / 2)) / (
                            2 * m * ((g * k) / m) ** (1 / 2)))) / (k ** 2 * (
                np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                    k / (g * m)) ** (1 / 2)))) / (
                         (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (1 / 2) + (((
                                                                                                                              m * np.log(
                                                                                                                          (
                                                                                                                                      k * t * v0 * np.cos(
                                                                                                                                  alpha)) / m + 1)) / k ** 2 - (
                                                                                                                              t * v0 * np.cos(
                                                                                                                          alpha)) / (
                                                                                                                              k * (
                                                                                                                                  (
                                                                                                                                              k * t * v0 * np.cos(
                                                                                                                                          alpha)) / m + 1))) ** 2 * (
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
                         (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2) - (3 * (
                (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k ** 2 - (t * v0 * np.cos(alpha)) / (
                    k * ((k * t * v0 * np.cos(alpha)) / m + 1))) ** 2 * (x0 - xL + (
                m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / ((x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (
                         5 / 2) - ((x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k) * ((2 * t * v0 * np.cos(alpha)) / (k ** 2 * ((k * t * v0 * np.cos(alpha)) / m + 1)) - (
                2 * m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k ** 3 + (t ** 2 * v0 ** 2 * np.cos(alpha) ** 2) / (
                                                 k * m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2))) / (
                         (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2) + (2 * ((
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
                                                                                                                                          1 / 2))) / k ** 2 - (
                                                                                                                                  m * (
                                                                                                                                      (
                                                                                                                                                  v0 * np.sin(
                                                                                                                                              t * (
                                                                                                                                                          (
                                                                                                                                                                      g * k) / m) ** (
                                                                                                                                                          1 / 2)) * np.sin(
                                                                                                                                              alpha)) / (
                                                                                                                                                  2 * g * m * (
                                                                                                                                                      k / (
                                                                                                                                                          g * m)) ** (
                                                                                                                                                              1 / 2)) - (
                                                                                                                                                  g * t * np.sin(
                                                                                                                                              t * (
                                                                                                                                                          (
                                                                                                                                                                      g * k) / m) ** (
                                                                                                                                                          1 / 2))) / (
                                                                                                                                                  2 * m * (
                                                                                                                                                      (
                                                                                                                                                                  g * k) / m) ** (
                                                                                                                                                              1 / 2)) + (
                                                                                                                                                  g * t * v0 * np.cos(
                                                                                                                                              t * (
                                                                                                                                                          (
                                                                                                                                                                      g * k) / m) ** (
                                                                                                                                                          1 / 2)) * np.sin(
                                                                                                                                              alpha) * (
                                                                                                                                                              k / (
                                                                                                                                                                  g * m)) ** (
                                                                                                                                                              1 / 2)) / (
                                                                                                                                                  2 * m * (
                                                                                                                                                      (
                                                                                                                                                                  g * k) / m) ** (
                                                                                                                                                              1 / 2)))) / (
                                                                                                                                  k * (
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
                                                                                                                                                  1 / 2)))) * (
                                                                                                                             (
                                                                                                                                         m * np.log(
                                                                                                                                     (
                                                                                                                                                 k * t * v0 * np.cos(
                                                                                                                                             alpha)) / m + 1)) / k ** 2 - (
                                                                                                                                         t * v0 * np.cos(
                                                                                                                                     alpha)) / (
                                                                                                                                         k * (
                                                                                                                                             (
                                                                                                                                                         k * t * v0 * np.cos(
                                                                                                                                                     alpha)) / m + 1))) * (
                                                                                                                             x0 - xL + (
                                                                                                                                 m * np.log(
                                                                                                                             (
                                                                                                                                         k * t * v0 * np.cos(
                                                                                                                                     alpha)) / m + 1)) / k)) / (
                         (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) / ((
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
                                                                                                                                          1 / 2))) / k) ** 2 / (
                                                                                                                              (
                                                                                                                                          x0 - xL + (
                                                                                                                                              m * np.log(
                                                                                                                                          (
                                                                                                                                                      k * t * v0 * np.cos(
                                                                                                                                                  alpha)) / m + 1)) / k) ** 2 + yL ** 2) + 1) - (
                      ((2 * ((m * np.log(
                          np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                      k / (g * m)) ** (1 / 2))) / k ** 2 - (m * (
                                  (v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (
                                      2 * g * m * (k / (g * m)) ** (1 / 2)) - (
                                              g * t * np.sin(t * ((g * k) / m) ** (1 / 2))) / (
                                              2 * m * ((g * k) / m) ** (1 / 2)) + (
                                              g * t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                                  k / (g * m)) ** (1 / 2)) / (2 * m * ((g * k) / m) ** (1 / 2)))) / (k * (
                                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                              alpha) * (k / (g * m)) ** (1 / 2)))) * (h0 - hL + (m * np.log(
                          np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                      k / (g * m)) ** (1 / 2))) / k)) / (
                                   (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) - (2 * (
                                  (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k ** 2 - (t * v0 * np.cos(alpha)) / (
                                      k * ((k * t * v0 * np.cos(alpha)) / m + 1))) * (x0 - xL + (
                                  m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) * (h0 - hL + (m * np.log(
                          np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                      k / (g * m)) ** (1 / 2))) / k) ** 2) / (
                                   (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** 2) * (((
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
                                                                                                                                           1 / 2))) / k ** 2 - (
                                                                                                                                   m * (
                                                                                                                                       (
                                                                                                                                                   v0 * np.sin(
                                                                                                                                               t * (
                                                                                                                                                           (
                                                                                                                                                                       g * k) / m) ** (
                                                                                                                                                           1 / 2)) * np.sin(
                                                                                                                                               alpha)) / (
                                                                                                                                                   2 * g * m * (
                                                                                                                                                       k / (
                                                                                                                                                           g * m)) ** (
                                                                                                                                                               1 / 2)) - (
                                                                                                                                                   g * t * np.sin(
                                                                                                                                               t * (
                                                                                                                                                           (
                                                                                                                                                                       g * k) / m) ** (
                                                                                                                                                           1 / 2))) / (
                                                                                                                                                   2 * m * (
                                                                                                                                                       (
                                                                                                                                                                   g * k) / m) ** (
                                                                                                                                                               1 / 2)) + (
                                                                                                                                                   g * t * v0 * np.cos(
                                                                                                                                               t * (
                                                                                                                                                           (
                                                                                                                                                                       g * k) / m) ** (
                                                                                                                                                           1 / 2)) * np.sin(
                                                                                                                                               alpha) * (
                                                                                                                                                               k / (
                                                                                                                                                                   g * m)) ** (
                                                                                                                                                               1 / 2)) / (
                                                                                                                                                   2 * m * (
                                                                                                                                                       (
                                                                                                                                                                   g * k) / m) ** (
                                                                                                                                                               1 / 2)))) / (
                                                                                                                                   k * (
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
                                                                                                                                                   1 / 2)))) / (
                                                                                                                                  (
                                                                                                                                              x0 - xL + (
                                                                                                                                                  m * np.log(
                                                                                                                                              (
                                                                                                                                                          k * t * v0 * np.cos(
                                                                                                                                                      alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (
                                                                                                                                  1 / 2) - (
                                                                                                                                  (
                                                                                                                                              (
                                                                                                                                                          m * np.log(
                                                                                                                                                      (
                                                                                                                                                                  k * t * v0 * np.cos(
                                                                                                                                                              alpha)) / m + 1)) / k ** 2 - (
                                                                                                                                                          t * v0 * np.cos(
                                                                                                                                                      alpha)) / (
                                                                                                                                                          k * (
                                                                                                                                                              (
                                                                                                                                                                          k * t * v0 * np.cos(
                                                                                                                                                                      alpha)) / m + 1))) * (
                                                                                                                                              x0 - xL + (
                                                                                                                                                  m * np.log(
                                                                                                                                              (
                                                                                                                                                          k * t * v0 * np.cos(
                                                                                                                                                      alpha)) / m + 1)) / k) * (
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
                                                                                                                                  (
                                                                                                                                              x0 - xL + (
                                                                                                                                                  m * np.log(
                                                                                                                                              (
                                                                                                                                                          k * t * v0 * np.cos(
                                                                                                                                                      alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (
                                                                                                                                  3 / 2))) / (
                      (h0 - hL + (m * np.log(
                          np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                      k / (g * m)) ** (1 / 2))) / k) ** 2 / (
                                  (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) + 1) ** 2

    return out