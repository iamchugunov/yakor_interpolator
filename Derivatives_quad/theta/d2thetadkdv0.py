import numpy as np
def  d2thetadkdv0(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = ((t * np.cos(alpha) * ((m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k ** 2 - (t * v0 * np.cos(alpha)) / (
                k * ((k * t * v0 * np.cos(alpha)) / m + 1))) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / (((k * t * v0 * np.cos(alpha)) / m + 1) * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) - (
                       (m * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (1 / 2)) / (k ** 2 * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2))) - (m * (
                           (np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (2 * g * m * (k / (g * m)) ** (1 / 2)) + (
                               g * t * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (1 / 2)) / (
                                       2 * m * ((g * k) / m) ** (1 / 2)))) / (k * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2))) + (m * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                           (v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (
                               2 * g * m * (k / (g * m)) ** (1 / 2)) - (g * t * np.sin(t * ((g * k) / m) ** (1 / 2))) / (
                                       2 * m * ((g * k) / m) ** (1 / 2)) + (
                                       g * t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                           1 / 2)) / (2 * m * ((g * k) / m) ** (1 / 2))) * (k / (g * m)) ** (1 / 2)) / (
                                   k * (np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(
                               alpha) * (k / (g * m)) ** (1 / 2)) ** 2)) / (
                       (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (1 / 2) + (
                       t * np.cos(alpha) * ((m * np.log(
                   np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2))) / k ** 2 - (m * (
                           (v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha)) / (
                               2 * g * m * (k / (g * m)) ** (1 / 2)) - (g * t * np.sin(t * ((g * k) / m) ** (1 / 2))) / (
                                       2 * m * ((g * k) / m) ** (1 / 2)) + (
                                       g * t * v0 * np.cos(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                                           1 / 2)) / (2 * m * ((g * k) / m) ** (1 / 2)))) / (k * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2)))) * (
                                   x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k)) / (
                       ((k * t * v0 * np.cos(alpha)) / m + 1) * (
                           (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) - (
                       3 * t * np.cos(alpha) * (
                           (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k ** 2 - (t * v0 * np.cos(alpha)) / (
                               k * ((k * t * v0 * np.cos(alpha)) / m + 1))) * (
                                   x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 * (h0 - hL + (
                           m * np.log(
                       np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                   k / (g * m)) ** (1 / 2))) / k)) / (((k * t * v0 * np.cos(alpha)) / m + 1) * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (5 / 2)) + (
                       t ** 2 * v0 * np.cos(alpha) ** 2 * (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) * (
                           h0 - hL + (m * np.log(
                       np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                                   k / (g * m)) ** (1 / 2))) / k)) / (m * ((k * t * v0 * np.cos(alpha)) / m + 1) ** 2 * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) + (
                       m * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                           (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k ** 2 - (t * v0 * np.cos(alpha)) / (
                               k * ((k * t * v0 * np.cos(alpha)) / m + 1))) * (
                                   x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) * (k / (g * m)) ** (
                                   1 / 2)) / (
                       k * ((x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** (3 / 2) * (
                           np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (
                               k / (g * m)) ** (1 / 2)))) / ((h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k) ** 2 / ((x0 - xL + (
                m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) + 1) - (((2 * t * np.cos(alpha) * (
                x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k) ** 2) / (((k * t * v0 * np.cos(alpha)) / m + 1) * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) ** 2) - (2 * m * np.sin(
        t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (1 / 2) * (h0 - hL + (m * np.log(
        np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (
                    1 / 2))) / k)) / (k * (
                (x0 - xL + (m * np.log((k * t * v0 * np.cos(alpha)) / m + 1)) / k) ** 2 + yL ** 2) * (
                                                  np.cos(t * ((g * k) / m) ** (1 / 2)) + v0 * np.sin(
                                              t * ((g * k) / m) ** (1 / 2)) * np.sin(alpha) * (k / (g * m)) ** (1 / 2)))) * (
                                                                                                   ((m * np.log(np.cos(t * ((
                                                                                                                                  g * k) / m) ** (
                                                                                                                             1 / 2)) + v0 * np.sin(
                                                                                                       t * ((
                                                                                                                        g * k) / m) ** (
                                                                                                                   1 / 2)) * np.sin(
                                                                                                       alpha) * (k / (
                                                                                                               g * m)) ** (
                                                                                                                         1 / 2))) / k ** 2 - (
                                                                                                                m * ((
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
                                                                                                                t * ((
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
                                                                                                               ((
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
