import numpy as np


def d2Vrdk2_lin(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):

    out = ((2 * (hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) * (
                (t * np.exp(-(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / k + (
                    g * m ** 2 * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (
                            m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k ** 2 + (
                            g * m * t) / k ** 2) + 2 * ((t * v0 * np.exp(-(k * t) / m) * np.cos(alpha)) / k + (
                m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k ** 2) * (
                        xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) * (
                       (v0 * np.exp(-(k * t) / m) * np.sin(alpha) + (g * m * (np.exp(-(k * t) / m) - 1)) / k) * (
                           (t * np.exp(-(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / k + (
                               g * m ** 2 * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (
                                       m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k ** 2 + (
                                       g * m * t) / k ** 2) + (
                                   (g * t * np.exp(-(k * t) / m)) / k + (g * m * (np.exp(-(k * t) / m) - 1)) / k ** 2 + (
                                       t * v0 * np.exp(-(k * t) / m) * np.sin(alpha)) / m) * (
                                   hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                                       g * m * t) / k) + v0 * np.exp(-(k * t) / m) * np.cos(alpha) * (
                                   (t * v0 * np.exp(-(k * t) / m) * np.cos(alpha)) / k + (
                                       m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k ** 2) + (
                                   t * v0 * np.exp(-(k * t) / m) * np.cos(alpha) * (
                                       xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / m)) / (
                      (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (3 / 2) - (3 * (2 * (
                hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) * ((
                                                                                                                                t * np.exp(
                                                                                                                            -(
                                                                                                                                        k * t) / m) * (
                                                                                                                                            v0 * np.sin(
                                                                                                                                        alpha) + (
                                                                                                                                                        g * m) / k)) / k + (
                                                                                                                                g * m ** 2 * (
                                                                                                                                    np.exp(
                                                                                                                                        -(
                                                                                                                                                    k * t) / m) - 1)) / k ** 3 + (
                                                                                                                                m * (
                                                                                                                                    v0 * np.sin(
                                                                                                                                alpha) + (
                                                                                                                                                g * m) / k) * (
                                                                                                                                            np.exp(
                                                                                                                                                -(
                                                                                                                                                            k * t) / m) - 1)) / k ** 2 + (
                                                                                                                                g * m * t) / k ** 2) + 2 * (
                                                                                          (t * v0 * np.exp(
                                                                                              -(k * t) / m) * np.cos(
                                                                                              alpha)) / k + (
                                                                                                      m * v0 * np.cos(
                                                                                                  alpha) * (np.exp(-(
                                                                                                          k * t) / m) - 1)) / k ** 2) * (
                                                                                          xL - x0 + (
                                                                                              m * v0 * np.cos(alpha) * (
                                                                                                  np.exp(-(
                                                                                                              k * t) / m) - 1)) / k)) ** 2 * (
                                                                                     (v0 * np.exp(-(k * t) / m) * np.sin(
                                                                                         alpha) + (g * m * (np.exp(
                                                                                         -(k * t) / m) - 1)) / k) * (
                                                                                                 hL - h0 + (m * (
                                                                                                     v0 * np.sin(alpha) + (
                                                                                                         g * m) / k) * (
                                                                                                                        np.exp(
                                                                                                                            -(
                                                                                                                                        k * t) / m) - 1)) / k + (
                                                                                                             g * m * t) / k) + v0 * np.exp(
                                                                                 -(k * t) / m) * np.cos(alpha) * (
                                                                                                 xL - x0 + (
                                                                                                     m * v0 * np.cos(
                                                                                                 alpha) * (np.exp(-(
                                                                                                         k * t) / m) - 1)) / k))) / (
                      4 * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (5 / 2)) - (
                      (v0 * np.exp(-(k * t) / m) * np.sin(alpha) + (g * m * (np.exp(-(k * t) / m) - 1)) / k) * (
                          (2 * t * np.exp(-(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / k ** 2 + (
                              4 * g * m ** 2 * (np.exp(-(k * t) / m) - 1)) / k ** 4 + (
                                      2 * m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (
                                      2 * g * m * t) / k ** 3 + (
                                      t ** 2 * np.exp(-(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / (k * m) + (
                                      2 * g * m * t * np.exp(-(k * t) / m)) / k ** 3) + (
                                  hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                                      g * m * t) / k) * ((2 * g * t * np.exp(-(k * t) / m)) / k ** 2 + (
                          2 * g * m * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (g * t ** 2 * np.exp(-(k * t) / m)) / (k * m) + (
                                                                     t ** 2 * v0 * np.exp(-(k * t) / m) * np.sin(
                                                                 alpha)) / m ** 2) + 2 * (
                                  (g * t * np.exp(-(k * t) / m)) / k + (g * m * (np.exp(-(k * t) / m) - 1)) / k ** 2 + (
                                      t * v0 * np.exp(-(k * t) / m) * np.sin(alpha)) / m) * (
                                  (t * np.exp(-(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / k + (
                                      g * m ** 2 * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (
                                              m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k ** 2 + (
                                              g * m * t) / k ** 2) + v0 * np.exp(-(k * t) / m) * np.cos(alpha) * (
                                  (2 * t * v0 * np.exp(-(k * t) / m) * np.cos(alpha)) / k ** 2 + (
                                      2 * m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (
                                              t ** 2 * v0 * np.exp(-(k * t) / m) * np.cos(alpha)) / (k * m)) + (
                                  2 * t * v0 * np.exp(-(k * t) / m) * np.cos(alpha) * (
                                      (t * v0 * np.exp(-(k * t) / m) * np.cos(alpha)) / k + (
                                          m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k**2)) / m + (
                                  t ** 2 * v0 * np.exp(-(k * t) / m) * np.cos(alpha) * (
                                      xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / m ** 2) / (
                      (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (1 / 2) + (((v0 * np.exp(-(k * t) / m) * np.sin(alpha) + (
                g * m * (np.exp(-(k * t) / m) - 1)) / k) * (hL - h0 + (
                m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) + v0 * np.exp(
        -(k * t) / m) * np.cos(alpha) * (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) * (2 * (
                xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) * ((2 * t * v0 * np.exp(-(k * t) / m) * np.cos(
        alpha)) / k ** 2 + (2 * m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (t ** 2 * v0 * np.exp(
        -(k * t) / m) * np.cos(alpha)) / (k * m)) + 2 * (hL - h0 + (
                m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) * ((2 * t * np.exp(
        -(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / k ** 2 + (4 * g * m ** 2 * (np.exp(-(k * t) / m) - 1)) / k ** 4 + (
                                                                                                                     2 * m * (
                                                                                                                         v0 * np.sin(
                                                                                                                     alpha) + (
                                                                                                                                     g * m) / k) * (
                                                                                                                                 np.exp(
                                                                                                                                     -(
                                                                                                                                                 k * t) / m) - 1)) / k ** 3 + (
                                                                                                                     2 * g * m * t) / k ** 3 + (
                                                                                                                     t ** 2 * np.exp(
                                                                                                                 -(
                                                                                                                             k * t) / m) * (
                                                                                                                                 v0 * np.sin(
                                                                                                                             alpha) + (
                                                                                                                                             g * m) / k)) / (
                                                                                                                     k * m) + (
                                                                                                                     2 * g * m * t * np.exp(
                                                                                                                 -(
                                                                                                                             k * t) / m)) / k ** 3) + 2 * (
                                                                                                                     (
                                                                                                                                 t * v0 * np.exp(
                                                                                                                             -(
                                                                                                                                         k * t) / m) * np.cos(
                                                                                                                             alpha)) / k + (
                                                                                                                                 m * v0 * np.cos(
                                                                                                                             alpha) * (
                                                                                                                                             np.exp(
                                                                                                                                                 -(
                                                                                                                                                             k * t) / m) - 1)) / k ** 2) ** 2 + 2 * (
                                                                                                                     (
                                                                                                                                 t * np.exp(
                                                                                                                             -(
                                                                                                                                         k * t) / m) * (
                                                                                                                                             v0 * np.sin(
                                                                                                                                         alpha) + (
                                                                                                                                                         g * m) / k)) / k + (
                                                                                                                                 g * m ** 2 * (
                                                                                                                                     np.exp(
                                                                                                                                         -(
                                                                                                                                                     k * t) / m) - 1)) / k ** 3 + (
                                                                                                                                 m * (
                                                                                                                                     v0 * np.sin(
                                                                                                                                 alpha) + (
                                                                                                                                                 g * m) / k) * (
                                                                                                                                             np.exp(
                                                                                                                                                 -(
                                                                                                                                                             k * t) / m) - 1)) / k ** 2 + (
                                                                                                                                 g * m * t) / k ** 2) ** 2)) / (
                      2 * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (3 / 2))

    return out

