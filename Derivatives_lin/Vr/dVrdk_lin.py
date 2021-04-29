import numpy as np


def dVrdk_lin(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = ((v0 * np.exp(-(k * t) / m) * np.sin(alpha) + (g * m * (np.exp(-(k * t) / m) - 1)) / k) * (
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
                   xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / m) / (
                  (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                  hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                  g * m * t) / k) ** 2 + yL ** 2) ** (1 / 2) - ((2 * (
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
                                                                      (t * v0 * np.exp(-(k * t) / m) * np.cos(
                                                                          alpha)) / k + (m * v0 * np.cos(
                                                                  alpha) * (np.exp(
                                                                  -(k * t) / m) - 1)) / k ** 2) * (
                                                                      xL - x0 + (m * v0 * np.cos(alpha) * (
                                                                      np.exp(-(
                                                                              k * t) / m) - 1)) / k)) * (
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
                  2 * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                  hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                  g * m * t) / k) ** 2 + yL ** 2) ** (3 / 2))

    return out
