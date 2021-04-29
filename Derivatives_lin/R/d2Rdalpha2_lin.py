import numpy as np 


def d2Rdaplpa2_lin(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):

    out = ((2 * m ** 2 * v0 ** 2 * np.cos(alpha) ** 2 * (np.exp(-(k * t) / m) - 1) ** 2) / k ** 2 + (
                2 * m ** 2 * v0 ** 2 * np.sin(alpha) ** 2 * (np.exp(-(k * t) / m) - 1) ** 2) / k ** 2 - (
                       2 * m * v0 * np.sin(alpha) * (np.exp(-(k * t) / m) - 1) * (
                           hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                               g * m * t) / k)) / k - (2 * m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1) * (
                xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / k) / (2 * (
                (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                    hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                        g * m * t) / k) ** 2 + yL ** 2) ** (1 / 2)) - ((2 * m * v0 * np.sin(alpha) * (
                np.exp(-(k * t) / m) - 1) * (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / k - (
                                                                                2 * m * v0 * np.cos(alpha) * (
                                                                                    np.exp(-(k * t) / m) - 1) * (
                                                                                            hL - h0 + (m * (
                                                                                                v0 * np.sin(alpha) + (
                                                                                                    g * m) / k) * (np.exp(
                                                                                        -(k * t) / m) - 1)) / k + (
                                                                                                        g * m * t) / k)) / k) ** 2 / (
                      4 * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (3 / 2))

    return out
