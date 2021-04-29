import numpy as np


def d2thetadv0dalpha_lin(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = (((m * np.sin(alpha) * (np.exp(-(k * t) / m) - 1)) / (
                k * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (1 / 2)) - (
                        m * np.cos(alpha) * (np.exp(-(k * t) / m) - 1) * (
                            xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) * (
                                    hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                                        g * m * t) / k)) / (
                        k * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (
                            3 / 2))) * ((2 * m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1) * (
                hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k)) / (
                                                    k * ((xL - x0 + (m * v0 * np.cos(alpha) * (
                                                        np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2)) + (
                                                    2 * m * v0 * np.sin(alpha) * (np.exp(-(k * t) / m) - 1) * (xL - x0 + (
                                                        m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) * (
                                                                hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (
                                                                    np.exp(-(k * t) / m) - 1)) / k + (
                                                                            g * m * t) / k) ** 2) / (k * (
                (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** 2))) / ((hL - h0 + (
                m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) ** 2 / ((
                                                                                                                         xL - x0 + (
                                                                                                                             m * v0 * np.cos(
                                                                                                                         alpha) * (
                                                                                                                                         np.exp(
                                                                                                                                             -(
                                                                                                                                                         k * t) / m) - 1)) / k) ** 2 + yL ** 2) + 1) ** 2 - (
                      (m * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / (
                          k * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (
                              1 / 2)) + (m * np.sin(alpha) * (np.exp(-(k * t) / m) - 1) * (
                          xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) * (hL - h0 + (
                          m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k)) / (
                                  k * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (
                                      3 / 2)) - (m ** 2 * v0 * np.cos(alpha) ** 2 * (np.exp(-(k * t) / m) - 1) ** 2 * (
                          xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / (k ** 2 * (
                          (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) + (
                                  m ** 2 * v0 * np.sin(alpha) ** 2 * (np.exp(-(k * t) / m) - 1) ** 2 * (
                                      xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / (k ** 2 * (
                          (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) + (
                                  m ** 2 * v0 * np.cos(alpha) * np.sin(alpha) * (np.exp(-(k * t) / m) - 1) ** 2 * (
                                      hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                                          g * m * t) / k)) / (k ** 2 * (
                          (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (3 / 2)) - (
                                  3 * m ** 2 * v0 * np.cos(alpha) * np.sin(alpha) * (np.exp(-(k * t) / m) - 1) ** 2 * (
                                      xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 * (hL - h0 + (
                                      m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                                                                                                                        g * m * t) / k)) / (
                                  k ** 2 * (
                                      (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + yL ** 2) ** (
                                              5 / 2))) / ((hL - h0 + (
                m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) ** 2 / ((
                                                                                                                         xL - x0 + (
                                                                                                                             m * v0 * np.cos(
                                                                                                                         alpha) * (
                                                                                                                                         np.exp(
                                                                                                                                             -(
                                                                                                                                                         k * t) / m) - 1)) / k) ** 2 + yL ** 2) + 1)

    return out