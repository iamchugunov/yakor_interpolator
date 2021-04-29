import numpy as np


def d2Vrdalpha2_lin(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = (v0 * np.exp(-(k * t) / m) * np.cos(alpha) * (
                xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) + v0 * np.exp(-(k * t) / m) * np.sin(alpha) * (
                       hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                           g * m * t) / k) - (
                       m * v0 ** 2 * np.exp(-(k * t) / m) * np.cos(alpha) ** 2 * (np.exp(-(k * t) / m) - 1)) / k - (
                       2 * m * v0 ** 2 * np.exp(-(k * t) / m) * np.sin(alpha) ** 2 * (np.exp(-(k * t) / m) - 1)) / k + (
                       m * v0 * np.sin(alpha) * (
                           v0 * np.exp(-(k * t) / m) * np.sin(alpha) + (g * m * (np.exp(-(k * t) / m) - 1)) / k) * (
                                   np.exp(-(k * t) / m) - 1)) / k) / (
                      (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (1 / 2) + (((v0 * np.exp(-(k * t) / m) * np.sin(alpha) + (
                g * m * (np.exp(-(k * t) / m) - 1)) / k) * (hL - h0 + (
                m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (g * m * t) / k) + v0 * np.exp(
        -(k * t) / m) * np.cos(alpha) * (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) * ((
                                                                                                                     2 * m ** 2 * v0 ** 2 * np.cos(
                                                                                                                 alpha) ** 2 * (
                                                                                                                                 np.exp(
                                                                                                                                     -(
                                                                                                                                                 k * t) / m) - 1) ** 2) / k **2 + (
                                                                                                                     2 * m ** 2 * v0 ** 2 * np.sin(
                                                                                                                 alpha) ** 2 * (
                                                                                                                                 np.exp(
                                                                                                                                     -(
                                                                                                                                                 k * t) / m) - 1) ** 2) / k ** 2 - (
                                                                                                                     2 * m * v0 * np.sin(
                                                                                                                 alpha) * (
                                                                                                                                 np.exp(
                                                                                                                                     -(
                                                                                                                                                 k * t) / m) - 1) * (
                                                                                                                                 hL - h0 + (
                                                                                                                                     m * (
                                                                                                                                         v0 * np.sin(
                                                                                                                                     alpha) + (
                                                                                                                                                     g * m) / k) * (
                                                                                                                                                 np.exp(
                                                                                                                                                     -(
                                                                                                                                                                 k * t) / m) - 1)) / k + (
                                                                                                                                             g * m * t) / k)) / k - (
                                                                                                                     2 * m * v0 * np.cos(
                                                                                                                 alpha) * (
                                                                                                                                 np.exp(
                                                                                                                                     -(
                                                                                                                                                 k * t) / m) - 1) * (
                                                                                                                                 xL - x0 + (
                                                                                                                                     m * v0 * np.cos(
                                                                                                                                 alpha) * (
                                                                                                                                                 np.exp(
                                                                                                                                                     -(
                                                                                                                                                                 k * t) / m) - 1)) / k)) / k)) / (
                      2 * ((xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) **2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (3 / 2)) - (3 * ((2 * m * v0 * np.sin(alpha) * (
                np.exp(-(k * t) / m) - 1) * (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / k - (
                                                                                           2 * m * v0 * np.cos(alpha) * (
                                                                                               np.exp(-(
                                                                                                           k * t) / m) - 1) * (
                                                                                                       hL - h0 + (m * (
                                                                                                           v0 * np.sin(
                                                                                                       alpha) + (
                                                                                                                       g * m) / k) * (
                                                                                                                              np.exp(
                                                                                                                                  -(
                                                                                                                                              k * t) / m) - 1)) / k + (
                                                                                                                   g * m * t) / k)) / k) ** 2 * (
                                                                                      (v0 * np.exp(-(k * t) / m) * np.sin(
                                                                                          alpha) + (g * m * (np.exp(
                                                                                          -(k * t) / m) - 1)) / k) * (
                                                                                                  hL - h0 + (m * (
                                                                                                      v0 * np.sin(
                                                                                                  alpha) + (
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
                              g * m * t) / k) ** 2 + yL ** 2) ** (5 / 2)) + (((2 * m * v0 * np.sin(alpha) * (
                np.exp(-(k * t) / m) - 1) * (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / k - (
                                                                                       2 * m * v0 * np.cos(alpha) * (
                                                                                           np.exp(-(k * t) / m) - 1) * (
                                                                                                   hL - h0 + (m * (
                                                                                                       v0 * np.sin(
                                                                                                   alpha) + (
                                                                                                                   g * m) / k) * (
                                                                                                                          np.exp(
                                                                                                                              -(
                                                                                                                                          k * t) / m) - 1)) / k + (
                                                                                                               g * m * t) / k)) / k) * (
                                                                                      v0 * np.exp(-(k * t) / m) * np.sin(
                                                                                  alpha) * (xL - x0 + (
                                                                                          m * v0 * np.cos(alpha) * (np.exp(-(
                                                                                              k * t) / m) - 1)) / k) - v0 * np.exp(
                                                                                  -(k * t) / m) * np.cos(alpha) * (
                                                                                                  hL - h0 + (m * (
                                                                                                      v0 * np.sin(
                                                                                                  alpha) + (
                                                                                                                  g * m) / k) * (
                                                                                                                         np.exp(
                                                                                                                             -(
                                                                                                                                         k * t) / m) - 1)) / k + (
                                                                                                              g * m * t) / k) - (
                                                                                                  m * v0 * np.cos(
                                                                                              alpha) * (v0 * np.exp(
                                                                                              -(k * t) / m) * np.sin(
                                                                                              alpha) + (g * m * (np.exp(-(
                                                                                                      k * t) / m) - 1)) / k) * (
                                                                                                              np.exp(-(
                                                                                                                          k * t) / m) - 1)) / k + (
                                                                                                  m * v0 ** 2 * np.exp(
                                                                                              -(k * t) / m) * np.cos(
                                                                                              alpha) * np.sin(alpha) * (
                                                                                                              np.exp(-(
                                                                                                                          k * t) / m) - 1)) / k)) / (
                      (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
                          hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                              g * m * t) / k) ** 2 + yL ** 2) ** (3 / 2)

    return out