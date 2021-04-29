
import numpy as np
def dRdk_lin(xL, yL, hL, t, v0, alpha, k, x0, h0, m, g, deltaR):
    out = -(2 * (hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
                g * m * t) / k) * (
                    (t * np.exp(-(k * t) / m) * (v0 * np.sin(alpha) + (g * m) / k)) / k + (
                    g * m ** 2 * (np.exp(-(k * t) / m) - 1)) / k ** 3 + (
                            m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k ** 2 + (
                            g * m * t) / k ** 2) + 2 * ((t * v0 * np.exp(-(k * t) / m) * np.cos(alpha)) / k + (
            m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k ** 2) * (
                    xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k)) / (2 * (
            (xL - x0 + (m * v0 * np.cos(alpha) * (np.exp(-(k * t) / m) - 1)) / k) ** 2 + (
            hL - h0 + (m * (v0 * np.sin(alpha) + (g * m) / k) * (np.exp(-(k * t) / m) - 1)) / k + (
            g * m * t) / k) ** 2 + yL ** 2) ** (1 / 2))
    return out
