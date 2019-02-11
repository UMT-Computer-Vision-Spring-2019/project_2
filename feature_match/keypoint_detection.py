import numpy as np
from scipy import signal


def pull_local_maxima(I, evaluation_fraction=1.):
    local_maxima = []
    for i in range(1, I.shape[0] - 1):
        for j in range(1, I.shape[1] - 1):
            if (I[i + 1][j] <= I[i][j]
                    and I[i - 1][j] <= I[i][j]
                    and I[i][j + 1] <= I[i][j]
                    and I[i + 1][j - 1] <= I[i][j]):
                local_maxima.append((i, j, I[i, j]))

    local_maxima.sort(key=lambda local_maxima: local_maxima[2], reverse=True)
    local_maxima = np.array(local_maxima)

    local_maxima = local_maxima[:local_maxima.shape[0] * evaluation_fraction]
    radii = np.zeros((local_maxima.shape[0], 1)) + float("inf")
    local_maxima = np.hstack((local_maxima, radii))

    for i in range(0, local_maxima.shape[0]):
        for j in range(0, local_maxima.shape[0]):
            distance = np.sqrt(
                (local_maxima[i][0] - local_maxima[j][0]) ** 2 + (local_maxima[i][1] - local_maxima[j][1]) ** 2)
        if (local_maxima[j][2] > local_maxima[i][2] and distance < local_maxima[i][3]):
            local_maxima[i][3] = distance

    indices = np.argsort(-local_maxima[:, 3])
    local_maxima = local_maxima[indices]
    return local_maxima[:100]


def harris_corner_detection(I):
    Su = np.matrix(
        [[-1, 0, 1],
         [-2, 0, 2],
         [-1, 0, 1]])
    w = np.matrix([
        [0.023528, 0.033969, 0.038393, 0.033969, 0.023528],
        [0.033969, 0.049045, 0.055432, 0.049045, 0.033969],
        [0.038393, 0.055432, 0.062651, 0.055432, 0.038393],
        [0.033969, 0.049045, 0.055432, 0.049045, 0.033969],
        [0.023528, 0.033969, 0.038393, 0.033969, 0.023528]
    ])

    Iu = signal.convolve2d(I, Su)
    Iv = signal.convolve2d(I, Su.T)

    Iuu = signal.convolve2d(np.multiply(Iu, Iu), w)
    Ivv = signal.convolve2d(np.multiply(Iv, Iv), w)
    Iuv = signal.convolve2d(np.multiply(Iu, Iv), w)

    H = np.divide(np.multiply(Iuu, Ivv) - np.multiply(Iuv, Iuv), Iuu + Ivv + 1e-10)

    local_maxima = pull_local_maxima(H)
    return local_maxima
