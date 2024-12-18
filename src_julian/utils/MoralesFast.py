import numpy as np
from scipy.ndimage import rotate
from scipy.ndimage import gaussian_filter
from scipy.interpolate import interp1d, interp2d
from scipy.ndimage.measurements import center_of_mass


class Tensor():

    def __init__(self, Exx, Exy, Exz, Eyx, Eyy, Eyz, Ezx, Ezy, Ezz):
        self.E1, self.E2, self.E3 = Exx.copy(), Exy.copy(), Exz.copy()
        self.E4, self.E5, self.E6 = Eyx.copy(), Eyy.copy(), Eyz.copy()
        self.E7, self.E8, self.E9 = Ezx.copy(), Ezy.copy(), Ezz.copy()

    def asmat(self):
        temp = np.array([[self.E1, self.E2, self.E3],
                         [self.E4, self.E5, self.E6],
                         [self.E7, self.E8, self.E9]]).transpose((2, 3, 4, 0, 1))
        return temp

    def asvoigt(self):
        return self.E1, self.E2, self.E3, self.E4, self.E5, self.E6, self.E7, self.E8, self.E9

    def transpose(self):
        return Tensor(self.E1, self.E4, self.E7, self.E2, self.E5, self.E8, self.E3, self.E6, self.E9)

    def identity_add(self):
        self.E1 += 1;
        self.E5 += 1;
        self.E9 += 1

    def identity_subtract(self):
        self.E1 -= 1;
        self.E5 -= 1;
        self.E9 -= 1

    @staticmethod
    def dot(X, Y):
        X1, X2, X3, X4, X5, X6, X7, X8, X9 = X.asvoigt()
        Y1, Y2, Y3, Y4, Y5, Y6, Y7, Y8, Y9 = Y.asvoigt()

        Z1, Z2, Z3 = X1 * Y1 + X2 * Y4 + X3 * Y7, X1 * Y2 + X2 * Y5 + X3 * Y8, X1 * Y3 + X2 * Y6 + X3 * Y9
        Z4, Z5, Z6 = X4 * Y1 + X5 * Y4 + X6 * Y7, X4 * Y2 + X5 * Y5 + X6 * Y8, X4 * Y3 + X5 * Y6 + X6 * Y9
        Z7, Z8, Z9 = X7 * Y1 + X8 * Y4 + X9 * Y7, X7 * Y2 + X8 * Y5 + X9 * Y8, X7 * Y3 + X8 * Y6 + X9 * Y9

        return Tensor(Z1, Z2, Z3, Z4, Z5, Z6, Z7, Z8, Z9)

class MyocardialStrain():

    def __init__(self, masklvmyo, com, flow):
        self.masklvmyo = masklvmyo
        self.com = com
        self.flow = flow

        assert len(masklvmyo.shape) == 3
        assert len(flow.shape) == 4
        assert masklvmyo.shape == flow.shape[:3]
        assert flow.shape[-1] == 3

    def calculate_strain(self, dx, dy, dz):
        cx, cy, cz = self.com
        # nx, ny, nz = self.masklvmyo.shape

        try:
            self.flow_rot = roll_to_center(self.flow, cx, cy)
            self.mask_rot = roll_to_center(self.masklvmyo, cx, cy)
        except Exception as e:
            print(e)
            self.flow_rot = self.flow
            self.mask_rot = self.masklvmyo

        # validation plot
        # import matplotlib.pyplot as plt
        # plt.figure()
        # plt.imshow(self.mask_rot[...,32])
        # plt.imshow(self.masklvmyo[...,32], alpha=.5)
        # plt.scatter(cx,cy)
        # plt.show()
        # make sure we squeeze only the last axis, otherwise this will fail for single slice data


        ux, uy, uz = np.array_split(self.flow_rot, 3, -1)
        if ux.shape[-2] == 1:
            print('single slice cube... ')
        Uxx, Uxy, Uxz = np.gradient(np.squeeze(ux, axis=-1),dx,dy,dz)
        Uyx, Uyy, Uyz = np.gradient(np.squeeze(uy, axis=-1),dx,dy,dz)
        Uzx, Uzy, Uzz = np.gradient(np.squeeze(uz, axis=-1),dx,dy,dz)

        F = Tensor(Uxx, Uxy, Uxz, Uyx, Uyy, Uyz, Uzx, Uzy, Uzz)

        F.identity_add()
        F = F.dot(F.transpose(), F)
        F.identity_subtract()

        self.Err, self.Ecc, self.Erc, self.Ecr = convert_to_polar(mask=self.mask_rot,
                                                                  E=0.5 * F.asmat()[:, :, :, :2, :2])


def roll(x, rx, ry):
    x = np.roll(x, rx, axis=0)
    return np.roll(x, ry, axis=1)


def roll_to_center(x, cx, cy):
    nx, ny = x.shape[:2]
    #return roll(x, int(nx // 2 - cx), int(ny // 2 - cy))
    try:
        return roll(x, int(nx // 2 - cx), int(ny // 2 - cy))
    except Exception as e: # dont roll
        return x


def polar_grid(nx=128, ny=128):
    x, y = np.meshgrid(np.linspace(-nx // 2, nx // 2, nx), np.linspace(-ny // 2, ny // 2, ny))
    phi = (np.rad2deg(np.arctan2(y, x)) + 180).T
    r = np.sqrt(x ** 2 + y ** 2 + 1e-8)
    return phi, r


def convert_to_polar(mask, E):
    phi = polar_grid(*E.shape[:2])[0]
    Err = np.zeros(mask.shape)
    Ecc = np.zeros(mask.shape)
    Erc = np.zeros(mask.shape)
    Ecr = np.zeros(mask.shape)
    for k in range(mask.shape[-1]):
        cos = np.cos(np.deg2rad(phi))
        sin = np.sin(np.deg2rad(phi))

        Exx, Exy, Eyx, Eyy = E[:, :, k, 0, 0], E[:, :, k, 0, 1], E[:, :, k, 1, 0], E[:, :, k, 1, 1]
        Err[:, :, k] += cos * (cos * Exx + sin * Exy) + sin * (cos * Eyx + sin * Eyy)
        Ecc[:, :, k] += -sin * (-sin * Exx + cos * Exy) + cos * (-sin * Eyx + cos * Eyy)
        Erc[:, :, k] += cos * (-sin * Exx + cos * Exy) + sin * (-sin * Eyx + cos * Eyy)
        Ecr[:, :, k] += -sin * (cos * Exx + sin * Exy) + cos * (cos * Eyx + sin * Eyy)

    return Err, Ecc, Erc, Ecr
