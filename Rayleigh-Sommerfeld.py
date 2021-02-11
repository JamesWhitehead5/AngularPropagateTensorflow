import numpy as np
import matplotlib.pyplot as plt


# Parameters
# `starting_field is a complex 2D field`
# `wavenumber` of the isotropic, homogeneous medium
# `dd` the pitch of the x, y sampling grid
# `distance` to be propagated
# `p1` is a 3-tuple of the propagation coordinate in 3D. Origin is at the center of the field
def propagate_to_point_np(starting_field, dd, wavenumber, p1):
    assert len(p1) == 3

    nx, ny = np.shape(starting_field)
    x = (np.arange(0, nx) - nx/2) * dd
    y = (np.arange(0, ny) - ny/2) * dd

    yy, xx = np.meshgrid(y, x)

    # x, y, and z components of the vector that connects input points to the ouput point
    r01_x = xx - p1[0]
    r01_y = yy - p1[1]
    r01_z = np.full(shape=(nx, ny), fill_value=p1[2])

    # distance between all input coordinates and the propagation point
    abs_r01 = np.sqrt(r01_x**2 + r01_y**2 + r01_z**2)

    # is equivalent to cosine of angle r01 make with normal
    cos_r01 = r01_z / abs_r01

    return wavenumber/(2*np.pi*1j) * np.sum(starting_field * np.exp(1j*wavenumber*abs_r01) / abs_r01 * cos_r01) * dd**2


def propagate_point_np_test():
    aperture = 1e-4
    x = np.linspace(-aperture/2, aperture/2, 100)
    y = x
    dd = x[1] - x[0]

    yy, xx = np.meshgrid(y, x)

    f = 1e-3
    wavenumber = 2*np.pi/633e-9
    phi = -wavenumber * np.sqrt(xx**2 + y**2 + f**2)
    starting_field = np.exp(1j*phi)

    e_field_p1 = propagate_to_point_np(starting_field, dd, wavenumber, (0., 0., f))
    print(np.abs(e_field_p1))

    plt.imshow(np.angle(starting_field))
    plt.colorbar()
    plt.show()

    def get_total_field(xx, yy):
        return propagate_to_point_np(starting_field, dd, wavenumber, (xx, yy, f))

    get_total_field = np.vectorize(get_total_field)

    total_field = get_total_field(xx, yy)
    intensity = np.abs(total_field)**2

    plt.imshow(intensity)
    plt.colorbar()
    plt.show()

    print(np.mean(intensity))





if __name__=='__main__':
    propagate_point_np_test()