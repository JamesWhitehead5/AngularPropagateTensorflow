import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# `starting_field` is a tensor that contains the complex x, y, z polarizations of a vector field in a shape = (nx, ny, 6)
# tensor. The last dimension contain the x_real, x_image, y_real... values.
# `wavenumber` of the isotropic, homogeneous medium
# `dd` the pitch of the x, y sampling grid
# `distance` to be propagated
# `p1` is a 3-tuple of the propagation coordinate in 3D. Origin is at the center of the field
def propagate(field, dd, k, p1, dtype):
    e_x = tf.complex(field[:, :, 0], field[:, :, 1], dtype=dtype['complex'])
    e_y = tf.complex(field[:, :, 2], field[:, :, 3], dtype=dtype['complex'])
    e_z = tf.complex(field[:, :, 4], field[:, :, 5], dtype=dtype['complex'])

    e_x_prop = propagate_scalar(e_x, dd, k, p1, dtype)
    e_y_prop = propagate_scalar(e_y, dd, k, p1, dtype)
    e_z_prop = propagate_scalar(e_z, dd, k, p1, dtype)

    propagated = tf.stack([
        tf.real(e_x_prop),
        tf.imag(e_x_prop),
        tf.real(e_y_prop),
        tf.imag(e_y_prop),
        tf.real(e_z_prop),
        tf.imag(e_z_prop),
    ], axis=-1)

    return propagated

def split_complex(complex_tensor):
    return tf.math.real(complex_tensor), tf.math.imag(complex_tensor)

def complex_mul(real1, image1, real2, image2):
    real = real1 * real2 - image1 * image2
    imag = real1 * image2 + real2 * image1
    return real, imag

# Parameters
# `starting_field is a complex 2D field`
# `wavenumber` of the isotropic, homogeneous medium
# `dd` the pitch of the x, y sampling grid
# `distance` to be propagated
# `p1` is a 3-tuple of the propagation coordinate in 3D. Origin is at the center of the field
def propagate_scalar(starting_field, dd, k, p1, dtype):
    nx, ny = tf.shape(starting_field)
    nx = int(nx)
    ny = int(ny)
    x = (tf.range(0, nx, dtype=dtype['real']) - nx / 2 + 0.5) * dd
    y = (tf.range(0, ny, dtype=dtype['real']) - ny / 2 + 0.5) * dd

    yy, xx = tf.meshgrid(y, x)

    # x, y, and z components of the vector that connects input points to the ouput point
    r01_x = xx - p1[0]
    r01_y = yy - p1[1]
    r01_z = tf.fill(dims=(nx, ny), value=tf.cast(p1[2], dtype=dtype['real']))

    # distance between all input coordinates and the propagation point
    abs_r01 = tf.sqrt(r01_x ** 2 + r01_y ** 2 + r01_z ** 2)

    # is equivalent to cosine of angle r01 make with normal
    cos_r01 = r01_z / abs_r01

    u_real, u_imag = split_complex(starting_field)

    result_image = dd**2 * k / (2. * np.pi) * tf.reduce_sum(
        (u_imag * tf.cos(k*abs_r01) + u_real * tf.sin(k * abs_r01)) * cos_r01 / abs_r01
    )

    result_real = -dd**2 * k / (2. * np.pi) * tf.reduce_sum(
        (u_real * tf.cos(k*abs_r01) - u_imag * tf.sin(k * abs_r01)) * cos_r01 / abs_r01
    )

    return result_real, result_image

def propagate_scalar_total(starting_field, dd, k, z, dtype):
    def get_total_field(xx, yy):
        return propagate_scalar(starting_field, dd, k, (xx, yy, z), dtype)

    get_total_field = np.vectorize(get_total_field)

    n_x, n_y = tf.shape(starting_field)
    n_x = int(n_x)
    n_y = int(n_y)

    x = np.linspace(-dd * n_x / 2., dd * n_x / 2., n_x)
    y = np.linspace(-dd * n_y / 2., dd * n_y / 2., n_y)
    yy, xx = tf.meshgrid(y, x)

    return tf.complex(*get_total_field(xx, yy))


def propagate_scalar_test():
    aperture = 1e-4
    x = np.linspace(-aperture / 2, aperture / 2, 100)
    y = x
    dd = x[1] - x[0]

    yy, xx = tf.meshgrid(y, x)

    f = 1e-3
    wavenumber = 2 * np.pi / 633e-9
    phi = -wavenumber * np.sqrt(xx ** 2 + y ** 2 + f ** 2)
    starting_field = np.exp(1j * phi)

    dtype = {'real': tf.float32, 'complex': tf.complex64}
    # dtype = {'real': tf.float64, 'complex': tf.complex128}

    starting_field = tf.constant(starting_field, dtype['complex'])

    e_field_p1 = propagate_scalar(starting_field, dd, wavenumber, (0., 0., f), dtype)

    print(np.abs(e_field_p1))

    plt.imshow(np.angle(starting_field))
    plt.colorbar()
    plt.show()

    def get_total_field(xx, yy):
        return propagate_scalar(starting_field, dd, wavenumber, (xx, yy, f), dtype)

    get_total_field = np.vectorize(get_total_field)
    prop_field = get_total_field(xx, yy)
    intensity = prop_field[0]**2 + prop_field[1]**2

    plt.imshow(intensity)
    plt.colorbar()
    plt.show()

    print(tf.reduce_mean(intensity))

if __name__=='__main__':
    propagate_scalar_test()