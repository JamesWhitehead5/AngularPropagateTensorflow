import numpy as np
import tensorflow as tf
from typing import List, Union
import logging
import matplotlib.pyplot as plt

def _assert_2d(field):
    """Checks if the array is 2D"""
    assert len(np.shape(field)) == 2, "Variable fields must be a 2D array"


def complex_mul(a_real, a_imag, b_real, b_imag):
    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real
    return real, imag

def complex_split(c):
    return tf.math.real(c), tf.math.imag(c)


def get_frequencies(nx, ny, dx, dy, float_type):
    k_x = tf.constant(np.fft.fftfreq(n=nx, d=dx) * 2 * np.pi, dtype=float_type)
    k_y = tf.constant(np.fft.fftfreq(n=ny, d=dy) * 2 * np.pi, dtype=float_type)
    k_Y, k_X = tf.meshgrid(k_y, k_x, indexing='xy')
    return k_X, k_Y

def propagate_angular(field: tf.Variable, k: float, z_list: List[float], dx: float, dy: float) -> tf.Variable:
    """Uses an angular propagation method to propagates a field to multiple planes

    Parameters:
        field (2D array): Complex field array to be propagated.
        k (complex): The wavenumber of the propagation medium. Can be calculated using 2*pi*n/wavelength
        z_list (1D array): Distances that the field should be propagated
        dx (float): physical size of each pixel in the x direction
        dy (float): physical size of each pixel in the y direction

    Returns:
        E_prop (3D array complex): Array of the propagated fields.
            The 1st, 2nd and 3rd dimensions correspond to the x, y, and z dimensions.
            The third dimension is the same as the length of z_list
    """
    if field.dtype == tf.complex128:
        float_type = tf.float64
    else:
        float_type = tf.float32

    z_list = tf.constant(z_list, dtype=float_type)

    # shape of the input field
    n_x, n_y = field.shape
    k_X, k_Y = get_frequencies(n_x, n_y, dx, dy, float_type)

    # define wavenumber for each wavevector in the direction of propagation
    k_tensor = k * tf.ones(dtype=float_type, shape=field.shape)
    kz_squared = tf.cast(k_tensor ** 2 - k_X ** 2 - k_Y ** 2, dtype=field.dtype)
    k_z = tf.sqrt(kz_squared)

    U = tf.signal.fft2d(field)

    # broadcast k_z into 3rd dimension
    k_Z = tf.broadcast_to(k_z, shape=(len(z_list), *k_z.shape))

    d_Z = tf.broadcast_to(
        tf.expand_dims(
            tf.expand_dims(
                z_list,
                axis=-1
            ),
            axis=-1
        ),
        k_Z.shape
    )

    phase_real = tf.math.real(k_Z) * d_Z
    phase_imag = tf.math.imag(k_Z) * d_Z

    H_real = tf.cos(phase_real) * tf.exp(-phase_imag)
    H_imag = tf.sin(phase_real) * tf.exp(-phase_imag)


    E_k_prop_real, E_k_prop_imag = complex_mul(
        *complex_split(U),
        H_real,
        H_imag,
    )

    E_prop = tf.signal.ifft2d(tf.complex(E_k_prop_real, E_k_prop_imag))

    # For DEBUG
    # import pickle
    # debug = {}
    # debug["k_X"] = k_X
    # debug["k_Y"] = k_Y
    # debug["E_in_k"] = E_in_k
    # debug["k_Z"] = k_Z
    # debug['phase'] = phase_real.numpy() + 1j*phase_imag.numpy()
    # debug['E_k_prop'] = E_k_prop_real.numpy() + 1j*E_k_prop_imag.numpy()
    # debug['field'] = field
    # debug['E_prop'] = E_prop
    #
    # pickle.dump(debug, open("tf.p", "wb"))

    return E_prop


def propagate_angular_bw_limited(field, k, z_list, dx, dy,):
    """Uses an angular propagation method to propagates a field to multiple planes

     Parameters:
         field (2D array): Complex field array to be propagated.
         k (complex): The wavenumber of the propagation medium. Can be calculated using 2*pi*n/wavelength
         z_list (1D array): Distances that the field should be propagated
         dx (float): physical size of each pixel in the x direction
         dy (float): physical size of each pixel in the y direction

     Returns:
         E_prop (3D array complex): Array of the propagated fields.
             The 1st, 2nd and 3rd dimensions correspond to the x, y, and z dimensions.
             The third dimension is the same as the length of z_list
     """

    if field.dtype == tf.complex128:
        float_type = tf.float64
    else:
        float_type = tf.float32


    z_list = tf.constant(z_list, dtype=float_type)

    # shape of the input field
    n_x, n_y = field.shape
    k_X, k_Y = get_frequencies(n_x, n_y, dx, dy, float_type)

    # define wavenumber for each wavevector in the direction of propagation
    # k_tensor = k * tf.ones(dtype=float_type, shape=field.shape)
    k_tensor = tf.fill(
        field.shape, k
    )

    kz_squared = tf.cast(k_tensor ** 2 - k_X ** 2 - k_Y ** 2, dtype=field.dtype)
    k_z = tf.sqrt(kz_squared)

    U = tf.signal.fft2d(field)

    # broadcast k_z into 3rd dimension
    k_Z = tf.broadcast_to(k_z, shape=(len(z_list), *k_z.shape))

    d_Z = tf.broadcast_to(
        tf.expand_dims(
            tf.expand_dims(
                z_list,
                axis=-1
            ),
            axis=-1
        ),
        k_Z.shape
    )

    phase_real = tf.math.real(k_Z) * d_Z
    phase_imag = tf.math.imag(k_Z) * d_Z

    H_real = tf.cos(phase_real) * tf.exp(-phase_imag)
    H_imag = tf.sin(phase_real) * tf.exp(-phase_imag)

    # Start: apply antialias filter to H
    # See paper 'Band-Limited Angular Spectrum Method for Numerical SImulation of Free-Space Propagation in Far and near fields'
    del_f_x = 1. / (2. * n_x * dx)
    del_f_y = 1. / (2. * n_y * dy)

    k_x_limit = k / tf.sqrt((2 * del_f_x * z_list) ** 2 + 1)
    k_y_limit = k / tf.sqrt((2 * del_f_y * z_list) ** 2 + 1)

    k_x_limit = tf.cast(k_x_limit, dtype=float_type)
    k_y_limit = tf.cast(k_y_limit, dtype=float_type)

    k_X_limit = tf.transpose(
        tf.broadcast_to(k_x_limit, shape=(*k_z.shape, len(z_list),)),
        perm=[2, 0, 1],
    )
    k_Y_limit = tf.transpose(
        tf.broadcast_to(k_y_limit, shape=(*k_z.shape, len(z_list),)),
        perm=[2, 0, 1],
    )

    k_XX = tf.broadcast_to(k_X, shape=(len(z_list), *k_X.shape))
    k_YY = tf.broadcast_to(k_Y, shape=(len(z_list), *k_Y.shape))

    kx_mask = (k_XX / k_X_limit) ** 2 + (k_YY / k) ** 2 <= 1.
    ky_mask = (k_YY / k_Y_limit) ** 2 + (k_XX / k) ** 2 <= 1.
    comb_mask = tf.logical_and(kx_mask, ky_mask)
    filter = tf.cast(comb_mask, dtype=float_type)
    # import matplotlib.pyplot as plt; plt.imshow(tf.math.abs(filter[0, :, :])); plt.show()


    H_real, H_imag = complex_mul(H_real, H_imag, filter, 0.)
    # Finish: apply antialias filter to H

    E_k_prop_real, E_k_prop_imag = complex_mul(
        *complex_split(U),
        H_real,
        H_imag,
    )
    # import matplotlib.pyplot as plt; plt.imshow(tf.math.abs(tf.complex(E_k_prop_real, E_k_prop_imag))[0, :, :]); plt.show()

    return tf.signal.ifft2d(tf.complex(E_k_prop_real, E_k_prop_imag))


def _pad(field, pad_factor=1.):
    n_x, n_y = np.shape(field)
    pad_x = int(n_x * pad_factor / 2)
    pad_y = int(n_y * pad_factor / 2)
    return tf.pad(field, paddings=tf.constant([[pad_x, pad_x], [pad_y, pad_y]]))


def _unpad(field, pad_factor=1.):
    if pad_factor == 0.:
        return field
    else:
        *_, n_x, n_y = np.shape(field)
        pad_x = int(n_x * pad_factor / (2 + 2 * pad_factor))
        pad_y = int(n_y * pad_factor / (2 + 2 * pad_factor))
        return field[:, pad_x:-pad_x, pad_y:-pad_y]


def propagate_padded(propagator, field, k, z_list, dx, dy, pad_factor=1.):
    padded_field = _pad(field, pad_factor)
    padded_propagated_field = propagator(padded_field, k, z_list, dx, dy,)
    propagated_field = _unpad(padded_propagated_field, pad_factor=pad_factor)
    return propagated_field
