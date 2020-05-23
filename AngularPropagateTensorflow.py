import numpy as np
import tensorflow as tf
from typing import List, Union


def _assert_2d(field):
    """Checks if the array is 2D"""
    assert len(np.shape(field)) == 2, "Variable fields must be a 2D array"


def fftfreq(n, d):
    # copied from np.fft.fftfreq implementation
    val = 1.0 / (n * d)
    # results = tf.zeros(shape=(n, ), dtype=tf.float64)
    N = (n - 1) // 2 + 1
    p1 = tf.range(0., N, dtype=tf.float64)
    # results[:N] = p1
    p2 = tf.range(-(n // 2) * 1., 0., dtype=tf.float64)
    # results[N:] = p2
    results = tf.concat((p1, p2[::-1]), axis=0)
    return results * val

def complex_matmul(a_real, a_imag, b_real, b_imag):
    real = a_real*b_real - a_imag*b_imag
    imag = a_real*b_imag + a_imag*b_real
    return real, imag


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
    # field = tf.convert_to_tensor(field, dtype=tf.complex128)

    # shape of the input field
    n_x, n_y = field.shape

    # # define spatial frequency vectors
    k_x = fftfreq(n=n_x, d=dx) * 2 * np.pi
    k_y = fftfreq(n=n_y, d=dy) * 2 * np.pi
    k_Y, k_X = tf.meshgrid(k_y, k_x, indexing='xy')
    # define wavenumber for each wavevector in the direction of propagation
    k_tensor = k*tf.ones(dtype=tf.float64, shape=field.shape)
    kz_squared = tf.cast(k_tensor ** 2 - k_X ** 2 - k_Y ** 2, dtype=tf.complex128)
    k_z = tf.sqrt(kz_squared)

    # fourier transform the input field
    E_in_k = tf.signal.fft2d(field)

    # broadcast E_in_k along the 3rd dimension
    # E_k = tf.keras.backend.repeat_elements(tf.expand_dims(E_in_k, axis=0), len(z_list), axis=0)
    E_k = tf.broadcast_to(E_in_k, shape=(len(z_list), *E_in_k.shape))

    # broadcast k_z into 3rd dimension
    # k_Z = tf.keras.backend.repeat_elements(tf.expand_dims(k_z, axis=0), len(z_list), axis=0)
    k_Z = tf.broadcast_to(k_z, shape=(len(z_list), *k_z.shape))


    d_Z = tf.broadcast_to(
        tf.expand_dims(
            tf.expand_dims(
                tf.constant(z_list,  dtype=tf.float64),
                axis=[-1]
            ),
            axis=[-1]
        ),
        k_Z.shape
    )

    phase_real = tf.math.real(k_Z) * d_Z
    phase_imag = tf.math.imag(k_Z) * d_Z

    phasor_real = tf.cos(phase_real) * tf.exp(-phase_imag)
    phasor_imag = tf.sin(phase_real) * tf.exp(-phase_imag)

    E_k_prop_real, E_k_prop_imag = complex_matmul(tf.math.real(E_in_k), tf.math.imag(E_in_k), phasor_real, phasor_imag)
    E_prop = tf.signal.ifft2d(tf.complex(E_k_prop_imag, E_k_prop_imag))
    E_prop = tf.transpose(E_prop, perm=[1, 2, 0])

    return E_prop


def propagate_angular_padded(field, k, z_list, dx, dy):
    """Pads the field with n-1 zeros to remove boundary reflections"""
    n_x, n_y = np.shape(field)
    pad_x = int(n_x)
    pad_y = int(n_y)

    padded_field = tf.Variable(tf.zeros(shape=(2 * pad_x + n_x, 2 * pad_y + n_y), dtype=tf.complex128))
    padded_field[pad_x:-pad_x, pad_y:-pad_y].assign(field)

    E_prop_padded = propagate_angular(padded_field, k, z_list, dx, dy)
    E_prop = E_prop_padded[pad_x:-pad_x, pad_y:-pad_y, :]

    return E_prop
