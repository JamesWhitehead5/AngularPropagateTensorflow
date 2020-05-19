import numpy as np
import tensorflow as tf

from typing import List, Union


def _assert_2d(field):
    """Checks if the array is 2D"""
    assert len(np.shape(field)) == 2, "Variable fields must be a 2D array"


def propagate_angular(field: np.ndarray, k: float, z_list: List[float], dx: float, dy: float):
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
    field = tf.convert_to_tensor(field, dtype=tf.complex128)

    # shape of the input field
    n_x, n_y = field.shape

    # define spatial frequency vectors
    k_x = tf.Variable(np.fft.fftfreq(n=n_x, d=dx) * 2 * np.pi, dtype=tf.complex128, trainable=False)
    k_y = tf.Variable(np.fft.fftfreq(n=n_y, d=dy) * 2 * np.pi, dtype=tf.complex128, trainable=False)
    k_Y, k_X = tf.meshgrid(k_y, k_x, indexing='xy')

    # fourier transform the input field
    E_in_k = tf.signal.fft2d(field)

    # define wavenumber for each wavevector in the direction of propagation
    k_z = tf.sqrt(k ** 2 - k_X ** 2 - k_Y ** 2)

    # broadcast E_in_k along the 3rd dimension
    E_k = tf.keras.backend.repeat_elements(E_in_k[np.newaxis, :, :], len(z_list), axis=0)


    # broadcast k_z into 3rd dimension
    k_Z = tf.keras.backend.repeat_elements(k_z[np.newaxis, :, :], len(z_list), axis=0)

    d_Z = tf.broadcast_to(tf.constant(z_list, shape=(len(z_list), 1, 1), dtype=tf.complex128), k_Z.shape)

    phase = k_Z * d_Z
    E_k_prop = E_k * np.exp(1j * phase)
    E_prop = tf.signal.ifft2d(E_k_prop)
    E_prop = tf.transpose(E_prop, perm=[1, 2, 0])
    #E_prop = E_k_prop

    return E_prop.numpy()


def propagate_angular_padded(field, k, z_list, dx, dy):
    """Pads the field with n-1 zeros to remove boundary reflections"""

    _assert_2d(field)
    n_x, n_y = np.shape(field)
    pad_x = int(n_x)
    pad_y = int(n_y)

    padded_field = np.zeros(shape=(2 * pad_x + n_x, 2 * pad_y + n_y), dtype=complex)
    padded_field[pad_x:-pad_x, pad_y:-pad_y] = field

    E_prop_padded = propagate_angular(padded_field, k, z_list, dx, dy)
    E_prop = E_prop_padded[pad_x:-pad_x, pad_y:-pad_y, :]

    return E_prop