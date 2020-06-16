import numpy as np
import tensorflow as tf
from typing import List, Union


def _assert_2d(field):
    """Checks if the array is 2D"""
    assert len(np.shape(field)) == 2, "Variable fields must be a 2D array"


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

    if field.dtype == tf.complex128:
        float_type = tf.float64
    else:
        float_type = tf.float32

    z_list = tf.constant(z_list, dtype=float_type)

    # field = tf.convert_to_tensor(field, dtype=tf.complex128)

    # shape of the input field
    n_x, n_y = field.shape

    k_x = tf.constant(np.fft.fftfreq(n=n_x, d=dx) * 2 * np.pi, dtype=float_type)
    k_y = tf.constant(np.fft.fftfreq(n=n_y, d=dy) * 2 * np.pi, dtype=float_type)
    k_Y, k_X = tf.meshgrid(k_y, k_x, indexing='xy')

    # define wavenumber for each wavevector in the direction of propagation
    k_tensor = k*tf.ones(dtype=float_type, shape=field.shape)
    kz_squared = tf.cast(k_tensor ** 2 - k_X ** 2 - k_Y ** 2, dtype=field.dtype)
    k_z = tf.sqrt(kz_squared)

    # fourier transform the input field
    E_in_k = tf.signal.fft2d(field)
    # broadcast E_in_k along the 3rd dimension
    E_k = tf.broadcast_to(E_in_k, shape=(len(z_list), *E_in_k.shape))

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
    del_k_x = 2.*np.pi/(2*n_x*dx)
    del_k_y = 2*np.pi/(2*n_y*dy)

    k_x_limit = k * ((del_k_x * z_list / np.pi) + 1) ** (-0.5)
    k_y_limit = k * ((del_k_y * z_list / np.pi) + 1) ** (-0.5)

    k_X_limit = tf.transpose(
        tf.broadcast_to(k_x_limit, shape=(len(k_x), len(k_y), len(z_list), )),
        perm=[2, 0, 1],
    )
    k_Y_limit = tf.transpose(
        tf.broadcast_to(k_y_limit, shape=(len(k_x), len(k_y), len(z_list), )),
        perm=[2, 0, 1],
    )

    k_XX = tf.broadcast_to(k_X, shape=(len(z_list), *k_X.shape))
    k_YY = tf.broadcast_to(k_Y, shape=(len(z_list), *k_Y.shape))

    kx_mask = (k_XX/k_X_limit)**2 + (k_YY/k)**2 <= 1.
    ky_mask = (k_YY/k_Y_limit)**2 + (k_XX/k)**2 <= 1.

    comb_mask = tf.cast(kx_mask, dtype=float_type) * tf.cast(ky_mask, dtype=float_type)

    # DFT H to apply filter (nothing is shifted)
    H_fft = tf.signal.fft2d(tf.complex(H_real, H_imag))

    filter = tf.signal.ifftshift(tf.signal.ifftshift(comb_mask, axes=[1, 2, ]), axes=[1, 2, ])
    # filter = tf.cast(filter, dtype=field.dtype)
    H = tf.signal.ifft2d(tf.complex(*complex_matmul(
        tf.math.real(H_fft),
        tf.math.imag(H_fft),
        tf.math.real(filter),
        tf.math.imag(filter),
    )))

    H_real = tf.math.real(H)
    H_imag = tf.math.imag(H)

    # Finish: apply antialias filter to H
    # E_in_k x H
    E_k_prop_real, E_k_prop_imag = complex_matmul(
        tf.math.real(E_in_k),
        tf.math.imag(E_in_k),
        H_real,
        H_imag,
    )

    comp = tf.complex(E_k_prop_real, E_k_prop_imag)
    E_prop = tf.signal.ifft2d(comp)


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


def propagate_angular_padded(field, k, z_list, dx, dy, pad_factor=1.):
    """Pads the field with n-1 zeros to remove boundary reflections"""
    n_x, n_y = np.shape(field)
    pad_x = int(n_x*pad_factor)
    pad_y = int(n_y*pad_factor)

    padded_field = tf.pad(field, paddings=tf.constant([[pad_x, pad_x], [pad_y, pad_y]]))
    E_prop_padded = propagate_angular(padded_field, k, z_list, dx, dy)
    E_prop = E_prop_padded[:, pad_x:-pad_x, pad_y:-pad_y]

    return E_prop
