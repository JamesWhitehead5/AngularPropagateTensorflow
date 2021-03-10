# import AngularPropagateTensorflow.AngularPropagateCPU as np_p
# import AngularPropagateTensorflow.AngularPropagateTensorflow as tf_p


import AngularPropagateTensorflow as ags
import RayleighSomerfeldTensorflow as rs

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

wavelength = 633e-9
dx = wavelength*8.
dy = dx
k = 2.*np.pi/wavelength


n = 64
lens_aperture = n*dx

f = 50.*lens_aperture
df = f/5.


x = tf.linspace(tf.constant(-lens_aperture/2., dtype=tf.float64), lens_aperture/2., num=n)
y = tf.linspace(tf.constant(-lens_aperture/2., dtype=tf.float64), lens_aperture/2., num=n)

Y, X = tf.meshgrid(y, x, indexing='xy')

phi = -k*tf.sqrt(f**2 + X**2 + Y**2)

field = tf.complex(tf.cos(phi), tf.sin(phi))

dtype = {'real': tf.float64, 'complex': tf.complex128}

planes_as = ags.propagate_padded(
    propagator=ags.propagate_angular_bw_limited,
    # propagator=tf_p.propagate_angular,
    field=field,
    k=k,
    z_list=[f - df, f, f + df],
    dx=dx,
    dy=dy,
    pad_factor=1.
)

near_as, perfect_as, far_as = planes_as

near_rs = rs.propagate_scalar_total(starting_field=field, dd=dx, k=k, z=f - df, dtype=dtype)
perfect_rs = rs.propagate_scalar_total(starting_field=field, dd=dx, k=k, z=f, dtype=dtype)
far_rs = rs.propagate_scalar_total(starting_field=field, dd=dx, k=k, z=f + df, dtype=dtype)


plt.figure()
plt.imshow(phi.numpy() % (2.*np.pi))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(near_as)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(perfect_as)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(far_as)**2)
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(tf.abs(near_rs)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(perfect_rs)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(far_rs)**2)
plt.colorbar()
plt.show()
