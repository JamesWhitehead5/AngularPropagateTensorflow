# import AngularPropagateTensorflow.AngularPropagateCPU as np_p
# import AngularPropagateTensorflow.AngularPropagateTensorflow as tf_p

import AngularPropagateCPU as np_p
import AngularPropagateTensorflow as tf_p

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

wavelength = 633e-9
dx = wavelength*2.
dy = dx
k = 2.*np.pi/wavelength


nx = 256
ny = nx
lens_aperture = nx*dx

f = 50.*lens_aperture
df = f/5.


x = tf.linspace(tf.constant(-lens_aperture/2., dtype=tf.float64), lens_aperture/2., num=nx)
y = tf.linspace(tf.constant(-lens_aperture/2., dtype=tf.float64), lens_aperture/2., num=ny)

Y, X = tf.meshgrid(y, x, indexing='xy')

phi = -k*tf.sqrt(f**2 + X**2 + Y**2)

field = tf.complex(tf.cos(phi), tf.sin(phi))


planes_tf = tf_p.propagate_padded(
    propagator=tf_p.propagate_angular_bw_limited,
    # propagator=tf_p.propagate_angular,
    field=field,
    k=k,
    z_list=[f - df, f, f + df],
    dx=dx,
    dy=dy,
    pad_factor=1.
)

planes_np = np_p.propagate_angular_padded(
    field=field.numpy(),
    k=k,
    z_list=[f - df, f, f + df],
    dx=dx,
    dy=dy,
)

tf_near, tf_perfect, tf_far = planes_tf
np_near, np_perfect, np_far = planes_np

plt.figure()
plt.imshow(phi.numpy() % (2.*np.pi))
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(np_near)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(np_perfect)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(np_far)**2)
plt.colorbar()
plt.show()


plt.figure()
plt.imshow(tf.abs(tf_near)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(tf_perfect)**2)
plt.colorbar()
plt.show()

plt.figure()
plt.imshow(tf.abs(tf_far)**2)
plt.colorbar()
plt.show()
