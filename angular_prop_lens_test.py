import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from AngularPropagateTensorflow.AngularPropagateCPU import propagate_angular_padded as np_pap
from AngularPropagateTensorflow.AngularPropagateTensorflow import propagate_angular_padded as tf_pap

lens_aperture = 1e-3
nx = 100
ny = 100
dx = lens_aperture/nx
dy = lens_aperture/ny
wavelength = 633e-9
k = 2.*np.pi/wavelength
f = 100e-3
df = 20e-3

x = tf.linspace(tf.constant(-nx/2., dtype=tf.float64), nx/2., num=nx)*dx
y = tf.linspace(tf.constant(-ny/2., dtype=tf.float64), ny/2., num=ny)*dy

Y, X = tf.meshgrid(y, x, indexing='xy')

phi = k*tf.sqrt(f**2 - X**2 - Y**2)

field = tf.complex(tf.cos(phi), tf.sin(phi))



planes_tf =  tf_pap(field=field, k=k, z_list=[f - df, f, f + df], dx=dx, dy=dy)
planes_np = np_pap(field=field.numpy(), k=k, z_list=[f - df, f, f + df], dx=dx, dy=dy)

tf_near, tf_perfect, tf_far = planes_tf
np_near, np_perfect, np_far = planes_np


plt.imshow(phi.numpy() % (2.*np.pi))
plt.show()

plt.imshow(tf.abs(np_near)**2)
plt.show()

plt.imshow(tf.abs(np_perfect)**2)
plt.show()

plt.imshow(tf.abs(np_far)**2)
plt.show()



plt.imshow(tf.abs(tf_near)**2)
plt.show()

plt.imshow(tf.abs(tf_perfect)**2)
plt.show()

plt.imshow(tf.abs(tf_far)**2)
plt.show()

