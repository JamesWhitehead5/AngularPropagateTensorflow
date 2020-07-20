# AngularPropagateTensorflow
Angular propagation implemented for the GPU using Tensorflow API


Example:
```python
  import AngularPropagateTensorflow as ap_tf
  import tensorflow as tf

  target_focal_length = 150e-6
  wavelength = 633e-9 # HeNe
  k = 2.*np.pi/wavelength
  lens_aperture = 50e-6
  dd = wavelength/2. # array element spacing
  slm_n = int(lens_aperture/dd)
  slm_shape = (slm_n, slm_n, )

  propagated_plane_distances = [target_focal_length, ]

  x = tf.linspace(tf.constant(-lens_aperture/2., dtype=tf.float64), lens_aperture/2., num=slm_shape[0])
  y = tf.linspace(tf.constant(-lens_aperture/2., dtype=tf.float64), lens_aperture/2., num=slm_shape[1])
  Y, X = tf.meshgrid(y, x, indexing='xy')
  phi = -k*tf.sqrt(target_focal_length**2 + X**2 + Y**2)
  complex_e_field = tf.complex(tf.cos(phi), tf.sin(phi))


  resultant_field_bw_lim = ap_tf.propagate_padded(
          propagator=ap_tf.propagate_angular_bw_limited, # propagate using bandwidth limited propagation
          field=complex_e_field,
          k=k,
          z_list=propagated_plane_distances,
          dx=dd,
          dy=dd,
          pad_factor=1.,
      )[0, :, :]


  resultant_field = ap_tf.propagate_padded(
          propagator=ap_tf.propagate_angular, # propagate using normal angular propagation
          field=complex_e_field,
          k=k,
          z_list=propagated_plane_distances,
          dx=dd,
          dy=dd,
          pad_factor=5.,
      )[0, :, :]
 ```
    
