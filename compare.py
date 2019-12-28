import time
import unittest
import numpy as np
import tensorflow as tf

import AngularPropagateCPU as ap_normal
import AngularPropagateTensorflowCPU as ap_tf_cpu


class MyTestCase(unittest.TestCase):

    def test_true(self):
        self.assertEqual(True, True)

    def test_iffts(self):
        np.random.seed(69)
        intput_shape = (2, 2, 2)
        input = np.random.rand(*intput_shape) + 1j*np.random.rand(*intput_shape)

        np_out = np.fft.ifft2(np.copy(input), axes=(0, 1))

        tf_input = tf.convert_to_tensor(np.copy(input), dtype=tf.complex64)

        tf_input = tf.transpose(tf_input, (2, 0, 1))
        tf_out = tf.signal.ifft2d(tf_input).numpy()
        tf_out = np.transpose(tf_out, (1, 2, 0))



        self.assertTrue(np.allclose(np_out, tf_out))

    def test_simple(self):
        nx = np.random.randint(3, 4)
        ny = np.random.randint(3, 4)
        input_field = np.random.rand(nx, ny)

        zs = [0, 1.5]
        k = 1.
        dx = 0.001
        dy = 0.001

        result_normal = ap_normal.propagate_angular(field=np.copy(input_field), k=k, z_list=zs, dx=dx, dy=dy)
        result_tf_cpu = ap_tf_cpu.propagate_angular(field=np.copy(input_field), k=k, z_list=zs, dx=dx, dy=dy)

        print(np.max(result_tf_cpu-result_normal))

        self.assertTrue(np.allclose(result_normal, result_tf_cpu, atol=1e-4))

    def test_hard_compare(self):
        np.random.seed(69)
        input_shape = (np.random.randint(100, 200), np.random.randint(100, 200))
        input_field = np.random.rand(*input_shape) + 1j*np.random.rand(*input_shape)

        zs = np.random.rand(np.random.randint(10, 20)).tolist()
        k = 1.
        dx = np.random.rand(1)[0]
        dy = np.random.rand(1)[0]

        result_normal = ap_normal.propagate_angular(field=np.copy(input_field), k=k, z_list=zs, dx=dx, dy=dy)
        result_tf_cpu = ap_tf_cpu.propagate_angular(field=np.copy(input_field), k=k, z_list=zs, dx=dx, dy=dy)

        print(np.max(result_tf_cpu - result_normal))

        self.assertTrue(np.allclose(result_normal, result_tf_cpu, atol=1e-4))


    def test_compare_speed(self):
        np.random.seed(69)
        input_shape = (2**7, 2**7)
        input_field = np.random.rand(*input_shape) + 1j * np.random.rand(*input_shape)

        zs = np.random.rand(100).tolist()
        k = 1.
        dx = np.random.rand(1)[0]
        dy = np.random.rand(1)[0]

        t_0 = time.time()
        print("Started numpy implementation")
        result_normal = ap_normal.propagate_angular(field=np.copy(input_field), k=k, z_list=zs, dx=dx, dy=dy)
        t_1 = time.time()
        print("Completed numpy implementation")

        print("Started tensorflow implementation")
        result_tf_cpu = ap_tf_cpu.propagate_angular(field=np.copy(input_field), k=k, z_list=zs, dx=dx, dy=dy)
        t_2 = time.time()
        print("Completed tensorflow implementation")

        print("Maximum error: {}".format(np.max(result_tf_cpu - result_normal)))

        self.assertTrue(np.allclose(result_normal, result_tf_cpu, atol=1e-4))

        print("Time for numpy implementation: {}".format(t_1-t_0))
        print("Time for tensorflow implementation: {}".format(t_2-t_1))

    def test_tensor_vector_multiply(self):
        x = tf.constant([[[1], [1]], [[1], [1]], [[1], [1]],])
        print(x.shape)
        b = tf.constant([[[1,2,3]]])
        print(x.numpy())
        print((tf.multiply(x, b)).numpy())
        tf.broadcast_to(b, (0, 1))

if __name__ == '__main__':
    unittest.main()
