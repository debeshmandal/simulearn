"""
Script of examples taken from Udemy Tensorflow 2.0 course

https://colab.research.google.com/drive/1CQmEJihmMrTNfAiMO9QbWQdY4R7gygDo
"""
import tensorflow as tf
import numpy as np

# constants are rank-N tensors initialised using tf.constant()
tensor = tf.constant([
    [1, 2], 
    [3, 4]
])

print(f'Shape: {tensor.shape}\n')
print(f'Numpy Array:\n{tensor.numpy()}\n')

# variables are rank-N tensors initialised using tf.Variable()
variable = tf.Variable([
    [
        [0., 1., 2.],
        [1., 5., 8.],
    ],
    [
        [6., 0., 1.],
        [3., 2., 4.],
    ]
])
print(f'variable:\n{variable}\n')

# use assign to change values in a variable
variable[0, 0, 0].assign(5.)
print(f'variable after assigning:\n{variable}')

# Some operations
print(f'Tensor:\n{tensor}\n')
print(f'Tensor * 2:\n{tensor * 2}\n')
print(f'Tensor + 5:\n{tensor + 5}\n')
print(f'Tensor + Tensor:\n{tensor + tensor}\n')
print(f'Numpy Squared Tensor:\n{np.square(tensor)}\n')
print(f'Dot product:\n{np.dot(tensor, tensor)}\n')

# Strings are initialised using tf.constant()/tf.Variable() as well
string = tf.constant('Hello!')

# operations on strings are carried out using tf.strings.<func>(string)
print(f'String: {string}')
print(f'Length: {tf.strings.length(string)}\n')

# strings can also be stored in arrays
str_array = tf.constant([
    'Hello',
    'World!'
])

print(f"String Array: {str_array}\n")
print(f"String Array per-item:")
for item in str_array:
    print(item)

