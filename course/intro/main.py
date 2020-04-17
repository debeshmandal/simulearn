#!/usr/bin/env python3
import tensorflow as tf

graph = tf.Graph()

with graph.as_default():
    a = tf.constant([2], name='constant_a')
    b = tf.constant([3], name='constant_b')
    c = tf.add(a, b)

with tf.compat.v1.Session(graph=graph) as session:
    result = session.run(c)
    print(result)

