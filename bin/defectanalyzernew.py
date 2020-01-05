from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import numpy as np
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph


def read_tensor_from_image_file(file_name,
        input_height=299,
        input_width=299,
        input_mean=0,
        input_std=255):
    input_name = "file_reader"
    output_name = "normalized"
    file_reader = tf.read_file(file_name, input_name)
    if file_name.endswith(".png"):
        image_reader = tf.image.decode_png(
                file_reader, channels=3, name="png_reader")
    elif file_name.endswith(".gif"):
        image_reader = tf.squeeze(
                tf.image.decode_gif(file_reader, name="gif_reader"))
    elif file_name.endswith(".bmp"):
        image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
    else:
        image_reader = tf.image.decode_jpeg(
                file_reader, channels=3, name="jpeg_reader")
    float_caster = tf.cast(image_reader, tf.float32)
    dims_expander = tf.expand_dims(float_caster, 0)
    resized = tf.image.resize_bilinear(
            dims_expander, [input_height, input_width])
    normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
    sess = tf.Session()
    result = sess.run(normalized)

    return result


def load_labels(label_file):
    label = []
    proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
    for l in proto_as_ascii_lines:
        label.append(l.rstrip())
    return label


def predict(img, graph1, labels1, graph2, labels2):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation1 = graph1.get_operation_by_name(input_name)
    output_operation1 = graph1.get_operation_by_name(output_name)
    input_operation2 = graph2.get_operation_by_name(input_name)
    output_operation2 = graph2.get_operation_by_name(output_name)

    with tf.Session(graph=graph1) as sess:
        results1 = sess.run(output_operation1.outputs[0], {
            input_operation1.outputs[0]: img
            })
        with tf.Session(graph=graph2) as sess:
            results2 = sess.run(output_operation2.outputs[0], {
                input_operation2.outputs[0]: img
                })

            results1 = labels1[np.argmax(results1)].capitalize()
    results2 = labels2[np.argmax(results2)].capitalize()

    if results1 == 'Normal':
        if results2 == 'Normal':
            printval = 'Non discontinuity'

        else:
            printval = f'Discontinuity - {results2}'
    else:
        if results1 == 'Normal':
            printval = f'Discontinuity - {results1}'
        else:
            printval = f'Discontinuity - {results1} and {results2}'

    return printval
