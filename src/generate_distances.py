import argparse
import os
import random
import sys

import tensorflow as tf
from scipy import misc
import numpy as np
from src import facenet

def main(args):

    image_size = args.image_size
    def no_align(image_path):
        img = misc.imread(image_path, mode='RGB')
        img = misc.imresize(img, (image_size, image_size), interp='bilinear')
        img = facenet.prewhiten(img)
        return img

    def no_align(image_path):
        def prewhiten(x):
            mean = tf.keras.backend.mean(x)
            std = tf.keras.backend.std(x)
            std_adj = tf.keras.backend.maximum(std, 1.0/tf.keras.backend.sqrt(tf.size(x)))
            y = tf.multiply(tf.subtract(x, mean), 1 / std_adj)
            return y
        image_string = tf.read_file(image_path)
        image_decoded = tf.cast(tf.image.decode_jpeg(image_string, channels=3), tf.int32)
        image_resized = tf.image.resize_images(image_decoded, (image_size, image_size))
        image_prewhitened = prewhiten(image_resized)
        return image_prewhitened


    def gen_dataset(args):
        dir_path = args.dir_path
        #Add logic to test out the model by giving reference images and test images
        classes = os.listdir(dir_path)
        n_reference = 10
        n_test = 10
        for cls in classes:
            filenames = []
            img_list = os.listdir(os.path.join(dir_path,cls))
            for i in range(n_reference):
                filenames.append(os.path.join(dir_path,cls,img_list[i]))
            for j in range(n_test):
                filenames.append(os.path.join(dir_path,cls,img_list[n_reference+j]))
        # now we have filenames - list(classes*(n_reference+n_test)) image paths
        filenames = tf.constant(filenames)
        dataset = tf.data.Dataset.from_tensor_slices(filenames)
        dataset = dataset.map(no_align, num_parallel_calls=4)
        dataset = dataset.batch(n_test + n_reference)
        return dataset

    def get_embeddings(args):
        dataset = gen_dataset(args)

        with tf.Graph().as_default():
            with tf.Session() as sess:
                # Load the model
                facenet.load_model(args.model)

                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")

                # Run forward pass to calculate embeddings
                feed_dict = {images_placeholder: dataset, phase_train_placeholder: False}
                emb = sess.run(embeddings, feed_dict=feed_dict)
                print(emb)

                nrof_images = len(args.image_files)

    get_embeddings(args)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('dir_path',type=str)
    parser.add_argument('model', type=str,
                        help='Could be either a directory containing the meta_file and ckpt_file or a model protobuf (.pb) file')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))


