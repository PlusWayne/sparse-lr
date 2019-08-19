import tensorflow as tf
import gzip


def parse_one_sample(line):
    line = list(map(lambda x: bytes(x, encoding='utf-8'), str(line, encoding='utf-8').split('\t')))
    label, time, duration = int(line[0]), int(line[1]), int(line[2])
    index = line[3::2]
    return label, time, duration, index


def to_tfrecords(file_path):
    tf_file_name = file_path[:-3] + '.tfrecord'
    file = gzip.open(file_path)
    with tf.io.TFRecordWriter(tf_file_name) as writer:
        while True:
            line = file.readline()
            if not line:
                break
            label, time, duration, list_index = parse_one_sample(line)
            tfrecord_feature = {
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'time': tf.train.Feature(int64_list=tf.train.Int64List(value=[time])),
                'duration': tf.train.Feature(int64_list=tf.train.Int64List(value=[duration])),
                'index': tf.train.Feature(bytes_list=tf.train.BytesList(value=list_index)),
            }
            example = tf.train.Example(features=tf.train.Features(feature=tfrecord_feature))
            serialized_example = example.SerializeToString()
            writer.write(serialized_example)
    print('Finished!')
    file.close()


if __name__ == '__main__':
    import argparse
    import multiprocessing
    import os
    import re

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default='.', help='input directory path for files')
    args = parser.parse_args()

    all_files_list = []
    for gz_file in os.listdir(args.input):
        if re.match('.*gz$', gz_file):
            all_files_list.append(gz_file)
    pool = multiprocessing.Pool(1)
    pool.map(to_tfrecords, all_files_list)
# example_ = tf.train.Example()
# example_.ParseFromString(serialized_example)
# print(example_.features)
