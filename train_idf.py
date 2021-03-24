import tensorflow as tf
from data_utils import *
from sklearn.model_selection import train_test_split
from vd_cnn import VDCNN
import sys


NUM_CLASS = 12
BATCH_SIZE = 128
NUM_EPOCHS = 10

print("Building dataset...")
path_train = sys.argv[1]
x, y, size_dict= build_dataset(path_train)
print(size_dict)

# train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.15)
# with tf.Session() as sess:
#     model = VDCNN(size_dict, size_dict, NUM_CLASS)
#
#     sess.run(tf.global_variables_initializer())
#     saver = tf.train.Saver(tf.global_variables())
#
#     train_batches = batch_iter(train_x, train_y, BATCH_SIZE, NUM_EPOCHS)
#     num_batches_per_epoch = (len(train_x) - 1) // BATCH_SIZE + 1
#     max_accuracy = 0
#
#     for x_batch, y_batch in train_batches:
#         train_feed_dict = {
#             model.x: x_batch,
#             model.y: y_batch,
#             model.is_training: True
#         }
#         _, step, loss = sess.run([model.optimizer, model.global_step, model.loss], feed_dict=train_feed_dict)
#         if step % 100 == 0:
#             print("step {0}: loss = {1}".format(step, loss))
#         if step % 2000 == 0:
#             # Test accuracy with validation data for each epoch.
#             valid_batches = batch_iter(valid_x, valid_y, BATCH_SIZE, 1)
#             sum_accuracy, cnt = 0, 0
#             for valid_x_batch, valid_y_batch in valid_batches:
#                 valid_feed_dict = {
#                     model.x: valid_x_batch,
#                     model.y: valid_y_batch,
#                     model.is_training: False
#                 }
#                 accuracy = sess.run(model.accuracy, feed_dict=valid_feed_dict)
#                 sum_accuracy += accuracy
#                 cnt += 1
#             valid_accuracy = sum_accuracy / cnt
#             print("\nValidation Accuracy = {1}\n".format(step // num_batches_per_epoch, sum_accuracy / cnt))
#
#             # Save model
#             if valid_accuracy > max_accuracy:
#                 max_accuracy = valid_accuracy
#                 saver.save(sess, "model_tfidf/model.ckpt", global_step=step)
#                 print("Model is saved.\n")
