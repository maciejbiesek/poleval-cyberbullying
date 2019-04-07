import os

import numpy as np
import tensorflow as tf
from sklearn.metrics import f1_score
from tensorflow.contrib.layers import fully_connected, dropout
from tensorflow.contrib.rnn import DropoutWrapper, GRUCell

from config import Parser


class GRU:
    def __init__(self, dataset, is_train=True):
        parser = Parser()
        args = parser.get_sections(['GENERAL', 'RNN'])
        self.dataset = dataset
        self.num_hidden = int(args['num_hidden'])
        self.hidden_size = int(args['hidden_size'])
        self.num_classes = int(args['num_classes'])
        self.num_epochs = int(args['epochs'])
        self.max_sent_length = int(args['max_sent_length'])
        self.saved_dir = args['saved_dir']
        self.display_freq = int(args['display_frequency'])
        self.rnn_keep_prob = float(args['rnn_keep_prob'])
        self.dense_keep_prob = float(args['dense_keep_prob'])

        self.x = tf.placeholder(tf.int32, [None, self.max_sent_length], name="x")
        self.y = tf.placeholder(tf.int32, [None], name="y")
        self.rnn_keep_prob_ph = tf.placeholder(tf.float32, name="rnn_keep_prob")
        self.dense_keep_prob_ph = tf.placeholder(tf.float32, name="dense_keep_prob")
        self.class_weight = tf.placeholder(tf.float32, name="class_weight")
        self.global_step = tf.Variable(0, trainable=False)

        if is_train:
            with tf.name_scope("embeddings"):
                init_embeddings = tf.constant(self.dataset.embeddings, dtype=tf.float32)
                embeddings = tf.get_variable("embeddings", initializer=init_embeddings, trainable=False)
                x_emb = tf.nn.embedding_lookup(embeddings, self.x)

            with tf.name_scope("encoder"):
                forward_cells = GRUCell(self.num_hidden)
                backward_cells = GRUCell(self.num_hidden)
                forward_cells = DropoutWrapper(forward_cells, output_keep_prob=self.rnn_keep_prob_ph)
                backward_cells = DropoutWrapper(backward_cells, output_keep_prob=self.rnn_keep_prob_ph)
                (encoder_outputs_fw, encoder_outputs_bw), _ = tf.nn.bidirectional_dynamic_rnn(forward_cells,
                                                                                              backward_cells,
                                                                                              x_emb, dtype=tf.float32)
                output = tf.concat([encoder_outputs_fw, encoder_outputs_bw], axis=2)[:, -1, :]

            with tf.name_scope("dense"):
                dense = fully_connected(output, self.hidden_size, activation_fn=tf.nn.relu)
                drop = dropout(dense, self.dense_keep_prob_ph)
                logits = fully_connected(drop, self.num_classes, activation_fn=tf.nn.relu)
                self.y_pred = tf.argmax(logits, axis=-1, output_type=tf.int32, name="y_pred")

            with tf.name_scope("loss"):
                weights = tf.gather(self.class_weight, self.y)
                xent = tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=self.y, weights=weights)
                self.loss = tf.reduce_mean(xent)
                self.optimizer = tf.train.AdamOptimizer().minimize(self.loss, global_step=self.global_step)

            with tf.name_scope("accuracy"):
                correct_predictions = tf.equal(self.y_pred, self.y)
                self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")

    def train(self):
        train_batches, val_x, val_y = self.dataset.parse_dataset()
        class_weight = self.dataset.get_class_weight()
        max_acc, max_f1, min_loss = 0, 0, 100
        os.makedirs(self.saved_dir, exist_ok=True)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver(tf.global_variables())
            for epoch in range(self.num_epochs):
                print(f"Training epoch: {epoch + 1}")
                for x_batch, y_batch in train_batches:
                    _, iteration, loss, acc = sess.run([self.optimizer, self.global_step, self.loss, self.accuracy],
                                                       feed_dict={self.x: x_batch,
                                                                  self.y: y_batch,
                                                                  self.rnn_keep_prob_ph: self.rnn_keep_prob,
                                                                  self.dense_keep_prob_ph: self.dense_keep_prob,
                                                                  self.class_weight: class_weight})
                    if iteration % self.display_freq == 0:
                        print(f"Global step {iteration:4d}: \t Loss={loss:.2f}, \t Training Accuracy={acc:.01%}")

                # validation after every epoch
                step, loss_valid, acc_valid, y_pred = sess.run([self.global_step, self.loss, self.accuracy,
                                                                self.y_pred],
                                                               feed_dict={self.x: val_x, self.y: val_y,
                                                                          self.rnn_keep_prob_ph: 1.0,
                                                                          self.dense_keep_prob_ph: 1.0,
                                                                          self.class_weight: class_weight})
                f1_valid = f1_score(val_y, y_pred, average='macro')
                print("-" * 10)
                print(f"Epoch: {epoch + 1}")
                print("Model evaluation: \n"
                      f"validation loss: {loss_valid:.2f}, "
                      f"validation accuracy: {acc_valid:.01%}, "
                      f"validation F1: {f1_valid:.01%}")
                print("-" * 10)

                if loss_valid <= min_loss and f1_valid > max_f1:
                    min_loss, max_acc, max_f1 = loss_valid, acc_valid, f1_valid
                    filename = os.path.join(self.saved_dir, "saved_model.ckpt")
                    saver.save(sess, filename, global_step=step)
                    print("Model is saved.\n")

    def tagging(self, dataset):
        test_batches = dataset.parse_dataset()
        y_pred = []

        checkpoint_file = tf.train.latest_checkpoint(self.saved_dir)
        graph = tf.Graph()
        with graph.as_default():
            with tf.Session() as sess:
                saver = tf.train.import_meta_graph(f"{checkpoint_file}.meta")
                saver.restore(sess, checkpoint_file)

                x_placeholder = graph.get_operation_by_name("x").outputs[0]
                y_placeholder = graph.get_operation_by_name("y").outputs[0]
                rnn_keep_prob_placeholder = graph.get_operation_by_name("rnn_keep_prob").outputs[0]
                dense_keep_prob_placeholder = graph.get_operation_by_name("dense_keep_prob").outputs[0]
                class_weight_placeholder = graph.get_operation_by_name("class_weight").outputs[0]
                y_pred_placeholder = graph.get_operation_by_name("dense/y_pred").outputs[0]

                for x_batch, y_batch in test_batches:
                    y_pred_batch = sess.run(y_pred_placeholder, feed_dict={x_placeholder: x_batch,
                                                                           y_placeholder: y_batch,
                                                                           rnn_keep_prob_placeholder: 1.0,
                                                                           dense_keep_prob_placeholder: 1.0,
                                                                           class_weight_placeholder:
                                                                               np.array([1.0] * self.num_classes)})
                    y_pred.extend(y_pred_batch)

        return y_pred
