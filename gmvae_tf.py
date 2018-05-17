import numpy as np
from scipy.stats import mode
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import xavier_initializer
import os.path

slim = tf.contrib.slim
mnist = input_data.read_data_sets("../data/mnist/", one_hot=True)

def log_bernoulli_with_logits(x, logits, eps=0.0, axis=-1):
    if eps > 0.0:
        max_val = np.log(1.0 - eps) - np.log(eps)
        logits = tf.clip_by_value(logits, -max_val, max_val,
                                  name='clipped_logit')
    return -tf.reduce_sum(
        tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=x), axis)

def log_normal(x, mu, var, eps=0.0, axis=-1):
    if eps > 0.0:
        var = tf.add(var, eps, name='clipped_var')
    return -0.5 * tf.reduce_sum(
        tf.log(2 * np.pi) + tf.log(var) + tf.square(x - mu) / var, axis)

def  qy_graph(x, k=10):
    """Network q(z|x)"""
    with slim.arg_scope([slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    reuse=tf.AUTO_REUSE):
        qy_logit = slim.fully_connected(x, 512, scope='fc1')
        qy_logit = slim.fully_connected(qy_logit, 512, scope='fc2')
        #add sbp_dropout layer in this
        qy_logit = slim.fully_connected(qy_logit, k, activation_fn=None, scope='logit')
        qy = tf.nn.softmax(qy_logit, name='prob')
    return qy_logit, qy

def qz_graph(x, y):
    with slim.arg_scope([slim.fully_connected],
                    activation_fn=tf.nn.relu,
                    weights_initializer=xavier_initializer(),
                    biases_initializer=tf.zeros_initializer(),
                    reuse=tf.AUTO_REUSE):
        mu_logvar = tf.concat([x,y],1, name='xy/concat')
        mu_logvar = slim.fully_connected(mu_logvar, 512, scope='fc3')
        mu_logvar = slim.fully_connected(mu_logvar, 512, scope='fc4')
        mu_logvar = slim.fully_connected(mu_logvar, 128, activation_fn=None, scope='fc5')
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)
        stddev = tf.sqrt(tf.exp(logvar))

        # Draw a z from the distribution
        epsilon = tf.random_normal(tf.shape(stddev))
        z = mu + tf.multiply(stddev, epsilon)
        return z, mu, logvar

def decoder(z, y):
    with slim.arg_scope([slim.fully_connected],
                        activation_fn=tf.nn.relu,
                        weights_initializer=xavier_initializer(),
                        biases_initializer=tf.zeros_initializer(),
                        reuse=tf.AUTO_REUSE): 
        # ---p(z)
        mu_logvar = slim.fully_connected(y, 128, activation_fn=None, scope='de_1')
        mu, logvar = tf.split(mu_logvar, num_or_size_splits=2, axis=1)
        # ---p(x)
        x_logit = slim.fully_connected(z, 512, scope='defc1')
        x_logit = slim.fully_connected(x_logit, 512, scope='defc2')
        x_logit = slim.fully_connected(x_logit, 784, activation_fn=None, scope='defc3')
        
    return mu, logvar, x_logit

def labeled_loss(x, px_logit, z, zm, zv, zm_prior, zv_prior):
    xy_loss = -log_bernoulli_with_logits(x, px_logit)
    xy_loss += log_normal(z, zm, zv) - log_normal(z, zm_prior, zv_prior)
    return xy_loss - np.log(0.1)

def test_acc(mnist, sess, qy_logit):
    logits = sess.run(qy_logit, feed_dict={'x:0': mnist.test.images})
    cat_pred = logits.argmax(1)
    real_pred = np.zeros_like(cat_pred)
    for cat in range(logits.shape[1]):
        idx = cat_pred == cat
        lab = mnist.test.labels.argmax(1)[idx]
        if len(lab) == 0:
            continue
        real_pred[cat_pred == cat] = mode(lab).mode[0]
    return np.mean(real_pred == mnist.test.labels.argmax(1))

def stream_print(f, string, pipe_to_file=True):
    print(string)
    if pipe_to_file and f is not None:
        f.write(string + '\n')
        f.flush()


def open_file(fname):
    if fname is None:
        return None
    else:
        i = 0
        while os.path.isfile('{:s}.{:d}'.format(fname, i)):
            i += 1
        return open('{:s}.{:d}'.format(fname, i), 'w')

def train(log_file, data, sess_info, epochs):
    (sess, qy_logit, nent, loss, train_step) = sess_info
    f = open_file(log_file)
    iterep = 500
    for i in range(iterep * epochs):
        sess.run(train_step, feed_dict={'x:0': mnist.train.next_batch(100)[0]})
        #message='i={:d}'.format(i + 1)
        #progbar(i, iterep, message)
        if (i + 1) % iterep == 0:
            a, b = sess.run([nent, loss], feed_dict=
                            {'x:0': mnist.train.images[np.random.choice(50000, 10000)]})
            c, d = sess.run([nent, loss], feed_dict={'x:0': mnist.test.images})
            a,b,c,d = -a.mean(), b.mean(), -c.mean(), d.mean()
            e = test_acc(mnist, sess, qy_logit)
            string = ('{:>10s},{:>10s},{:>10s},{:>10s},{:>10s},{:>10s}'
                      .format('tr_ent', 'tr_loss', 't_ent', 't_loss', 't_acc', 'epoch'))
            stream_print(f, string, i <= iterep)
            string = ('{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10.2e},{:10d}'
                      .format(a, b, c, d, e, (i + 1) // iterep))
            stream_print(f, string)
    if f is not None: 
        f.close()

tf.reset_default_graph()
x = tf.placeholder(tf.float32,[None, 784], name='x')

with tf.name_scope('x_binarized'):
    xb = tf.cast(tf.greater(x, tf.random_uniform(tf.shape(x), 0, 1)), tf.float32)
with tf.name_scope('y_'):
    y_ = tf.fill(tf.stack([tf.shape(x)[0], 10]), 0.0)

qy_logit, qy = qy_graph(xb)
z, zm, zv, zm_prior, zv_prior, px_logit = [[None] * 10 for i in range(6)]
for i in range(10):
    with tf.name_scope('graphs/hot_at{:d}'.format(i)):
        y = tf.add(y_, tf.constant(np.eye(10)[i], dtype=tf.float32, name='hot_at_{:d}'.format(i)))
        z[i], zm[i], zv[i] = qz_graph(xb, y)
        zm_prior[i], zv_prior[i], px_logit[i] = decoder(z[i], y)

with tf.name_scope('loss'):
    with tf.name_scope('neg_entropy'):
        nent = tf.reduce_sum(qy * tf.nn.log_softmax(qy_logit), 1)
    losses = [None] * 10
    for i in range(10):
        with tf.name_scope('loss_at{:d}'.format(i)):
            losses[i] = labeled_loss(xb, px_logit[i], z[i], zm[i], tf.exp(zv[i]), zm_prior[i], tf.exp(zv_prior[i]))
    with tf.name_scope('final_loss'):
        loss = tf.add_n([nent] + [qy[:, i] * losses[i] for i in range(10)])

train_step = tf.train.AdamOptimizer().minimize(loss)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess_info = (sess, qy_logit, nent, loss, train_step)
train('logs/gmvae_mnist.log', mnist, sess_info, epochs=100)
