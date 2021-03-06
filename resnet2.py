import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


TRAIN_EPOCHS=300
LEARNING_RATE=0.001
BATCH=100


def global_avg_pooling(x):
    return tf.reduce_mean(x, [1,2])


def max_pool(x):
    return tf.layers.max_pooling2d(x,[3,3],2)

def conv2d(inputs,filters,activation=tf.nn.relu,ks=3,stride=1):
    return tf.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=[ks,ks],
        strides=(stride,stride),
        padding='same',
        activation=activation)

def res_module(x,filters):
    scale=int(filters/x.shape[3].value)

    stride1 = 1
    stride2 = 1

    if scale == 2:
        stride1 = 2
        stride2 = 1
        short_cut = conv2d(x,filters,ks=1,stride=stride1)
    else:
        short_cut = x
        pass
    l1 = conv2d(x,filters,stride=stride1)
    l2 = conv2d(l1,filters,None,stride=stride2)

    return tf.nn.relu(short_cut+l2)

def build_resnet(x,stacks=3):
    filters = 64
    net = conv2d(x,filters,ks=7,stride=2)
    net = max_pool(net)
    for i in range(1,stacks*3+1):
        net = res_module(net,filters)
        if i%3==0:
            filters *= 2
    return net


dims = 28
num_of_class = 10
xs = tf.placeholder(tf.float32, [None, dims*dims])
ys = tf.placeholder(tf.float32, [None, num_of_class])

images = tf.reshape(xs, [-1, dims, dims, 1])
net = build_resnet(images)
out = global_avg_pooling(net)
w_out = tf.Variable(tf.random_normal([out.shape[1].value,num_of_class]))
b_out = tf.Variable(tf.constant(0.1, shape=[num_of_class]))
classify_layer = tf.matmul(out,w_out)+b_out

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=classify_layer, labels=ys))
opt = tf.train.AdamOptimizer(learning_rate=LEARNING_RATE)
train_op = opt.minimize(loss)
init = tf.global_variables_initializer()


mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

with tf.Session(config=None) as sess:
    sess.run(init)
    for i in range(TRAIN_EPOCHS):
        batch_xs, batch_ys = mnist.train.next_batch(BATCH)
        _, train_loss = sess.run([train_op,loss], feed_dict={xs: batch_xs, ys: batch_ys})
        if i%50==0:
            print('step:{},loss:{}'.format(i,train_loss))





