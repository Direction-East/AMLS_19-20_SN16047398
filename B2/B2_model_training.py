import tensorflow as tf
import numpy as np
import os

num_labels = 5
img_size = 500
num_channels = 3
session = tf.Session()
x = tf.placeholder(tf.float32, shape = [None, img_size, img_size, num_channels], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, num_labels], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.05))
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):
    weights = create_weights(shape = [conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1,1,1,1], padding='SAME')
    biases = create_biases(num_filters)
    layer += biases
    layer = tf.nn.max_pool(value=layer,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')
    layer = tf.nn.relu(layer)

    return layer

def create_dropout_layer(layer, rate):
    layer = tf.nn.dropout(layer, rate)
    return layer

def create_flatten_layer(layer):
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer,[-1,num_features])

    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    weights = create_weights(shape=[num_inputs,num_outputs])
    biases = create_biases(num_outputs)

    layer = tf.matmul(input,weights)+biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer

filter_size_conv1 =3
num_filters_conv1 = 32

filter_size_conv2 =3
num_filters_conv2=32

filter_size_conv3 = 3
num_filters_conv3 = 64

fc_layer_size = 128

layer_conv1 = create_convolutional_layer(input=x, num_input_channels=num_channels, conv_filter_size=filter_size_conv1, num_filters=num_filters_conv1)

# layer_conv1_dropped = create_dropout_layer(layer_conv1, 0.2)

layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

# layer_conv2_dropped = create_dropout_layer(layer_conv1, 0.5)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)

# layer_conv3_dropped = create_dropout_layer(layer_conv3, 0.5)

layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements(),
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_labels,
                     use_relu=False)

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                    labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer())

def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0
batch_size = 50
epoch_size = 10

def determine_nextbatch(iteration, batch_size, x, y):
    if ((iteration+1)*batch_size > x.shape[0] and (iteration*batch_size < x.shape[0])):
        x_batch = np.vstack(x[iteration*batch_size:,:,:,:],x[:(iteration+1)*batch_size-x.shape[0],:,:,:])
        y_batch = np.vstack(y[iteration*batch_size:],y[:(iteration+1)*batch_size-x.shape[0]])
        return x_batch, y_batch
    if (iteration*batch_size > x.shape[0]):
        start = iteration*batch_size%x.shape[0]
        x_batch = x[start:start+batch_size,:,:,:]
        y_batch = y[start:start+batch_size]
        return x_batch, y_batch
    else:
        start = iteration*batch_size
        x_batch = x[start:start+batch_size,:,:,:]
        y_batch = y[start:start+batch_size]
        return x_batch, y_batch

saver = tf.train.Saver()
def B2_train(num_iteration, x_train_eyecolour, y_train_eyecolour, x_test_eyecolour, y_test_eyecolour, model_name):
    global total_iterations
    train_acc = 0
    for i in range(total_iterations,
                   total_iterations + num_iteration):

#         x_batch, y_true_batch, _, cls_batch = data.train.next_batch(batch_size)
#         x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(batch_size)
        x_batch, y_true_batch= determine_nextbatch(i, batch_size, x_train_eyecolour, y_train_eyecolour)
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_tr)
        # print(i)
        if ((i%epoch_size) == (epoch_size-1)):
            x_valid_batch, y_valid_batch = determine_nextbatch(i, batch_size, x_test_eyecolour, y_test_eyecolour)
            feed_dict_val = {x: x_valid_batch,
                      y_true: y_valid_batch}
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i//epoch_size)

            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            model_name = os.path.join('Dataset'+model_name)
            saver.save(session, os.path.join(os.getcwd(),model_name))
        if (i == total_iterations + num_iteration)
        train_acc = session.run(accuracy, feed_dict=feed_dict_tr)

    total_iterations += num_iteration
    return train_acc
