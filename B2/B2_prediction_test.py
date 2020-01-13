import tensorflow.compat.v1 as tf
import numpy as np

def show_test_accuracy(x_test, model_path, checkpoint_path, y_test):
    sess = tf.Session()
    tf.disable_eager_execution()
    ############# load saved tf model #####################
    saver = tf.train.import_meta_graph(model_path)
    print("loading saved model")
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))
    print("model successfully loaded")
    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")

    ############# prediction and test #####################
    num_correct = 0
    batch_size = 50
    total_test_num = len(x_test)
    num_batch =  total_test_num//batch_size
    for i in range(num_batch):
        print("testing {0}/{1} batch".format(i+1, num_batch))
        x_test_batch = x_test[(i*batch_size):((i+1)*batch_size),:,:,:]
        y_test_batch = np.zeros((x_test_batch.shape[0],5))
        feed_dict_testing = {x: x_test_batch, y_true: y_test_batch}
        results = sess.run(y_pred, feed_dict = feed_dict_testing)
        for j in range(results.shape[0]):
            if(np.argmax(results[j]) == np.argmax(y_test[j])):
                    num_correct+=1
        # print(round(num_correct/((i+1)*batch_size),2))
    accuracy = num_correct/(num_batch*batch_size)
    accuracy = round(accuracy,2)
    return accuracy
