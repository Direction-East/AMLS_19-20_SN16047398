import tensorflow.compat.v1 as tf
import numpy as np

def predict_using_saved_tf_model(x_test, model_path, checkpoint_path):
    sess = tf.Session()
    tf.disable_eager_execution()
    ############# load saved tf model #####################
    saver = tf.train.import_meta_graph(model_path)
    print("loading saved model")
    saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

    graph = tf.get_default_graph()
    y_pred = graph.get_tensor_by_name("y_pred:0")
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")
    y_test_images = np.zeros((len(x_test),5))

    ############# prediction and test #####################
    feed_dict_testing = {x: x_test, y_true: y_test_images}
    results = sess.run(y_pred, feed_dict = feed_dict_testing)

    return results


def show_test_accuracy(x_test, model_path, checkpoint_path, y_test):
    results = predict_using_saved_tf_model(x_test, model_path, checkpoint_path)
    ############# show result #############################
    num_correct = 0
    for i in range(results.shape[0]):
        if(np.argmax(results[i]) == np.argmax(y_test[i])):
            num_correct+=1
    accuracy = num_correct/len(results)
    accuracy = round(accuracy,2)
    return accuracy
