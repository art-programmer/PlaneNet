import tensorflow as tf
import numpy as np

sg = tf.contrib.bayesflow.stochastic_graph
st = tf.contrib.bayesflow.stochastic_tensor

def createModel(x):
    f = tf.contrib.layers.fully_connected(x, 10)
    y = tf.contrib.layers.fully_connected(f, 10, activation_fn=None)
    c = tf.contrib.layers.fully_connected(f, 10, activation_fn=None)    
    return y, c

def buildLoss(x, y, c):
    selection = st.StochasticTensor(tf.contrib.distributions.Bernoulli(logits=c))
    yc = y * tf.cast(selection, tf.float32)
    #distr = tf.contrib.distributions.Bernoulli(logits = c)
    #yc = y * tf.cast(tf.contrib.distributions.Bernoulli(probs = tf.sigmoid(c)).sample(), tf.float32)
    #yc = tf.concat([x, x], axis=1) * tf.cast(tf.contrib.distributions.Bernoulli(probs = tf.sigmoid(c)).sample(), tf.float32)
    #sparsity_loss = tf.reduce_mean(1 - tf.sigmoid(plane_confidence_pred)) * 100
    
    diff = tf.abs(tf.expand_dims(x, -1) - tf.expand_dims(yc, -2))
    diff_loss = tf.reduce_mean(tf.reduce_min(diff, axis=-1), axis=1, keep_dims=True)
    confidence_loss = tf.reduce_mean(tf.sigmoid(c))
    #loss = diff_loss + confidence_loss
    diff_loss = sg.surrogate_loss([diff_loss])
    diff_loss = tf.reduce_mean(diff_loss)
    loss = diff_loss * 100 + confidence_loss
    #loss = tf.reduce_mean(sg.surrogate_loss([diff_loss])) + confidence_loss
    return loss, diff_loss, confidence_loss, yc

if __name__=='__main__':
    batchSize = 16
    with st.value_type(st.SampleValue()):
    #with True:
        x = tf.placeholder(tf.float32,shape=(batchSize, 5),name='x')
        y, c = createModel(x)
        loss, diff_loss, confidence_loss, yc = buildLoss(x, y, c)
        pass
    

    optimizer = tf.train.AdamOptimizer(1e-4)
    train_op = optimizer.minimize(loss)


    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())
    with tf.Session() as sess:
        sess.run(init_op)
        for i in xrange(10000):
            _, total_loss, loss_1, loss_2 = sess.run([train_op, loss, diff_loss, confidence_loss], feed_dict = {x: np.random.random((batchSize, 5))})
            if i % 100 == 0:
                print((total_loss, loss_1, loss_2))
                pass
            continue
        _, total_loss, x_, y_, c_, yc_ = sess.run([train_op, loss, x, y, c, yc], feed_dict = {x: np.tile(np.expand_dims(np.arange(5) + 1, 0), [batchSize, 1])})
        print(x_[0])
        print(y_[0])
        print(c_[0])
        print(yc_[0])        
        pass
        
