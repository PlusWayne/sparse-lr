import tensorflow as tf


# multi-class logistic regression
class MultiClassLogisticRegression(object):
    def __init__(self, feature_dim, class_dim, iteration=2000):
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, self.class_dim), dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(self.class_dim,), value=0.01, dtype=tf.float64))

    def train_step(self, train, label, optimizer):
        with tf.GradientTape() as tape:
            predict = tf.add(tf.matmul(train, self.w), self.b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict))
        gradients = tape.gradient(loss, [self.w, self.b])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b]))

    def fit(self, train, label):
        optimizer = tf.optimizers.Adam()
        label = tf.one_hot(tf.cast(label, tf.int64), self.class_dim)
        for i in range(self.iteration):
            self.train_step(train, label, optimizer)
            if i % 100 == 0:
                predict = tf.add(tf.matmul(train, self.w), self.b)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict))
                tf.print(loss)

    def predict(self, train):
        predict = tf.add(tf.matmul(train, self.w), self.b)
        predict = tf.nn.sigmoid(predict)
        predict = tf.cast(tf.argmax(predict, axis=1), dtype=tf.float64)
        return tf.reshape(predict, (-1, 1))


# two class logistic regression
class LogisticRegression(object):
    def __init__(self, feature_dim, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, 1), dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))

    def train_step(self, train, label, optimizer):
        with tf.GradientTape() as tape:
            predict = tf.add(tf.matmul(train, self.w), self.b)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
        gradients = tape.gradient(loss, [self.w, self.b])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b]))

    def fit(self, train, label):
        optimizer = tf.optimizers.Adam()
        for i in range(self.iteration):
            self.train_step(train, label, optimizer)
            if i % 100 == 0:
                # t_label = tf.one_hot(label, 1)
                predict = tf.add(tf.matmul(train, self.w), self.b)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
                tf.print(loss)

    def predict(self, train):
        predict = tf.add(tf.matmul(train, self.w), self.b)
        predict = tf.nn.sigmoid(predict)
        predict = tf.math.round(predict)
        return tf.reshape(predict, (-1, 1))


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np

    raw_data = load_digits(n_class=10)
    X = np.array(raw_data.data, dtype=np.float64)
    y = np.array(raw_data.target, dtype=np.float64)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    y_train = tf.reshape(y_train, (-1, 1))
    y_test = tf.reshape(y_test, (-1, 1))
    lr = MultiClassLogisticRegression(feature_dim=64, class_dim=10, iteration=10000)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    tf.print(tf.reduce_mean(tf.cast(tf.equal(pred, y_test), dtype=tf.float64)))
