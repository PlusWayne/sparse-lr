import tensorflow as tf


class LogisticRegression(object):
    def __init__(self, feature_dim, class_dim, iteration=2000):
        self.feature_dim = feature_dim
        self.class_dim = class_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, self.class_dim), dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(self.class_dim,), value=0.01, dtype=tf.float64))

    def train_step(self, train, label, optimizer):
        label = tf.one_hot(label, self.class_dim)
        with tf.GradientTape() as tape:
            predict = tf.add(tf.matmul(train, self.w), self.b)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=predict))
        gradients = tape.gradient(loss, [self.w, self.b])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b]))

    def fit(self, train, label):
        optimizer = tf.optimizers.Adam()
        for i in range(self.iteration):
            self.train_step(train, label, optimizer)
            if i % 100 == 0:
                t_label = tf.one_hot(label, self.class_dim)
                predict = tf.add(tf.matmul(train, self.w), self.b)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t_label, logits=predict))
                tf.print(loss)

    def predict(self, train):
        predict = tf.add(tf.matmul(train, self.w), self.b)
        predict = tf.nn.sigmoid(predict)
        predict = tf.cast(tf.argmax(predict, axis=1), dtype=tf.int32)
        return predict


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np

    raw_data = load_digits(n_class=2)
    X = np.array(raw_data.data, dtype=np.float64)
    y = np.array(raw_data.target)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    lr = LogisticRegression(feature_dim=64, class_dim=2, iteration=2000)
    lr.fit(X_train, y_train)
    pred = lr.predict(X_test)
    tf.print(tf.reduce_mean(tf.cast(tf.equal(pred, y_test), dtype=tf.float64)))
