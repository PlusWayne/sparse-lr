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

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b] = variables


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

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b] = variables


# sparse two class logistic regression
class SparseLogisticRegression(object):
    def __init__(self, feature_dim, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, 1), dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))

    def train_step(self, sparse_ids, sparse_vals, label, optimizer):
        with tf.GradientTape() as tape:
            z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals, combiner='sum')
            predict = tf.nn.bias_add(z, self.b)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
            # tf.print(loss)
        gradients = tape.gradient(loss, [self.w, self.b])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b]))

    def batch_sparse_data(self, train_id, train_val):
        # convert sparse batch matrix for embedding lookup
        # train_id with shape [batch_size, sparse_feature_number]
        # train_val with shape [batch_size, sparse_feature_value]
        indices, ids, values = [], [], []
        for i, (id, value) in enumerate(zip(train_id, train_val)):
            if len(id) == 0:
                indices.append((i, 0))
                ids.append(0)
                values.append(0.0)
                continue
            indices.extend([(i, t) for t in range(len(id))])
            ids.extend(id)
            values.extend(value)
        shape = (len(train_id), self.feature_dim)
        return indices, ids, values, shape

    def fit(self, train_id, train_val, label):
        indices, ids, values, shape = self.batch_sparse_data(train_id, train_val)
        sparse_ids = tf.SparseTensor(indices=indices, values=ids, dense_shape=shape)
        sparse_vals = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        optimizer = tf.optimizers.Adam()
        for i in range(self.iteration):
            self.train_step(sparse_ids, sparse_vals, label, optimizer)
            if i % 100 == 0:
                z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                  combiner='sum')
                predict = tf.nn.bias_add(z, self.b)
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
                template = "The training loss at {} iteration is {}"
                tf.print(template.format(i, loss))

    def predict(self, test_id, test_val):
        indices, ids, values, shape = self.batch_sparse_data(test_id, test_val)
        sparse_ids = tf.SparseTensor(indices=indices, values=ids, dense_shape=shape)
        sparse_vals = tf.SparseTensor(indices=indices, values=values, dense_shape=shape)
        z = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals, combiner='sum')
        predict = tf.nn.sigmoid(z)
        predict = tf.math.round(predict)
        return tf.reshape(predict, (-1, 1))

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b] = variables


if __name__ == '__main__':
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split
    import numpy as np
    import os

    """ test sparse logistic regression """
    # train_id = [[1, 2], [0, 1]]
    # train_val = [[1.0, 1.0], [1.0, 1.0]]
    # label = np.array([[1.0], [0.0]], dtype=np.float64)
    # print(label)
    # slr = SparseLogisticRegression(feature_dim=3)
    # slr.fit(train_id, train_val, label)

    """ test sparse logistic regression with digits data"""
    from scipy.sparse import csc_matrix

    raw_data = load_digits(n_class=2)
    X = np.array(raw_data.data, dtype=np.float64)
    y = np.array(raw_data.target, dtype=np.float64)


    def to_sparse_matrix(X):
        train_id = []
        train_val = []
        for x in X:
            t = csc_matrix(x)
            _, col = t.nonzero()
            val = x[col]
            train_id.append(col)
            train_val.append(val)
        return train_id, train_val


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    train_id, train_val = to_sparse_matrix(X_train)
    test_id, test_val = to_sparse_matrix(X_test)
    y_train = tf.reshape(y_train, (-1, 1))
    y_test = tf.reshape(y_test, (-1, 1))
    slr = SparseLogisticRegression(feature_dim=64)
    slr.fit(train_id, train_val, y_train)
    pred = slr.predict(test_id, test_val)

    # print accuracy
    tf.print('the accuarcy is {}'.format(tf.reduce_mean(tf.cast(tf.equal(pred, y_test), dtype=tf.float64))))

    """ test multi class logistic regression with digits data"""
    # raw_data = load_digits(n_class=10)
    # X = np.array(raw_data.data, dtype=np.float64)
    # y = np.array(raw_data.target, dtype=np.float64)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # y_train = tf.reshape(y_train, (-1, 1))
    # y_test = tf.reshape(y_test, (-1, 1))
    # lr = MultiClassLogisticRegression(feature_dim=64, class_dim=10, iteration=1000)
    # if 'parameters.txt' in os.listdir('.'):
    #     lr.load_model()
    # lr.fit(X_train, y_train)
    # lr.save_model()
    # pred = lr.predict(X_test)
    # tf.print(tf.reduce_mean(tf.cast(tf.equal(pred, y_test), dtype=tf.float64)))
