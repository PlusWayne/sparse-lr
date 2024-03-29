import tensorflow as tf


# two class fm
class FactorizationMachines(object):
    def __init__(self, feature_dim, factor_size, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, 1), dtype=tf.float64))
        self.v = tf.Variable(
            tf.random.normal(shape=(self.feature_dim, factor_size), mean=0.02, stddev=0.1, dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))

    def train_step(self, train, label, optimizer):
        with tf.GradientTape() as tape:
            linear_part = tf.add(tf.matmul(train, self.w), self.b)
            sum_part = tf.square(tf.matmul(train, self.v))
            square_part = tf.matmul(tf.square(train), tf.square(self.v))
            second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
            second_part = tf.reshape(second_part, (-1, 1))
            predict = linear_part + second_part
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
        gradients = tape.gradient(loss, [self.w, self.b, self.v])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b, self.v]))

    def fit(self, train, label):
        optimizer = tf.optimizers.Adam()
        for i in range(self.iteration):
            self.train_step(train, label, optimizer)
            if i % 100 == 0:
                # t_label = tf.one_hot(label, 1)
                linear_part = tf.add(tf.matmul(train, self.w), self.b)
                sum_part = tf.square(tf.matmul(train, self.v))
                square_part = tf.matmul(tf.square(train), tf.square(self.v))
                second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
                second_part = tf.reshape(second_part, (-1, 1))
                predict = linear_part + second_part
                loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
                tf.print(loss)

    def predict(self, train):
        linear_part = tf.add(tf.matmul(train, self.w), self.b)
        sum_part = tf.square(tf.matmul(train, self.v))
        square_part = tf.matmul(tf.square(train), tf.square(self.v))
        second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_part, square_part), 1)
        second_part = tf.reshape(second_part, (-1, 1))
        predict = linear_part + second_part
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
class SparseFactorizationMachines(object):
    def __init__(self, feature_dim, factor_size, iteration=2000):
        self.feature_dim = feature_dim
        self.iteration = iteration
        self.w = tf.Variable(tf.random.normal(shape=(self.feature_dim, 1), dtype=tf.float64))
        self.v = tf.Variable(tf.random.normal(shape=(self.feature_dim, factor_size), stddev=0.05, dtype=tf.float64))
        self.b = tf.Variable(tf.constant(shape=(1,), value=0.01, dtype=tf.float64))

    def train_step(self, sparse_ids, sparse_vals, label, optimizer):
        with tf.GradientTape() as tape:
            linear_part = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                        combiner='sum')
            embedding = tf.nn.embedding_lookup_sparse(params=self.v, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                      combiner='sum')
            embedding_square = tf.nn.embedding_lookup_sparse(params=tf.square(self.v), sp_ids=sparse_ids,
                                                             sp_weights=tf.square(sparse_vals), combiner='sum')
            sum_square = tf.square(embedding)
            second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_square, embedding_square), 1)
            second_part = tf.reshape(second_part, (-1, 1))
            predict = tf.nn.bias_add(linear_part + second_part, self.b)
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=predict))
            # tf.print(loss)
        gradients = tape.gradient(loss, [self.w, self.b, self.v])
        optimizer.apply_gradients(zip(gradients, [self.w, self.b, self.v]))

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
        optimizer = tf.optimizers.Adadelta()
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
        linear_part = tf.nn.embedding_lookup_sparse(params=self.w, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                    combiner='sum')
        embedding = tf.nn.embedding_lookup_sparse(params=self.v, sp_ids=sparse_ids, sp_weights=sparse_vals,
                                                  combiner='sum')
        embedding_square = tf.nn.embedding_lookup_sparse(params=tf.square(self.v), sp_ids=sparse_ids,
                                                         sp_weights=tf.square(sparse_vals), combiner='sum')
        sum_square = tf.square(embedding)
        second_part = 0.5 * tf.reduce_sum(tf.subtract(sum_square, embedding_square), 1)
        second_part = tf.reshape(second_part, (-1, 1))
        predict = tf.nn.bias_add(linear_part + second_part, self.b)
        predict = tf.sigmoid(predict)
        predict = tf.math.round(predict)
        return tf.reshape(predict, (-1, 1))

    def save_model(self):
        import pickle
        with open('parameters.txt', 'wb+') as file:
            pickle.dump([self.w, self.b, self.v], file)

    def load_model(self):
        import pickle
        with open('parameters.txt', 'rb') as file:
            variables = pickle.load(file)
            [self.w, self.b, self.v] = variables


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
    # slr = SparseFactorizationMachines(feature_dim=3, factor_size=2)
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
    slr = SparseFactorizationMachines(feature_dim=64, factor_size=8, iteration=1000)
    slr.load_model()
    slr.fit(train_id, train_val, y_train)
    pred = slr.predict(test_id, test_val)

    # print accuracy
    tf.print('the accuarcy is {}'.format(tf.reduce_mean(tf.cast(tf.equal(pred, y_test), dtype=tf.float64))))

    """ test two class fm  with digits data"""
    # raw_data = load_digits(n_class=2)
    # X = np.array(raw_data.data, dtype=np.float64)
    # y = np.array(raw_data.target, dtype=np.float64)
    #
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    # y_train = tf.reshape(y_train, (-1, 1))
    # y_test = tf.reshape(y_test, (-1, 1))
    # lr = FactorizationMachines(feature_dim=64, factor_size=8, iteration=1000)
    # lr.fit(X_train, y_train)
    # pred = lr.predict(X_test)
    # tf.print(tf.reduce_mean(tf.cast(tf.equal(pred, y_test), dtype=tf.float64)))
