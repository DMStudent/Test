# coding: utf-8

import tensorflow as tf


class TCNNConfig(object):
    """CNN配置参数"""

    embedding_dim = 64  # 词向量维度
    seq_length = 30  # 序列长度
    num_classes = 3  # 类别数
    num_filters = 256  # 卷积核数目
    kernel_size = 5  # 卷积核尺寸
    vocab_size = 5000  # 词汇表达小

    hidden_dim = 256  # 全连接层神经元

    dropout_keep_prob = 0.5  # dropout保留比例
    learning_rate = 1e-3  # 学习率

    batch_size = 1000  # 每批训练大小
    num_epochs = 10  # 总迭代轮次

    print_per_batch = 100  # 每多少轮输出一次结果
    save_per_batch = 100  # 每多少轮存入tensorboard


class TextCNN(object):
    """文本分类，CNN模型"""

    def __init__(self, config, input_x, input_y):
        self.config = config

        # 三个待输入的数据
        self.input_x = input_x
        self.input_y = input_y
        self.keep_prob = config.dropout_keep_prob

        self.cnn()

    def cnn(self):
        """CNN模型"""
        # 词向量映射
        with tf.device('/cpu:0'):
            self.embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            self.embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            self.conv = tf.layers.conv1d(self.embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
            # global max pooling layer
            self.gmp = tf.reduce_max(self.conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            self.fc = tf.layers.dense(self.gmp, self.config.hidden_dim, name='fc1')
            self.fc = tf.nn.dropout(self.fc, self.keep_prob)
            self.fc = tf.nn.relu(self.fc)

            # 分类器
            self.logits = tf.layers.dense(self.fc, self.config.num_classes, name='fc2')
            self.y_pred_score = tf.nn.softmax(self.logits)
            self.y_pred_cls = tf.argmax(self.y_pred_score, 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            #cross_entropy = tf.nn.weighted_cross_entropy_with_logits(logits=self.logits, targets=self.input_y, pos_weight=0)
            # your class weights
            class_weights = tf.constant([[1.0, 1.0, 1.0]])
            y_float = tf.cast(self.input_y, tf.float32)
            # deduce weights for batch samples based on their true label
            weights = tf.reduce_sum(class_weights * y_float, axis=1)
            # compute your (unweighted) softmax cross entropy loss
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            # apply the weights, relying on broadcasting of the multiplication
            weighted_losses = unweighted_losses * weights
            # reduce the result to get your final loss
            cross_entropy = tf.reduce_mean(weighted_losses)

            tv = tf.trainable_variables()  # 得到所有可以训练的参数，即所有trainable=True 的tf.Variable/tf.get_variable
            regularization_cost = 0.001 * tf.reduce_sum([tf.nn.l2_loss(v) for v in tv])  # 0.001是lambda超参数

            self.loss = tf.reduce_mean(cross_entropy)+regularization_cost
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            self.correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))


    def char_cnn(self):
        """CNN模型"""
        conv_layers = [[256, 7, 3],
                       [256, 7, 3],
                       [256, 3, None],
                       [256, 3, None],
                       [256, 3, None],
                       [256, 3, 3]]

        fc_layers = [1024, 1024]

        # 词向量映射
        with tf.device('/cpu:0'):
            embedding = tf.get_variable('embedding', [self.config.vocab_size, self.config.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(embedding, self.input_x)

        # with tf.name_scope("cnn"):
        #     # CNN layer
        #     conv = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, name='conv')
        #     # global max pooling layer
        #     gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
        with tf.name_scope("cnn1"):
            # CNN layer
            conv1 = tf.layers.conv1d(embedding_inputs, self.config.num_filters, self.config.kernel_size, strides=1, padding='same', name='conv1')
            # global max pooling layer
            max_pool_1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=4, strides=4, padding='same', name='pool_1')

        with tf.name_scope("cnn2"):
            # CNN layer
            conv2 = tf.layers.conv1d(max_pool_1, self.config.num_filters, self.config.kernel_size, strides=1, padding='same',  name='conv2')
            # global max pooling layer
            max_pool_2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=4, strides=4, padding='same', name='pool_2')

        with tf.name_scope("cnn3"):
            # CNN layer
            conv3 = tf.layers.conv1d(max_pool_2, self.config.num_filters, self.config.kernel_size, strides=1, padding='same',  name='conv3')
            # global max pooling layer
            # max_pool_3 = tf.layers.max_pooling1d(inputs=conv3, pool_size=50, strides=50, padding='same', name='pool_3')
            gmp = tf.reduce_max(conv3, reduction_indices=[1], name='gmp')



        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden_dim, name='fc1')
            fc = tf.nn.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.num_classes, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
