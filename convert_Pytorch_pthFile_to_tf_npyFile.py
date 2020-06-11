# import tensorflow.compat.v1 as tf 
# import tensorflow as tf
import torch
import numpy as np
# from tensorflow.python import pywrap_tensorflow
# from tensorflow.python.platform import gfile


# def convert(bin_path, ckptpath):
#     with tf.Session() as sess:
#         for var_name, value in torch.load(bin_path, map_location='gpu').items():
#             tf.Variable(initial_value=value.data.numpy(), name=var_name)
#         saver = tf.train.Saver()
#         sess.run(tf.global_variables_initializer())
#         saver.save(sess, ckpt_path)

bin_path = './resnet34-333f7ec4.pth'
ckpt_path = 'bert_model.ckpt'

dictionary = {}
# with tf.Session() as sess:
i=0
for var_name, value in torch.load(bin_path).items():
    # print(i)
    # try:
    #     if len(value.shape==4):
    #         dict[var_name] = np.transpose(value.data.numpy(),[0,2,3,1])
    #     else:
    #         dict[var_name] = value.data.numpy()
    # except:
    #     print('\n')
    #     print(var_name)
    #     # tf.Variable(value.data.numpy(), name=var_name)
    # print(var_name)
    # print(dict[var_name].shape)
    # print(value.shape)

    numpy_value = value.data.numpy()
    if len(numpy_value.shape)==4:
        numpy_value = np.transpose(numpy_value, [2,3,1,0])
    dictionary[var_name] = numpy_value
print(type(dictionary))

np.save('resnet34.npy', dictionary)
data_dict = np.load('./resnet34.npy', allow_pickle=True).item()
# print(type(data_dict))
print(data_dict['layer1.0.conv1.weight'].shape)


    # sess.run(tf.global_variables_initializer())
# 
    # saver = tf.train.Saver()
    # saver.save(sess, ckpt_path)


# with tf.Session() as sess:
    # tf.Variable(np.random.rand(1,2,3),name='test_var')

    # sess.run(tf.global_variables_initializer())
    # saver = tf.train.Saver()
    # saver.save(sess, 'test.ckpt')

# import os

# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('bert_model.ckpt.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./'))

#     print('0')
#     reader = pywrap_tensorflow.NewCheckpointReader('./')

#     print('1')
#     var_to_shape_map = reader.get_variable_to_shape_map()
#     print('2')
#     for key in var_to_shape_map:
#         print("tensor_name: ", key)
#         print(reader.get_tensor(key)) 


# graph = tf.get_default_graph()
# graphdef = graph.as_graph_def()
# _ = tf.train.import_meta_graph("./bert_model.ckpt.meta")
# summary_write = tf.summary.FileWriter("./" , graph)


# with tf.Session() as sess:
#     saver = tf.train.import_meta_graph('bert_model.ckpt.meta')
#     saver.restore(sess, tf.train.latest_checkpoint('./'))

#     g_list = tf.global_variables()
#     # for g in g_list:
#         # print(sess.run(g))

#     tf.summary.FileWriter(logdir='./log',graph=sess.graph)

# def bn_layer(x,is_training, name='BatchNorm',moving_decay=0.9,eps=1e-5):
#         # 获取输入维度并判断是否匹配卷积层(4)或者全连接层(2)
#         shape = x.shape
#         assert len(shape) in [2,4]

#         param_shape = shape[-1]
#         with tf.variable_scope(name):
        
#             # 声明BN中唯一需要学习的两个参数，y=gamma*x+beta
#             gamma = tf.get_variable('gamma',param_shape,initializer=tf.constant_initializer(1))
#             beta  = tf.get_variable('beta', param_shape,initializer=tf.constant_initializer(0))
            
#             # 计算当前整个batch的均值与方差
#             axes = list(range(len(shape)-1))
#             batch_mean, batch_var = tf.nn.moments(x,axes,name='moments')

#             # 采用滑动平均更新均值与方差
#             ema = tf.train.ExponentialMovingAverage(moving_decay)

#             def mean_var_with_update():
#                 ema_apply_op = ema.apply([batch_mean,batch_var])
#                 with tf.control_dependencies([ema_apply_op]):
#                     return tf.identity(batch_mean), tf.identity(batch_var)

#             # 训练时，更新均值与方差，测试时使用之前最后一次保存的均值与方差
#             mean, var = tf.cond(tf.equal(is_training,True),mean_var_with_update,
#             lambda:(ema.average(batch_mean),ema.average(batch_var)))

#             # 后执行batch normalization
#             return tf.nn.batch_normalization(x,mean,var,beta,gamma,eps)

# gpu_options = tf.GPUOptions(allow_growth=True)
# tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# with tf.name_scope('con1'):

#     x = tf.placeholder(tf.float32, [32,10], name = 'input')
#     y_true = tf.placeholder(tf.float32, [32,2])

#     weight = tf.Variable(np.random.rand(10,5), dtype = tf.float32, name = 'weight1')
#     y1 = tf.nn.relu(tf.matmul(x, weight))
#     beta = 2 #offset
#     gamma = 2 #scale
#     # b = bn_layer(y1,is_training=True,name='BatchNorm',moving_decay=0.9,eps=1e-5)
#     # b =  tf.nn.batch_normalization(y1,mean=2,variance=2,offset=beta,scale=gamma,variance_epsilon=1e-6)
#     b1 = tf.contrib.layers.batch_norm(y1, decay = 0.9, center = True, scale = True,epsilon=1e-9,updates_collections=tf.GraphKeys.UPDATE_OPS,
#                                     is_training=True, scope='BN1.1')

#     weight2 = tf.Variable(np.random.rand(5,2), dtype = tf.float32, name='weight2')
#     y2 = tf.matmul(b1, weight2)

#     b2 = tf.contrib.layers.batch_norm(y2, decay = 0.9, center = True, scale = True,epsilon=1e-9,updates_collections=tf.GraphKeys.UPDATE_OPS,
#                                     is_training=True, scope='BN1.2')



# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_true, logits = b2))
# optimizer = tf.train.MomentumOptimizer(0.01,momentum=0.9 , name='MOMENTUM')

# update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
# with tf.control_dependencies(update_ops):
#     train = optimizer.minimize(cost)


# test_tensor = tf.Variable(np.ones((2,2)), dtype = tf.float32, name ='test_tensor')
# test_tensor_2 = test_tensor
# test_tensor_2 +=  1


# var_list = tf.trainable_variables()
# g_list = tf.global_variables()
# bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
# bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
# var_list += bn_moving_vars

# with tf.Session() as sess:

#     sess.run(tf.global_variables_initializer())

#     sess.run(train, feed_dict = {x:np.random.rand(32,10), y_true:np.random.rand(32,2)})

#     sess.run(tf.assign(tf.get_default_graph().get_tensor_by_name('BN1.1/moving_mean:0'),[100,100,100,100,100]))

#     var_list = tf.trainable_variables()
#     g_list = tf.global_variables()
#     bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
#     bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
#     var_list += bn_moving_vars

#     saver = tf.train.Saver()
#     # saver.save(sess, ckpt_path)
#     var = tf.global_variables()
#     for i in var:
#         print(i)
#     print('\n')
#     for i in var_list:
#         print(i)
#     print('\n')
#     a = np.array([1,2,3,4,6])
#     print(a.shape)
#     print(sess.run(tf.get_default_graph().get_tensor_by_name('BN1.1/moving_mean:0')))
#     print(tf.get_default_graph().get_tensor_by_name('BN1.1/moving_mean:0').shape==a.shape)
#     print(sess.run(test_tensor))
#     print(sess.run(test_tensor_2))
#     print(sess.run(test_tensor))