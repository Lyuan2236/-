#中文分词，各种分类算法的使用和性能比较

import os
import shutil
import jieba
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras import models
from keras import layers
from keras.utils.np_utils import to_categorical
from keras.preprocessing.text import Tokenizer
from sklearn import svm
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


warnings.filterwarnings('ignore')


# -------------------------------------------
# 路径（换成自己当前的数据集路径）
# filepath = '/home/kesci/work/xiaozhi/text_classification.zip'
# savepath = '/home/kesci/work/xiaozhi/'
# delectpath = '/home/kesci/work/xiaozhi/text_classification'

# filepath = 'D:/workspace_pycharm/text_classification.zip'
# savepath = 'D:/workspace_pycharm/'
# delectpath = 'D:/workspace_pycharm/text_classification'
#
# # -----------------------------------------
# # 检索并删除文件夹
# if os.path.exists(delectpath):
#     print('\n存在该文件夹，正在进行删除，防止解压重命名失败......\n')
#     shutil.rmtree(delectpath)
# else:
#     print('\n不存在该文件夹, 请放心处理......\n')
#
# # -----------------------------------------
# # 解压并处理中文名字乱码的问题
# z = zipfile.ZipFile(filepath, 'r')
# for file in z.namelist():
#     # 中文乱码需处理
#     filename = file.encode('cp437').decode('gbk')  # 先使用 cp437 编码，然后再使用 gbk 解码
#     z.extract(file, savepath)  # 解压 ZIP 文件
#     # 解决乱码问题
#     os.chdir(savepath)  # 切换到目标目录
#     os.rename(file, filename)  # 将乱码重命名文件


def read_text(path, text_list):
    '''
    path: 必选参数，文件夹路径
    text_list: 必选参数，文件夹 path 下的所有 .txt 文件名列表
    return: 返回值
        features 文本(特征)数据，以列表形式返回; 
        labels 分类标签，以列表形式返回
    '''
    
    features, labels = [], [] 
    for text in text_list:
        if text.split('.')[-1] == 'txt':
            try:
                with open(path + text, encoding='gbk') as fp:
                    features.append(fp.read())  # 特征
                    labels.append(path.split('/')[-2])  # 标签
            except Exception as erro:
                print('\n>>>发现错误, 正在输出错误信息...\n', erro)
                
    return features, labels


def merge_text(train_or_test, label_name):
    '''
    train_or_test: 必选参数，train 训练数据集 or test 测试数据集
    label_name: 必选参数，分类标签的名字
    return: 返回值
        merge_features 合并好的所有特征数据，以列表形式返回;
        merge_labels   合并好的所有分类标签数据，以列表形式返回
    '''
    
    print('\n>>>文本读取和合并程序已经启动, 请稍候...')
    
    merge_features, merge_labels = [], []  # 函数全局变量
    for name in label_name:
        # path = '/home/kesci/work/xiaozhi/text_classification/'+ train_or_test +'/'+ name +'/'
        path = 'D:/workspace_pycharm/text_classification/' + train_or_test + '/' + name + '/'
        text_list = os.listdir(path)
        features, labels = read_text(path=path, text_list=text_list)  # 调用函数
        merge_features += features  # 特征
        merge_labels   += labels    # 标签
        
    # 可以自定义添加一些想要知道的信息
    print('\n>>>你正在处理的数据类型是...\n', train_or_test)
    print('\n>>>[', train_or_test ,']数据具体情况如下...')
    print('样本数量\t', len(merge_features), '\t类别名称\t', set(merge_labels))   
    print('\n>>>文本读取和合并工作已经处理完毕...\n')
    
    return merge_features, merge_labels


# 获取训练集
train_or_test = 'train'
label_name = ['女性', '体育', '校园', '文学']
X_train, y_train = merge_text(train_or_test, label_name)

# 获取测试集
train_or_test = 'test'
label_name = ['女性', '体育', '校园', '文学']
X_test, y_test = merge_text(train_or_test, label_name)


# ### 3. 中文文本分词
# 训练集
X_train_word = [jieba.cut(words) for words in X_train]
X_train_cut = [' '.join(word) for word in X_train_word]

# 测试集
X_test_word = [jieba.cut(words) for words in X_test]
X_test_cut = [' '.join(word) for word in X_test_word]


# 4. 停止词使用
# 加载停止词语料
# stoplist = [word.strip() for word in open('/home/kesci/work/xiaozhi/text_classification/stopword.txt',
#                                           encoding='utf-8').readlines()]
stoplist = [word.strip() for word in open('D:/workspace_pycharm/text_classification/stop/stopword.txt',
                                          encoding='utf-8').readlines()]

# ### 5. 编码器处理文本标签
le = LabelEncoder()
y_train_le = le.fit_transform(y_train)
y_test_le = le.fit_transform(y_test)

# ### 文本数据转换成数据值数据矩阵
count = CountVectorizer(stop_words=stoplist)

'''注意：
这里要先 count.fit() 训练所有训练和测试集，保证特征数一致，
这样在算法建模时才不会报错
'''
count.fit(list(X_train_cut) + list(X_test_cut))
X_train_count = count.transform(X_train_cut)
X_test_count = count.transform(X_test_cut)

X_train_count = X_train_count.toarray()
X_test_count = X_test_count.toarray()

print(X_train_count.shape, X_test_count.shape)

# ### 6. 算法模型
# 封装一个函数，提高复用率，使用时只需调用函数即可
# 用于存储所有算法的名字，准确率和所消耗的时间
estimator_list, score_list, time_list = [], [], []


def get_text_classification(estimator, X, y, X_test, y_test):
    '''
    estimator: 分类器，必选参数
            X: 特征训练数据，必选参数
            y: 标签训练数据，必选参数
       X_test: 特征测试数据，必选参数
        y_tes: 标签测试数据，必选参数
       return: 返回值
           y_pred_model: 预测值
             classifier: 分类器名字
                  score: 准确率
                      t: 消耗的时间
                  matrix: 混淆矩阵
                  report: 分类评价函数
                       
    '''
    start = time.time()
    
    # print('\n>>>算法正在启动，请稍候...')
    model = estimator
    
    # print('\n>>>算法正在进行训练，请稍候...')
    model.fit(X, y)
    print(model)
    
    # print('\n>>>算法正在进行预测，请稍候...')
    y_pred_model = model.predict(X_test)
    print(y_pred_model)
    
    # print('\n>>>算法正在进行性能评估，请稍候...')
    score = metrics.accuracy_score(y_test, y_pred_model)
    matrix = metrics.confusion_matrix(y_test, y_pred_model)
    report = metrics.classification_report(y_test, y_pred_model)

    print('>>>准确率\n', score)
    print('\n>>>混淆矩阵\n', matrix)
    print('\n>>>召回率\n', report)
    # print('>>>算法程序已经结束...')
    
    end = time.time()
    t = end - start
    print('\n>>>算法消耗时间为：', t, '秒\n')
    classifier = str(model).split('(')[0]
    
    return y_pred_model, classifier, score, round(t, 2), matrix, report
  

# # 常规算法——方法1——k 近邻算法
# knc = KNeighborsClassifier()
# result = get_text_classification(knc, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# # 常规算法——方法2——决策树
# dtc = DecisionTreeClassifier()
# result = get_text_classification(dtc, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# 常规算法——方法3——多层感知器
mlpc = MLPClassifier()
result = get_text_classification(mlpc, X_train_count, y_train_le, X_test_count, y_test_le)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])

# # 常规算法——方法4——伯努力贝叶斯算法
# bnb = BernoulliNB()
#
# result = get_text_classification(bnb, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# # #### 常规算法——方法5——高斯贝叶斯
#
# # In[18]:
#
#
# gnb = GaussianNB()
#
# result = get_text_classification(gnb, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# # #### 常规算法——方法6——多项式朴素贝叶斯
#
# # In[19]:
#
#
# mnb = MultinomialNB()
#
# result = get_text_classification(mnb, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# #### 常规算法——方法7——逻辑回归算法

# In[20]:


lgr = LogisticRegression()

result = get_text_classification(lgr, X_train_count, y_train_le, X_test_count, y_test_le)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# # #### 常规算法——方法8——支持向量机算法
#
# # In[21]:
#
#
# svc = svm.SVC()
#
# result = get_text_classification(svc, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# #### 集成学习算法——方法1——随机森林算法

# In[22]:


rfc = RandomForestClassifier()

result = get_text_classification(rfc, X_train_count, y_train_le, X_test_count, y_test_le)
estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# # #### 集成学习算法——方法2——自增强算法
#
# # In[23]:
#
#
# abc = AdaBoostClassifier()
#
# result = get_text_classification(abc, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# # #### 集成学习算法——方法3——lightgbm算法
#
# # In[24]:
#
#
# gbm = lightgbm.LGBMClassifier()
#
# result = get_text_classification(gbm, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# # #### 集成学习算法——方法4——xgboost算法
# # xgboost 模型运行有点慢，这里需要等待一阵子
#
# # In[25]:
#
#
# xgb = xgboost.XGBClassifier()
#
# result = get_text_classification(xgb, X_train_count, y_train_le, X_test_count, y_test_le)
# estimator_list.append(result[1]), score_list.append(result[2]), time_list.append(result[3])


# #### 深度学习算法——方法1——多分类前馈神经网络
# 虽然 Keras 也是一个高级封装的接口，但对初学者来说也会很容易混淆一些地方，所以小知同学来说一些概念。
# ```
# 1 算法流程：
# 创建神经网络——添加神经层——编译神经网络——训练神经网络——预测——性能评估——保存模型
# 
# 2 添加神经层
# 至少要有两层神经层，第一层必须是输入神经层，最后一层必须是输出层；
# 输入神经层主要设置输入的维度，而最后一层主要是设置激活函数的类型来指明是分类还是回归问题
# 
# 3 编译神经网络
# 分类问题的 metrics，一般以 accuracy 准确率来衡量
# 回归问题的 metrics, 一般以 mae 平均绝对误差来衡量
# ```
# 暂时就说这些比较容易混淆的知识点

# In[26]:


# start = time.time()
# # --------------------------------
# # np.random.seed(0)     # 设置随机数种子
# feature_num = X_train_count.shape[1]     # 设置所希望的特征数量
#
# # ---------------------------------
# # 独热编码目标向量来创建目标矩阵
# y_train_cate = to_categorical(y_train_le)
# y_test_cate = to_categorical(y_test_le)
# print(y_train_cate)
#
#
# # In[27]:
#
#
# # ----------------------------------------------------
# # 1 创建神经网络
# network = models.Sequential()
#
# # ----------------------------------------------------
# # 2 添加神经连接层
# # 第一层必须有并且一定是 [输入层], 必选
# network.add(layers.Dense(     # 添加带有 relu 激活函数的全连接层
#                          units=128,
#                          activation='relu',
#                          input_shape=(feature_num, )
#                          ))
#
# # 介于第一层和最后一层之间的称为 [隐藏层]，可选
# network.add(layers.Dense(     # 添加带有 relu 激活函数的全连接层
#                          units=128,
#                          activation='relu'
#                          ))
# network.add(layers.Dropout(0.8))
# # 最后一层必须有并且一定是 [输出层], 必选
# network.add(layers.Dense(     # 添加带有 softmax 激活函数的全连接层
#                          units=4,
#                          activation='sigmoid'
#                          ))
#
# # -----------------------------------------------------
# # 3 编译神经网络
# network.compile(loss='categorical_crossentropy',  # 分类交叉熵损失函数
#                 optimizer='rmsprop',
#                 metrics=['accuracy']              # 准确率度量
#                 )
#
# # -----------------------------------------------------
# # 4 开始训练神经网络
# history = network.fit(X_train_count,     # 训练集特征
#             y_train_cate,        # 训练集标签
#             epochs=20,          # 迭代次数
#             batch_size=300,    # 每个批量的观测数  可做优化
#             validation_data=(X_test_count, y_test_cate)  # 验证测试集数据
#             )
# network.summary()
#
#
# # In[28]:
#
#
# # -----------------------------------------------------
# # 5 模型预测
#
# y_pred_keras = network.predict(X_test_count)
#
# # y_pred_keras[:20]
#
#
# # In[29]:
#
#
# # -----------------------------------------------------
# # 6 性能评估
# print('>>>多分类前馈神经网络性能评估如下...\n')
# score = network.evaluate(X_test_count,
#                         y_test_cate,
#                         batch_size=32)
# print('\n>>>评分\n', score)
# print()
# end = time.time()
#
# estimator_list.append('前馈网络')
# score_list.append(score[1])
# time_list.append(round(end-start, 2))
#
#
# # In[30]:
#
#
# # 损失函数情况
# train_loss = history.history["loss"]
# valid_loss = history.history["val_loss"]
# epochs = [i for i in range(len(train_loss))]
# plt.plot(epochs, train_loss,linewidth=3.0)
# plt.plot(epochs, valid_loss,linewidth=3.0)
#
#
# # In[31]:
#
#
# # 准确率情况
# train_loss = history.history["acc"]
# valid_loss = history.history["val_acc"]
# epochs = [i for i in range(len(train_loss))]
# plt.plot(epochs, train_loss,linewidth=3.0)
# plt.plot(epochs, valid_loss,linewidth=3.0)
#
#
# # In[32]:
#
#
# # ----------------------------------------------------
# # 7 保存/加载模型
#
# # 保存
# print('\n>>>你正在进行保存模型操作, 请稍候...\n')
#
# network.save('/home/kesci/work/xiaozhi/my_network_model.h5')
#
# print('>>>保存工作已完成...\n')
#
#
# # 加载和使用
# print('>>>你正在加载已经训练好的模型, 请稍候...\n')
#
# my_load_model = models.load_model('/home/kesci/work/xiaozhi/my_network_model.h5')
#
# print('>>>你正在使用加载的现成模型进行预测, 请稍候...\n')
# print('>>>预测部分结果如下...')
#
# my_load_model.predict(X_test_count)[:20]


# #### 深度学习算法——方法2——LSTM 神经网络
# 本案例的数据使用 LSTM 跑模型

# In[33]:


# ---------------------------------------------

# # 设置所希望的特征数
# feature_num = X_train_count.shape[1]
#
# # 使用单热编码目标向量对标签进行处理
#
# y_train_cate = to_categorical(y_train_le)
# y_test_cate = to_categorical(y_test_le)
#
# print(y_train_cate)


# In[34]:


# # ----------------------------------------------
# # 1 创建神经网络
# lstm_network = models.Sequential()

# # ----------------------------------------------
# # 2 添加神经层
# lstm_network.add(layers.Embedding(input_dim=feature_num,  # 添加嵌入层
#                                   output_dim=4))

# lstm_network.add(layers.LSTM(units=128))                 # 添加 128 个单元的 LSTM 神经层

# lstm_network.add(layers.Dense(units=4,
#                               activation='sigmoid'))     # 添加 sigmoid 分类激活函数的全连接层

# # ----------------------------------------------
# # 3 编译神经网络
# lstm_network.compile(loss='binary_crossentropy',
#                      optimizer='Adam',
#                      metrics=['accuracy']
#                      )

# # ----------------------------------------------
# # 4 开始训练模型
# lstm_network.fit(X_train_count,
#                  y_train_cate,
#                  epochs=5,
#                  batch_size=128,
#                  validation_data=(X_test_count, y_test_cate)
#                  )


# ###  7.算法之间性能比较

# In[35]:



df = pd.DataFrame()
df['分类器'] = estimator_list
df['准确率'] = score_list
df['消耗时间/s'] = time_list
df


# 综上 DataFrame 展示，结合消耗时间和准确率来看，可以得出以下结论：  
# 
# 在同一训练集和测试集、分类器默认参数设置（都未进行调参）的情况下：  
# * 综合效果最好的是:
# ```
# MultinomialNB 多项式朴素贝叶斯分类算法：
# 其准确率达到了 90.5% 并且所消耗的的时间才 0.55 s```
# 
# * 综合效果最差的是：
# ```
# SVC	支持向量机
# 其准确率才 0.575 并且消耗时间高达 380.72s```
# 
# * 准确率最低的是：0.570
# ```
# AdaBoostClassifier 自适应增强集成学习算法```
# 
# * 消耗时间最高的是：566.59s
# ```
# XGBClassifier 集成学习算法```
# 

# In[ ]:




