import numpy as np
import matplotlib.pyplot as plt

#  随机生成 600 个两维样本
np.random.seed(0)
class1 = np.random.uniform(low=-1, high=3, size=(300, 2))
class2 = np.random.uniform(low=5, high=8, size=(300, 2))

#  合并数据作为训练集
X = np.vstack((class1, class2))
y = np.hstack((np.zeros(class1.shape[0]), np.ones(class2.shape[0])))

#  计算两个类别的均值向量
mean_vectors = []
mean_vectors.append(np.mean(class1, axis=0))
mean_vectors.append(np.mean(class2, axis=0))

# 计算类内散度矩阵
S_W = np.zeros((2,2))
for cl, mv in zip(range(2), mean_vectors):
    class_sc_mat = np.zeros((2,2))  # 每个类的散度矩阵
    for row in X[y == cl]:
        row, mv = row.reshape(2,1), mv.reshape(2,1)  # 列向量
        class_sc_mat += (row-mv).dot((row-mv).T)
    S_W += class_sc_mat  # 累加类散度矩阵

# 计算类间散度矩阵
overall_mean = np.mean(X, axis=0)
S_B = np.zeros((2,2))
for i, mean_vec in enumerate(mean_vectors):
    n = X[y==i,:].shape[0]
    mean_vec = mean_vec.reshape(2,1)  # 列向量
    overall_mean = overall_mean.reshape(2,1)  # 列向量
    S_B += n * (mean_vec - overall_mean).dot((mean_vec - overall_mean).T)

#  选择最大特征值对应的特征向量作为最佳投影方向
eig_vals, eig_vecs = np.linalg.eig(np.linalg.inv(S_W).dot(S_B))
w = eig_vecs[:, np.argmax(eig_vals)]


# 8. 将数据投影到新的空间
X_lda = X.dot(w)

# 测试数据
test_point = np.array([2, 3])
test_point_transformed = test_point.dot(w)

mean_class1 = np.mean(class1, axis=0)
mean_class2 = np.mean(class2, axis=0)
decision_boundary = (mean_class1 + mean_class2) / 2

# 画出投影前数据分布以及判别函数位置
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.scatter(class1[:, 0], class1[:, 1], alpha=0.5, label='Class 1')
plt.scatter(class2[:, 0], class2[:, 1], color = 'y',alpha=0.5, label='Class 2')
plt.scatter(test_point[0], test_point[1], color='red', marker='*', label='Test Point [2,3]')
plt.axline(decision_boundary, slope=-1/w[1], color='black', linestyle='--', label='Decision Boundary')
plt.xlabel('Feature x')
plt.ylabel('Feature y')
plt.legend()
plt.title('Original')

# 画出投影到一维空间后样本分布
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:300], np.zeros(300), alpha=0.5, label='Class 1')
plt.scatter(X_lda[300:], np.zeros(300), color = 'g',alpha=0.5, label='Class 2')
plt.scatter(test_point_transformed, 0, color='b', marker='*', label='Test Point [2,3]')
plt.xlabel('LD1')
plt.title('LDA Result')
plt.axis('tight')
plt.legend()
plt.tight_layout()
plt.show()

# 判别测试点属于哪一类
print('Test point [2,3] belongs to Class', '1' if test_point_transformed < 0 else '2')