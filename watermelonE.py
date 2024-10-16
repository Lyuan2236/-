import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# DecisionTreeClassifier使用方法可查看网址https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
# 1. 数据读取和处理
data = pd.read_csv('melon_data1.csv', encoding='gbk')
print('西瓜的数据形态：', data.shape)
print('读取西瓜数据集：')
df=pd.DataFrame(data)
print(pd.DataFrame(data))

# 将target目标值转变为数字
data.loc[data['好瓜与否'] != '是', '好瓜与否'] = 0
data.loc[data['好瓜与否'] == '是', '好瓜与否'] = 1
data['好瓜与否'] = data['好瓜与否'].astype('int')
print('修改目标值之后的数据集：')
print(pd.DataFrame(data))

# 使用get_dummies将文本数据转化为数值
data_X = pd.get_dummies(data.iloc[:, 1:-1])
print('转换之后的特征值：')
df=pd.DataFrame(data_X)
print(df)
df.to_csv('melonFeature.csv', index=False)

# 获取特征向量X和分类标签y
X = data_X.values
y = data.iloc[:, -1].values
# 打印数据形态
print('特征形态：{} 标签形态：{}'.format(X.shape, y.shape))

# 2. 用决策树建模并做出预测
# 拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
### 填空1：使用信息增益构建决策树模型并进行预测，给出预测结果和精度 ###


# 3. 决策树的分类过程展示
# 导入graphviz工具 https://blog.csdn.net/qq_45956730/article/details/126689318
import graphviz
# 导入决策树中输出graphviz的接口
from sklearn.tree import export_graphviz

### 填空2：使用信息增益显示决策树图形（使用graphviz工具） ###

