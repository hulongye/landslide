#!/usr/bin/env python
# coding: utf-8

"""
作者：胖哥
微信公众号：胖哥真不错
微信号: zy10178083

为了防止大家在运行项目时报错(项目都是运行好的，报错基本都是版本不一致 导致的)，
胖哥把项目中用到的库文件版本在这里说明：

pandas == 1.1.5
matplotlib == 3.3.4
numpy == 1.19.5
seaborn == 0.11.1
scikit-learn == 0.24.1
tensorflow == 2.4.1
Keras == 2.4.3

"""

# 导入第三方库
import numpy.random as rd  # 导入随机数生成方法
import matplotlib.pyplot as plt  # 数据可视化库
from sklearn.model_selection import train_test_split  # 数据集拆分工具
import warnings, pandas as pd, numpy as np, math  # 告警库 数据处理库  科学计算库
from keras.models import Sequential  # 导入序贯模型
from keras.utils import plot_model  # 导入 模型绘图工具
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score  # 模型评估方法
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
import keras.layers as layers  # 导入 层 工具包
import keras.backend as K  # 导入 后端 工具包

warnings.filterwarnings('ignore')  # 忽略告警


# 定义灰狼优化算法实现函数
def sanitized_gwo(X_train, X_test, y_train, y_test, SearchAgents_no, T, dim, lb, ub):
    Alpha_position = [0, 0]  # 初始化Alpha灰狼的位置
    Beta_position = [0, 0]  # 初始化Beta灰狼的位置
    Delta_position = [0, 0]  # 初始化Delta灰狼的位置

    Alpha_score = float("inf")  # 初始化Alpha灰狼目标函数的值
    Beta_score = float("inf")  # 初始化Beta灰狼目标函数的值
    Delta_score = float("inf")  # 初始化Delta灰狼目标函数的值

    Positions = np.dot(rd.rand(SearchAgents_no, dim), (ub - lb)) + lb  # 初始化第一个搜索位置

    Convergence_curve = np.zeros((1, T))  # 初始化收敛曲线

    iterations = []  # 定义迭代次数列表
    accuracy = []  # 定义准确率列表

    # 迭代求解
    t = 0
    while t < T:  # 循环

        # 迭代每只灰狼位置
        for i in range(0, (Positions.shape[0])):
            # 如果搜索位置超出了搜索空间，则需要返回到搜索空间
            for j in range(0, (Positions.shape[1])):
                Flag4ub = Positions[i, j] > ub  # 大于最大值
                Flag4lb = Positions[i, j] < lb  # 小于最小值
                # 如果灰狼的位置在最大值和最小值之间，则不需要调整位置，如果超过最大值，则返回最大值边界；如果低于最小值，则返回最小值边界 
                if Flag4ub:  # 判断
                    Positions[i, j] = ub  # 赋值
                if Flag4lb:  # 判断
                    Positions[i, j] = lb  # 赋值

            # 建立卷积神经网络分类模型并训练
            cnn_model = Sequential()  # 序贯模型
            cnn_model.add(
                Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1), activation='relu',
                       padding='valid'))  # 1维卷积层
            cnn_model.add(MaxPooling1D(strides=1, padding='same'))  # 1维最大池化层
            cnn_model.add(Flatten())  # 展平层
            cnn_model.add(Dense(units=int(abs(Positions[i][0])) * 10, activation='relu'))  # 全连接层
            cnn_model.add(Dense(1, activation='sigmoid'))  # 输出层
            cnn_model.compile(loss='binary_crossentropy',
                              optimizer='adam',
                              metrics=['acc'])  # 编译
            cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(abs(Positions[i][1])), batch_size=64)  # 拟合
            score = cnn_model.evaluate(X_test, y_test, batch_size=128)  # 模型评估

            # 使错误率降到最低

            fitness_value = (1 - float(score[1])) * 100  # 错误率 赋值 适应度函数值
            if fitness_value < Alpha_score:  # 如果目标函数值小于Alpha灰狼的目标函数值
                Alpha_score = fitness_value  # 然后将Alpha灰狼的目标函数值更新为最优目标函数值
                Alpha_position = Positions[i]  # 同时更新Alpha灰狼的位置到最佳位置
            if fitness_value > Alpha_score and fitness_value < Beta_score:  # 如果目标函数值大于Alpha灰狼的目标函数值并且小于Beta灰狼的目标函数值
                Beta_score = fitness_value  # 然后将Beta灰狼的目标函数值更新为最优目标函数值
                Beta_position = Positions[i]  # 同时更新Beta灰狼的位置到最佳位置
            # 如果目标函数值大于Alpha灰狼的目标函数值并且大于Beta灰狼的目标函数值并且小于Delta灰狼的目标函数值
            if fitness_value > Alpha_score and fitness_value > Beta_score and fitness_value < Delta_score:
                Delta_score = fitness_value  # 然后将Delta灰狼的目标函数值更新为最优目标函数值
                Delta_position = Positions[i]  # 同时更新Delta灰狼的位置到最佳位置

        a = 2 - t * (2 / T)  # 收敛因子从2线性递减到0

        # 循环更新灰狼个体的位置
        for i in range(0, (Positions.shape[0])):
            # 遍历每个维度
            for j in range(0, (Positions.shape[1])):
                # 包围猎物，更新位置
                r1 = rd.random(1)  # 生成0~1之间的随机数
                r2 = rd.random(1)  # 生成0~1之间的随机数
                A1 = 2 * a * r1 - a  # 计算系数向量A
                C1 = 0.5 + (0.5 * math.exp(-j / 500)) + (1.4 * (math.sin(j) / 30))  # 通过时变加速度常数 计算系数向量C

                # alpha灰狼位置更新
                D_alpha = abs(C1 * Alpha_position[j] - Positions[i, j])  # alpha灰狼与其它个体的距离
                X1 = Alpha_position[j] - A1 * D_alpha  # alpha灰狼当前的位置

                r1 = rd.random(1)  # 生成0~1之间的随机数
                r2 = rd.random(1)  # 生成0~1之间的随机数

                A2 = 2 * a * r1 - a  # 计算系数向量A
                C2 = 1 + (1.4 * (1 - math.exp(-j / 500))) + (1.4 * (math.sin(j) / 30))  # 基于差分均值的摄动时变参数 计算系数向量C

                # Beta灰狼位置更新
                D_beta = abs(C2 * Beta_position[j] - Positions[i, j])  # Beta灰狼与其它个体的距离
                X2 = Beta_position[j] - A2 * D_beta  # Beta灰狼当前的位置

                r1 = rd.random(1)  # 生成0~1之间的随机数
                r2 = rd.random(1)  # 生成0~1之间的随机数

                A3 = 2 * a * r1 - a  # 计算系数向量A
                C3 = (1 / (1 + math.exp(-0.0001 * j / T))) + (
                        (0.5 - 2.5) * ((j / T) ** 2))  # 基于sigmoid函数的加速度系数 计算系数向量C

                # Delta灰狼位置更新
                D_delta = abs(C3 * Delta_position[j] - Positions[i, j])  # Delta灰狼与其它个体的距离
                X3 = Delta_position[j] - A3 * D_delta  # Delta灰狼当前的位置

                # 位置更新
                Positions[i, j] = (X1 + X2 + X3) / 3

        t = t + 1
        iterations.append(t)  # 迭代次数存入列表中
        accuracy.append(abs((100 - Alpha_score) / 100))  # 计算准确率
        print('----------------迭代次数----------------' + str(t))

    best_units = Alpha_position[0]  # 最优位置  即最优参数值
    best_epochs = Alpha_position[1]  # 最优位置  即最优参数值

    return best_units, best_epochs, iterations, accuracy  # 返回数据


if __name__ == '__main__':
    print('----------------1. 数据加载/预处理/探索性数据分析/特征工程-------------------')
    file_name = 'data.xlsx'  # 定义数据文件
    df = pd.read_excel(file_name)  # 读取数据

    # 查看数据前5行
    print('*************查看数据前5行*****************')
    print(df.head())

    # 数据缺失值统计
    print('**************数据缺失值统计****************')
    print(df.info())

    # 描述性统计分析
    print(df.describe())
    print('******************************')

    #  y变量柱状图
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    # kind='bar' 绘制柱状图
    df['y'].value_counts().plot(kind='bar')  # 绘制柱状图
    plt.xlabel("y变量")  # 设置x轴的名称
    plt.ylabel("数量")  # 设置y轴的名称
    plt.title('y变量柱状图')  # 设置标题的名称
    plt.show()  # 展示图片

    # y=1样本x1变量分布直方图
    fig = plt.figure(figsize=(8, 5))  # 设置画布大小
    plt.rcParams['font.sans-serif'] = 'SimHei'  # 设置中文显示
    plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    data_tmp = df.loc[df['y'] == 1, 'x1']  # 过滤出y=1的样本
    # 绘制直方图  bins：控制直方图中的区间个数 auto为自动填充个数  color：指定柱子的填充色
    plt.hist(data_tmp, bins='auto', color='g')
    plt.xlabel('x1')  # 设置x轴的名称
    plt.ylabel('数量')  # 设置y轴的名称
    plt.title('y=1样本x1变量分布直方图')  # 设置标题的名称
    plt.show()  # 展示图片

    # 数据的相关性分析
    import seaborn as sns

    sns.heatmap(df.corr(), cmap="YlGnBu", annot=True)  # 绘制热力图
    plt.title('相关性分析热力图')  # 设置标题的名称
    plt.show()  # 展示图片

    # 提取特征变量和标签变量
    y = df.y  # 标签
    X = df.drop('y', axis=1)  # 特征

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    X_train = layers.Lambda(lambda X_train: K.expand_dims(X_train, axis=-1))(X_train)  # 增加维度

    print('***********************查看训练集的形状**************************')
    print(X_train.shape)  # 查看训练集的形状

    X_test = layers.Lambda(lambda X_test: K.expand_dims(X_test, axis=-1))(X_test)  # 增加维度
    print('***********************查看测试集的形状**************************')
    print(X_test.shape)  # 查看测试集的形状

    print('----------------2. 初始化参数------------')
    SearchAgents_no = 20  # 灰狼个数
    T = 2  # 最大迭代次数
    dim = 2  # 维度 需要优化两个变量 - units and epochs
    lb = 1  # 最小值限制
    ub = 10  # 最大值限制

    print('----------------3.调用灰狼算法函数-----------------')
    best_units, best_epochs, iterations, accuracy = sanitized_gwo(X_train, X_test, y_train, y_test, SearchAgents_no, T,
                                                                  dim,
                                                                  lb, ub)

    print('----------------4. 最优结果展示-----------------')
    print("The best units is " + str(int(abs(best_units)) * 10))  # 输出数据
    print("The best epochs is " + str(int(abs(best_epochs)) * 10))  # 输出数据

    print('----------------5. 应用优化后的最优参数值构建卷积神经网络分类模型-----------------')
    # 应用优化后的最优参数值构建卷积神经网络分类模型
    cnn_model = Sequential()  # 序贯模型
    cnn_model.add(Conv1D(filters=5, kernel_size=(4,), input_shape=(X_train.shape[1], 1),
                         activation='relu'))  # 1维卷积层
    cnn_model.add(MaxPooling1D(strides=1, padding='same'))  # 1维最大池化层
    cnn_model.add(Flatten())  # 展平层
    cnn_model.add(Dense(int(abs(best_units)) * 10, activation='relu'))  # 全连接层
    cnn_model.add(Dense(1, activation='sigmoid'))  # 输出层
    cnn_model.compile(loss='binary_crossentropy',
                      optimizer='adam',
                      metrics=['acc'])  # 编译
    history = cnn_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=int(abs(best_epochs)) * 10,
                            batch_size=64)  # 拟合
    print('*************************输出模型摘要信息*******************************')
    print(cnn_model.summary())  # 输出模型摘要信息

    plot_model(cnn_model, to_file='model.png', show_shapes=True)  # 保存模型结构信息


    # 定义绘图函数：损失曲线图和准确率曲线图
    def show_history(history):
        loss = history.history['loss']  # 获取损失
        val_loss = history.history['val_loss']  # 测试集损失
        epochs = range(1, len(loss) + 1)  # 迭代次数
        plt.figure(figsize=(12, 4))  # 设置图片大小
        plt.subplot(1, 2, 1)  # 增加子图
        plt.plot(epochs, loss, 'r', label='Training loss')  # 绘制曲线图
        plt.plot(epochs, val_loss, 'b', label='Test loss')  # 绘制曲线图
        plt.title('Training and Test loss')  # 设置标题名称
        plt.xlabel('Epochs')  # 设置x轴名称
        plt.ylabel('Loss')  # 设置y轴名称
        plt.legend()  # 添加图例
        acc = history.history['acc']  # 获取准确率
        val_acc = history.history['val_acc']  # 获取测试集准确率
        plt.subplot(1, 2, 2)  # 增加子图
        plt.plot(epochs, acc, 'r', label='Training acc')  # 绘制曲线图
        plt.plot(epochs, val_acc, 'b', label='Test acc')  # 绘制曲线图
        plt.title('Training and Test accuracy')  # 设置标题名称
        plt.xlabel('Epochs')  # 设置x轴名称
        plt.ylabel('Accuracy')  # 设置y轴名称
        plt.legend()  # 添加图例
        plt.show()  # 显示图片


    show_history(history)  # 调用绘图函数

    y_pred = cnn_model.predict(X_test, batch_size=10)  # 预测
    y_pred = np.round(y_pred)  # 转化为类别

    print('----------------模型评估-----------------')
    # 模型评估
    print('**************************输出测试集的模型评估指标结果*******************************')

    print('卷积神经网络分类模型-最优参数-准确率分值: {0:0.4f}'.format(accuracy_score(y_test, y_pred)))
    print("卷积神经网络分类模型-最优参数-查准率 :", round(precision_score(y_test, y_pred), 4))
    print("卷积神经网络分类模型-最优参数-召回率 :", round(recall_score(y_test, y_pred), 4))
    print("卷积神经网络分类模型-最优参数-F1分值:", round(f1_score(y_test, y_pred), 4))

    from sklearn.metrics import classification_report  # 导入分类报告工具

    # 分类报告
    print(classification_report(y_test, y_pred))

    from sklearn.metrics import confusion_matrix  # 导入混淆矩阵工具
    import seaborn as sns  # 统计数据可视化

    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)

    # 构建数据框
    cm_matrix = pd.DataFrame(data=cm, columns=['Actual :0', 'Actual :1'],
                             index=['Predict :0', 'Predict :1'])

    sns.heatmap(cm_matrix, annot=True, fmt='d', cmap='YlGnBu')  # 热力图展示
    plt.show()  # 展示图片
