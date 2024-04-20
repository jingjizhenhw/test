import os
import math
import keras
import traceback
import warnings
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from keras.models import Sequential
from sklearn.metrics import r2_score
from keras.layers import LSTM, Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed

warnings.filterwarnings("ignore")  # 忽略警告信息
pd.set_option('mode.chained_assignment', None)  # 允许进行链式索引赋值操作而不会产生警告。
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']  # 设置绘图中显示中文字符
plt.rcParams['axes.unicode_minus'] = False  # 设置绘图中正常显示负号


# 数据处理
def data_processing():
    try:
        data = pd.read_csv(os.getcwd() + "\\data\\household_power_consumption.txt", sep=';',
                           header=0, low_memory=False, infer_datetime_format=True,
                           parse_dates={'datetime': [0, 1]}, index_col=['datetime'])
        print("查看数据集")
        print(data.shape)

        print("查看缺失值情况")
        print(data.describe(include='all'))

        data.replace('?', np.nan, inplace=True)
        data = data.astype('float32')

        print("查看缺失值")
        print(data.isna().sum())

        values = data.values
        one_day = 60 * 24
        for row in range(values.shape[0]):
            for col in range(values.shape[1]):
                if np.isnan(values[row, col]):
                    values[row, col] = values[row - one_day, col]

        print("查看填充后的缺失值")
        print(data.isna().sum())

        values = data.values
        data['sub_metering_4'] = (values[:, 0] * 1000 / 60) - (values[:, 4] + values[:, 5] + values[:, 6])

        data.to_csv(os.getcwd() + "\\data\\household_power_consumption.csv")

        dataset = pd.read_csv(os.getcwd() + "\\data\\household_power_consumption.csv", header=0, infer_datetime_format=True,
                              parse_dates=['datetime'], index_col=['datetime'])
        daily_groups = dataset.resample('D')
        daily_data = daily_groups.sum()
        print('daily data shape:')
        print(daily_data.shape)
        print('daily data head:')
        print(daily_data.head())
        daily_data.to_csv(os.getcwd() + "\\data\\household_power_consumption_days.csv")
    except Exception as e:
        print("数据处理过程中发生错误：", e)
        # 打印详细的异常堆栈信息 有助于快速定位和解决问题
        traceback.print_exc()


# 以周为单位切分训练数据和测试数据
def split_dataset(data):
    # data为按天的耗电量统计数据，shape为(1442, 8)
    # 测试集取最后一年的46周（322天）数据，剩下的159周（1113天）数据为训练集，以下的切片实现此功能。
    train, test = data[1:-328], data[-328:-6]
    train = np.array(np.split(train, len(train) / 7))  # 将数据划分为按周为单位的数据
    test = np.array(np.split(test, len(test) / 7))
    return train, test


# 根据预期值评估一个或多个周预测损失
def evaluate_forecasts(actual, predicted):
    # 计算周期RMSE--scores（选取的周期为周之七天 即7个周期）
    scores = list()
    for i in range(actual.shape[1]):
        mse = skm.mean_squared_error(actual[:, i], predicted[:, i])
        rmse = math.sqrt(mse)
        scores.append(rmse)

    # 计算总RMSE--score（所有预测值与真实值之间的总体均方根误差）
    s = 0
    for row in range(actual.shape[0]):
        for col in range(actual.shape[1]):
            s += (actual[row, col] - predicted[row, col]) ** 2
    score = math.sqrt(s / (actual.shape[0] * actual.shape[1]))
    print('actual.shape[0]:{}, actual.shape[1]:{}'.format(actual.shape[0], actual.shape[1]))

    # 计算r2（决定系数）
    print('r2:')
    print(r2_score(actual.flatten(), predicted.flatten()))

    return score, scores


# 输出评估分数
def summarize_scores(name, score, scores):
    s_scores = ', '.join(['%.1f' % s for s in scores])
    print('%s: [%.3f] %s\n' % (name, score, s_scores))


# 滑动窗口截取序列数据 窗口宽度为7 滑动步长为1
def sliding_window(train, sw_width=7, n_out=7, in_start=0):
    data = train.reshape((train.shape[0] * train.shape[1], train.shape[2]))  # 将以周为单位的样本展平为以天为单位的序列
    X, y = [], []

    for _ in range(len(data)):
        in_end = in_start + sw_width
        out_end = in_end + n_out

        # 保证截取样本完整，最大元素索引不超过原序列索引，则截取数据；否则丢弃该样本
        if out_end < len(data):
            # 训练数据以滑动步长1截取
            train_seq = data[in_start:in_end, 0]
            train_seq = train_seq.reshape((len(train_seq), 1))
            X.append(train_seq)
            y.append(data[in_end:out_end, 0])
        in_start += 1

    return np.array(X), np.array(y)


# 定义 Encoder-Decoder LSTM 模型
def cnn_lstm_model(train, sw_width, in_start=0, verbose_set=0, epochs_num=20, batch_size_set=4):
    train_x, train_y = sliding_window(train, sw_width, in_start=0)

    n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]

    train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

    model = Sequential()
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu',
                                     input_shape=(n_timesteps, n_features)))
    model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
    model.add(tf.keras.layers.MaxPooling1D(pool_size=2))
    model.add(tf.keras.layers.Flatten())
    model.add(RepeatVector(n_outputs))
    model.add(LSTM(200, activation='relu', return_sequences=True))
    model.add(TimeDistributed(Dense(100, activation='relu')))
    # 输出层只有一个神经元，因此输出第一特征列（总负荷）预测列
    model.add(TimeDistributed(Dense(1)))

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])
    print(model.summary())

    model.fit(train_x, train_y,
              epochs=epochs_num, batch_size=batch_size_set, verbose=verbose_set)
    return model


# 预测输入数据
def forecast(model, pred_seq, sw_width):
    data = np.array(pred_seq)
    data = data.reshape((data.shape[0] * data.shape[1], data.shape[2]))

    input_x = data[-sw_width:, 0]  # 获取输入数据的最后一周的数据
    input_x = input_x.reshape((1, len(input_x), 1))  # 重塑形状[1, sw_width, 1]

    yhat = model.predict(input_x, verbose=0)  # 预测下周数据
    yhat = yhat[0]  # 获取预测向量
    return yhat


# 评估模型
def evaluate_model(model, train, test, sd_width):
    history_fore = [x for x in train]
    predictions = list()  # 用于保存每周的前向验证结果；

    for i in range(len(test)):
        yhat_sequence = forecast(model, history_fore, sd_width)  # 预测下周的数据
        predictions.append(yhat_sequence)  # 保存预测结果
        history_fore.append(test[i, :])  # 得到真实的观察结果并添加到历史中以预测下周
    print(predictions)

    predictions = np.array(predictions)  # 评估一周中每天的预测结果
    score, scores = evaluate_forecasts(test[:, :, 0], predictions)  # test取第一特征列（总负荷）

    return score, scores, predictions


# 绘制RMSE折线图
def model_plot(score, scores, days, name):
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot(days, scores, marker='o', label=name)
    plt.grid(linestyle='--', alpha=0.5)
    plt.ylabel(r'$RMSE$', size=15)
    plt.title('CNN-LSTM 模型预测评估', size=18)
    plt.legend()

    # 保存图片
    plt.savefig('img/CNN-LSTM模型RMSE折线图.png')

    # 显示图形
    plt.show()


# 绘制点状图
def dot_chart(predictions):
    # 定义数据
    week = list(range(46))  # 46周
    pre_sun = list()
    pre_mon = list()
    pre_tue = list()
    pre_wed = list()
    pre_thr = list()
    pre_fri = list()
    pre_sat = list()

    # 填充数据
    for i in range(len(predictions)):
        if i % 7 == 0:
            pre_sun.append(predictions[i])
        elif i % 7 == 1:
            pre_mon.append(predictions[i])
        elif i % 7 == 2:
            pre_tue.append(predictions[i])
        elif i % 7 == 3:
            pre_wed.append(predictions[i])
        elif i % 7 == 4:
            pre_thr.append(predictions[i])
        elif i % 7 == 5:
            pre_fri.append(predictions[i])
        elif i % 7 == 6:
            pre_sat.append(predictions[i])

    # 绘制子图（周一至周日）
    ax1 = plt.subplot(3, 3, 1)
    plt.title("SUN")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_sun, "oc:")
    ax2 = plt.subplot(3, 3, 2)
    plt.title("MON")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_mon, "oc:")
    ax3 = plt.subplot(3, 3, 3)
    plt.title("TUE")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_tue, "oc:")
    ax4 = plt.subplot(3, 3, 4)
    plt.title("WED")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_wed, "oc:")
    ax5 = plt.subplot(3, 3, 5)
    plt.title("THR")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_thr, "oc:")
    ax6 = plt.subplot(3, 3, 6)
    plt.title("FRI")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_fri, "oc:")
    ax7 = plt.subplot(3, 3, 7)
    plt.title("SAT")
    plt.xlabel("week")
    plt.ylabel("global_active_power")
    plt.plot(week, pre_sat, "oc:")

    # 自动调整子图布局
    plt.tight_layout()

    # 保存图片
    plt.savefig('img/预测数据周期总负荷点状图.png')

    # 显示图形
    plt.show()


# 绘制折线图
def line_chart(predicted_data, test_data):
    # x轴标签--日期
    dates = range(1, len(test_data) + 1)  # 从1开始，每个日期对应一条数据

    # 绘制测试数据（实线）
    plt.plot(dates, test_data, label='Actual', color='blue', linestyle='-')

    # 绘制预测数据（虚线）
    plt.plot(dates, predicted_data, label='Predicted', color='red', linestyle='--')

    # 添加标题和标签
    plt.title('Actual vs Predicted Daily Data')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()

    # 保存图片
    plt.savefig('img/测试数据实际与预测对比图.png')

    # 显示图形
    plt.show()


# 绘制预测数据点状图
def predictions_plot(predictions, test):
    # 绘制点状图
    dot_chart(predictions)

    # 绘制折线图
    line_chart(predictions, test)


# 主函数
def main(dataset, sw_width, days, name, in_start, verbose, epochs, batch_size):
    try:
        # 数据处理
        data_processing()

        # 划分训练集和测试集
        train, test = split_dataset(dataset.values)

        # 训练模型
        model = cnn_lstm_model(train, sw_width, in_start, verbose_set=0, epochs_num=20, batch_size_set=4)

        # 计算RMSE
        score, scores, predictions = evaluate_model(model, train, test, sw_width)

        # 打印分数
        summarize_scores(name, score, scores)

        # 保存 keras 格式模型
        model.save("企业电力营销模型.keras")

        # 加载模型
        # loaded_model = keras.models.load_model("企业电力营销模型.keras")

        # 获取一维数据
        test_feature = test[:, :, 0]  # 提取测试数据的第一个特征列
        test_feature_flat = test_feature.flatten()  # 转换为一维数组
        predictions_flat = list(np.array(predictions).flatten())

        # 绘图
        predictions_plot(predictions_flat, test_feature_flat)
        model_plot(score, scores, days, name)
    except Exception as e:
        print("主函数运行过程中发生错误：", e)
        traceback.print_exc()


# 运行主函数
if __name__ == "__main__":
    # 读取数据
    dataset = pd.read_csv(os.getcwd() + "\\data\\household_power_consumption_days.csv", header=0,
                          infer_datetime_format=True, engine='c',
                          parse_dates=['datetime'], index_col=['datetime'])

    # 设置所需数值
    days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
    name = 'CNN-LSTM'
    sliding_window_width = 14
    input_sequence_start = 0
    epochs_num = 20
    batch_size_set = 4
    verbose_set = 0

    # 运行主函数
    main(dataset, sliding_window_width, days, name, input_sequence_start,
         verbose_set, epochs_num, batch_size_set)
