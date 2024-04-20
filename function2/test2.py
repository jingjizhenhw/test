import pandas as pd
import numpy as np
import pymysql
import warnings

from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


warnings.filterwarnings("ignore")

# 数据库配置
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'hw123321',
    'database': 'electric'
}


# 数据导入模块
def import_data():
    try:
        # 连接数据库
        conn = pymysql.connect(**db_config)
        # 读取数据 存储在data_df中
        data_df = pd.read_sql_query(
            "SELECT user_id, DATE_FORMAT(money_date,'%Y-%m-%d') as money_date, user_money FROM user_info",
            conn, parse_dates=['money_date'])
        # 关闭数据库连接
        conn.close()

        # 判断查询数据是否为空
        if data_df.empty:
            print("No user data found.")
            return None
        return data_df
    except Exception as e:
        print("Error while importing data:", e)
        return None


# 数据清洗模块
def clean_data(data):
    if data is None:
        return None

    try:
        # 记录数据清洗前的数据数量
        original_count = len(data)

        # 处理缺失值 删除包含缺失值的行（测试无缺失值）
        data.dropna(inplace=True)

        # 处理异常时间戳（测试有1条记录）
        data = remove_outlier_timestamps(data, 'money_date', '2018-03-01', '2019-03-31')

        # 处理离群值（测试有6条记录）
        data = remove_outliers(data, 'user_money')  # 使用四分位范围法

        # 处理重复缴费数据（测试无重复缴费）
        data = remove_duplicate_payments(data)

        # 处理异常值
        data = data[data['user_money'] > 0]  # 剔除缴费金额为0的记录（测试无异常值）
        data = remove_users_with_few_rows(data)  # 剔除数据少的用户（测试有14条记录）

        # 去除没有缴费日期为2019年3月的用户数据——确保预测一致性（测试有28条数据）
        data = data[data['user_id'].isin(data[data['money_date'].dt.strftime('%Y-%m') == '2019-03']['user_id'])]

        # 记录数据清洗后的数据数量
        new_count = len(data)

        # 输出剔除前后的数据数量差异
        print(f"Original data count: {original_count}")
        print(f"New data count after removing outliers: {new_count}")
        print(f"Number of rows removed: {original_count - new_count}")

        # 假设data是经过groupby操作后的DataFrame
        # for user_id, group_data in data.groupby('user_id'):
        #     print(f"User ID: {user_id}")
        #     print(group_data)
        #     print("----------------------------------")

        return data
    except Exception as e:
        print("Error while cleaning data:", e)
        return None


# 剔除异常时间戳
def remove_outlier_timestamps(data, timestamp_column, start_date, end_date):
    data[timestamp_column] = pd.to_datetime(data[timestamp_column])
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)
    data = data[(data[timestamp_column] >= start_date) & (data[timestamp_column] <= end_date)]
    return data


# 剔除离群值
def remove_outliers(data, column):
    q1 = data[column].quantile(0.25)
    q3 = data[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]
    return data


# 剔除重复缴费数据
def remove_duplicate_payments(data):
    # 添加月份列以用于分组
    data['money_date'] = data['money_date'].dt.to_period('M')

    # 找到重复缴费的行
    # duplicate_payments = data[data.duplicated(subset=['user_id', 'money_date'], keep=False)]

    # 保留第一次缴费记录，删除其余重复缴费记录
    cleaned_data = data.drop_duplicates(subset=['user_id', 'money_date'], keep='first')

    return cleaned_data


# 剔除数据较少的用户
def remove_users_with_few_rows(data, threshold=5):
    # 计算每个用户编号对应的行数
    user_row_counts = data['user_id'].value_counts()

    # 找到行数小于阈值的用户编号
    users_to_remove = user_row_counts[user_row_counts < threshold].index

    # 剔除行数过少的用户
    data = data[~data['user_id'].isin(users_to_remove)]

    return data


# 添加个性化建模模块
def personalized_modeling(data):
    # 获取每个用户的缴费时间序列和缴费金额序列
    time_series = data.groupby('user_id')['money_date'].apply(list)
    money_series = data.groupby('user_id')['user_money'].apply(list)

    # 初始化结果列表
    users_forecast = []

    # 初始化机器学习模型管道
    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # 特征标准化
        ('model', LinearRegression())  # 使用线性回归模型
    ])

    # 遍历每个用户的时间序列
    for user_id, ts in time_series.items():
        # 获取用户的缴费金额序列
        user_money_series = money_series[user_id]

        # 准备特征
        X = np.arange(len(ts)).reshape(-1, 1)  # 时间特征

        # 准备目标变量
        y = np.array(user_money_series)  # 缴费金额序列作为目标变量

        # 拟合模型
        pipeline.fit(X, y)

        # 预测未来六个月的缴费金额
        next_six_months = np.arange(len(ts) + 1, len(ts) + 7).reshape(-1, 1)  # 2019-04-01~2019-09-30
        forecast = pipeline.predict(next_six_months)  # 使用模型进行预测

        # 将预测结果保存到结果列表中
        users_forecast.append({'user_id': user_id, 'forecast_money': forecast})

    return users_forecast


# 用户价值评估模块
def evaluate_user_value(history_data, forecast_data):
    # 剔除历史缴费日期 便于评估
    combined_data = history_data.groupby('user_id').apply(lambda x: x.sort_values('money_date')['user_money'].tolist())

    # 将预测数据按用户编号合并到历史数据中
    for item in forecast_data:
        user_id = item['user_id']
        forecast_money = item['forecast_money']
        if user_id in combined_data:
            combined_data[user_id].extend(forecast_money)

    # 计算每个用户的指标
    user_indicators = []
    for user_id, payments in combined_data.items():
        total_payment = sum(payments)
        average_payment = total_payment / len(payments)
        payment_growth_rate = (payments[-1] - payments[0]) / payments[0] * 100 if len(payments) > 1 else 0
        user_indicators.append(
            {'user_id': user_id, 'total_payment': total_payment, 'average_payment': average_payment,
             'payment_growth_rate': payment_growth_rate})

    # 标准化指标数据
    total_payments = [user['total_payment'] for user in user_indicators]
    average_payments = [user['average_payment'] for user in user_indicators]
    payment_growth_rates = [user['payment_growth_rate'] for user in user_indicators]

    total_payment_normalized = (total_payments - np.mean(total_payments)) / np.std(total_payments)
    average_payment_normalized = (average_payments - np.mean(average_payments)) / np.std(average_payments)
    payment_growth_rate_normalized = (payment_growth_rates - np.mean(payment_growth_rates)) / np.std(
        payment_growth_rates)

    # 定义权重
    weights = [0.4, 0.3, 0.3]

    # 计算加权平均评分
    weighted_scores = [weights[0] * total + weights[1] * average + weights[2] * growth for total, average, growth in
                       zip(total_payment_normalized, average_payment_normalized, payment_growth_rate_normalized)]

    # 创建DataFrame存储用户数据和加权评分
    user_data = pd.DataFrame(user_indicators)
    user_data['weighted_score'] = weighted_scores

    # 将用户指标上传至csv文件中
    user_data.to_csv('data/user_data.csv', index=False)

    # 根据加权评分降序排列用户数据，并选取Top 5高价值用户
    top5_high_value_users = user_data.sort_values(by='weighted_score', ascending=False).head(5)

    return top5_high_value_users


# 结果排序模块
def sort_results(top5_users):
    print("预测最可能成为高价值类型的客户TOP5：")
    # 输出用户ID和加权评分
    for index, row in top5_users.iterrows():
        print(f"用户ID: {row['user_id']}, 加权评分: {row['weighted_score']}")


# top5用户信息可视化模块
def top5_user_information_visualization(history_data, forecast_data, top5_users):
    # 提取top5用户的用户ID
    top5_user_ids = top5_users['user_id'].tolist()

    # 初始化图形
    plt.figure(figsize=(15, 6))

    # 定义历史数据
    top5_history_money = []  # top5用户历史缴费金额
    top5_history_date = []  # top5用户历史缴费时间
    merge_history_date = set()  # 合并top5用户历史缴费时间 设置不重复

    # 遍历获得top5用户历史各数据
    for user_id in top5_user_ids:
        # 依据用户编号获得对应用户总数据user_history_data
        user_history_data = history_data[history_data['user_id'] == user_id]

        # 依据user_history_data分别获取用户缴费金额和缴费时间 并转为对应的类型
        user_history_date = user_history_data['money_date'].astype(str).tolist()
        user_history_money = user_history_data['user_money'].astype(float).tolist()

        # 数据汇总
        top5_history_date.append(user_history_date)
        top5_history_money.append(user_history_money)
        merge_history_date.update(user_history_date)  # 合并缴费时间

    # 将合并的缴费时间转为列表并排序
    merge_history_date = list(merge_history_date)
    merge_history_date.sort()

    # 定义预测数据
    last_payment_date_str = merge_history_date[-1]

    # 将缴费日期转换为 datetime 对象
    last_payment_date = datetime.strptime(last_payment_date_str, '%Y-%m')

    # 遍历用户
    for i, user_id in enumerate(top5_user_ids):
        # 获取用户历史缴费数据
        user_history_date = top5_history_date[i]  # 单个用户历史缴费时间
        user_history_money = top5_history_money[i]  # 单个用户历史缴费金额

        # 绘制历史缴费金额折线图
        plt.plot(merge_history_date, [user_history_money[user_history_date.index(date)] if date in user_history_date else None for date in merge_history_date],
                 marker='o', label=f'User {user_id} (Historical)')

        # 推测用户预测缴费日期（未来6个月）
        forecast_date = [f"{(last_payment_date + relativedelta(months=i)).strftime('%Y-%m')}" for i in range(1, 7)]

        # 获取用户预测缴费金额
        user_forecast_data = next((item['forecast_money'] for item in forecast_data if item['user_id'] == user_id),
                                  None)

        # 绘制预测缴费金额折线图
        plt.plot(forecast_date, user_forecast_data,
                 marker='o',
                 linestyle='--',
                 label=f'User {user_id} (Forecast)')

    # 添加图例
    plt.legend()

    # 设置标题和标签
    plt.title('Top 5 Users - Historical and Forecasted Payments')
    plt.xlabel('Time')
    plt.ylabel('Payment Amount')

    # 保存图片
    plt.savefig('data/高价值TOP5用户预测缴费折线图.png')

    # 显示图形
    plt.grid(True)
    plt.show()


# 主函数
def main():
    try:
        # 1 导入数据
        data = import_data()

        # 2 数据清洗
        data = clean_data(data)

        # 3 线性回归模型 个性化建模
        forecast_data = personalized_modeling(data)

        # # 4 评估用户价值
        top5_users = evaluate_user_value(data, forecast_data)

        # 5 结果输出
        sort_results(top5_users)

        # 6 可视化图表
        top5_user_information_visualization(data, forecast_data, top5_users)
    except Exception as e:
        print("An error occurred in the main function:", e)


# 运行主函数
if __name__ == "__main__":
    main()
