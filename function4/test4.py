import os
import joblib
import warnings
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

warnings.filterwarnings("ignore")
pd.set_option('mode.chained_assignment', None)
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False

# 设置显示的最大行数和列数
pd.set_option('display.max_rows', None)  # None 表示显示所有行
pd.set_option('display.max_columns', None)  # None 表示显示所有列

# 禁用省略号
pd.set_option('display.expand_frame_repr', False)


# 数据获取
def data_acquisition():
    try:
        data = pd.read_csv(
            os.getcwd() + "\\data\\Monroe_County_Single_Family_Residential__Building_Assets_and_Energy_Consumption__2017-2019.csv")
        data.drop(
            columns=['Ethnic group', 'NYSERDA Energy Efficiency Program Participation',
                     'Average annual electric use (MMBtu)',
                     'Average annual gas use (MMBtu)', 'Average annual total energy use (MMBtu)'], inplace=True)
        print("查看数据集")
        print(data.shape)

        print("查看缺失值情况")
        # 根据输出结果，该数据集中不存在缺失值。
        print(data.describe(include='all'))

        return data
    except Exception as e:
        print("数据获取出现异常：", e)
        return None


# 初始绘图
def plotting_1(data):
    # 绘制带有核密度估计曲线（KDE）和直方图的分布图
    plt.figure(figsize=(9, 5))
    sns.distplot(data["Average annual electric use (kWh)"])
    plt.xlim(0, 40000)
    # 保存图片
    plt.savefig('img/电力使用分布图.png')
    # 显示图形
    plt.show()

    # 绘制热力图 显示不同收入范围和居住人数条件下的平均年电力使用量
    dataset = data.pivot_table(index='Median income range', columns='Number of occupants',
                               values='Average annual electric use (kWh)', aggfunc='mean')
    plt.figure(figsize=(9, 5))
    sns.heatmap(dataset)
    # 保存图片
    plt.savefig('img/电力使用热力图.png')
    # 显示图形
    plt.show()

    # 绘制三个子条形图 分别展示不同属性（浴室数量、厨房数量、壁炉数量）与平均年度电力使用量之间的关系
    plt.figure(figsize=(15, 6))
    # 子图1——浴室数量
    ax1 = plt.subplot(1, 3, 1)
    plt.title("bathrooms")
    sns.barplot(x="Total number of bathrooms", y="Average annual electric use (kWh)", hue="Total number of bathrooms",
                data=data, log=True)
    # 子图2——厨房数量
    ax2 = plt.subplot(1, 3, 2)
    plt.title("kitchens")
    sns.barplot(x="Number of kitchens", y="Average annual electric use (kWh)", hue="Number of kitchens", data=data,
                log=True)
    # 子图3——壁炉数量
    ax3 = plt.subplot(1, 3, 3)
    plt.title("fireplaces")
    sns.barplot(x="Number of fireplaces", y="Average annual electric use (kWh)", hue="Number of fireplaces", data=data,
                log=True)
    plt.subplots_adjust(wspace=0.5, hspace=15)
    # 保存图片
    plt.savefig('img/多子图条形图.png')
    # 显示图形
    plt.show()


# 数据处理
def data_processing(data):
    categorical_cols = ['Square footage range', 'Number of bedrooms', 'Total number of bathrooms', 'Number of kitchens',
                        'Number of fireplaces', 'Number of occupants', 'Median income range']
    numeric_cols = list(set(data.columns) - set(categorical_cols))

    ohe = OneHotEncoder(drop='first', sparse=False)
    data = np.hstack((ohe.fit_transform(data[categorical_cols]), data[numeric_cols]))
    cols = sum([(categorical_cols[i] + '_' + ohe.categories_[i][1:]).tolist() for i in range(len(categorical_cols))],
               []) + numeric_cols
    data = pd.DataFrame(data, columns=cols)

    id = list(data.index)
    data['id'] = id
    data.to_csv(os.getcwd() + "\\data\\data_power_consumption.csv")


# 构建模型 同时为了便于后期可视化分析 建立一个绘制图像的类
class EnergyFingerPrints():

    def __init__(self, data):
        # 统计每个聚类簇的中心点
        self.means = []
        self.data = data

    # 肘部法
    def elbow_method(self, n_clusters, num):
        # 拐点计算，计算不同聚类簇数量下各簇内样本点到中心点距离之和，对比选择最合适的聚类簇数n_clusters
        fig, ax = plt.subplots(figsize=(8, 4))
        distortions = []

        # 通过迭代尝试不同数量的聚类，并计算每个聚类数量下的到中心点的距离之和（称为“失真”）
        for i in range(1, n_clusters):
            km = KMeans(n_clusters=i,
                        init='k-means++',  # 初始中心簇的获取方式，k-means++一种比较快的收敛的方法
                        n_init=10,  # 初始中心簇的迭代次数
                        max_iter=300,  # 数据分类的迭代次数
                        random_state=0)  # 初始化中心簇的方式
            km.fit(self.data)
            distortions.append(km.inertia_)  # inertia计算样本点到最近的中心点的距离之和

        # 绘制出聚类数量与失真之间的关系图。通过观察图形，可以找到失真开始快速下降的拐点
        # 该拐点对应的聚类数量通常被认为是最佳的选择
        plt.plot(range(1, n_clusters), distortions, marker='o', lw=1)
        plt.xlabel('聚类数量')
        plt.ylabel('至中心点距离之和')
        # 保存图片
        if num == 1:
            plt.savefig('img/聚类数量与失真关系图-用户基本属性特征.png')
        if num == 2:
            plt.savefig('img/聚类数量与失真关系图-电器设备使用特征.png')
        # 显示图形
        plt.show()

    def get_cluster_counts(self):  # 返回各簇中用户的数量
        return pd.Series(self.predictions).value_counts()

    def get_cluster(self):  # 获取聚类标签
        return pd.DataFrame(self.predictions)

    def labels(self, n_clusters):  # 按照指定的聚类簇数量对给定数据进行聚类分析
        self.n_clusters = n_clusters
        return KMeans(self.n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0).fit(self.data).labels_

    def fit(self, n_clusters):  # 基于划分簇的数量，对数据进行聚类分析
        self.n_clusters = n_clusters
        self.kmeans = KMeans(self.n_clusters)
        self.predictions = self.kmeans.fit_predict(self.data)


# 用户基本属性特征
def basic_user_attribute_characteristics(dataset):
    # 获取特征
    data = dataset[['Square footage range_<= 1,500', 'Square footage range_>=2500', 'Median income range_$50k - $100k',
                    'Median income range_< $50k', 'Median income range_> $150k', 'Average annual electric use (kWh)',
                    'Number of occupants_Less than 3', 'Number of occupants_More than 4']]

    # 集群划分
    cls_data = cluster_model(data, 1)

    return cls_data


# 电器设备使用特征
def usage_characteristics_of_electrical_equipment(dataset):
    # 获取特征
    data = dataset[['Number of bedrooms_3', 'Number of bedrooms_4 or more', 'Total number of bathrooms_2 or 2.5',
                    'Total number of bathrooms_3 or more', 'Number of kitchens_2 or more',
                    'Average annual electric use (kWh)', 'Number of fireplaces_1 or more']]

    # 集群划分
    cls_data = cluster_model(data, 2)

    return cls_data


# 集群划分-聚类模型
def cluster_model(data, num):
    # 数据转换为numpy数组 便于分析
    data = np.array(data)

    # 实例化模型对象
    energy_clusters = EnergyFingerPrints(data)
    # 肘部法=>最佳聚类簇--4
    energy_clusters.elbow_method(n_clusters=13, num=num)
    # 聚类分析
    energy_clusters.fit(n_clusters=4)

    # 保存模型
    joblib.dump(energy_clusters, '电力用户集群分析模型.pkl')
    # # 加载模型
    # loaded_kmeans = joblib.load('电力用户集群分析模型.pkl')

    # 获取各簇数量
    count = energy_clusters.get_cluster_counts()
    print("统计各个簇的数量")
    print(count)

    # 将用户id与其所属的聚类簇进行关联，并将结果保存在DataFrame中
    # 获取样本所属簇标签
    group = energy_clusters.labels(n_clusters=4)
    # 读取文件 获取用户id列
    data2 = pd.read_csv(os.getcwd() + "\\data\\data_power_consumption.csv")
    num = data2['id']
    # 创建DataFrame 其中包含用户id
    cls = pd.DataFrame(list(num))
    # 将聚类标签列添加到DataFrame中。
    cls['cluster'] = list(group)
    # 更改DataFrame列名
    cls.columns = ['id', 'cluster']
    # 根据聚类标签排序
    cls = cls.sort_values(by='cluster', ascending=True)
    # 重置DataFrame索引 确保索引连续
    cls.reset_index(drop=True)

    # 输出各簇类样本的用户id
    print("# 第一类:")
    print(np.array(cls.loc[cls.cluster == 0].id))
    print("# 第二类:")
    print(np.array(cls.loc[cls.cluster == 1].id))
    print("# 第三类:")
    print(np.array(cls.loc[cls.cluster == 2].id))
    print("# 第四类:")
    print(np.array(cls.loc[cls.cluster == 3].id))

    # 将聚类结果与原始数据合并 并重置索引
    # 获取各样本簇类标签
    cls = energy_clusters.get_cluster()
    # 将原始数据转为dataframe类型
    cls_data = pd.DataFrame(data)
    # 合并
    cls_data = pd.merge(cls_data, cls, left_index=True, right_index=True)
    # 重置索引
    cls_data = cls_data.reset_index()

    return cls_data


# 聚类绘图
def plotting_2(cls_data, num):
    # 绘制集群划分散点图
    color = ["red", "pink", "orange", "gray"]
    plt.figure(figsize=(15, 10))
    for i in range(4):
        plt.scatter(cls_data.loc[cls_data['0_y'] == i, 'index'], cls_data.loc[cls_data['0_y'] == i, 5]
                    # 就是取出y_pred是0，1，2，3的那一簇的X
                    , marker='o'  # 点的形状
                    , s=8  # 点的大小
                    , c=color[i]
                    , label=f'Cluster {i}'
                    )
    # 添加图例
    plt.legend(loc='upper left')  # 设置图例位置
    plt.xlabel('样本索引')  # 设置 x 轴标签
    plt.ylabel('年均用电量')  # 设置 y 轴标签
    # 保存图片
    if num == 1:
        plt.savefig('img/集群划分散点图-用户基本属性特征.png')
    if num == 2:
        plt.savefig('img/集群划分散点图-电器设备使用特征.png')
    # 显示图形
    plt.show()

    # 绘制集群划分折线图
    plt.figure(figsize=(15, 10))
    for i in range(4):
        cluster_data = cls_data[cls_data['0_y'] == i]
        plt.plot(cluster_data.index, cluster_data[5], label=f'Cluster {i}')
    # 添加图例
    plt.legend(loc='upper left')  # 设置图例位置
    plt.xlabel('样本索引')  # 设置 x 轴标签
    plt.ylabel('年均用电量')
    # 保存图片
    if num == 1:
        plt.savefig('img/集群划分折线图-用户基本属性特征.png')
    if num == 2:
        plt.savefig('img/集群划分折线图-电器设备使用特征.png')
    # 显示图形
    plt.show()


# 依据不同特征对用户进行集群划分
def feature_partitioning():
    # 获取总数据
    dataset = pd.read_csv(os.getcwd() + "\\data\\data_power_consumption.csv")

    # 根据不同的标准对用户进行集群划分
    # 用户基本属性
    cls_data_1 = basic_user_attribute_characteristics(dataset)
    # 电器设备使用
    cls_data_2 = usage_characteristics_of_electrical_equipment(dataset
                                                               )
    return cls_data_1, cls_data_2


# 主函数
def main():
    # 数据获取
    data = data_acquisition()

    if data is not None:
        # 初始绘图（3张图 保存至./img）
        plotting_1(data)

        # 数据处理
        data_processing(data)

        # 依据不同特征对用户进行集群划分
        cls_data_1, cls_data_2 = feature_partitioning()

        # 集群划分绘图（散点图 + 折线图）
        plotting_2(cls_data_1, 1)
        plotting_2(cls_data_2, 2)
    else:
        print("无法进行数据处理和分析，因为数据获取失败。")


# 运行主函数
if __name__ == "__main__":
    main()
