from function1.MySQLHelper import *

# 数据库配置
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'hw123321',
    'database': 'electric'
}


# main
def classify_users():
    # 查询用户用电信息
    with MySQLHelper(**db_config) as helper:
        user_data = helper.query("SELECT user_id, SUM(user_money) as money_sum, COUNT(*) as times FROM user_info "
                                 "GROUP BY user_id")
        if not user_data:
            print("No user data found.")
            return

    # 计算全体用户缴费总和、缴费次数及平均值
    total_money = sum(user[1] for user in user_data)
    total_times = sum(user[2] for user in user_data)
    user_num = len(user_data)  # 用户人数

    average_money = total_money / user_num if user_num > 0 else 0
    average_times = total_times / user_num if user_num > 0 else 0

    # 批量插入数据 减少数据库交互次数
    # 已成功执行 故注释
    # with MySQLHelper(**db_config) as helper:
    #     for user in user_data:
    #         user_id = user[0]
    #         money_sum = user[1]
    #         times = user[2]
    #
    #         user_type = classify_user(money_sum, times, average_money, average_times)
    #
    #         # 添加数据到user_classify表
    #         insert_sql = '''
    #                         INSERT INTO user_classify (user_id, money_sum, times, ave_money, ave_times, user_type)
    #                         VALUES (%s, %s, %s, %s, %s, %s)
    #                     '''
    #         values = {
    #             'user_id': user_id,
    #             'money_sum': money_sum,
    #             'times': times,
    #             'ave_money': average_money,
    #             'ave_times': average_times,
    #             'user_type': user_type
    #         }
    #         helper.insert('user_classify', values)


# 分类客户--初步
def classify_user(money_sum, times, average_money, average_times):
    if money_sum >= average_money:
        if times >= average_times:
            return '高价值型客户'
        else:
            return '潜力型客户'
    else:
        if times >= average_times:
            return '大众型客户'
        else:
            return '低价值型客户'


if __name__ == "__main__":
    classify_users()

