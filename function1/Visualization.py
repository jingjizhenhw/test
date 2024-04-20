# localhost:5000
import os

from flask import *
from function1.MySQLHelper import *

# 数据库配置
db_config = {
    'host': 'localhost',
    'port': 3306,
    'user': 'root',
    'password': 'hw123321',
    'database': 'electric'
}

app = Flask(__name__, template_folder='./templates')


@app.route('/')
def liandong():
    data1 = []
    data2 = []
    with MySQLHelper(**db_config) as helper:
        type_data = helper.query('select user_type,count(*) as type_num from user_classify group by user_type')
    for i in type_data:
        data1.append(i[0])
        data2.append(i[1])
    print("图表联动(路由)！")
    return render_template("Chart_combination.html", data=type_data, data1=data1, data2=data2)


if __name__ == "__main__":
    app.run()
