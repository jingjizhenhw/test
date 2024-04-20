import pymysql


class MySQLHelper:
    def __init__(self, host, port, user, password, database):
        self.host = host  # 主机号
        self.port = port  # 端口号
        self.user = user  # 用户名
        self.password = password  # 密码
        self.database = database  # 数据库名
        # 连接对象pymysql赋值至类实例self的conn属性，用于接收数据库连接
        self.conn = pymysql.connect(host=host, port=port, user=user, password=password, database=database)  #

    # 上下文管理器：使用 with 语句管理数据库连接，以确保在退出时正确关闭连接。
    # 这可以通过实现 __enter__ 和 __exit__ 方法来实现。
    # 可用于文件处理、数据库连接、网络连接和锁管理等
    # 上下文管理器使得资源管理更加方便、清晰，并有助于避免资源泄漏和错误处理。
    # 在 f-string 中，变量名被放置在 {} 中，并在运行时替换为变量的实际值。
    # 这种方法比使用传统的字符串连接或格式化操作更简洁和易读。

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:  # 检查数据库连接对象是否存在，避免潜在异常
            self.conn.close()  # 关闭数据库连接

    # 执行sql语句 无返回值
    def execute(self, sql: str, params: tuple = ()):
        try:
            with self.conn.cursor() as cursor:  # 游标对象 用于执行sql语句
                cursor.execute(sql, params)  # sql有占位符则使用params内具体数值替代，否则直接执行语句
            self.conn.commit()  # 提交事务，确保数据库一致性（慎重提交！！！）
        except Exception as e:
            print(f"Error executing SQL: {e}")

    # 查询并返回全部数据
    def query(self, sql: str, params: tuple = ()):
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(sql, params)
                result = cursor.fetchall()
            return result
        except Exception as e:
            print(f"Error querying the database: {e}")
            return None

    # 使用占位符和参数绑定可防止SQL注入攻击
    # 它确保了用户输入的数据不会直接嵌入到SQL语句中，而是通过参数传递，数据库会对输入进行适当的处理
    # 数据库对输入的处理主要涉及到参数化查询/使用预处理语句
    # 参数化查询将 SQL 查询语句中的变量部分使用占位符表示，然后将实际的参数值通过参数绑定的方式传递给数据库执行
    # 预处理语句将参数值在执行时被传递给预处理语句，而不是直接嵌入到 SQL 语句中。是一种在执行之前对 SQL 语句进行编译的方式。
    # f'' 是一种称为 f-string（格式化字符串字面值）的字符串格式。
    # 它允许在字符串中嵌入表达式，并在运行时将它们求值。
    # {table}、{fields} 和 {values} 是在运行时替换为相应变量的占位符。

    # 插入一条数据
    def insert(self, table, data: dict):
        fields = ','.join(data.keys())
        values = ','.join(['%s'] * len(data))
        sql = 'INSERT INTO %s (%s) VALUES (%s)' % (table, fields, values)
        self.execute(sql, tuple(data.values()))

    # 修改数据
    def update(self, table, data: dict, where: str = '1=2', where_params: tuple = ()):
        set_values = ','.join(['%s = %s' % (field, '%s') for field in data.keys()])
        sql = 'UPDATE %s SET %s WHERE %s' % (table, set_values, where)
        # where内存在占位符则需要where_params传递数值
        params = tuple(data.values()) + where_params
        self.execute(sql, params)

    # 删除数据
    def delete(self, table, where: str = '1=2', where_params: tuple = ()):
        sql = 'DELETE FROM %s WHERE %s' % (table, where)
        self.execute(sql, where_params)


# 测试
if __name__ == "__main__":
    with MySQLHelper("localhost", 3306, "root", "hw123321", "electric") as helper:
        result = helper.query("select * from user_info")
        print(result)
