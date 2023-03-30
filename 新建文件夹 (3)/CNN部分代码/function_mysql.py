import pymysql

def conn_mysql(account, begin, end):
    """
    根据用户账号、起始时间、截止时间来获取相应的用户生理数据
    :param account: 用户账号
    :param begin: 起始时间
    :param end: 截止时间
    :return:
    """
    try:
        connc = pymysql.Connect(
            host='122.9.34.21',
            user='root',
            # password="hhIWm$XS47scbUX*!Q45i6rQ",
            password="hhIWm!XS47scbUX_!Q34i5rQ",
            database='jytech',
            port=3306,
            charset='utf8'
        )
        cur = connc.cursor()
        sql = "SELECT * FROM test_normal_second WHERE account=%s AND time BETWEEN %s AND %s; "
        time_list = [account, begin, end]
        cur.execute(sql, time_list)
        result = cur.fetchall()
        cur.close()
        connc.close()
        return result
    except Exception as e:
        return e


