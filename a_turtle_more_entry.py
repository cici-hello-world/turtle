import a_stock_backtest_more as a_more

# STOCK_CODE = "300308.SZ"
# STOCK_NAME = "中际旭创"
# STOCK_CODE = "002230"
# STOCK_NAME = "科大讯飞"
# STOCK_CODE = "300059.SZ"
# STOCK_NAME = "东方财富"
# STOCK_CODE = "002475.SZ"
# STOCK_NAME = "立讯精密"
# STOCK_CODE = "002371.SZ"
# STOCK_NAME = "北方华创"
# STOCK_CODE = "002594.SZ"
# STOCK_NAME = "比亚迪"
# STOCK_CODE = "600036.SH"
# STOCK_NAME = "招商银行"
# STOCK_CODE = "000625.SZ"
# STOCK_NAME = "长安汽车"
# STOCK_CODE = "601899.SH"
# STOCK_NAME = "紫金矿业"

# STOCK_CODE = "000400.SZ"
# STOCK_NAME = "许继电气"
# STOCK_CODE = "688981.SH"
# STOCK_NAME = "中芯国际"

#================工业富联================
#2018-06-08上市
# a_more.STOCK_CODE = "601138.SH"
# a_more.STOCK_NAME = "工业富联"

#================长安汽车================
a_more.STOCK_CODE = "000625.SZ"
a_more.STOCK_NAME = "长安汽车"
a_more.INITIAL_CAPITAL = 100000.0


#================华天科技================
# a_more.STOCK_CODE = "002185.SZ"
# a_more.STOCK_NAME = "华天科技"
# a_more.INITIAL_CAPITAL = 100000.0


#================中国电影================
# a_more.STOCK_CODE = "600977.SH"
# a_more.STOCK_NAME = "中国电影"
# a_more.INITIAL_CAPITAL = 100000.0


#================特锐德================
# a_more.STOCK_CODE = "300001.SZ"
# a_more.STOCK_NAME = "特锐德"
# a_more.INITIAL_CAPITAL = 100000.0


def main():
    try:
        a_more.main()
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()