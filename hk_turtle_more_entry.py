import hk_stock_backtest_more as hk_more
import pandas as pd

#'2023-09-01', '2023-09-08'受台风影响临时休市
#================腾讯================
# hk_more.STOCK_CODE = "00700.HK"
# hk_more.STOCK_NAME = "腾讯"
# hk_more.dividend_info = pd.DataFrame({
#     'ex_date': ['20260515'], # 假设未来还有几次分红
#     'dividend': [5.3]             # 对应的分红金额
# })
# # # 确保 ex_date 是字符串格式，方便后续匹配
# hk_more.dividend_info['ex_date'] = hk_more.dividend_info['ex_date'].astype(str)
# hk_more.INITIAL_CAPITAL = 400000.0

#================泡泡玛特================
# hk_more.STOCK_CODE = "09992.HK"
# hk_more.STOCK_NAME = "泡泡玛特"
# hk_more.dividend_info = pd.DataFrame({
#     'ex_date': ['20260515'], # 假设未来还有几次分红
#     'dividend': [2.7248]             # 对应的分红金额
# })
# hk_more.dividend_info['ex_date'] = hk_more.dividend_info['ex_date'].astype(str)
# hk_more.YEARS = 5
# hk_more.PER_SHARES = 200

#================老铺黄金================
# hk_more.STOCK_CODE = "06181.HK"
# hk_more.STOCK_NAME = "老铺黄金"
# hk_more.dividend_info = pd.DataFrame({
#     'ex_date': ['20251120', '20260512'], # 假设未来还有几次分红
#     'dividend': [10.52, 13.65]             # 对应的分红金额
# })
# hk_more.dividend_info['ex_date'] = hk_more.dividend_info['ex_date'].astype(str)
# hk_more.INITIAL_CAPITAL = 400000.0
# hk_more.YEARS = 2

#================美团================
# #2018-09-20开市
# hk_more.STOCK_CODE = "03690.HK"
# hk_more.STOCK_NAME = "美团"

#================小米================
#2018-07-09开市
hk_more.STOCK_CODE = "01810.HK"
hk_more.STOCK_NAME = "小米"
hk_more.INITIAL_CAPITAL = 100000.0


def main():
    try:
        hk_more.main()
    except Exception as e:
        print(f"❌ 错误：{e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()