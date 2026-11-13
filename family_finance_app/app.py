#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
家庭资产十年推演 —— Streamlit 网页应用
======================================
客户输入自己的资产负债表、收入、支出、投资参数，
实时推演未来 10 年（可调）的资产增长，支持导出 Excel 报表。

运行：
    streamlit run family_finance_app/app.py
"""
import io
import os
import sys
from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

# 复用命令行脚本的计算引擎（run_simulation / build_excel）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import family_balance_sheet as engine

st.set_page_config(page_title="家庭净资产计算器", page_icon="🏠", layout="wide")



# ---------------------------------------------------------------------------
# 图表样式：分类调色板（固定槽位 1-6，色盲安全）、浅色面墨色规范
# ---------------------------------------------------------------------------
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ITEMS = ("现金", "房产", "定投基金", "股票", "投资类保险", "公积金账户", "有息负债")
ASSETS = ("现金", "房产", "定投基金", "股票", "投资类保险", "公积金账户")

# 意见反馈：默认追加写入本地 CSV；上线后如需云端收集，
# 把 FEEDBACK_WEBHOOK 设为接收 POST 的服务地址（金数据/问卷星/自建接口均可）
FEEDBACK_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "feedback.csv")
FEEDBACK_WEBHOOK = ""


def save_feedback(text):
    """把反馈追加写入本地 CSV；配置了 webhook 时同时上报。"""
    if not text or not text.strip():
        return False
    row = {"时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
           "反馈": text.strip()}
    df = pd.DataFrame([row])
    if not os.path.exists(FEEDBACK_FILE):
        df.to_csv(FEEDBACK_FILE, index=False, encoding="utf-8-sig")
    else:
        df.to_csv(FEEDBACK_FILE, mode="a", header=False, index=False,
                  encoding="utf-8-sig")
    if FEEDBACK_WEBHOOK:
        try:
            import requests
            requests.post(FEEDBACK_WEBHOOK, json=row, timeout=5)
        except Exception:
            pass
    return True


def net(b):
    return sum(b[n] for n in ASSETS) - b["有息负债"]


def style_altair(chart, height=380):
    """统一的图表外观：细线、弱化网格/轴、顶部图例、悬停提示。"""
    return (chart
            .properties(height=height, background=SURFACE)
            .configure_view(strokeWidth=0)
            .configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor=GRID,
                            domainColor=AXIS, tickColor=AXIS)
            .configure_legend(titleColor=MUTED, labelColor=INK, orient="top",
                              padding=4, columns=8)
            .configure_title(color=INK, fontSize=15, anchor="start"))


def line_chart(df, x, y, color, domain, slots, tooltip_fields, x_sort=None):
    """按调色板规范画多序列折线图（2px 线、圆点、悬停提示）。

    x_sort: 可选，显式指定横轴类别顺序；用于让月度现金流的月份顺序
            与 Excel「第N期现金流」sheet 的「月份」列完全一致。
    """
    x_kwargs = {"axis": alt.Axis(labelAngle=0)}
    if x_sort is not None:
        x_kwargs["sort"] = x_sort
    return alt.Chart(df).mark_line(
        strokeWidth=2,
        point=alt.OverlayMarkDef(size=45, filled=True),
    ).encode(
        x=alt.X("{}:O".format(x), **x_kwargs),
        y=alt.Y("{}:Q".format(y), axis=alt.Axis(format="~s")),
        color=alt.Color("{}:N".format(color),
                        scale=alt.Scale(domain=domain, range=slots),
                        legend=alt.Legend(title=None)),
        tooltip=tooltip_fields,
    )


def parse_insurance_df(df):
    """解析保险月份明细编辑表 -> {月份: 金额}"""
    out = {}
    if df is None or df.empty:
        return out
    for row in df.to_dict("records"):
        m = pd.to_numeric(row.get("月份"), errors="coerce")
        v = pd.to_numeric(row.get("金额"), errors="coerce")
        if pd.isna(m) and pd.isna(v):
            continue
        if pd.isna(m) or pd.isna(v):
            raise ValueError("「保险月份明细」存在未填完整的行")
        m = int(m)
        if not 1 <= m <= 12:
            raise ValueError("「保险月份明细」月份必须在 1-12：{}".format(m))
        out[m] = float(v)
    return out


def parse_one_off(df):
    """解析一次性开支编辑表 -> {(期数, 月份): 金额}"""
    out = {}
    if df is None or df.empty:
        return out
    for row in df.to_dict("records"):
        y = pd.to_numeric(row.get("期数"), errors="coerce")
        m = pd.to_numeric(row.get("月份"), errors="coerce")
        v = pd.to_numeric(row.get("金额"), errors="coerce")
        if pd.isna(y) and pd.isna(m) and pd.isna(v):
            continue
        if pd.isna(y) or pd.isna(m) or pd.isna(v):
            raise ValueError("「一次性开支」存在未填完整的行")
        y, m = int(y), int(m)
        if y < 1 or not 1 <= m <= 12:
            raise ValueError("「一次性开支」期数/月份不合法：第{}期{}月".format(y, m))
        out[(y, m)] = float(v)
    return out


def parse_living_adj(df):
    """解析生活费调整编辑表 -> [((期数, 月份), 新月生活费)]"""
    out = []
    if df is None or df.empty:
        return out
    for row in df.to_dict("records"):
        y = pd.to_numeric(row.get("期数"), errors="coerce")
        m = pd.to_numeric(row.get("月份"), errors="coerce")
        v = pd.to_numeric(row.get("新月生活费"), errors="coerce")
        if pd.isna(y) and pd.isna(m) and pd.isna(v):
            continue
        if pd.isna(y) or pd.isna(m) or pd.isna(v):
            raise ValueError("「生活费调整」存在未填完整的行")
        y, m = int(y), int(m)
        if y < 1 or not 1 <= m <= 12:
            raise ValueError("「生活费调整」期数/月份不合法：第{}期{}月".format(y, m))
        out.append(((y, m), float(v)))
    return out


# ---------------------------------------------------------------------------
# 页面：标题与输入区
# ---------------------------------------------------------------------------

st.title("家庭净资产计算器")
st.caption("输入资产负债表、收入、支出与投资参数，计算未来每年资产增长率。"
           "系统不保存资产数据，可下载Excel自行保存计算结果；全部参数可随时调整，结果实时更新。")
D = engine.CONFIG # 默认值取命令行脚本的当前配置
with st.expander("① 第0期资产负债表", expanded=True):
    c = st.columns(3)
    cash0 = c[0].number_input("现金", value=0.0, step=10000.0)
    house0 = c[1].number_input("房产", value=0.0, step=10000.0)
    sp0 = c[2].number_input("定投基金", value=0.0, step=10000.0)
    c = st.columns(5)
    stock0 = c[0].number_input("股票", value=0.0, step=10000.0)
    ins_inv0 = c[1].number_input("投资类保险", value=0.0, step=10000.0)
    gjj0 = c[2].number_input("公积金", value=0.0, step=10000.0)
    debt0 = c[3].number_input("有息负债(剩余本金)", value=0.0, step=10000.0)
    gjj_base_month = c[4].number_input("0期记账月份", 1, 12, 1)
    c = st.columns(1)
    years = c[0].slider("计算年数", 1, 30, 10)

with st.expander("② 收入与支出（元）(现金流)", expanded=True):
    c = st.columns(3)
    loan_rate_pct = c[0].number_input("贷款年利率(%)", 0.0, 10.0, float(D["贷款年利率"]) * 100, 0.1)
    loan_months = c[1].number_input("贷款剩余期数(月)", 0, 360, int(D["贷款剩余期数_月"]), 12)
    loan_mode = c[2].selectbox("贷款类型", ["等额本息", "等额本金"], index=0)
    # 月供提示：随上方贷款参数实时计算
    loan_annuity = engine.monthly_payment(debt0, loan_rate_pct / 100.0, int(loan_months))
    if loan_mode == "等额本金":
        loan_first = ((debt0 / loan_months + debt0 * (loan_rate_pct / 100.0) / 12.0)
                      if loan_months > 0 else 0.0)
        st.caption("月供（等额本金）：首月 {:,.0f} 元，逐月递减（贷款还清后为 0）".format(
            loan_first))
    else:
        st.caption("月供（等额本息）：{:,.0f} 元/月（若贷款已还清则为 0）".format(
            loan_annuity))
    c = st.columns(3)
    gjj_offset = c[0].toggle("是否公积金冲贷", value=False)
    gjj_monthly = c[1].number_input("每月缴存公积金", value=0.0, step=500.0)
    gjj_month = c[2].number_input("公积金提取月份", 1, 12, 1)
    # 规则：额度由页面按「当年预估全年月供合计与提取月账户余额取小」自动算出，
    #       作为真实数值传入引擎（不可手工修改，不再传 None）；
    #       第2期起引擎仍按前12个月实付月供之和自动计算，且始终不超过账户余额；
    #       冲贷开 = 0。
    annual_mortgage = engine.year_mortgage_total(
        debt0, loan_mode, loan_annuity,
        (debt0 / loan_months) if loan_months > 0 else 0.0,
        loan_rate_pct / 100.0 / 12.0)
    gjj_month_acct = gjj0 + ((12 if int(gjj_month)<int(gjj_base_month) else 0) + int(gjj_month) - int(gjj_base_month))* gjj_monthly
    default_cap = (0.0 if gjj_offset
                   else min(annual_mortgage,gjj_month_acct))
    gjj_cap = st.number_input("可提取公积金(每年)（自动计算，不可修改）",
                              max_value=float(gjj_month_acct),
                              value=default_cap, step=10000.0, disabled=True)
    c = st.columns(3)
    salary = c[0].number_input("月薪(税后)", value=0.0, step=1000.0)
    bonus = c[1].number_input("年终奖(每年)", value=0.0, step=10000.0)
    bonus_month = c[2].number_input("年终奖发放月份", 1, 12, 1)
    c = st.columns(2)
    living = c[0].number_input("每月生活费", value=0.0, step=500.0)
    social = c[1].number_input("每月社保", value=0.0, step=100.0)

with st.expander("③ 投资参数", expanded=True):
    c = st.columns(3)
    dca1 = c[0].number_input("基金定投-第1年每月", value=0.0, step=1000.0)
    dca2 = c[1].number_input("基金定投-第2年起每月", value=0.0, step=1000.0)
    r_pct = c[2].slider("定投基金 年化收益率 R(%)", 0.0, 20.0, 0.0, 0.1)
    c = st.columns(3)
    stock_ret_pct = c[0].number_input("股票年收益率(%)", -50.0, 50.0, 0.0, 0.5)
    house_gain_pct = c[1].number_input("房产年涨幅(%)", -10.0, 20.0, 0.0, 0.5)
    ins_return_pct = c[2].number_input("保险年收益率(%)", -10.0, 20.0, 0.0, 0.5)

with st.expander("④ 其他设置（保险 / 一次性开支 / 生活费调整）", expanded=False):
    st.write("**保险月份明细**：按月份缴纳，未列出的月份为 0。")
    insurance_df = st.data_editor(
        pd.DataFrame(columns=["月份", "金额"]),
        num_rows="dynamic", key="insurance_editor", width="stretch")
    st.write("**一次性开支**：指定期数+月份的一笔额外支出（如第1期6月装修支出 100,000）。")
    one_off_df = st.data_editor(
        pd.DataFrame(columns=["期数", "月份", "金额"]),
        num_rows="dynamic", key="one_off_editor", width="stretch")
    st.write("**生活费调整**：自指定期数+月份起，每月生活费变为新值。")
    living_adj_df = st.data_editor(
        pd.DataFrame(columns=["期数", "月份", "新月生活费"]),
        num_rows="dynamic", key="living_adj_editor", width="stretch")

# ---------------------------------------------------------------------------
# 组装配置并运行引擎
# ---------------------------------------------------------------------------

try:
    cfg = {
        "现金": cash0, "房产": house0, "定投基金": sp0, "股票": stock0,
        "投资类保险": ins_inv0, "公积金账户": gjj0, "有息负债": debt0,
        "0期记账月份": int(gjj_base_month),
        "贷款类型": loan_mode,
        "贷款年利率": loan_rate_pct / 100.0, "贷款剩余期数_月": int(loan_months),
        "月薪": salary, "公积金冲贷": bool(gjj_offset),
        "每月缴存公积金": gjj_monthly,
        "可提取公积金(每年)": gjj_cap, "年终奖": bonus,
        "年终奖月份": int(bonus_month), "公积金月份": int(gjj_month),
        "每月生活费": living, "每月社保": social,
        "保险月份明细": parse_insurance_df(insurance_df),
        "一次性开支": parse_one_off(one_off_df),
        "生活费调整": parse_living_adj(living_adj_df),
        "定投基金定投_第1年每月": dca1, "定投基金定投_第2年起每月": dca2,
        "R": r_pct / 100.0,
        "股票年收益率": stock_ret_pct / 100.0, "房产年涨幅": house_gain_pct / 100.0,
        "保险年收益率": ins_return_pct / 100.0,
        "年数": int(years),
        "excel输出路径": None, "现金流期数": None,
    }
    results = engine.run_simulation(cfg)
except ValueError as e:
    st.error("参数有误：{}".format(e))
    st.stop()

records = results["records"]
last_curr = records[-1][2]
net0, netN = net(records[0][1]), net(last_curr)
growth = netN - net0
cagr = (netN / net0) ** (1.0 / years) - 1.0 if net0 > 0 else float("nan")

# ---------------------------------------------------------------------------
# 结果区
# ---------------------------------------------------------------------------

st.divider()
st.subheader("推演结果")

c = st.columns(4)
c[0].metric("第{}期净资产".format(years), "{:,.0f} 元".format(netN),
            "{:+,.0f} 元".format(growth))
c[1].metric("净资产年化复合增速", "{:.1%}".format(cagr) if not pd.isna(cagr) else "—")
c[2].metric("第{}期现金".format(years), "{:,.0f} 元".format(last_curr["现金"]))
c[3].metric("第{}期定投基金".format(years), "{:,.0f} 元".format(last_curr["定投基金"]))

# --- 图1：净资产与各科目走势 ---
balance_domain = ["净资产", "现金", "房产", "定投基金", "股票", "投资类保险", "公积金", "有息负债"]
BALANCE_KEYS = {"净资产": None, "现金": "现金", "房产": "房产", "定投基金": "定投基金",
                "股票": "股票", "投资类保险": "投资类保险",
                "公积金": "公积金账户", "有息负债": "有息负债"}


def bal_value(bs, name):
    """按显示名取值（公积金显示名对应配置键 公积金账户）。"""
    if name == "净资产":
        return net(bs)
    return bs[BALANCE_KEYS[name]]


bs_rows = []
for name in balance_domain:
    bs_rows.append({
        "科目": name,
        "第0期": bal_value(records[0][1], name),
        **{"第{}期".format(y): bal_value(cu, name) for y, _, cu in records},
    })
bs_df = pd.DataFrame(bs_rows).set_index("科目")

chart_rows = []
for name in balance_domain:
    for y, v in zip([0] + [y for y, _, _ in records], bs_df.loc[name]):
        chart_rows.append({"期数": y, "科目": name, "金额(元)": v})
chart_df = pd.DataFrame(chart_rows)

st.altair_chart(style_altair(line_chart(
    chart_df, "期数", "金额(元)", "科目", balance_domain, SLOTS[:8],
    [alt.Tooltip("期数:O", title="期数"), alt.Tooltip("科目:N"),
     alt.Tooltip("金额(元):Q", format=",.0f")])), width="stretch")

# --- 图2：历年收入 / 支出 / 净现金流 ---
flow_domain = ["收入", "支出", "净现金流"]
cf_rows = []
for e in results["yearly"]:
    income = sum(e["inc"].values())
    expense = sum(e["exp"].values()) + e["mortgage"]["月供"]
    for name, v in (("收入", income), ("支出", expense), ("净现金流", e["net_cash"])):
        cf_rows.append({"期数": e["year"], "科目": name, "金额(元)": v})
cf_df = pd.DataFrame(cf_rows)

st.altair_chart(style_altair(line_chart(
    cf_df, "期数", "金额(元)", "科目", flow_domain, SLOTS[:3],
    [alt.Tooltip("期数:O", title="期数"), alt.Tooltip("科目:N"),
     alt.Tooltip("金额(元):Q", format=",.0f")])), width="stretch")

# --- 逐年资产负债表 ---
st.subheader("逐年资产负债表（元）")
st.dataframe(bs_df.style.format("{:,.0f}"), width="stretch")

# --- 月度现金流 ---
st.subheader("月度现金流")
flow_year = st.selectbox("选择期数查看月度现金流", list(range(1, years + 1)),
                         index=years - 1)
monthly = results["monthly_by_year"][flow_year]
# 横轴顺序直接取自写入 Excel 的同一份月度明细（自 0期记账月次月起 12 个月，
# 跨年回绕），避免图表把月份按数值升序重排成 1、2…12
month_order = [str(m["月份"]) for m in monthly]
monthly_rows = []
for m in monthly:
    for name in flow_domain:
        monthly_rows.append({"月份": str(m["月份"]), "科目": name, "金额(元)": m[name]})
monthly_df = pd.DataFrame(monthly_rows)
st.altair_chart(style_altair(line_chart(
    monthly_df, "月份", "金额(元)", "科目", flow_domain, SLOTS[:3],
    [alt.Tooltip("月份:O", title="月份"), alt.Tooltip("科目:N"),
     alt.Tooltip("金额(元):Q", format=",.0f")], x_sort=month_order), 300), width="stretch")

# --- Excel 下载（含所选期的现金流量表） ---
flow = {"year": flow_year,
        "monthly": results["monthly_by_year"][flow_year],
        "entry": results["yearly"][flow_year - 1]}
wb = engine.build_excel(cfg, records, flow)
if wb is None:
    st.warning("未安装 openpyxl，无法生成 Excel（请检查 requirements.txt）")
else:
    buf = io.BytesIO()
    wb.save(buf)
    st.download_button(
        "下载 Excel 报表（含逐年资产负债表与第{}期现金流量表）".format(flow_year),
        buf.getvalue(), "家庭资产负债表.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("模型假设说明", expanded=False):
    st.markdown("""
- 逐月模拟：月薪每月入现金；公积金自0期记账月次月起按月缴存入账户（资产负债表「公积金」科目），
  按配置月份提取入现金（额度始终自动计算，不可手工修改：冲贷关 = min(当年月供合计〔第1期〕
  / 前12个月实付月供之和〔第2期起〕, 账户余额)，冲贷开 = 0；无贷款时不提取，
  提取在当月月供抵扣之后进行）；
  开启冲贷时每月月供先用公积金账户余额抵扣；年终奖按配置月份发放
- 支出：每月生活费、每月社保、保险（按月份明细）、月供（贷款支持等额本金/等额本息两种还款方式，由本金/利率/剩余期数推算）
- 定投基金：每月定投（月末买入，当月不计收益），持仓按月复利，年化收益率 R；第1年与第2年起定投额可不同
- 股票收益、房产涨幅、投资类保险按输入的年度比率计算（投资类保险是资产科目，
  与④高级设置中的保费支出相互独立）；一次性开支与生活费调整按高级设置执行
- 所有投资增值均为账面浮盈，不产生现金流入
""")

# ---------------------------------------------------------------------------
# 意见反馈（侧边栏）
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.subheader("意见反馈")
    st.caption("欢迎提出新需求或问题。")
    feedback_text = st.text_area("反馈内容", height=110, key="feedback_text")
    if st.button("提交反馈", key="feedback_submit"):
        if save_feedback(feedback_text):
            st.success("感谢反馈！已记录。")
        else:
            st.warning("反馈内容为空，请填写后再提交。")
