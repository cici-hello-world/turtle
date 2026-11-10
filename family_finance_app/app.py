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

import altair as alt
import pandas as pd
import streamlit as st

# 复用命令行脚本的计算引擎（run_simulation / build_excel）
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
import family_balance_sheet as engine

st.set_page_config(page_title="家庭资产十年推演", page_icon="🏠", layout="wide")

# ---------------------------------------------------------------------------
# 图表样式：分类调色板（固定槽位 1-6，色盲安全）、浅色面墨色规范
# ---------------------------------------------------------------------------
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK = "#0b0b0b"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

ITEMS = ("现金", "房产", "定投基金", "股票", "有息负债")
ASSETS = ("现金", "房产", "定投基金", "股票")


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
                              padding=4, columns=6)
            .configure_title(color=INK, fontSize=15, anchor="start"))


def line_chart(df, x, y, color, domain, slots, tooltip_fields):
    """按调色板规范画多序列折线图（2px 线、圆点、悬停提示）。"""
    return alt.Chart(df).mark_line(
        strokeWidth=2,
        point=alt.OverlayMarkDef(size=45, filled=True),
    ).encode(
        x=alt.X("{}:O".format(x), axis=alt.Axis(labelAngle=0)),
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

st.title("家庭资产十年推演")
st.caption("输入资产负债表、收入、支出与投资参数，推演未来资产增长。"
           "数据仅在本机计算，不会上传；全部参数可随时调整，结果实时更新。")

D = engine.CONFIG  # 默认值取命令行脚本的当前配置

with st.expander("① 第0期资产负债表", expanded=True):
    c = st.columns(5)
    cash0 = c[0].number_input("现金", value=float(D["现金"]), step=10000.0)
    house0 = c[1].number_input("房产", value=float(D["房产"]), step=10000.0)
    sp0 = c[2].number_input("定投基金", value=float(D["定投基金"]), step=10000.0)
    stock0 = c[3].number_input("股票", value=float(D["股票"]), step=10000.0)
    debt0 = c[4].number_input("有息负债(剩余本金)", value=float(D["有息负债"]),
                              step=10000.0)
    c = st.columns(4)
    loan_rate_pct = c[0].number_input("贷款年利率(%)", 0.0, 10.0,
                                      float(D["贷款年利率"]) * 100, 0.1)
    loan_months = c[1].number_input("贷款剩余期数(月)", 0, 360,
                                    int(D["贷款剩余期数_月"]), 12)
    loan_mode = c[2].selectbox("贷款类型", ["等额本息", "等额本金"],
                               index=0 if D.get("贷款类型", "等额本息") == "等额本息" else 1)
    years = c[3].slider("推演年数", 1, 30, int(D["年数"]))

with st.expander("② 收入与支出（元）(现金流入流出)", expanded=True):
    c = st.columns(5)
    salary = c[0].number_input("月薪(税后)", value=float(D["月薪"]), step=1000.0)
    gjj = c[1].number_input("公积金(每年)", value=float(D["公积金"]), step=10000.0)
    bonus = c[2].number_input("年终奖(每年)", value=float(D["年终奖"]), step=10000.0)
    bonus_month = c[3].number_input("年终奖月份", 1, 12, int(D["年终奖月份"]))
    gjj_month = c[4].number_input("公积金提取月份", 1, 12, int(D["公积金月份"]))
    c = st.columns(2)
    living = c[0].number_input("每月生活费", value=float(D["每月生活费"]), step=500.0)
    social = c[1].number_input("每月社保", value=float(D["每月社保"]), step=100.0)

with st.expander("③ 投资参数", expanded=True):
    c = st.columns(5)
    dca1 = c[0].number_input("基金定投-第1年每月", value=float(D["定投基金定投_第1年每月"]),
                             step=1000.0)
    dca2 = c[1].number_input("基金定投-第2年起每月", value=float(D["定投基金定投_第2年起每月"]),
                             step=1000.0)
    r_pct = c[2].slider("定投基金 年化收益率 R(%)", 0.0, 20.0, float(D["R"]) * 100, 0.1)
    stock_ret_pct = c[3].number_input("股票年收益率(%)", -50.0, 50.0,
                                      float(D["股票年收益率"]) * 100, 0.5)
    house_gain_pct = c[4].number_input("房产年涨幅(%)", -10.0, 20.0,
                                       float(D["房产年涨幅"]) * 100, 0.5)

with st.expander("④ 高级设置（保险 / 一次性开支 / 生活费调整）", expanded=False):
    st.write("**保险月份明细**：按月份缴纳，未列出的月份为 0。")
    insurance_df = st.data_editor(
        pd.DataFrame([{"月份": m, "金额": v}
                      for m, v in sorted(D["保险月份明细"].items())]),
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
        "有息负债": debt0,
        "贷款类型": loan_mode,
        "贷款年利率": loan_rate_pct / 100.0, "贷款剩余期数_月": int(loan_months),
        "月薪": salary, "公积金": gjj, "年终奖": bonus,
        "年终奖月份": int(bonus_month), "公积金月份": int(gjj_month),
        "每月生活费": living, "每月社保": social,
        "保险月份明细": parse_insurance_df(insurance_df),
        "一次性开支": parse_one_off(one_off_df),
        "生活费调整": parse_living_adj(living_adj_df),
        "定投基金定投_第1年每月": dca1, "定投基金定投_第2年起每月": dca2,
        "R": r_pct / 100.0,
        "股票年收益率": stock_ret_pct / 100.0, "房产年涨幅": house_gain_pct / 100.0,
        "年数": int(years),
        "excel输出路径": None, "现金流期数": None,
    }
    results = engine.run_simulation(cfg)
except ValueError as e:
    st.error("参数有误：{}".format(e))
    st.stop()

records = results["records"]
payment = results["payment"]
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

if loan_mode == "等额本金":
    st.caption("等额本金：首月月供 {:,.0f} 元，逐月递减（贷款还清后为 0）".format(
        results["first_payment"]))
else:
    st.caption("等额本息月供 {:,.0f} 元/月（若贷款已还清则为 0）".format(payment))

# --- 图1：净资产与各科目走势 ---
balance_domain = ["净资产", "现金", "房产", "定投基金", "股票", "有息负债"]
bs_rows = []
for name in balance_domain:
    bs_rows.append({
        "科目": name,
        "第0期": net(records[0][1]) if name == "净资产" else records[0][1][name],
        **{"第{}期".format(y): (net(cu) if name == "净资产" else cu[name])
           for y, _, cu in records},
    })
bs_df = pd.DataFrame(bs_rows).set_index("科目")

chart_rows = []
for name in balance_domain:
    for y, v in zip([0] + [y for y, _, _ in records], bs_df.loc[name]):
        chart_rows.append({"期数": y, "科目": name, "金额(元)": v})
chart_df = pd.DataFrame(chart_rows)

st.altair_chart(style_altair(line_chart(
    chart_df, "期数", "金额(元)", "科目", balance_domain, SLOTS[:6],
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
monthly_rows = []
for m in monthly:
    for name in flow_domain:
        monthly_rows.append({"月份": m["月份"], "科目": name, "金额(元)": m[name]})
monthly_df = pd.DataFrame(monthly_rows)
st.altair_chart(style_altair(line_chart(
    monthly_df, "月份", "金额(元)", "科目", flow_domain, SLOTS[:3],
    [alt.Tooltip("月份:O", title="月份"), alt.Tooltip("科目:N"),
     alt.Tooltip("金额(元):Q", format=",.0f")]), 300), width="stretch")

# --- Excel 下载 ---
wb = engine.build_excel(cfg, records)
buf = io.BytesIO()
wb.save(buf)
st.download_button(
    "下载 Excel 报表（含逐年资产负债表）", buf.getvalue(),
    "家庭资产负债表.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

with st.expander("模型假设说明", expanded=False):
    st.markdown("""
- 逐月模拟：月薪每月入现金；公积金每年一次按配置月份提取入现金；年终奖按配置月份发放
- 支出：每月生活费、每月社保、保险（按月份明细）、月供（贷款支持等额本金/等额本息两种还款方式，由本金/利率/剩余期数推算）
- 定投基金：每月定投（月末买入，当月不计收益），持仓按月复利，年化收益率 R；第1年与第2年起定投额可不同
- 股票收益、房产涨幅按输入的年度比率计算；一次性开支与生活费调整按高级设置执行
- 所有投资增值均为账面浮盈，不产生现金流入
""")
