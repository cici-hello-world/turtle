#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
家庭资产负债表年度推演
======================
输入第0期（期初）资产负债表的初始值，以及每年的收入与支出，
逐月模拟现金流，输出第1期（一年后）资产负债表各科目的余额。

科目：现金、房产、定投基金、股票、投资类保险、公积金、有息负债（公积金贷款，20年期，支持等额本息/等额本金）

模拟假设
--------
- 收入：月薪（每月）、年终奖（每年一次）、公积金（自0期记账月次月起按月缴存，
  按配置月份提取入现金，额度 = min(可提取额度, 账户余额)，
  缺省：冲贷关 = 当年月供合计（第1期）/ 前12个月实付月供之和（第2期起）、
  冲贷开 = 0；无贷款时不提取）
- 支出：每月生活费、月供（由贷款本金/利率/剩余期数推算）、每月社保、
  保险（按月份明细缴纳：1月8400、4月3000、6月7000、9月3000）
- 支持一次性开支（指定期数+月份）与生活费调整（自指定期数+月份起变为新值），
  均计入当期现金流出，反映在资产负债表与现金流表中
- 定投基金：每月定投（从现金转入，月末买入、当月不计收益），持仓按月复利，
  复合年化收益率 R（变量，自行输入）；第1年每月25,000，第2年起每月5,000
- 股票：收益暂定为 0，持仓不变；房产：默认不增值（可设年涨幅）
- 公积金单列「公积金」科目：自0期记账月次月起每月按「每月缴存公积金」缴存入账户，
  按配置月份提取入现金（额度 = min(可提取额度, 账户余额)，无贷款时不提取；
  提取在当月月供抵扣之后进行，缺省额度：冲贷关 = 当年月供合计（第1期）
  / 前12个月实付月供之和（第2期起）、冲贷开 = 0）；
  开启「公积金冲贷」时，每月月供先用公积金账户余额抵扣

用法
----
修改下方 CONFIG 中的参数后运行：
    python3 scripts/family_balance_sheet.py
可用命令行临时覆盖 R、年数或输出某期月度现金流：
    python3 scripts/family_balance_sheet.py --R 0.10 --年数 5
    python3 scripts/family_balance_sheet.py --现金流 3     # 输出第3期月度现金流表和曲线
除终端报表外，每期资产负债表会写入 Excel（默认 scripts/家庭资产负债表.xlsx，
可在 CONFIG 的「excel输出路径」中修改）；月度现金流曲线输出为 PNG
（scripts/家庭现金流_第N期.png，也可在 CONFIG 的「现金流期数」中设置）。
"""

import os

CONFIG = {
    # ============ 第0期资产负债表（单位：元） ============
    "现金":    1000000.0, #430000.0
    "房产":    700000.0, #2140000.0
    "定投基金":  0,
    "股票":     410000.0, #340000.0
    "投资类保险": 0,        # 投资类保险资产（与「保险月份明细」的保费支出相互独立）
    "公积金账户": 0,        # 公积金账户余额（按月缴存，按年提取入现金/冲抵月供）
    "0期记账月份": 1,       # 第0期资产负债表的记账月份（1-12），用于计算提取月账户余额
    "有息负债": 0,        # 公积金贷款剩余本金800000.0

    # ============ 贷款参数 ============
    "贷款类型":     "等额本息",     # "等额本息"（月供固定）或 "等额本金"（月供逐月递减）
    "贷款年利率":     0.026,      # 公积金贷款年利率
    "贷款剩余期数_月": 240,         # 20 年等额本息 = 240 期；已还部分请相应减少

    # ============ 收入 ============
    "月薪":     11000.0,          # 元/月
    "公积金冲贷": False,          # True：每月月供先用公积金账户余额抵扣
    "每月缴存公积金": 5000.0,       # 元/月，缴存入公积金账户
    "可提取公积金(每年)": None,     # 元/年；None：缺省=上一年总还款额（冲贷开时为0，无贷款不提取）
    "年终奖":   60000.0,         # 元/年
    "年终奖月份": 1,              # 年终奖发放月份（1-12）
    "公积金月份": 5,             # 公积金每年提取月份（1-12），该月全额提取入现金

    # ============ 支出 ============
    "每月生活费": 5000.0,         # 元/月
    "每月社保":  880.0,          # 元/月（若月薪已是税后到手，填 0）
    "保险月份明细": {              # 元/年，按月份缴纳（未列出的月份为 0）
        1: 8400.0,
        4: 3000.0,
        6: 7000.0,
        9: 3000.0,
    },
    "一次性开支": {               # {(期数, 月份): 金额}，如 (1, 6): 50000 表示第1期6月一次性支出5万
        (1, 6): 100000,
    },
    "生活费调整": [               # [((期数, 月份), 新月生活费)]，自该期该月起每月生活费变为新值
        ((1, 7), 7000.0),
    ],

    # ============ 投资 ============
    "定投基金定投_第1年每月": 66700.0,   # 元/月，从现金转入定投基金（第1年）25000.0
    "定投基金定投_第2年起每月": 10000.0,  # 元/月（第2年及以后）10000
    "R": 0.074,                    # 定投基金 复合年化收益率（变量，自行输入）
    "股票年收益率": 0.0,           # 暂定为 0
    "房产年涨幅":  0.0,           # 默认不增值
    "保险年收益率": 0.0,           # 投资类保险资产的年化收益率

    # ============ 其他 ============
    "年数": 10,                     # 模拟年数（依次输出第1期…第N期）
    "excel输出路径": None,           # None：输出到脚本所在目录 scripts/家庭资产负债表.xlsx
    "现金流期数": 10,              # 非 None 时输出该期月度现金流表和曲线图，如 3
}

# 资产负债表科目：(配置键, 显示名)
BALANCE_ITEMS = (
    ("现金", "现金"),
    ("房产", "房产"),
    ("定投基金", "定投基金"),
    ("股票", "股票"),
    ("投资类保险", "投资类保险"),
    ("公积金账户", "公积金"),
    ("有息负债", "有息负债"),
)
BALANCE_ASSETS = ("现金", "房产", "定投基金", "股票", "投资类保险", "公积金账户")


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def pad_cjk(text, width):
    """按终端显示宽度补齐（中文占 2 格）。"""
    text = str(text)
    disp = sum(2 if ord(ch) > 127 else 1 for ch in text)
    return text + " " * max(width - disp, 0)


def pad_cjk_r(text, width):
    """按终端显示宽度右对齐。"""
    text = str(text)
    disp = sum(2 if ord(ch) > 127 else 1 for ch in text)
    return " " * max(width - disp, 0) + text


def money_row(label, value):
    print(pad_cjk(label, 18) + "{:>20,.2f}".format(value))


def text_row(label, text):
    print(pad_cjk(label, 18) + str(text))


def monthly_payment(principal, annual_rate, months):
    """等额本息月供金额。"""
    if principal <= 0.0 or months <= 0:
        return 0.0
    r = annual_rate / 12.0
    if r == 0.0:
        return principal / months
    return principal * r * (1.0 + r) ** months / ((1.0 + r) ** months - 1.0)


def repay_one_month(balance, payment, monthly_rate):
    """还一个月月供，返回 (剩余本金, 当月利息, 当月偿还本金)。"""
    if balance <= 0.0 or payment <= 0.0:
        return 0.0, 0.0, 0.0
    interest = balance * monthly_rate
    principal_part = min(max(payment - interest, 0.0), balance)
    return balance - principal_part, interest, principal_part


def year_mortgage_total(loan, loan_mode, payment, principal_per_month, r_loan):
    """推算当年 12 个月的月供合计（贷款提前还清则按实际结束）。"""
    bal = loan
    total = 0.0
    for _ in range(12):
        if bal <= 0.0:
            break
        if loan_mode == "等额本金":
            interest = bal * r_loan
            principal_part = min(principal_per_month, bal)
            total += principal_part + interest
            bal -= principal_part
        else:
            total += payment
            bal, _, _ = repay_one_month(bal, payment, r_loan)
    return total


def validate(cfg):
    if 1.0 + cfg["R"] <= 0.0:
        raise ValueError("R 必须大于 -100%")
    if cfg["有息负债"] > 0 and cfg["贷款剩余期数_月"] <= 0:
        raise ValueError("有息负债大于 0 时，贷款剩余期数_月必须大于 0")
    if cfg.get("贷款类型", "等额本息") not in ("等额本金", "等额本息"):
        raise ValueError("贷款类型必须是「等额本金」或「等额本息」")
    for key in ("年终奖月份", "公积金月份", "0期记账月份"):
        if not 1 <= cfg[key] <= 12:
            raise ValueError("{} 必须在 1-12 之间".format(key))
    for month in cfg["保险月份明细"]:
        if not 1 <= month <= 12:
            raise ValueError("保险月份明细中的月份必须在 1-12 之间：{}".format(month))
    for (y, m), v in cfg["一次性开支"].items():
        if y < 1 or not 1 <= m <= 12:
            raise ValueError("一次性开支的(期数, 月份)不合法：({}, {})".format(y, m))
        if v < 0:
            raise ValueError("一次性开支金额不能为负：({}, {})".format(y, m))
    for (y, m), v in cfg["生活费调整"]:
        if y < 1 or not 1 <= m <= 12:
            raise ValueError("生活费调整的(期数, 月份)不合法：({}, {})".format(y, m))
        if v < 0:
            raise ValueError("生活费调整的新值不能为负：({}, {})".format(y, m))


# ---------------------------------------------------------------------------
# 报表输出
# ---------------------------------------------------------------------------

def print_balance_sheet(year, prev, curr):
    print()
    print("=" * 64)
    print("第 {} 期资产负债表（单位：元）".format(year))
    print("=" * 64)
    print(pad_cjk("科目", 10)
          + pad_cjk_r("第{}期".format(year - 1), 18)
          + pad_cjk_r("第{}期".format(year), 18)
          + pad_cjk_r("变动", 18))
    print("-" * 64)
    for key, label in BALANCE_ITEMS:
        print(pad_cjk(label, 10)
              + "{:>18,.2f}".format(prev[key])
              + "{:>18,.2f}".format(curr[key])
              + "{:>18,.2f}".format(curr[key] - prev[key]))
    net0 = sum(prev[n] for n in BALANCE_ASSETS) - prev["有息负债"]
    net1 = sum(curr[n] for n in BALANCE_ASSETS) - curr["有息负债"]
    print("-" * 64)
    print(pad_cjk("净资产", 10)
          + "{:>18,.2f}".format(net0)
          + "{:>18,.2f}".format(net1)
          + "{:>18,.2f}".format(net1 - net0))


def print_cashflow(inc, exp, mortgage, dca_total, net_cash):
    print()
    print("全年现金流（单位：元）")
    income_total = sum(inc.values())
    expense_total = sum(exp.values()) + mortgage["月供"]
    text_row("【收入】", "")
    money_row("  月薪", inc["月薪"])
    money_row("  公积金(年提取)", inc["公积金"])
    money_row("  年终奖", inc["年终奖"])
    money_row("  合计", income_total)
    text_row("【支出】", "")
    money_row("  生活费", exp["生活费"])
    money_row("  社保", exp["社保"])
    money_row("  保险", exp["保险"])
    if exp.get("一次性", 0.0):
        money_row("  一次性", exp["一次性"])
    money_row("  月供", mortgage["月供"])
    if mortgage.get("公积金冲抵", 0.0):
        money_row("    其中公积金冲抵", mortgage["公积金冲抵"])
    money_row("    其中利息", mortgage["利息"])
    money_row("    其中本金", mortgage["本金"])
    money_row("  合计", expense_total)
    text_row("【资金转移】", "")
    money_row("  定投转入定投基金", dca_total)
    money_row("现金净变动(期末-期初)", net_cash)


def print_details(prev, curr, dca_total, mortgage):
    print()
    print("投资与贷款明细（单位：元）")
    sp_gain = curr["定投基金"] - prev["定投基金"] - dca_total
    money_row("定投基金 定投", dca_total)
    money_row("定投基金 账面收益", sp_gain)
    money_row("定投基金 期末市值", curr["定投基金"])
    money_row("贷款 偿还本金", mortgage["本金"])
    money_row("贷款 支付利息", mortgage["利息"])
    money_row("贷款 期末本金", curr["有息负债"])


def print_monthly_cashflow(year, months):
    """打印指定期数的月度现金流明细表。"""
    print()
    print("=" * 66)
    print("第 {} 期月度现金流（单位：元）".format(year))
    print("=" * 66)
    print(pad_cjk("月份", 4) + pad_cjk_r("收入", 12) + pad_cjk_r("支出", 12)
          + pad_cjk_r("定投", 12) + pad_cjk_r("净现金流", 12) + pad_cjk_r("月末现金", 14))
    print("-" * 66)
    for m in months:
        print("{:>4}".format(m["月份"])
              + "{:>12,.2f}".format(m["收入"])
              + "{:>12,.2f}".format(m["支出"])
              + "{:>12,.2f}".format(m["定投"])
              + "{:>12,.2f}".format(m["净现金流"])
              + "{:>14,.2f}".format(m["月末现金"]))
    print("-" * 66)
    print(pad_cjk("合计", 4)
          + "{:>12,.2f}".format(sum(m["收入"] for m in months))
          + "{:>12,.2f}".format(sum(m["支出"] for m in months))
          + "{:>12,.2f}".format(sum(m["定投"] for m in months))
          + "{:>12,.2f}".format(sum(m["净现金流"] for m in months))
          + "{:>14,.2f}".format(months[-1]["月末现金"]))


def plot_cashflow_curve(year, months, path):
    """输出指定期数的月度现金流曲线（PNG）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ 未安装 matplotlib，跳过曲线输出（pip3 install matplotlib）")
        return None

    # 中文字体（macOS / Windows 常用字体依次回退）
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS",
                                       "Hiragino Sans GB", "Heiti TC",
                                       "Microsoft YaHei", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False

    # 参考调色板（浅色面）：分类槽位 1/2/3，按固定顺序取用
    series = [("收入", "#2a78d6"), ("支出", "#eb6834"), ("净现金流", "#1baf7a")]
    surface = "#fcfcfb"    # 图表表面
    ink = "#0b0b0b"        # 主墨色
    muted = "#898781"      # 弱化色（轴、刻度）
    grid = "#e1e0d9"       # 网格线
    axis = "#c3c2b7"       # 轴线 / 零线

    # x 轴按模拟顺序编号（月份可能回绕，如 …11、12、1），刻度显示对应日历月份
    x = list(range(1, len(months) + 1))
    month_labels = [str(m["月份"]) for m in months]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)

    for name, color in series:
        y = [m[name] for m in months]
        ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=5,
                label=name, zorder=3)
        # 线尾直接标注（文字用灰墨，不带序列色）
        ax.annotate(name, (x[-1], y[-1]), xytext=(6, 0),
                    textcoords="offset points", color=muted,
                    fontsize=10, va="center")

    ax.axhline(0, color=axis, linewidth=1, linestyle="--", zorder=1)

    ax.set_title("第 {} 期月度现金流".format(year), color=ink,
                 fontsize=14, fontweight="bold", loc="left", pad=12)
    ax.set_xlabel("月份", color=muted)
    ax.set_ylabel("金额（元）", color=muted)
    ax.set_xticks(x)
    ax.set_xticklabels(month_labels)
    ax.tick_params(colors=muted)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(axis)
    ax.grid(axis="y", color=grid, linewidth=1)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", frameon=False, fontsize=10)

    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    print("现金流曲线已输出：{}".format(path))
    return path


# ---------------------------------------------------------------------------
# Excel 输出
# ---------------------------------------------------------------------------

def build_excel(cfg, records, flow=None):
    """构建资产负债表的 Excel 工作簿（供保存到文件或内存下载）。

    第 1 个 sheet 为「汇总」（科目 × 各期期末值），
    之后每期一个 sheet（科目、期初、期末、变动）。
    records: [(year, prev, curr), ...]
    flow: 可选，{"year": 期数, "monthly": 月度明细, "entry": 该期年度汇总}，
          提供时额外生成「第N期现金流」sheet（月度明细 + 全年现金流汇总）。
    返回 openpyxl Workbook；未安装 openpyxl 时返回 None。
    """
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("⚠ 未安装 openpyxl，跳过 Excel 输出（pip3 install openpyxl）")
        return None

    items = BALANCE_ITEMS
    assets = BALANCE_ASSETS
    money_fmt = "#,##0.00"
    thin = Side(style="thin", color="999999")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    header_font = Font(bold=True)
    header_fill = PatternFill("solid", fgColor="D9E1F2")
    title_font = Font(bold=True, size=12)
    net_fill = PatternFill("solid", fgColor="FFF2CC")

    def net(b):
        return sum(b[n] for n in assets) - b["有息负债"]

    wb = Workbook()

    # ---------- 汇总 sheet ----------
    ws = wb.active
    ws.title = "汇总"
    ws.cell(row=1, column=1, value="家庭资产负债表汇总（单位：元）").font = title_font
    header = ["科目", "第0期"] + ["第{}期".format(y) for y, _, _ in records]
    for c, name in enumerate(header, start=1):
        cell = ws.cell(row=2, column=c, value=name)
        cell.font = header_font
        cell.fill = header_fill
        cell.border = border
        cell.alignment = Alignment(horizontal="center")
    for r, (key, label) in enumerate(items, start=3):
        ws.cell(row=r, column=1, value=label).border = border
        cell = ws.cell(row=r, column=2, value=records[0][1][key])
        cell.number_format = money_fmt
        cell.border = border
        for c, (_, _, curr) in enumerate(records, start=3):
            cell = ws.cell(row=r, column=c, value=curr[key])
            cell.number_format = money_fmt
            cell.border = border
    r_net = 3 + len(items) + 1          # 空一行后放净资产
    ws.cell(row=r_net, column=1, value="净资产").font = header_font
    ws.cell(row=r_net, column=1).border = border
    cell = ws.cell(row=r_net, column=2, value=net(records[0][1]))
    cell.number_format = money_fmt
    cell.font = header_font
    cell.fill = net_fill
    cell.border = border
    for c, (_, _, curr) in enumerate(records, start=3):
        cell = ws.cell(row=r_net, column=c, value=net(curr))
        cell.number_format = money_fmt
        cell.font = header_font
        cell.fill = net_fill
        cell.border = border
    ws.column_dimensions["A"].width = 12
    for c in range(2, len(header) + 1):
        ws.column_dimensions[get_column_letter(c)].width = 16

    # ---------- 每期一个 sheet ----------
    for year, prev, curr in records:
        ws = wb.create_sheet("第{}期".format(year))
        ws.cell(row=1, column=1,
                value="第 {} 期资产负债表（单位：元）".format(year)).font = title_font
        header = ["科目", "第{}期".format(year - 1), "第{}期".format(year), "变动"]
        for c, name in enumerate(header, start=1):
            cell = ws.cell(row=2, column=c, value=name)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
            cell.alignment = Alignment(horizontal="center")
        for r, (key, label) in enumerate(items, start=3):
            ws.cell(row=r, column=1, value=label).border = border
            for c, v in enumerate(
                    (prev[key], curr[key], curr[key] - prev[key]), start=2):
                cell = ws.cell(row=r, column=c, value=v)
                cell.number_format = money_fmt
                cell.border = border
        r_net = 3 + len(items) + 1
        ws.cell(row=r_net, column=1, value="净资产").font = header_font
        ws.cell(row=r_net, column=1).border = border
        for c, v in enumerate(
                (net(prev), net(curr), net(curr) - net(prev)), start=2):
            cell = ws.cell(row=r_net, column=c, value=v)
            cell.number_format = money_fmt
            cell.font = header_font
            cell.fill = net_fill
            cell.border = border
        ws.column_dimensions["A"].width = 12
        for c in range(2, 5):
            ws.column_dimensions[get_column_letter(c)].width = 18

    # ---------- 所选期现金流量表 sheet ----------
    if flow:
        fy = flow["year"]
        monthly = flow["monthly"]
        e = flow["entry"]
        ws = wb.create_sheet("第{}期现金流".format(fy))
        ws.cell(row=1, column=1,
                value="第 {} 期现金流量表（单位：元）".format(fy)).font = title_font
        mh = ["月份", "收入", "支出", "定投", "净现金流", "月末现金"]
        for c, name in enumerate(mh, start=1):
            cell = ws.cell(row=2, column=c, value=name)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
            cell.alignment = Alignment(horizontal="center")
        for i, m in enumerate(monthly):
            r = 3 + i
            for c, v in enumerate((m["月份"], m["收入"], m["支出"], m["定投"],
                                  m["净现金流"], m["月末现金"]), start=1):
                cell = ws.cell(row=r, column=c, value=v)
                if c > 1:
                    cell.number_format = money_fmt
                cell.border = border
        r_total = 3 + len(monthly)
        totals = ["合计"] + [sum(m[k] for m in monthly)
                             for k in ("收入", "支出", "定投", "净现金流")] \
            + [monthly[-1]["月末现金"]]
        for c, v in enumerate(totals, start=1):
            cell = ws.cell(row=r_total, column=c, value=v)
            if c > 1:
                cell.number_format = money_fmt
            cell.font = header_font
            cell.fill = net_fill
            cell.border = border

        # 全年现金流汇总（放在 H/I 列，避开左侧月度表的列宽）
        r = r_total + 2
        ws.cell(row=r, column=8, value="全年现金流汇总").font = header_font
        r += 1
        for c, name in ((8, "项目"), (9, "金额(元)")):
            cell = ws.cell(row=r, column=c, value=name)
            cell.font = header_font
            cell.fill = header_fill
            cell.border = border
            cell.alignment = Alignment(horizontal="center")
        r += 1

        def put(label, value=None, bold=False, fill=None):
            nonlocal r
            c1 = ws.cell(row=r, column=8, value=label)
            c1.border = border
            if bold:
                c1.font = header_font
            if fill:
                c1.fill = fill
            if value is not None:
                c2 = ws.cell(row=r, column=9, value=value)
                c2.number_format = money_fmt
                c2.border = border
                if bold:
                    c2.font = header_font
                if fill:
                    c2.fill = fill
            r += 1

        inc, exp, mortgage = e["inc"], e["exp"], e["mortgage"]
        income_total = sum(inc.values())
        expense_total = sum(exp.values()) + mortgage["月供"]
        put("收入", bold=True)
        put("  月薪", inc["月薪"])
        put("  公积金(年提取)", inc["公积金"])
        put("  年终奖", inc["年终奖"])
        put("  合计", income_total, bold=True, fill=net_fill)
        put("支出", bold=True)
        put("  生活费", exp["生活费"])
        put("  社保", exp["社保"])
        put("  保险", exp["保险"])
        if exp.get("一次性", 0.0):
            put("  一次性", exp["一次性"])
        put("  月供", mortgage["月供"])
        if mortgage.get("公积金冲抵", 0.0):
            put("    其中公积金冲抵", mortgage["公积金冲抵"])
        put("    其中利息", mortgage["利息"])
        put("    其中本金", mortgage["本金"])
        put("  合计", expense_total, bold=True, fill=net_fill)
        put("资金转移", bold=True)
        put("  定投转入定投基金", e["dca_total"])
        put("现金净变动(期末-期初)", e["net_cash"], bold=True, fill=net_fill)

        ws.column_dimensions["A"].width = 8
        for c in range(2, 7):
            ws.column_dimensions[get_column_letter(c)].width = 14
        ws.column_dimensions["H"].width = 24
        ws.column_dimensions["I"].width = 16

    return wb


def write_excel(cfg, records, path, flow=None):
    """把每期资产负债表写入 Excel 文件（命令行输出用）。"""
    wb = build_excel(cfg, records, flow)
    if wb is None:
        return None
    wb.save(path)
    print()
    print("Excel 已输出：{}".format(path))
    return path


# ---------------------------------------------------------------------------
# 模拟主流程
# ---------------------------------------------------------------------------

def run_simulation(cfg):
    """纯计算引擎：输入配置，返回结构化结果（不打印，供命令行与网页应用共用）。

    返回 dict：
      years            模拟年数
      payment          等额本息月供
      records          [(year, prev, curr), ...] 每期资产负债表
      monthly_by_year  {year: [12个月的现金流明细]}
      yearly           [{year, prev, curr, inc, exp, mortgage, dca_total, net_cash}]
    """
    validate(cfg)
    years = int(cfg["年数"])
    r_sp = (1.0 + cfg["R"]) ** (1.0 / 12.0) - 1.0      # 定投基金 月化收益率
    r_loan = cfg["贷款年利率"] / 12.0                    # 贷款月利率

    cash, house, sp500, stocks, ins_invest, gjj_acct, loan = (
        cfg["现金"], cfg["房产"], cfg["定投基金"], cfg["股票"],
        cfg.get("投资类保险", 0.0), cfg.get("公积金账户", 0.0), cfg["有息负债"])
    payment = monthly_payment(loan, cfg["贷款年利率"], cfg["贷款剩余期数_月"])
    loan_mode = cfg.get("贷款类型", "等额本息")
    loan_months = cfg["贷款剩余期数_月"]
    loan_initial = loan                 # 期初本金（等额本金首月月供用）
    # 等额本金：每月偿还本金固定 = 期初本金 / 剩余期数
    principal_per_month = (loan / loan_months) if (loan > 0.0 and loan_months > 0) else 0.0

    prev = {"现金": cash, "房产": house, "定投基金": sp500,
            "股票": stocks, "投资类保险": ins_invest, "公积金账户": gjj_acct,
            "有息负债": loan}
    records = []                     # 每期资产负债表，用于 Excel 输出
    monthly_by_year = {}             # 每年 12 个月的现金流明细，用于曲线输出
    yearly = []                      # 每年现金流汇总，供报表打印
    mortgage_history = []            # 每月实付月供，用于「前12个月月供之和」

    for year in range(1, years + 1):
        inc = {"月薪": 0.0, "公积金": 0.0, "年终奖": 0.0}
        exp = {"生活费": 0.0, "社保": 0.0, "保险": 0.0, "一次性": 0.0}
        mortgage = {"月供": 0.0, "本金": 0.0, "利息": 0.0,
                    "现金支付": 0.0, "公积金冲抵": 0.0}
        dca_total = 0.0
        dca_monthly = (cfg["定投基金定投_第1年每月"] if year == 1
                       else cfg["定投基金定投_第2年起每月"])
        monthly = []                 # 当期 12 个月的现金流明细
        # 当年月供合计（按贷款计划推算，公积金提取额度上限）
        annual_mortgage = year_mortgage_total(loan, loan_mode, payment,
                                              principal_per_month, r_loan)

        # 自 0期记账月份 的次月起循环 12 个日历月（超过 12 月时回绕到 1 月）
        start_month = cfg["0期记账月份"]
        for offset in range(12):
            month = (start_month + offset) % 12 + 1
            salary = cfg["月薪"]
            bonus = cfg["年终奖"] if month == cfg["年终奖月份"] else 0.0
            # 缴存自0期记账月次月起：模拟自0期次月开始，故每个月均缴存
            gjj_acct += cfg.get("每月缴存公积金", 0.0)
            living = cfg["每月生活费"]
            for (sy, sm), new_living in cfg["生活费调整"]:   # 自指定期+月起生效
                if (year, month) >= (sy, sm):
                    living = new_living
            social = cfg["每月社保"]
            insurance = cfg["保险月份明细"].get(month, 0.0)
            one_off = cfg["一次性开支"].get((year, month), 0.0)

            cash += salary + bonus

            mortgage_paid = 0.0
            covered = 0.0
            if loan > 0.0:
                if loan_mode == "等额本金":
                    interest = loan * r_loan
                    principal_part = min(principal_per_month, loan)
                    mortgage_paid = principal_part + interest
                    loan -= principal_part
                else:                       # 等额本息
                    mortgage_paid = payment
                    loan, interest, principal_part = repay_one_month(loan, payment, r_loan)
                # 公积金冲贷：月供先用公积金账户余额抵扣
                covered = min(gjj_acct, mortgage_paid) if cfg.get("公积金冲贷", False) else 0.0
                gjj_acct -= covered
                cash -= mortgage_paid - covered
                mortgage["月供"] += mortgage_paid
                mortgage["现金支付"] += mortgage_paid - covered
                mortgage["公积金冲抵"] += covered
                mortgage["利息"] += interest
                mortgage["本金"] += principal_part
            mortgage_history.append(mortgage_paid)

            # 公积金按年提取：当月月供抵扣之后再进行（无贷款时不提取）
            # 额度 = min(可提取额度, 账户余额)；缺省：冲贷关 = 当年月供合计（第1期）
            # / 前12个月实付月供之和（第2期起），冲贷开 = 0
            gjj_withdraw = 0.0
            if month == cfg["公积金月份"] and annual_mortgage > 0.0:
                cap = cfg.get("可提取公积金(每年)")
                if cfg.get("公积金冲贷", False):
                    cap = 0.0                        # 冲贷开：不额外提取
                elif year == 1:
                    # 第1期：优先用手动指定额度；自动（None）时用当年预估全年月供合计
                    if cap is None:
                        cap = annual_mortgage
                else:
                    # 第2期起：按前 12 个月实际月供之和自动计算
                    cap = sum(mortgage_history[-12:])
                # 无论哪种来源都不能超过账户余额（也不能为负）
                gjj_withdraw = min(max(float(cap), 0.0), gjj_acct)
                gjj_acct -= gjj_withdraw
                cash += gjj_withdraw

            cash -= living + social + insurance + one_off

            # 定投基金：持仓按月复利，月末定投买入
            sp500 = sp500 * (1.0 + r_sp) + dca_monthly
            dca_total += dca_monthly
            cash -= dca_monthly

            inc["月薪"] += salary
            inc["公积金"] += gjj_withdraw
            inc["年终奖"] += bonus
            exp["生活费"] += living
            exp["社保"] += social
            exp["保险"] += insurance
            exp["一次性"] += one_off

            # 月度现金流明细（用于曲线输出，均为现金口径）
            income_m = salary + bonus + gjj_withdraw
            expense_m = living + social + insurance + (mortgage_paid - covered) + one_off
            monthly.append({
                "月份": month,
                "收入": income_m,
                "支出": expense_m,
                "定投": dca_monthly,
                "净现金流": income_m - expense_m - dca_monthly,
                "月末现金": cash,
            })

        monthly_by_year[year] = monthly

        # 年度末：房产、股票、投资类保险增值（默认 0）
        house *= 1.0 + cfg["房产年涨幅"]
        stocks *= 1.0 + cfg["股票年收益率"]
        ins_invest *= 1.0 + cfg.get("保险年收益率", 0.0)

        curr = {"现金": cash, "房产": house, "定投基金": sp500,
                "股票": stocks, "投资类保险": ins_invest, "公积金账户": gjj_acct,
                "有息负债": loan}
        records.append((year, prev.copy(), curr.copy()))
        yearly.append({
            "year": year,
            "prev": prev.copy(),
            "curr": curr.copy(),
            "inc": inc,
            "exp": exp,
            "mortgage": mortgage,
            "dca_total": dca_total,
            "net_cash": curr["现金"] - prev["现金"],
        })
        prev = curr

    return {
        "cfg": cfg,
        "years": years,
        "payment": payment,          # 等额本息的固定月供（等额本金模式下仅供参考）
        "loan_mode": loan_mode,
        "first_payment": (payment if loan_mode == "等额本息"
                          else principal_per_month + loan_initial * r_loan),
        "records": records,
        "monthly_by_year": monthly_by_year,
        "yearly": yearly,
    }


def simulate(cfg):
    """命令行入口：调用 run_simulation 并打印报表、输出 Excel 与现金流曲线。"""
    res = run_simulation(cfg)
    years = res["years"]
    payment = res["payment"]

    print("=" * 64)
    print("模拟参数")
    print("=" * 64)
    text_row("模拟年数", "{} 年".format(years))
    text_row("R(定投基金复合年化)", "{:.2%}".format(cfg["R"]))
    money_row("定投-第1年每月", cfg["定投基金定投_第1年每月"])
    money_row("定投-第2年起每月", cfg["定投基金定投_第2年起每月"])
    text_row("贷款类型", cfg.get("贷款类型", "等额本息"))
    if cfg.get("贷款类型", "等额本息") == "等额本金":
        text_row("月供(等额本金)", "首月 {:,} 元，逐月递减".format(res["first_payment"]))
    else:
        money_row("月供(等额本息)", payment)
    text_row("贷款剩余期数", "{} 月".format(cfg["贷款剩余期数_月"]))
    text_row("公积金冲贷", "是" if cfg.get("公积金冲贷", False) else "否")
    gjj_cap = cfg.get("可提取公积金(每年)")
    text_row("公积金", "0期记账{}月；每月缴存 {:,}；{} 月提取（额度 {}）".format(
        cfg.get("0期记账月份", 1), cfg.get("每月缴存公积金", 0.0), cfg["公积金月份"],
        "{:,.0f} 元".format(gjj_cap) if gjj_cap is not None
        else "默认=前12个月月供之和(冲贷开为0)"))
    if cfg["一次性开支"]:
        text_row("一次性开支", "; ".join(
            "第{}期{}月 {:,}".format(y, m, v)
            for (y, m), v in sorted(cfg["一次性开支"].items())))
    if cfg["生活费调整"]:
        text_row("生活费调整", "; ".join(
            "第{}期{}月起 {:,}".format(y, m, v) for (y, m), v in cfg["生活费调整"]))
    print()
    print("注：定投于每月末买入（当月不计收益）；股票收益率 0；投资增值为账面浮盈。")

    for entry in res["yearly"]:
        print_balance_sheet(entry["year"], entry["prev"], entry["curr"])
        print_cashflow(entry["inc"], entry["exp"], entry["mortgage"],
                       entry["dca_total"], entry["net_cash"])
        print_details(entry["prev"], entry["curr"], entry["dca_total"],
                      entry["mortgage"])
        if entry["curr"]["现金"] < 0:
            print()
            print("⚠ 第 {} 期末现金为负，请检查收支是否平衡".format(entry["year"]))

    # 月度现金流：输入期数，输出该期的月度现金流表、曲线图，并写入 Excel
    flow = None
    flow_year = cfg.get("现金流期数")
    if flow_year is not None:
        flow_year = int(flow_year)
        if 1 <= flow_year <= years:
            flow = {"year": flow_year,
                    "monthly": res["monthly_by_year"][flow_year],
                    "entry": res["yearly"][flow_year - 1]}
            print_monthly_cashflow(flow_year, res["monthly_by_year"][flow_year])
            png_path = os.path.join(
                os.path.dirname(os.path.abspath(__file__)),
                "家庭现金流_第{}期.png".format(flow_year))
            plot_cashflow_curve(flow_year, res["monthly_by_year"][flow_year], png_path)
        else:
            print()
            print("⚠ 现金流期数 {} 超出模拟范围 1-{}，已跳过".format(flow_year, years))

    path = cfg.get("excel输出路径") or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "家庭资产负债表.xlsx")
    write_excel(cfg, res["records"], path, flow=flow)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="家庭资产负债表年度推演")
    parser.add_argument("--R", type=float, default=None,
                        help="覆盖 CONFIG 中的 R（定投基金 复合年化收益率）")
    parser.add_argument("--年数", type=int, default=None,
                        help="覆盖 CONFIG 中的模拟年数")
    parser.add_argument("--现金流", type=int, default=None,
                        help="输出指定期数的月度现金流表和曲线，如 --现金流 3")
    args = parser.parse_args()
    if args.R is not None:
        CONFIG["R"] = args.R
    if args.年数 is not None:
        CONFIG["年数"] = args.年数
    if args.现金流 is not None:
        CONFIG["现金流期数"] = args.现金流
    simulate(CONFIG)
