import streamlit as st
import pandas as pd
import plotly.express as px
import os
from pathlib import Path
import numpy as np
import json

os.chdir(Path(__file__).resolve().parent)

# =========================
# Page config
# =========================
st.set_page_config(page_title="机构行为分析", layout="wide")

st.markdown(
    """
<style>
.stMarkdown h1 { font-size: 28px; }
.stMarkdown h2 { font-size: 18px; }
.stMarkdown h3 { font-size: 16px; }
.stMarkdown h4 { font-size: 14px; }

.topnav {
  display: flex;
  gap: 22px;
  margin: 6px 0 12px 0;
}
.topnav a {
  font-weight: 600;
  font-size: 20px;
  color: #6b7280;
  text-decoration: none;
  padding: 6px 0;
  border-bottom: 2px solid transparent;
}
.topnav a.active {
  color: #1f77b4;
  border-bottom-color: #1f77b4;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div style="font-size:34px; font-weight:700; margin-bottom: 8px;">
      机构行为分析
    </div>
    """,
    unsafe_allow_html=True,
)

# =========================
# File path
# 用脚本所在目录定位数据文件，避免 Streamlit Cloud 找不到 CSV
# =========================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_CSV = DATA_DIR / "bond_trades_all_in_one.csv"
FACTORS_PARQUET = DATA_DIR / "mvp_factors_enhanced.parquet"
FACTORS_META = DATA_DIR / "mvp_factors_enhanced.meta.json"

LEGACY_SOURCE_CSV = BASE_DIR / "bond_trades_all_in_one.csv"

# =========================
# Data loader
# =========================
@st.cache_data
def load_data(file_path: str, file_signature: tuple[float, int]):
    try:
        df = pd.read_csv(file_path)
        df["日期"] = pd.to_datetime(df["日期"], errors="coerce")
        df = df.dropna(subset=["日期"])

        for col in ["机构类型", "资产类别", "期限"]:
            if col in df.columns:
                df[col] = df[col].astype("string").str.strip()

        for col in ["Net", "Buy", "Sell", "Turnover"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df
    except FileNotFoundError:
        st.error(f"文件未找到: {file_path}")
        return None


def current_source_csv() -> Path:
    if SOURCE_CSV.exists():
        return SOURCE_CSV
    return LEGACY_SOURCE_CSV


def get_file_signature(p: Path) -> tuple[float, int]:
    try:
        return (os.path.getmtime(p), os.path.getsize(p))
    except OSError:
        return (0.0, 0)


file_path = current_source_csv()
file_signature = get_file_signature(file_path)

df = load_data(str(file_path), file_signature)


@st.cache_data
def compute_mvp_factors(file_path: str, file_signature: tuple[float, int], start_date: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.columns = [str(c).strip() for c in df.columns]

    has_a = {"日期", "机构类型", "期限", "资产类别", "Net"}.issubset(df.columns)
    has_b = {"日期", "机构类型", "期限", "资产类别", "成交净额"}.issubset(df.columns)
    if not (has_a or has_b):
        raise ValueError(
            "CSV 列名不符合预期。\n"
            f"当前列名：{df.columns.tolist()}\n"
            "期望 schema A: 日期,机构类型,期限,资产类别,Net,Buy,Sell,Turnover\n"
            "或 schema B: 日期,机构类型,期限,资产类别,成交净额"
        )

    for c in ["机构类型", "期限", "资产类别"]:
        df[c] = df[c].astype("string").str.strip()

    df["日期"] = pd.to_datetime(df["日期"].astype("string").str.strip(), errors="coerce")
    df = df.dropna(subset=["日期"]).copy()

    def to_num(s: pd.Series) -> pd.Series:
        s = s.astype("string").str.strip()
        s = s.replace({"--": "0", "—": "0", "-": "0", "nan": "0", "None": "0"})
        return pd.to_numeric(s, errors="coerce").fillna(0.0)

    if has_a:
        for c in ["Net", "Buy", "Sell", "Turnover"]:
            if c in df.columns:
                df[c] = to_num(df[c])
        if "Turnover" not in df.columns:
            df["Turnover"] = df.get("Buy", 0.0) + df.get("Sell", 0.0)
    else:
        df["Net"] = to_num(df["成交净额"])
        df["Buy"] = 0.0
        df["Sell"] = 0.0
        df["Turnover"] = np.nan

    df = df[~df["机构类型"].astype("string").str.contains("机构分类|期限分类", na=False)].copy()
    df = df[df["期限"] != "合计"].copy()
    df = df[df["日期"] >= pd.Timestamp(start_date)].copy()

    rate_assets = {
        "国债-新债",
        "国债-老债",
        "政策性金融债-新债",
        "政策性金融债-老债",
        "地方政府债",
    }
    df = df[df["资产类别"].isin(rate_assets)].copy()

    tenor_dur_w = {
        "≦1Y": 0.5,
        "≤1Y": 0.5,
        "<=1Y": 0.5,
        "1Y以下": 0.5,
        "1-3Y": 2.0,
        "3-5Y": 4.0,
        "5-7Y": 6.0,
        "7-10Y": 8.5,
        "10-15Y": 12.0,
        "15-20Y": 17.0,
        "20-30Y": 25.0,
        ">30Y": 35.0,
    }
    long_buckets = {"7-10Y", "10-15Y", "15-20Y", "20-30Y", ">30Y"}
    short_buckets = {"≦1Y", "≤1Y", "<=1Y", "1Y以下", "1-3Y"}

    df["DurW"] = df["期限"].map(tenor_dur_w).fillna(0.0)
    df["is_long"] = df["期限"].isin(long_buckets)
    df["is_short"] = df["期限"].isin(short_buckets)

    def sector(asset: str) -> str:
        if asset.startswith("国债"):
            return "国债"
        if asset.startswith("政策性金融债"):
            return "政金"
        if asset == "地方政府债":
            return "地方"
        return "其他"

    df["Sector"] = df["资产类别"].map(sector)

    def compute_daily_factors(d: pd.DataFrame) -> pd.Series:
        net_long = d.loc[d["is_long"], "Net"].sum()

        if d["Turnover"].notna().any() and d["Turnover"].sum() > 0:
            turn_long = d.loc[d["is_long"], "Turnover"].sum()
            f2 = net_long / (turn_long + 1e-12)
            turnover_total = d["Turnover"].sum()
        else:
            abs_long = d.loc[d["is_long"], "Net"].abs().sum()
            f2 = net_long / (abs_long + 1e-12)
            turnover_total = np.nan

        f3 = (d["Net"] * d["DurW"]).sum()

        net_short = d.loc[d["is_short"], "Net"].sum()
        f4 = net_long - net_short

        gov_net = d.loc[d["Sector"] == "国债", "Net"].sum()
        pfb_net = d.loc[d["Sector"] == "政金", "Net"].sum()
        lg_net = d.loc[d["Sector"] == "地方", "Net"].sum()
        f5 = pfb_net - gov_net
        f6 = lg_net - pfb_net

        gov_new = d.loc[d["资产类别"] == "国债-新债", "Net"].sum()
        gov_old = d.loc[d["资产类别"] == "国债-老债", "Net"].sum()
        f7 = gov_new - gov_old

        pfb_new = d.loc[d["资产类别"] == "政策性金融债-新债", "Net"].sum()
        pfb_old = d.loc[d["资产类别"] == "政策性金融债-老债", "Net"].sum()
        f8 = pfb_new - pfb_old

        inst = d.groupby("机构类型")["Net"].sum()
        abs_sum = inst.abs().sum() + 1e-12
        crowd_buy = inst.clip(lower=0).sum() / abs_sum
        crowd_sell = (-inst).clip(lower=0).sum() / abs_sum
        top3_share = inst.abs().sort_values(ascending=False).head(3).sum() / abs_sum
        f9 = crowd_buy * top3_share
        f10 = crowd_sell * top3_share

        return pd.Series(
            {
                "F1_长端净买入(7Y+)_亿": net_long,
                "F2_长端净买入强度": f2,
                "F3_久期倾向DurFlow": f3,
                "F4_曲线偏好(长端-短端)_亿": f4,
                "F5_政金相对国债偏好(PFB-GOV)_亿": f5,
                "F6_地方相对政金偏好(LG-PFB)_亿": f6,
                "F7_国债新老偏好(新-老)_亿": f7,
                "F8_政金新老偏好(新-老)_亿": f8,
                "F9_买方拥挤×集中度": f9,
                "F10_卖方拥挤×集中度": f10,
                "CrowdBuy": crowd_buy,
                "CrowdSell": crowd_sell,
                "Top3_share_|Net|": top3_share,
                "净买入合计(利率债)_亿": d["Net"].sum(),
                "成交合计(利率债,Turnover)_亿": turnover_total,
            }
        )

    factors = (
        df.groupby("日期", sort=True)
        .apply(compute_daily_factors)
        .reset_index()
        .sort_values("日期")
        .reset_index(drop=True)
    )

    zscore_win = 60
    ewma_halflife_10 = 10
    ewma_halflife_20 = 20

    def rolling_zscore(s: pd.Series, win: int) -> pd.Series:
        m = s.rolling(win, min_periods=max(10, win // 3)).mean()
        sd = s.rolling(win, min_periods=max(10, win // 3)).std(ddof=0)
        return (s - m) / (sd + 1e-12)

    core_cols = [
        "F1_长端净买入(7Y+)_亿",
        "F2_长端净买入强度",
        "F3_久期倾向DurFlow",
        "F4_曲线偏好(长端-短端)_亿",
        "F5_政金相对国债偏好(PFB-GOV)_亿",
        "F6_地方相对政金偏好(LG-PFB)_亿",
        "F7_国债新老偏好(新-老)_亿",
        "F8_政金新老偏好(新-老)_亿",
        "F9_买方拥挤×集中度",
        "F10_卖方拥挤×集中度",
    ]

    out = factors.copy()
    for c in core_cols:
        out[f"{c}_z{zscore_win}"] = rolling_zscore(out[c], zscore_win)
        out[f"{c}_ewmHL{ewma_halflife_10}"] = out[c].ewm(
            halflife=ewma_halflife_10, adjust=False
        ).mean()
        out[f"{c}_ewmHL{ewma_halflife_20}"] = out[c].ewm(
            halflife=ewma_halflife_20, adjust=False
        ).mean()

    zF3 = out[f"F3_久期倾向DurFlow_z{zscore_win}"]
    zF4 = out[f"F4_曲线偏好(长端-短端)_亿_z{zscore_win}"]
    zF5 = out[f"F5_政金相对国债偏好(PFB-GOV)_亿_z{zscore_win}"]
    zF6 = out[f"F6_地方相对政金偏好(LG-PFB)_亿_z{zscore_win}"]
    zF9 = out[f"F9_买方拥挤×集中度_z{zscore_win}"]
    zF10 = out[f"F10_卖方拥挤×集中度_z{zscore_win}"]

    out["Score_A_久期(加久期倾向)"] = 0.7 * zF3 + 0.3 * zF4
    out["Score_B_RV(券种轮动)"] = 0.5 * zF5 + 0.5 * zF6
    out["Score_C_执行(追涨风险)"] = 0.6 * zF9 - 0.4 * zF10

    def tri_signal(x: pd.Series, th: float = 0.5) -> np.ndarray:
        return np.where(x > th, 1, np.where(x < -th, -1, 0))

    out["Signal_A_久期"] = tri_signal(out["Score_A_久期(加久期倾向)"])
    out["Signal_B_RV"] = tri_signal(out["Score_B_RV(券种轮动)"])
    out["Signal_C_执行"] = tri_signal(out["Score_C_执行(追涨风险)"])

    return out

if df is not None:
    footnote_texts = (
        df.loc[
            df["机构类型"].astype("string").str.contains("机构分类|期限分类", na=False),
            "机构类型",
        ]
        .dropna()
        .astype("string")
        .str.strip()
        .unique()
        .tolist()
    )
    df = df[~df["机构类型"].astype("string").str.contains("机构分类|期限分类", na=False)]
    start_date = pd.Timestamp("2024-12-31")

    with st.expander("数据文件", expanded=False):
        st.write(f"当前数据文件：{file_path.name}")
        st.write(f"文件位置：{file_path}")
        st.write(f"文件大小：{file_signature[1]:,} bytes")

        uploaded = st.file_uploader("上传新的数据文件（CSV）", type=["csv"], key="uploaded_csv")
        col_u1, col_u2 = st.columns([1, 3])
        with col_u1:
            do_save = st.button("保存", key="save_upload")
        with col_u2:
            do_save_and_compute = st.button("保存并计算因子", key="save_compute")

        if uploaded is not None and (do_save or do_save_and_compute):
            SOURCE_CSV.write_bytes(uploaded.getvalue())
            st.cache_data.clear()
            st.success("已保存新数据文件。")

            if do_save_and_compute:
                new_sig = get_file_signature(SOURCE_CSV)
                with st.spinner("计算因子并写入缓存..."):
                    enh = compute_mvp_factors(str(SOURCE_CSV), new_sig, "1900-01-01")
                    enh.to_parquet(FACTORS_PARQUET, index=False)
                    FACTORS_META.write_text(
                        json.dumps(
                            {
                                "source_mtime": new_sig[0],
                                "source_size": new_sig[1],
                            },
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                st.success("因子已计算并缓存。")

            st.rerun()

        if FACTORS_PARQUET.exists() and FACTORS_META.exists():
            try:
                meta = json.loads(FACTORS_META.read_text(encoding="utf-8"))
                st.write(
                    f"已缓存因子：source_size={meta.get('source_size')}, source_mtime={meta.get('source_mtime')}"
                )
            except Exception:
                st.write("已缓存因子：meta 读取失败")

    tab = st.query_params.get("tab", "trade")
    if isinstance(tab, list):
        tab = tab[0] if tab else "trade"
    if tab not in {"trade", "factors"}:
        tab = "trade"

    st.markdown(
        f"""
<div class="topnav">
  <a href="?tab=trade" class="{'active' if tab == 'trade' else ''}">现券成交分机构数据可视化</a>
  <a href="?tab=factors" class="{'active' if tab == 'factors' else ''}">机构行为因子（市场层面）</a>
</div>
""",
        unsafe_allow_html=True,
    )

    if tab == "trade":

        col1, col2 = st.columns(2)

        with col1:
            institutions = sorted(df["机构类型"].dropna().unique().tolist())
            default_institution = "大型银行"
            institution_index = (
                institutions.index(default_institution)
                if default_institution in institutions
                else 0
            )
            selected_institution = st.selectbox(
                "选择机构类型",
                institutions,
                index=institution_index,
                key="selected_institution",
            )

        with col2:
            assets = sorted(df["资产类别"].dropna().unique().tolist())
            default_asset = "国债-老债"
            asset_index = assets.index(default_asset) if default_asset in assets else 0
            selected_asset = st.selectbox(
                "选择资产类别",
                assets,
                index=asset_index,
                key="selected_asset",
            )

        filtered_df = df[
            (df["机构类型"] == selected_institution) & (df["资产类别"] == selected_asset)
        ].copy()
        filtered_df = filtered_df[filtered_df["日期"] >= start_date]
        filtered_df = filtered_df[filtered_df["期限"] != "合计"]
        filtered_df = filtered_df.sort_values("日期")
        has_filtered = not filtered_df.empty

        if has_filtered:
            term_order = [
                "≦1Y",
                "1-3Y",
                "3-5Y",
                "5-7Y",
                "7-10Y",
                "10-15Y",
                "15-20Y",
                "20-30Y",
                ">30Y",
            ]

            chart_df = (
                filtered_df.groupby(["日期", "期限"], as_index=False)
                .agg(
                    Net=("Net", "sum"),
                    Buy=("Buy", "sum"),
                    Sell=("Sell", "sum"),
                    Turnover=("Turnover", "sum"),
                )
                .sort_values("日期")
            )

            date_order = chart_df["日期"].dt.strftime("%Y-%m-%d").unique().tolist()
            chart_df["日期"] = chart_df["日期"].dt.strftime("%Y-%m-%d")

            available_terms = chart_df["期限"].dropna().unique().tolist()
            ordered_terms = [t for t in term_order if t in available_terms]
            missing_terms = [t for t in available_terms if t not in ordered_terms]
            ordered_terms.extend(sorted(missing_terms))
            chart_df["期限"] = pd.Categorical(
                chart_df["期限"], categories=ordered_terms, ordered=True
            )

            fig = px.bar(
                chart_df,
                x="日期",
                y="Net",
                color="期限",
                title=f"{selected_institution} - {selected_asset} 每日成交净额",
                labels={"Net": "成交净额 (亿元)", "日期": "日期", "期限": "期限"},
                hover_data=["Buy", "Sell", "Turnover"],
                category_orders={"期限": ordered_terms, "日期": date_order},
            )

            fig.update_layout(
                xaxis_title="日期",
                yaxis_title="成交净额（亿）",
                barmode="relative",
                hovermode="x unified",
                showlegend=False,
            )
            fig.update_xaxes(type="category")

            for trace in fig.data:
                trace.offsetgroup = "stack"
                trace.alignmentgroup = "stack"
                trace.hovertemplate = "%{fullData.name}: %{y:.2f}<extra></extra>"

            st.plotly_chart(fig, width="stretch")

            st.markdown("---")

            st.markdown("### 期限结构热力图")
            heat_df = (
                filtered_df.groupby(["日期", "期限"], as_index=False)["Net"].sum()
            )
            heat_pivot = heat_df.pivot(index="期限", columns="日期", values="Net")
            heat_pivot = heat_pivot.reindex([t for t in ordered_terms if t in heat_pivot.index])

            fig_heat = px.imshow(
                heat_pivot,
                aspect="auto",
                color_continuous_scale="RdBu_r",
                labels={"x": "日期", "y": "期限", "color": "净买入"},
                title=f"{selected_institution} - {selected_asset} 期限结构热力图",
            )
            st.plotly_chart(fig_heat, width="stretch")

            st.markdown("### 期限净买入排名")
            rank_df = (
                filtered_df.groupby("期限", as_index=False)["Net"].sum().sort_values("Net", ascending=False)
            )
            rank_df["期限"] = pd.Categorical(rank_df["期限"], categories=ordered_terms, ordered=True)
            st.dataframe(rank_df, width="stretch", hide_index=True)
        else:
            st.warning("该筛选条件下没有数据。")

        if footnote_texts:
            st.markdown("---")
            st.caption("\n\n".join(footnote_texts))

    else:
        factors_enh = None
        meta_ok = False

        if FACTORS_PARQUET.exists() and FACTORS_META.exists() and file_signature != (0.0, 0):
            try:
                meta = json.loads(FACTORS_META.read_text(encoding="utf-8"))
                meta_ok = (
                    float(meta.get("source_mtime", -1)) == float(file_signature[0])
                    and int(meta.get("source_size", -1)) == int(file_signature[1])
                )
            except Exception:
                meta_ok = False

        if meta_ok:
            @st.cache_data
            def load_cached_factors(parquet_path: str, sig: tuple[float, int]) -> pd.DataFrame:
                return pd.read_parquet(parquet_path)

            factors_enh = load_cached_factors(str(FACTORS_PARQUET), file_signature)

        if factors_enh is None or factors_enh.empty:
            st.info("未发现可用的缓存因子。请在上方‘数据文件’里点击‘保存并计算因子’。")
        else:
            display_start = pd.Timestamp("2025-01-01")
            factors_view = factors_enh[factors_enh["日期"] >= display_start].copy()
            if factors_view.empty:
                st.info("2025-01-01 以来没有可展示的因子数据。")
                factors_view = factors_enh.tail(1).copy()

            def render_explain(text: str) -> str:
                t = text
                if t.startswith(">"):
                    t = "\\" + t
                if t.startswith("<"):
                    t = "\\" + t
                t = t.replace("\n>", "\n\\>").replace("\n<", "\n\\<")
                return t.replace("\n", "  \n")

            def raw_factor_explain(f: str) -> str:
                mapping: dict[str, str] = {
                    "F3_久期倾向DurFlow": ">0：市场整体在“加久期”（利率下行/做平的风更大）；<0：整体在“减久期”（利率上行风险/资金面偏紧/去杠杆信号）",
                    "F1_长端净买入(7Y+)_亿": ">0：长端被净买，长端更强；<0：长端被净卖，长端承压",
                    "F4_曲线偏好(长端-短端)_亿": ">0：长端相对短端更强 → 倾向“做平/长端牛”\n\n<0：短端更强 → 倾向“做陡/偏防御/资金面因素更重”",
                    "F2_长端净买入强度": "0：长端净买入更集中、更“方向性”\n\n<0：长端净卖出更集中\n\n绝对值越大：方向越明确；接近0：更偏对冲/双边。",
                    "F5_政金相对国债偏好(PFB-GOV)_亿": ">0：偏政金（利差/票息更受欢迎，政金利差更可能收敛）\n\n<0：偏国债（更防御、更重流动性/安全）",
                    "F6_地方相对政金偏好(LG-PFB)_亿": ">0：偏地方（地方利差压力小/可能收敛）\n\n<0：偏政金（地方利差可能走阔，配置更谨慎）",
                    "F7_国债新老偏好(新-老)_亿": ">0（偏新券）：交易化、对冲/流动性需求更强（也更容易出现拥挤）\n\n<0（偏老券）：配置盘“捡便宜”、看 carry/roll 或相对价值更明显",
                    "F8_政金新老偏好(新-老)_亿": ">0（偏新券）：交易化、对冲/流动性需求更强（也更容易出现拥挤）\n\n<0（偏老券）：配置盘“捡便宜”、看 carry/roll 或相对价值更明显",
                    "F9_买方拥挤×集中度": "高：买盘一致且集中（“最后一棒”风险↑）→ 配置盘不宜追",
                    "F10_卖方拥挤×集中度": "高：卖盘一致且集中（被动卖压概率↑）→ 可能出现“好价接货”窗口，但要等卖压衰减",
                }
                return mapping.get(f, "")

            def score_explain(s: str) -> str:
                mapping: dict[str, str] = {
                    "Score_A_久期(加久期倾向)": "Score_A_久期（加久期倾向） = 0.7z(F3) + 0.3z(F4)\n\n它回答：现在加久期是不是顺风？顺风程度有多强？\n\n>0：加久期顺风；<0：减久期/谨慎\n\nz(F3) 是主力：整体久期需求\n\nz(F4) 是辅助：风来自长端还是短端（长端强会拉高 Score_A）\n\n投资建议：\n\nScore_A 持续为正（比如连续 3-5 天）→ 可以把久期逐步加到目标上限附近\n\nScore_A 由正转负 → 停止加仓，或把执行拆更细",
                    "Score_B_RV(券种轮动)": "Score_B_RV（券种轮动） = 0.5z(F5) + 0.5z(F6)\n\n它回答：券种上更应该偏政金还是国债？地方要不要上？\n\n>0：更偏“政金/地方”（相对国债更受欢迎）；<0：更偏“国债/政金”（地方更弱）\n\n投资建议：\n\nScore_B 上行：国债预算可以向政金倾斜；若 F6 也偏正，地方可以更积极（仍要看我行具体地方债约束）\n\nScore_B 下行：回到更防御：国债权重↑、地方更谨慎",
                    "Score_C_执行(追涨风险)": "Score_C_执行（追涨风险） = 0.6z(F9) - 0.4z(F10)\n\n它回答：现在追进去会不会“接最后一棒”？执行风险大不大？\n\n>0：追涨风险高（买方拥挤强于卖方拥挤）；<0：更像“卖压阶段”（可能存在好价，但要确认卖压衰减）\n\n投资建议：\n\nScore_C 高：不追；必须买就拆单、慢执行、用更流动券过桥\n\nScore_C 低：更适合接货（但配合买卖压力衰减判断）",
                }
                return mapping.get(s, "")

            def signal_explain(s: str) -> str:
                mapping: dict[str, str] = {
                    "Signal_A_久期": "设定的阈值 |score| > 0.5 点灯（±1），否则 0。；+1：强信号（偏积极）；0：中性/噪声区（不要过度解读）；-1：反向强信号（偏保守或反向操作）；+1：加久期窗口更明确；-1：减久期/谨慎窗口更明确；0：不要用久期做方向下注，更多做 RV 或等待",
                    "Signal_B_RV": "设定的阈值 |score| > 0.5 点灯（±1），否则 0。；+1：强信号（偏积极）；0：中性/噪声区（不要过度解读）；-1：反向强信号（偏保守或反向操作）；+1：券种轮动更偏“政金/地方”（相对国债）；-1：更偏国债/更防御；0：券种信号不清晰，不要大幅切换配比",
                    "Signal_C_执行": "设定的阈值 |score| > 0.5 点灯（±1），否则 0。；+1：强信号（偏积极）；0：中性/噪声区（不要过度解读）；-1：反向强信号（偏保守或反向操作）；+1：追涨风险显著（买方拥挤）→ 别追/慢执行；-1：卖压/可接货阶段（相对更适合做 bid 接）；0：执行风险不极端，按常规节奏做即可",
                }
                return mapping.get(s, "")

            st.markdown("### 机构行为因子")
            raw_factors = [
                "F1_长端净买入(7Y+)_亿",
                "F2_长端净买入强度",
                "F3_久期倾向DurFlow",
                "F4_曲线偏好(长端-短端)_亿",
                "F5_政金相对国债偏好(PFB-GOV)_亿",
                "F6_地方相对政金偏好(LG-PFB)_亿",
                "F7_国债新老偏好(新-老)_亿",
                "F8_政金新老偏好(新-老)_亿",
                "F9_买方拥挤×集中度",
                "F10_卖方拥挤×集中度",
            ]
            col_mode_label, col_mode = st.columns([1, 5])
            with col_mode_label:
                st.markdown("展示方式")
            with col_mode:
                mode = st.radio(
                    "展示方式",
                    ["只看 zscore60", "原值 + zscore60"],
                    index=0,
                    horizontal=True,
                    label_visibility="collapsed",
                    key="factor_mode",
                )
            selected_raw = st.selectbox(
                "选择因子（原始）",
                raw_factors,
                index=2,
                key="selected_raw_factor",
            )
            selected_z = f"{selected_raw}_z60"

            if mode == "原值 + zscore60":
                fig_raw = px.line(
                    factors_view,
                    x="日期",
                    y=selected_raw,
                    title=f"{selected_raw}（原值）",
                )
                st.plotly_chart(fig_raw, width="stretch")
                explain = raw_factor_explain(selected_raw)
                if explain:
                    st.markdown(render_explain(explain))

            fig_z = px.line(
                factors_view,
                x="日期",
                y=selected_z,
                title=f"{selected_raw}（zscore60）",
            )
            fig_z.add_hline(y=0, line_width=1, line_dash="solid", line_color="#666")
            fig_z.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="#999")
            fig_z.add_hline(y=-0.5, line_width=1, line_dash="dash", line_color="#999")
            fig_z.add_hline(y=1.0, line_width=1, line_dash="dot", line_color="#bbb")
            fig_z.add_hline(y=-1.0, line_width=1, line_dash="dot", line_color="#bbb")
            st.plotly_chart(fig_z, width="stretch")
            st.caption("阈值线：虚线为±0.5（点灯阈值），点线为±1.0（更极端）。")
            explain = raw_factor_explain(selected_raw)
            if explain:
                st.markdown(render_explain(explain))

            st.markdown("### 开关评分")
            score_cols = [
                "Score_A_久期(加久期倾向)",
                "Score_B_RV(券种轮动)",
                "Score_C_执行(追涨风险)",
            ]
            score_df = factors_enh[["日期", *score_cols]].copy()
            score_df = score_df[score_df["日期"] >= display_start].copy()
            if score_df.empty:
                score_df = factors_enh[["日期", *score_cols]].copy()
            score_long = score_df.melt(id_vars=["日期"], var_name="Score", value_name="Value")
            fig_scores = px.line(score_long, x="日期", y="Value", color="Score")
            y_min = float(score_long["Value"].min())
            y_max = float(score_long["Value"].max())
            y_pad = max(0.25, (y_max - y_min) * 0.05)
            y0 = y_min - y_pad
            y1 = y_max + y_pad
            fig_scores.add_hrect(y0=y0, y1=-0.5, fillcolor="#d62728", opacity=0.08, line_width=0)
            fig_scores.add_hrect(y0=-0.5, y1=0.5, fillcolor="#7f7f7f", opacity=0.04, line_width=0)
            fig_scores.add_hrect(y0=0.5, y1=y1, fillcolor="#2ca02c", opacity=0.08, line_width=0)
            fig_scores.add_hline(y=0.5, line_width=1, line_dash="dash", line_color="#999")
            fig_scores.add_hline(y=-0.5, line_width=1, line_dash="dash", line_color="#999")
            fig_scores.update_yaxes(range=[y0, y1])
            st.plotly_chart(fig_scores, width="stretch")

            explain = "\n\n".join(
                [
                    "#### Score_A_久期(加久期倾向)",
                    score_explain("Score_A_久期(加久期倾向)"),
                    "#### Score_B_RV(券种轮动)",
                    score_explain("Score_B_RV(券种轮动)"),
                    "#### Score_C_执行(追涨风险)",
                    score_explain("Score_C_执行(追涨风险)"),
                ]
            )
            st.markdown(render_explain(explain))

            st.markdown("### 信号灯")
            signal_cols = ["Signal_A_久期", "Signal_B_RV", "Signal_C_执行"]
            sig = factors_enh[["日期", *signal_cols]].copy()
            sig = sig[sig["日期"] >= display_start].copy()
            if sig.empty:
                sig = factors_enh[["日期", *signal_cols]].copy()
            sig["日期"] = sig["日期"].dt.strftime("%Y-%m-%d")
            z = np.vstack([sig[c].to_numpy() for c in signal_cols])
            fig_sig = px.imshow(
                z,
                x=sig["日期"].tolist(),
                y=signal_cols,
                zmin=-1,
                zmax=1,
                color_continuous_scale=[
                    [0.0, "#d62728"],
                    [0.5, "#f0f0f0"],
                    [1.0, "#2ca02c"],
                ],
                aspect="auto",
            )
            fig_sig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig_sig, width="stretch")

            explain = "\n\n".join(
                [
                    "#### Signal_A_久期",
                    signal_explain("Signal_A_久期"),
                    "#### Signal_B_RV",
                    signal_explain("Signal_B_RV"),
                    "#### Signal_C_执行",
                    signal_explain("Signal_C_执行"),
                ]
            )
            st.markdown(render_explain(explain))
