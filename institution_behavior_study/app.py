import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Set page configuration
st.set_page_config(page_title="现券成交数据可视化", layout="wide")

# Title
st.markdown(
    """
    <div style="font-size:22px; font-weight:600; margin-bottom: 8px;">
      现券成交分机构数据可视化
    </div>
    """,
    unsafe_allow_html=True,
)

# Load data
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

from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import os

BASE_DIR = Path(__file__).resolve().parent
file_path = BASE_DIR / "bond_trades_all_in_one.csv"
try:
    file_signature = (os.path.getmtime(file_path), os.path.getsize(file_path))
except OSError:
    file_signature = (0.0, 0)

if st.button("刷新数据"):
    st.cache_data.clear()
    st.rerun()

df = load_data(str(file_path), file_signature)

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

    # Sidebar or Top filters
    col1, col2 = st.columns(2)
    
    with col1:
        # Get unique institutions
        institutions = sorted(df["机构类型"].dropna().unique().tolist())
        default_institution = "大型银行"
        institution_index = (
            institutions.index(default_institution) if default_institution in institutions else 0
        )
        selected_institution = st.selectbox(
            "选择机构类型",
            institutions,
            index=institution_index,
            key="selected_institution",
        )
        
    with col2:
        # Get unique asset classes
        assets = sorted(df["资产类别"].dropna().unique().tolist())
        default_asset = "国债-老债"
        asset_index = assets.index(default_asset) if default_asset in assets else 0
        selected_asset = st.selectbox(
            "选择资产类别",
            assets,
            index=asset_index,
            key="selected_asset",
        )
        
    # Filter data
    filtered_df = df[(df["机构类型"] == selected_institution) & (df["资产类别"] == selected_asset)]
    filtered_df = filtered_df[filtered_df["日期"] >= start_date]
    filtered_df = filtered_df[filtered_df["期限"] != "合计"]
    
    # Sort by date
    filtered_df = filtered_df.sort_values("日期")
    
    if not filtered_df.empty:
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
            .agg(Net=("Net", "sum"), Buy=("Buy", "sum"), Sell=("Sell", "sum"), Turnover=("Turnover", "sum"))
            .sort_values("日期")
        )

        date_order = chart_df["日期"].dt.strftime("%Y-%m-%d").unique().tolist()
        chart_df["日期"] = chart_df["日期"].dt.strftime("%Y-%m-%d")

        available_terms = chart_df["期限"].dropna().unique().tolist()
        ordered_terms = [t for t in term_order if t in available_terms]
        missing_terms = [t for t in available_terms if t not in ordered_terms]
        ordered_terms.extend(sorted(missing_terms))
        chart_df["期限"] = pd.Categorical(chart_df["期限"], categories=ordered_terms, ordered=True)

        # Create bar chart
        # Using '期限' for color to stack bars by maturity term
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
        
        # Update layout for better visibility
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
        
        # Display chart
        st.plotly_chart(fig, width="stretch")
            
    else:
        st.warning("该筛选条件下没有数据。")

    if footnote_texts:
        st.markdown("---")
        st.caption("\n\n".join(footnote_texts))
