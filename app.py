# -*- coding: utf-8 -*-
import time
from typing import Callable

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from curl_cffi import requests as cffi_requests

# ======================================================================
# 常數設定
# ======================================================================
PLATFORMS = ["PChome", "MOMO"]
PLATFORM_COLORS = {"PChome": "#2E86DE", "MOMO": "#E84393"}

# --- PChome ---
PCHOME_API = "https://ecshweb.pchome.com.tw/search/v3.3/all/results"
PCHOME_HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
SORT_OPTIONS = {
    "銷量高到低": "sale/dc",
    "價格低到高": "prc/ac",
    "價格高到低": "prc/dc",
    "最新上架": "new/dc",
    "相關度": "rnk/dc",
}

# --- MOMO ---
MOMO_HOME = "https://www.momoshop.com.tw/"
MOMO_API = "https://apisearch.momoshop.com.tw/momoSearchCloud/moec/textSearch"
MOMO_HEADERS = {
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "zh-TW,zh;q=0.9,en;q=0.8",
    "Content-Type": "application/json;charset=UTF-8",
    "Origin": "https://www.momoshop.com.tw",
    "Referer": "https://www.momoshop.com.tw/",
}

st.set_page_config(page_title="PChome vs MOMO 價格比較", page_icon="⚖️", layout="wide")
st.title("⚖️ PChome vs MOMO 商品價格比較")

ProgressCb = Callable[[int, int], None]  # (目前頁, 總頁數)


# ======================================================================
# 爬蟲:PChome
# ======================================================================
def fetch_pchome(keyword: str, pages: int, sort: str, delay: float, report: ProgressCb) -> pd.DataFrame:
    frames = []
    for i in range(1, pages + 1):
        report(i, pages)
        resp = requests.get(
            PCHOME_API,
            params={"q": keyword, "page": i, "sort": sort},  # requests 會自動處理中文編碼
            headers=PCHOME_HEADERS,
            timeout=15,
        )
        resp.raise_for_status()
        prods = resp.json().get("prods", [])
        if not prods:
            break
        page_df = pd.DataFrame(prods)
        page_df["頁碼"] = i
        frames.append(page_df)
        if i < pages and delay > 0:
            time.sleep(delay)

    if not frames:
        return pd.DataFrame()
    raw = pd.concat(frames, ignore_index=True)
    if not {"name", "price"}.issubset(raw.columns):
        return pd.DataFrame()

    df = raw[["name", "price", "頁碼"]].rename(columns={"name": "商品名稱", "price": "價格"})
    df["價格"] = pd.to_numeric(df["價格"], errors="coerce")
    return df.dropna(subset=["商品名稱", "價格"]).reset_index(drop=True)


# ======================================================================
# 爬蟲:MOMO
# ======================================================================
def build_momo_payload(keyword: str, page: int, price_s: int, price_e: int) -> dict:
    return {
        "host": "momoshop",
        "flag": "searchEngine",
        "data": {
            "searchValue": keyword,
            "searchType": "1",
            "cateLevel": "-1",
            "curPage": str(page),
            "priceS": str(price_s),
            "priceE": str(price_e),
            "showType": "chessboardType",
            "isFuzzy": "0",
            "rtnCateDatainfo": {
                "cateCode": "",
                "cateLv": "-1",
                "keyword": keyword,
                "curPage": str(page),  # 與外層一致
                "historyDoPush": "false",
                "timestamp": int(time.time() * 1000),  # 用當下時間
            },
            "flag": 2018,
            "serviceCode": "MT01",
            "addressSearchData": {},
        },
    }


def fetch_momo(
    keyword: str, pages: int, price_s: int, price_e: int, delay: float, report: ProgressCb
) -> pd.DataFrame:
    session = cffi_requests.Session(impersonate="chrome")  # 模擬 Chrome 的 TLS 指紋與 UA
    session.get(MOMO_HOME, headers=MOMO_HEADERS, timeout=20)  # 先取得 cookie

    rows = []
    for i in range(1, pages + 1):
        report(i, pages)
        resp = session.post(
            MOMO_API,
            headers=MOMO_HEADERS,
            json=build_momo_payload(keyword, i, price_s, price_e),
            timeout=20,
        )
        if resp.status_code != 200:
            raise RuntimeError(f"第 {i} 頁請求失敗 (HTTP {resp.status_code}):{resp.text[:200]}")
        products = resp.json().get("rtnSearchData", {}).get("goodsInfoList", [])
        if not products:
            break
        for p in products:
            rows.append(
                {"商品名稱": p.get("goodsName", ""), "價格": p.get("goodsPrice", ""), "頁碼": i}
            )
        if i < pages and delay > 0:
            time.sleep(delay)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # 價格清理:取第一組數字並移除千分位逗號(例如 "$1,273" -> 1273)
    price = df["價格"].astype(str).str.extract(r"(\d[\d,]*)")[0].str.replace(",", "", regex=False)
    df["價格"] = pd.to_numeric(price, errors="coerce")
    df = df.dropna(subset=["價格"])
    return df[df["商品名稱"] != ""].reset_index(drop=True)


# ======================================================================
# 資料處理
# ======================================================================
def add_price_band(df: pd.DataFrame, n_bins: int):
    """以兩個平台合併後的價格範圍切分區間,讓兩邊可直接比較。回傳 (df, 區間標籤順序)"""
    df = df.copy()
    lo, hi = df["價格"].min(), df["價格"].max()
    if lo == hi:
        label = f"{int(lo)}"
        df["價格區間"] = label
        return df, [label]

    n_bins = max(1, min(n_bins, int(df["價格"].nunique())))
    edges = np.linspace(lo, hi, n_bins + 1)
    labels = [f"{int(round(edges[k]))}–{int(round(edges[k + 1]))}" for k in range(n_bins)]
    if len(set(labels)) < len(labels):  # 價差很小時避免標籤重複
        labels = [f"區間{k + 1}" for k in range(n_bins)]
    df["價格區間"] = pd.cut(
        df["價格"], bins=edges, labels=labels, include_lowest=True, ordered=True
    ).astype(str)
    return df, labels


def short(text: str, n: int = 18) -> str:
    return text if len(text) <= n else text[:n] + "…"


def make_reporter(bar, name: str, idx: int, total: int) -> ProgressCb:
    def report(i: int, pages: int) -> None:
        bar.progress((idx + (i - 1) / pages) / total, text=f"[{name}] 正在抓取第 {i}/{pages} 頁...")

    return report


# ======================================================================
# 側邊欄:搜尋條件
# ======================================================================
with st.sidebar:
    st.header("搜尋條件")
    with st.form("search_form"):
        keyword = st.text_input("產品名稱", value="行動電源")
        pages = st.number_input("每個平台抓取頁數", min_value=1, max_value=20, value=3, step=1)
        c1, c2 = st.columns(2)
        price_s = c1.number_input("最低價", min_value=0, value=0, step=100)
        price_e = c2.number_input("最高價", min_value=0, value=9999999, step=1000)
        sort_label = st.selectbox("PChome 排序方式", list(SORT_OPTIONS.keys()))
        delay = st.slider("每頁間隔秒數", 0.0, 10.0, 1.0, 0.5)
        submitted = st.form_submit_button("🔍 開始爬取並比較", use_container_width=True)

if submitted:
    if not keyword.strip():
        st.error("請輸入產品名稱")
    elif price_s > price_e:
        st.error("最低價不可大於最高價")
    else:
        kw_in = keyword.strip()
        bar = st.progress(0, text="準備中...")
        frames, errors = [], []
        for idx, name in enumerate(PLATFORMS):
            report = make_reporter(bar, name, idx, len(PLATFORMS))
            try:
                if name == "PChome":
                    part = fetch_pchome(kw_in, int(pages), SORT_OPTIONS[sort_label], delay, report)
                else:
                    part = fetch_momo(kw_in, int(pages), int(price_s), int(price_e), delay, report)
            except Exception as e:
                errors.append(f"{name} 爬取失敗:{e}")
                continue
            if part.empty:
                errors.append(f"{name} 沒有取得任何資料")
                continue
            part.insert(0, "平台", name)
            frames.append(part)
        bar.progress(1.0, text="抓取完成")
        bar.empty()

        if frames:
            all_df = pd.concat(frames, ignore_index=True)
            # 兩個平台一律套用相同的價格範圍,比較才公平
            all_df = all_df[(all_df["價格"] >= price_s) & (all_df["價格"] <= price_e)]
            all_df["價格"] = all_df["價格"].round().astype(int)
            all_df = all_df.reset_index(drop=True)
            all_df.insert(1, "序號", all_df.groupby("平台").cumcount() + 1)
        else:
            all_df = pd.DataFrame()

        st.session_state["df"] = all_df
        st.session_state["keyword"] = kw_in
        st.session_state["errors"] = errors

# ======================================================================
# 主畫面
# ======================================================================
for msg in st.session_state.get("errors", []):
    st.warning(msg)

df = st.session_state.get("df")
if df is None:
    st.info("👈 請在左側輸入產品名稱與頁數,再按「開始爬取並比較」。")
    st.stop()
if df.empty:
    st.error("沒有取得任何資料,請換個關鍵字或稍後再試。")
    st.stop()

kw = st.session_state.get("keyword", "")
platforms = [p for p in PLATFORMS if p in df["平台"].unique()]
if len(platforms) < len(PLATFORMS):
    st.caption(f"目前只有 {'、'.join(platforms)} 的資料,無法進行完整比較。")

# ---------- 統計摘要 ----------
stats = (
    df.groupby("平台")["價格"]
    .agg(商品數="count", 平均價格="mean", 中位數="median", 最高價格="max", 最低價格="min")
    .round(0)
    .astype(int)
    .reindex(platforms)
)

st.subheader(f"「{kw}」比較結果")
cols = st.columns(len(platforms))
for col, p in zip(cols, platforms):
    with col:
        st.markdown(f"**{p}**")
        a, b = st.columns(2)
        a.metric("商品數", f"{stats.loc[p, '商品數']:,}")
        b.metric("平均價格", f"{stats.loc[p, '平均價格']:,}")
        a.metric("最高價格", f"{stats.loc[p, '最高價格']:,}")
        b.metric("最低價格", f"{stats.loc[p, '最低價格']:,}")

if len(platforms) == 2:
    cheaper = stats["平均價格"].idxmin()
    other = [p for p in platforms if p != cheaper][0]
    diff = stats.loc[other, "平均價格"] - stats.loc[cheaper, "平均價格"]
    st.caption(f"以平均價格來看,{cheaper} 較低,比 {other} 低約 {diff:,} 元(僅代表本次抓取到的商品)。")

n_bins = st.slider("價格區間數量(用於圓餅圖、柱狀圖與旭日圖)", 2, 8, 5)
dfb, labels = add_price_band(df, n_bins)
band_colors = dict(zip(labels, px.colors.qualitative.Set2 * 2))

tab_line, tab_bar, tab_pie, tab_sun, tab_data = st.tabs(
    ["📈 折線圖", "📊 柱狀圖", "🥧 圓餅圖", "☀️ 旭日圖", "📋 資料表"]
)

# ---------- 折線圖 ----------
with tab_line:
    fig = px.line(
        df,
        x="序號",
        y="價格",
        color="平台",
        color_discrete_map=PLATFORM_COLORS,
        category_orders={"平台": platforms},
        hover_name="商品名稱",
        markers=True,
        title=f"「{kw}」各平台商品售價",
    )
    for p in platforms:
        fig.add_hline(
            y=float(stats.loc[p, "平均價格"]),
            line_dash="dash",
            line_color=PLATFORM_COLORS[p],
            annotation_text=f"{p} 平均 {stats.loc[p, '平均價格']:,}",
            annotation_position="top left",
        )
    fig.update_layout(xaxis_title="商品序號", yaxis_title="價格", height=520)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 柱狀圖 ----------
with tab_bar:
    st.markdown("##### 價格統計比較")
    long_stats = (
        stats.reset_index()
        .melt(
            id_vars="平台",
            value_vars=["最低價格", "平均價格", "中位數", "最高價格"],
            var_name="統計量",
            value_name="價格",
        )
    )
    fig = px.bar(
        long_stats,
        x="統計量",
        y="價格",
        color="平台",
        barmode="group",
        text_auto=",.0f",
        color_discrete_map=PLATFORM_COLORS,
        category_orders={"平台": platforms, "統計量": ["最低價格", "平均價格", "中位數", "最高價格"]},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("##### 價格區間商品數比較")
    counts = (
        dfb.groupby(["價格區間", "平台"])
        .size()
        .unstack("平台", fill_value=0)
        .reindex(labels, fill_value=0)
        .stack()
        .reset_index(name="商品數")
    )
    fig = px.bar(
        counts,
        x="價格區間",
        y="商品數",
        color="平台",
        barmode="group",
        text_auto=True,
        color_discrete_map=PLATFORM_COLORS,
        category_orders={"平台": platforms, "價格區間": labels},
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 圓餅圖 ----------
with tab_pie:
    pie_cols = st.columns(len(platforms))
    for col, p in zip(pie_cols, platforms):
        sub = (
            dfb[dfb["平台"] == p]
            .groupby("價格區間")
            .size()
            .reindex(labels, fill_value=0)
            .reset_index(name="商品數")
        )
        sub = sub[sub["商品數"] > 0]
        fig = px.pie(
            sub,
            names="價格區間",
            values="商品數",
            color="價格區間",
            color_discrete_map=band_colors,
            hole=0.3,
            title=f"{p} 價格區間分布",
        )
        fig.update_traces(sort=False, textinfo="label+percent")
        fig.update_layout(height=480, showlegend=False)
        col.plotly_chart(fig, use_container_width=True)

# ---------- 旭日圖 ----------
with tab_sun:
    show_items = st.checkbox("顯示商品層級", value=False)
    sun = dfb.copy()
    sun["商品"] = sun["商品名稱"].map(short)
    path = ["平台", "價格區間"] + (["商品"] if show_items else [])
    title = f"「{kw}」平台 → 價格區間" + (" → 商品" if show_items else "")
    fig = px.sunburst(
        sun,
        path=path,
        color="平台",
        color_discrete_map=PLATFORM_COLORS,
        title=title,
    )
    fig.update_traces(textinfo="label+value")
    fig.update_layout(height=680)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 資料表 ----------
with tab_data:
    st.markdown("##### 統計摘要")
    st.dataframe(stats.reset_index(), use_container_width=True, hide_index=True)

    pick = st.multiselect("篩選平台", platforms, default=platforms)
    view = df[df["平台"].isin(pick)]
    st.dataframe(view, use_container_width=True, hide_index=True)
    st.download_button(
        "⬇️ 下載 CSV",
        data=view.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"比較_{kw}.csv",
        mime="text/csv",
    )
