"""
Primetrade.ai — Trader Performance vs Market Sentiment
Streamlit Dashboard

Run with:  streamlit run streamlit_app.py

Before running, make sure all packages are installed:
    pip install seaborn matplotlib pandas numpy scikit-learn xgboost streamlit
"""

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# ── Page config ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Primetrade.ai — Sentiment Dashboard",
    page_icon="📊",
    layout="wide",
)

# ── Constants ────────────────────────────────────────────────────
FEAR_COLOR  = "#E24B4A"
GREED_COLOR = "#1D9E75"
PALETTE     = {"Fear": FEAR_COLOR, "Greed": GREED_COLOR}

sns.set_theme(style="whitegrid")
plt.rcParams.update({
    "figure.dpi"      : 120,
    "axes.titlesize"  : 12,
    "axes.titleweight": "bold",
    "axes.labelsize"  : 10,
})

# ── Helper ───────────────────────────────────────────────────────
def add_bar_labels(ax, fmt="%.2f"):
    for container in ax.containers:
        ax.bar_label(container, fmt=fmt, padding=3, fontsize=9)

# ── Load data ────────────────────────────────────────────────────
@st.cache_data
def load_data():
    merged_path  = "outputs/merged_daily.csv"
    profile_path = "outputs/account_profile.csv"

    if not os.path.exists(merged_path):
        st.error(
            "❌ `outputs/merged_daily.csv` not found.\n\n"
            "**Fix:** Open `analysis.ipynb` in VS Code and run all cells "
            "(Cell 23 exports the required files). Then re-run the dashboard."
        )
        st.stop()

    df = pd.read_csv(merged_path, parse_dates=["date"])

    # Ensure clean sentiment labels
    df["sentiment"] = df["sentiment"].astype(str).str.strip()
    df["sentiment"] = df["sentiment"].replace({
        "Extreme Fear" : "Fear",
        "Extreme Greed": "Greed",
    })
    df = df[df["sentiment"].isin(["Fear", "Greed"])].copy()

    profile = (
        pd.read_csv(profile_path)
        if os.path.exists(profile_path)
        else pd.DataFrame()
    )
    return df, profile

df, profile = load_data()

# ── Sidebar ──────────────────────────────────────────────────────
st.sidebar.title("🎛️ Filters")

# Date range
min_date   = df["date"].min().date()
max_date   = df["date"].max().date()
date_range = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

# Sentiment
sentiment_filter = st.sidebar.multiselect(
    "Sentiment",
    options=["Fear", "Greed"],
    default=["Fear", "Greed"],
)

# Leverage
raw_lev_min = float(df["avg_leverage"].min())
raw_lev_max = float(df["avg_leverage"].quantile(0.99))
if raw_lev_max <= raw_lev_min:
    raw_lev_max = raw_lev_min + 1.0

lev_range = st.sidebar.slider(
    "Avg leverage range",
    min_value=round(raw_lev_min, 1),
    max_value=round(raw_lev_max, 1),
    value=(round(raw_lev_min, 1), round(raw_lev_max, 1)),
    step=0.1,
)

# Apply filters
try:
    start_date = pd.Timestamp(date_range[0])
    end_date   = pd.Timestamp(date_range[1])
except Exception:
    start_date = df["date"].min()
    end_date   = df["date"].max()

filtered = df[
    (df["date"]         >= start_date) &
    (df["date"]         <= end_date) &
    (df["sentiment"].isin(sentiment_filter)) &
    (df["avg_leverage"] >= lev_range[0]) &
    (df["avg_leverage"] <= lev_range[1])
].copy()

if filtered.empty:
    st.warning("⚠️ No data matches the current filters. Please widen the filter range.")
    st.stop()

# ── Header & KPIs ────────────────────────────────────────────────
st.title("📊 Primetrade.ai — Trader Performance vs Market Sentiment")
st.caption("Hyperliquid historical data × Bitcoin Fear/Greed Index")
st.markdown("---")

fear_df  = filtered[filtered["sentiment"] == "Fear"]
greed_df = filtered[filtered["sentiment"] == "Greed"]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Rows (filtered)", f"{len(filtered):,}")
k2.metric("Fear days",       f"{fear_df['date'].nunique():,}")
k3.metric("Greed days",      f"{greed_df['date'].nunique():,}")
k4.metric(
    "Avg PnL — Fear",
    f"{fear_df['total_pnl'].mean():.4f}" if len(fear_df) else "N/A",
)
k5.metric(
    "Avg PnL — Greed",
    f"{greed_df['total_pnl'].mean():.4f}" if len(greed_df) else "N/A",
    delta=(
        f"{greed_df['total_pnl'].mean() - fear_df['total_pnl'].mean():.4f}"
        if len(fear_df) and len(greed_df) else None
    ),
)
st.markdown("---")

# ── Tabs ─────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    ["📈 Performance", "🧠 Behavior", "🔵 Segments", "📋 Raw data"]
)

# ═════════════════════════════════════════════════════════════════
# TAB 1 — Performance
# ═════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Performance metrics: Fear vs Greed")

    # ── Row 1: 3 bar charts ──────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, metric, title, fmt in zip(
        [c1, c2, c3],
        ["total_pnl",  "win_rate",  "avg_leverage"],
        ["Avg daily PnL", "Avg win rate", "Avg leverage"],
        ["%.4f",       "%.3f",     "%.2f"],
    ):
        means  = filtered.groupby("sentiment")[metric].mean().reindex(["Fear", "Greed"])
        colors = [PALETTE.get(s, "#aaa") for s in means.index]
        fig, ax = plt.subplots(figsize=(4, 3.5))
        means.plot(kind="bar", ax=ax, color=colors, width=0.5, edgecolor="white")
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=0)
        add_bar_labels(ax, fmt=fmt)
        col.pyplot(fig)
        plt.close(fig)

    # ── PnL box plot ─────────────────────────────────────────────
    st.markdown("#### PnL distribution (1st–99th percentile)")
    p1, p99 = filtered["total_pnl"].quantile([0.01, 0.99])
    clipped = filtered.copy()
    clipped["total_pnl"] = clipped["total_pnl"].clip(p1, p99)

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(
        data=clipped, x="sentiment", y="total_pnl",
        palette=PALETTE, order=["Fear", "Greed"],
        width=0.45, flierprops={"markersize": 3, "alpha": 0.4},
        ax=ax,
    )
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.set_title("Daily PnL by sentiment")
    ax.set_ylabel("Total PnL")
    ax.set_xlabel("")
    st.pyplot(fig)
    plt.close(fig)

    # ── PnL timeline ─────────────────────────────────────────────
    st.markdown("#### PnL over time")
    daily_agg = (
        filtered.groupby(["date", "sentiment"])["total_pnl"]
        .mean().reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    for s, grp in daily_agg.groupby("sentiment"):
        ax.scatter(
            grp["date"], grp["total_pnl"],
            label=s, color=PALETTE.get(s, "#888"),
            alpha=0.65, s=18,
        )
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.4)
    ax.set_ylabel("Avg daily PnL")
    ax.legend(title="Sentiment")
    fig.autofmt_xdate()
    st.pyplot(fig)
    plt.close(fig)

    # ── Summary stats table ───────────────────────────────────────
    st.markdown("#### Summary statistics")
    metric_cols = [c for c in
                   ["total_pnl", "win_rate", "avg_leverage", "trade_count", "long_ratio"]
                   if c in filtered.columns]
    st.dataframe(
        filtered.groupby("sentiment")[metric_cols]
        .agg(["mean", "median", "std"]).round(4),
        use_container_width=True,
    )

# ═════════════════════════════════════════════════════════════════
# TAB 2 — Behavior
# ═════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Trader behavior by sentiment")

    # ── Violin plots ─────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)
    for col, metric, title in zip(
        [c1, c2, c3],
        ["trade_count", "avg_leverage", "long_ratio"],
        ["Trades per day", "Avg leverage", "Long ratio"],
    ):
        fig, ax = plt.subplots(figsize=(4, 4))
        sns.violinplot(
            data=filtered, x="sentiment", y=metric,
            palette=PALETTE, order=["Fear", "Greed"],
            inner="quartile", ax=ax,
        )
        ax.set_title(title)
        ax.set_xlabel("")
        col.pyplot(fig)
        plt.close(fig)

    # ── Leverage KDE ─────────────────────────────────────────────
    st.markdown("#### Leverage density")
    fig, ax = plt.subplots(figsize=(10, 4))
    plotted = False
    for s, grp in filtered.groupby("sentiment"):
        lev = grp["avg_leverage"].clip(0, grp["avg_leverage"].quantile(0.99))
        if lev.nunique() > 1:
            lev.plot.kde(ax=ax, label=s, color=PALETTE.get(s, "#888"), linewidth=2)
            plotted = True
    if not plotted:
        ax.text(0.5, 0.5, "Not enough variance to plot KDE",
                ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Avg leverage")
    ax.set_title("Leverage distribution — Fear vs Greed")
    ax.legend(title="Sentiment")
    st.pyplot(fig)
    plt.close(fig)

    # ── Trade count bar ───────────────────────────────────────────
    st.markdown("#### Avg trades per day")
    tc = filtered.groupby("sentiment")["trade_count"].mean().reindex(["Fear", "Greed"])
    fig, ax = plt.subplots(figsize=(5, 3))
    tc.plot(kind="bar", ax=ax,
            color=[PALETTE.get(s, "#aaa") for s in tc.index],
            width=0.45, edgecolor="white")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    add_bar_labels(ax, fmt="%.1f")
    ax.set_title("Avg daily trade count")
    st.pyplot(fig)
    plt.close(fig)

# ═════════════════════════════════════════════════════════════════
# TAB 3 — Segments
# ═════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Trader segment analysis")

    seg_cols_present = all(
        c in filtered.columns for c in ["lev_seg", "freq_seg", "perf_seg"]
    )

    if not seg_cols_present:
        st.info(
            "ℹ️ Segment columns not found in `outputs/merged_daily.csv`.\n\n"
            "Re-run the notebook all the way through Cell 23, then refresh this page."
        )
    else:
        seg_choice = st.selectbox(
            "Choose segment",
            options=["lev_seg", "freq_seg", "perf_seg"],
            format_func=lambda x: {
                "lev_seg" : "High vs Low leverage",
                "freq_seg": "Frequent vs Infrequent",
                "perf_seg": "Consistent winners vs Inconsistent",
            }[x],
        )

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("**Avg PnL by segment & sentiment**")
            piv = (
                filtered.groupby([seg_choice, "sentiment"])["total_pnl"]
                .mean().unstack("sentiment")
            )
            for c in ["Fear", "Greed"]:
                if c not in piv.columns:
                    piv[c] = 0.0
            fig, ax = plt.subplots(figsize=(6, 4))
            piv[["Fear", "Greed"]].plot(
                kind="bar", ax=ax,
                color=[FEAR_COLOR, GREED_COLOR],
                width=0.5, edgecolor="white",
            )
            ax.set_xlabel("")
            ax.tick_params(axis="x", rotation=20)
            ax.legend(title="Sentiment", fontsize=8)
            add_bar_labels(ax, fmt="%.2f")
            st.pyplot(fig)
            plt.close(fig)

        with c2:
            st.markdown("**Win rate heatmap**")
            hm = (
                filtered.groupby([seg_choice, "sentiment"])["win_rate"]
                .mean().unstack("sentiment")
            )
            for c in ["Fear", "Greed"]:
                if c not in hm.columns:
                    hm[c] = np.nan
            fig, ax = plt.subplots(figsize=(5, 3))
            sns.heatmap(
                hm[["Fear", "Greed"]], annot=True, fmt=".2f",
                cmap="RdYlGn", vmin=0.3, vmax=0.7,
                linewidths=0.5, ax=ax,
                cbar_kws={"shrink": 0.8},
            )
            ax.set_ylabel("")
            ax.set_xlabel("Sentiment")
            st.pyplot(fig)
            plt.close(fig)

    # ── Clustering archetypes ─────────────────────────────────────
    if not profile.empty and "archetype" in profile.columns:
        st.markdown("---")
        st.subheader("Behavioral archetypes (clustering)")

        if "pca1" in profile.columns and "pca2" in profile.columns:
            n_clusters    = int(profile["cluster"].nunique())
            palette_clust = sns.color_palette("tab10", n_clusters)
            fig, ax = plt.subplots(figsize=(8, 5))
            for c, grp in profile.groupby("cluster"):
                name = grp["archetype"].iloc[0] if not grp.empty else f"Cluster {c}"
                ax.scatter(
                    grp["pca1"], grp["pca2"],
                    label=name,
                    color=palette_clust[int(c) % n_clusters],
                    alpha=0.7, s=30,
                )
            ax.set_xlabel("PC1")
            ax.set_ylabel("PC2")
            ax.set_title("Trader archetypes — PCA projection")
            ax.legend(fontsize=8)
            st.pyplot(fig)
            plt.close(fig)

        st.markdown("**Cluster mean statistics**")
        exclude = {"account", "cluster", "archetype", "pca1", "pca2",
                   "lev_seg", "freq_seg", "perf_seg"}
        cols_p = [c for c in profile.columns if c not in exclude]
        if cols_p:
            st.dataframe(
                profile.groupby("archetype")[cols_p].mean().round(3),
                use_container_width=True,
            )

# ═════════════════════════════════════════════════════════════════
# TAB 4 — Raw data
# ═════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("Filtered dataset")
    st.write(f"Showing up to 1,000 of **{len(filtered):,}** rows")
    st.dataframe(filtered.head(1000), use_container_width=True)

    csv_bytes = filtered.to_csv(index=False).encode("utf-8")
    st.download_button(
        label    = "⬇️ Download full filtered CSV",
        data     = csv_bytes,
        file_name= "filtered_trader_data.csv",
        mime     = "text/csv",
    )

    with st.expander("📋 Dataset info"):
        st.write(f"**Shape:** {filtered.shape}")
        null_info = filtered.isnull().sum()
        null_info = null_info[null_info > 0]
        if len(null_info):
            st.dataframe(
                pd.DataFrame({
                    "null count": null_info,
                    "null %"    : (null_info / len(filtered) * 100).round(2),
                }),
                use_container_width=True,
            )
        else:
            st.success("No null values ✅")

# ── Footer ────────────────────────────────────────────────────────
st.markdown("---")
st.caption("Primetrade.ai — Data Science Intern Assignment · Built with Streamlit")