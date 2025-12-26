# app.py
# Streamlit dashboard for ecommerce behavior analytics (polished layout + fast mode)

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import timedelta

from src.data_loader import load_data
from src.analytics import (
    compute_kpis,
    event_type_distribution,
    trends_summary,
    funnel_by_session,
    funnel_by_category,
    funnel_by_price_bucket,
    funnel_by_hour,
    category_revenue,
    category_price_revenue,
    repeat_purchase_stats,
    time_to_purchase_distribution,
    high_view_low_buy_categories,
)

# ---------- DATA LOADING ---------- #
@st.cache_data
def get_data() -> pd.DataFrame:
    return load_data()


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.header("Filters")

    min_date = df["event_date"].min()
    max_date = df["event_date"].max()

    # Default date range = last 14 days (or from min_date if data is shorter)
    default_start = max_date - timedelta(days=13)
    if default_start < min_date:
        default_start = min_date

    date_range = st.sidebar.date_input(
        "Date range",
        value=(default_start, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, (tuple, list)):
        start_date, end_date = date_range
    else:
        start_date = end_date = date_range

    filtered = df[
        (df["event_date"] >= start_date) &
        (df["event_date"] <= end_date)
    ]

    # Event type filter
    events = sorted(filtered["event_type"].dropna().unique())
    selected = st.sidebar.multiselect("Event types", events, default=events)
    if selected:
        filtered = filtered[filtered["event_type"].isin(selected)]

    # Category filter
    if "category_code" in filtered.columns:
        cats = sorted(filtered["category_code"].dropna().unique())
        chosen_cats = st.sidebar.multiselect("Category", cats, default=[])
        if chosen_cats:
            filtered = filtered[filtered["category_code"].isin(chosen_cats)]

    # Brand filter
    if "brand" in filtered.columns:
        brands = sorted(filtered["brand"].dropna().unique())
        chosen_brands = st.sidebar.multiselect("Brand", brands, default=[])
        if chosen_brands:
            filtered = filtered[filtered["brand"].isin(chosen_brands)]

    st.sidebar.markdown(f"**Rows after filters:** {len(filtered):,}")

    return filtered

def main():
    st.set_page_config(page_title="E-Commerce Behavior Dashboard", layout="wide")

    st.title("🛒 E-Commerce Behavior Dashboard")
    st.caption(
        "Built with Streamlit using sampled clickstream data from a large multi-category ecommerce dataset."
    )

    st.info(
        "This dashboard runs on a ~1M-row sample derived from an original 8GB+ dataset. "
        "In production, these metrics would be powered by data pipelines or a warehouse."
    )

    # ---- Load data ----
    try:
        df = get_data()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        st.stop()

    filtered = apply_filters(df)

    if filtered.empty:
        st.warning("No data for the selected filters. Try expanding the date range or event types.")
        st.stop()

    # ---------- FAST MODE / SAMPLING FOR HEAVY TABS ---------- #
    st.sidebar.markdown("### ⚡ Performance")
    fast_mode = st.sidebar.toggle(
        "Fast mode (recommended)",
        value=True,
        help="Uses a random sample for heavy charts (Funnel / Categories / Products) to keep the app responsive.",
    )

    MAX_ROWS_DETAILED = 300_000
    if fast_mode and len(filtered) > MAX_ROWS_DETAILED:
        filtered_heavy = filtered.sample(MAX_ROWS_DETAILED, random_state=42)
    else:
        filtered_heavy = filtered

    tab_overview, tab_funnel, tab_categories, tab_products = st.tabs(
        ["📊 Overview", "🔁 Funnel", "🏷 Categories", "📦 Products"]
    )

    # ===================== OVERVIEW =====================
    # (Uses full filtered data for high-level KPIs)
    with tab_overview:
        st.subheader("Business Overview")

        kpis = compute_kpis(filtered)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Revenue", f"${kpis['total_revenue']:,.2f}")
        c2.metric("Orders (sessions w/ purchase)", f"{kpis['total_sessions']:,}")
        c3.metric("Unique Users", f"{kpis['total_users']:,}")
        c4.metric("Conversion Rate", f"{kpis['conversion_rate']*100:.2f}%")
        c5.metric("Avg Order Value", f"${kpis['avg_order_value']:,.2f}")

        st.markdown("---")

        purchases = filtered[filtered["event_type"] == "purchase"]

        col_main_left, col_main_right = st.columns((2, 1))

        with col_main_left:
            st.markdown("#### Revenue Over Time")
            if not purchases.empty:
                rev_daily = (
                    purchases.groupby("event_date")["price"]
                    .sum()
                    .reset_index(name="revenue")
                )
                fig_rev = px.line(
                    rev_daily,
                    x="event_date",
                    y="revenue",
                    markers=True,
                    labels={"event_date": "Date", "revenue": "Revenue"},
                )
                fig_rev.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
                st.plotly_chart(fig_rev, use_container_width=True)
            else:
                st.info("No purchase events in current selection to show revenue trend.")

        with col_main_right:
            st.markdown("#### Active Users & Sessions (Daily)")
            daily = (
                filtered.groupby("event_date")
                .agg(
                    users=("user_id", "nunique"),
                    sessions=("user_session", "nunique"),
                )
                .reset_index()
            )
            if not daily.empty:
                fig_us = px.line(
                    daily,
                    x="event_date",
                    y=["users", "sessions"],
                    markers=True,
                    labels={"value": "Count", "event_date": "Date"},
                )
                fig_us.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
                st.plotly_chart(fig_us, use_container_width=True)
            else:
                st.info("No activity to show users/sessions trend.")

        st.markdown("---")

        col_small_left, col_small_mid, col_small_right = st.columns((1, 1, 1))

        with col_small_left:
            st.markdown("#### Event Type Mix")
            ev_dist = event_type_distribution(filtered)
            if not ev_dist.empty:
                fig_ev = px.bar(
                    ev_dist,
                    x="event_type",
                    y="count",
                    labels={"event_type": "Event type", "count": "Count"},
                )
                fig_ev.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
                st.plotly_chart(fig_ev, use_container_width=True)
            else:
                st.info("No events to show type mix.")

        with col_small_mid:
            st.markdown("#### Avg Order Value Trend")
            if not purchases.empty:
                rev_daily = (
                    purchases.groupby("event_date")["price"]
                    .sum()
                    .reset_index(name="revenue")
                )
                orders_daily = (
                    purchases.groupby("event_date")["user_session"]
                    .nunique()
                    .reset_index(name="orders")
                )
                aov = rev_daily.merge(orders_daily, on="event_date")
                aov["aov"] = aov["revenue"] / aov["orders"].replace(0, pd.NA)
                aov = aov.dropna(subset=["aov"])
                if not aov.empty:
                    fig_aov = px.line(
                        aov,
                        x="event_date",
                        y="aov",
                        markers=True,
                        labels={"event_date": "Date", "aov": "AOV"},
                    )
                    fig_aov.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=260)
                    st.plotly_chart(fig_aov, use_container_width=True)
                else:
                    st.info("Not enough data to compute AOV trend.")
            else:
                st.info("No purchases to compute AOV.")

        with col_small_right:
            st.markdown("#### Highlights (Auto Summary)")
            st.write(trends_summary(filtered))

        st.markdown("---")

        col_bottom_left, col_bottom_right = st.columns((1, 1))

        with col_bottom_left:
            st.markdown("#### Top Categories by Revenue")
            top_cat = category_revenue(filtered, top_n=5)
            if not top_cat.empty:
                st.dataframe(top_cat, use_container_width=True, hide_index=True)
            else:
                st.info("No category revenue to show.")

        with col_bottom_right:
            st.markdown("#### Top Brands by Purchasing Users")
            if not purchases.empty and "brand" in purchases.columns:
                top_brand_buyers = (
                    purchases.dropna(subset=["brand"])
                    .groupby("brand")["user_id"]
                    .nunique()
                    .reset_index(name="buyers")
                    .sort_values("buyers", ascending=False)
                    .head(5)
                )
                st.dataframe(top_brand_buyers, use_container_width=True, hide_index=True)
            else:
                st.info("No brand-level purchase data in selection.")

    # ===================== FUNNEL =====================
    # (Uses filtered_heavy for performance)
    with tab_funnel:
        st.subheader("View → Cart → Purchase Funnel")

        funnel_df = funnel_by_session(filtered_heavy)
        if funnel_df.empty:
            st.info("Not enough session data to compute funnel.")
        else:
            row = funnel_df.iloc[0]

            c1, c2, c3 = st.columns(3)
            c1.metric("Sessions w/ View", f"{row['sessions_with_view']:,}")
            c2.metric("Sessions w/ Cart", f"{row['sessions_with_cart']:,}")
            c3.metric("Sessions w/ Purchase", f"{row['sessions_with_purchase']:,}")

            st.markdown("#### Drop-off Rates")

            conv_df = pd.DataFrame(
                {
                    "Step": ["View → Cart", "Cart → Purchase"],
                    "Conversion (%)": [
                        row["view_to_cart_rate"] * 100,
                        row["cart_to_purchase_rate"] * 100,
                    ],
                }
            )
            fig_conv = px.bar(
                conv_df,
                x="Step",
                y="Conversion (%)",
                text="Conversion (%)",
            )
            fig_conv.update_traces(texttemplate="%{text:.1f}", textposition="outside")
            fig_conv.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
            st.plotly_chart(fig_conv, use_container_width=True)

            st.markdown("##### Step Counts")
            st.dataframe(
                pd.DataFrame(
                    {
                        "Step": ["View", "Cart", "Purchase"],
                        "Sessions": [
                            row["sessions_with_view"],
                            row["sessions_with_cart"],
                            row["sessions_with_purchase"],
                        ],
                    }
                ),
                hide_index=True,
                use_container_width=True,
            )

        st.markdown("#### Event Type Trend Over Time")
        events_daily = (
            filtered_heavy.groupby(["event_date", "event_type"])["user_id"]
            .nunique()
            .reset_index(name="users")
        )
        if not events_daily.empty:
            fig_ftrend = px.line(
                events_daily,
                x="event_date",
                y="users",
                color="event_type",
                markers=True,
                labels={"event_date": "Date", "users": "Users", "event_type": "Event type"},
            )
            fig_ftrend.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=320)
            st.plotly_chart(fig_ftrend, use_container_width=True)
        else:
            st.info("No data to show event-type trend.")

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Funnel by Category (Top 5)")
            fc = funnel_by_category(filtered_heavy, top_n=5)
            if not fc.empty:
                df_long = fc.melt(
                    id_vars="category_code",
                    value_vars=["view_to_cart_rate", "cart_to_purchase_rate"],
                    var_name="step",
                    value_name="rate",
                )
                df_long["rate_percent"] = df_long["rate"] * 100
                fig = px.bar(
                    df_long,
                    x="category_code",
                    y="rate_percent",
                    color="step",
                    barmode="group",
                    labels={"rate_percent": "Conversion (%)", "category_code": "Category"},
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No purchase/category data available for current filters.")

        with col2:
            st.markdown("#### Funnel by Price Bucket")
            fp = funnel_by_price_bucket(filtered_heavy)
            if not fp.empty:
                df_long = fp.melt(
                    id_vars="price_bucket",
                    value_vars=["view_to_cart_rate", "cart_to_purchase_rate"],
                    var_name="step",
                    value_name="rate",
                )
                df_long["rate_percent"] = df_long["rate"] * 100
                fig = px.bar(
                    df_long,
                    x="price_bucket",
                    y="rate_percent",
                    color="step",
                    barmode="group",
                    labels={"price_bucket": "Price bucket", "rate_percent": "Conversion (%)"},
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price information available for funnel breakdown.")

        st.markdown("#### Time-of-Day Conversion")
        fh = funnel_by_hour(filtered_heavy)
        if not fh.empty:
            plot_df = fh.copy()
            plot_df["view_to_cart_%"] = plot_df["view_to_cart_rate"] * 100
            plot_df["cart_to_purchase_%"] = plot_df["cart_to_purchase_rate"] * 100
            fig = px.line(
                plot_df,
                x="hour",
                y=["view_to_cart_%", "cart_to_purchase_%"],
                markers=True,
                labels={"value": "Conversion (%)", "hour": "Hour of day"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough hourly data to show time-of-day funnel.")

    # ===================== CATEGORIES =====================
    with tab_categories:
        st.subheader("Category Performance")

        st.markdown("#### Revenue Leaders")
        cr = category_revenue(filtered, top_n=10)
        if not cr.empty:
            fig = px.bar(
                cr,
                x="category_code",
                y="revenue",
                labels={"category_code": "Category", "revenue": "Revenue"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No purchase events for selected filters.")

        st.markdown("#### Estimated Margin vs Revenue (30% margin assumption)")
        cpr = category_price_revenue(filtered, top_n=10)
        if not cpr.empty:
            fig = px.bar(
                cpr,
                x="category_code",
                y=["revenue", "margin_estimate"],
                barmode="group",
                labels={"category_code": "Category", "value": "Amount"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No revenue data to estimate margins.")

        # ---- Advanced behavior metrics (slower) ----
        show_advanced = st.checkbox(
            "Show advanced behavior metrics (slower to compute)",
            value=False,
        )

        if show_advanced:
            st.markdown("#### Repeat Purchase Distribution (by User)")
            rp = repeat_purchase_stats(filtered)
            if not rp.empty:
                fig = px.bar(
                    rp,
                    x="purchase_sessions_bucket",
                    y="users",
                    labels={
                        "purchase_sessions_bucket": "Number of purchase sessions",
                        "users": "Users",
                    },
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No repeat purchase behavior found in current selection.")

            st.markdown("#### Time-to-Purchase Distribution (minutes)")
            ttp = time_to_purchase_distribution(filtered)
            if not ttp.empty:
                fig = px.histogram(
                    ttp,
                    x="minutes_to_purchase",
                    nbins=40,
                    labels={"minutes_to_purchase": "Minutes from first view to first purchase"},
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Cannot compute time-to-purchase with current data.")

            st.markdown("#### High-View / Low-Buy Categories")
            hv = high_view_low_buy_categories(filtered, min_views=200)
            if not hv.empty:
                st.dataframe(hv)
            else:
                st.info(
                    "No categories meet the 'high-view / low-buy' condition for current filters. "
                    "Try expanding the date range."
                )
        else:
            st.info(
                "Enable 'advanced behavior metrics' above to see repeat purchase, "
                "time-to-purchase and high-view/low-buy analysis."
            )

    # ===================== PRODUCTS =====================
    with tab_products:
        st.subheader("Product & Brand Performance")

        purchases_heavy = filtered_heavy[filtered_heavy["event_type"] == "purchase"]

        st.markdown("#### Top Brands by Purchasing Users")
        if not purchases_heavy.empty and "brand" in purchases_heavy.columns:
            top_brand_buyers = (
                purchases_heavy.dropna(subset=["brand"])
                .groupby("brand")["user_id"]
                .nunique()
                .reset_index(name="buyers")
                .sort_values("buyers", ascending=False)
                .head(10)
            )
            fig = px.bar(
                top_brand_buyers,
                x="brand",
                y="buyers",
                labels={"brand": "Brand", "buyers": "Unique buyers"},
            )
            fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No purchase data with brand information in current selection.")

        st.markdown("#### Product Purchase Rate by Brand (View vs Cart vs Purchase)")
        if "brand" in filtered_heavy.columns:
            brand_event = (
                filtered_heavy.dropna(subset=["brand"])
                .groupby(["brand", "event_type"])["user_id"]
                .nunique()
                .reset_index(name="users")
            )

            if not brand_event.empty:
                brand_totals = (
                    brand_event.groupby("brand")["users"]
                    .sum()
                    .sort_values(ascending=False)
                    .head(10)
                    .index
                )
                brand_event = brand_event[brand_event["brand"].isin(brand_totals)]

                fig = px.bar(
                    brand_event,
                    x="brand",
                    y="users",
                    color="event_type",
                    barmode="group",
                    labels={"users": "Users", "brand": "Brand", "event_type": "Event type"},
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=350)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No brand-level events to show.")
        else:
            st.info("Brand field not available in this dataset.")

            st.markdown("#### Price Range of Purchased Products")
        if not purchases_heavy.empty:
            prices = purchases_heavy["price"].dropna()
            if not prices.empty:
                bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, 10_000]
                labels = ["0–50", "50–100", "100–200", "200–500",
                          "500–1000", "1k–2k", "2k–5k", "5k+"]
                bucket = pd.cut(prices, bins=bins, labels=labels, right=False)
                dist = (
                    bucket.value_counts()
                    .sort_index()
                    .rename_axis("price_range")
                    .reset_index(name="count")
                )
                fig = px.bar(
                    dist,
                    x="price_range",
                    y="count",
                    labels={"price_range": "Price range", "count": "Purchased products"},
                )
                fig.update_layout(margin=dict(l=0, r=0, t=10, b=0), height=300)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No price information available on purchase events.")
        else:
            st.info("No purchase events in current selection.")


        st.markdown("#### Top Purchased Products (by Revenue)")
        if not purchases_heavy.empty:
            top_products = (
                purchases_heavy.groupby("product_id")["price"]
                .sum()
                .reset_index(name="revenue")
                .sort_values("revenue", ascending=False)
                .head(20)
            )
            st.dataframe(top_products)
        else:
            st.info("No purchased products to display for the selected filters.")


if __name__ == "__main__":
    main()
