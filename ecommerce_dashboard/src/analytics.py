# src/analytics.py
# Reusable analytics logic (optimized + bug-fixes)

import pandas as pd


# =========================
# -------- KPIs -----------
# =========================
def compute_kpis(df: pd.DataFrame) -> dict:
    total_users = df["user_id"].nunique()
    total_sessions = df["user_session"].nunique()

    purchases = df[df["event_type"] == "purchase"]
    total_revenue = purchases["price"].sum()

    if total_sessions > 0:
        sessions_with_purchase = purchases["user_session"].nunique()
        conversion_rate = sessions_with_purchase / total_sessions
    else:
        conversion_rate = 0.0

    avg_order_value = (
        purchases.groupby("user_session")["price"].sum().mean()
        if not purchases.empty else 0.0
    )

    return {
        "total_users": total_users,
        "total_sessions": total_sessions,
        "total_revenue": float(total_revenue),
        "conversion_rate": float(conversion_rate),
        "avg_order_value": float(avg_order_value),
    }


# =========================
# ---- Overview Charts ----
# =========================
def events_over_time(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("event_date")["event_type"]
        .count()
        .reset_index(name="events_count")
        .sort_values("event_date")
    )


def event_type_distribution(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df["event_type"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "event_type"})
    )


def bounce_rate(df: pd.DataFrame) -> float:
    """Percentage of sessions with only 1 event (simple bounce definition)."""
    session_sizes = df.groupby("user_session")["event_type"].count()
    if session_sizes.empty:
        return 0.0
    bounces = (session_sizes == 1).sum()
    total_sessions = len(session_sizes)
    return float(bounces / total_sessions) if total_sessions else 0.0


def session_depth_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of number of events per session (1, 2–3, 4–5, 6–10, 10+)."""
    session_sizes = (
        df.groupby("user_session")["event_type"]
        .count()
        .reset_index(name="events_per_session")
    )

    if session_sizes.empty:
        return pd.DataFrame(columns=["depth_bucket", "count"])

    session_sizes["depth_bucket"] = pd.cut(
        session_sizes["events_per_session"],
        bins=[0, 1, 3, 5, 10, 100],
        labels=["1", "2–3", "4–5", "6–10", "10+"],
    )

    return (
        session_sizes["depth_bucket"]
        .value_counts()
        .reset_index(name="count")
        .rename(columns={"index": "depth_bucket"})
        .sort_values("depth_bucket")
    )


def trends_summary(df: pd.DataFrame) -> str:
    events_df = events_over_time(df)

    if len(events_df) < 2:
        return "Not enough data to generate a trend summary."

    first = events_df.iloc[0]
    last = events_df.iloc[-1]

    change_pct = (
        (last["events_count"] - first["events_count"])
        / max(first["events_count"], 1)
    ) * 100

    if change_pct > 5:
        direction = "increased"
    elif change_pct < -5:
        direction = "decreased"
    else:
        direction = "stayed relatively stable"

    return (
        f"Traffic has {direction} over the selected period. "
        f"Events on the first day: {int(first['events_count']):,}, "
        f"last day: {int(last['events_count']):,} "
        f"({change_pct:.1f}% change)."
    )


# =========================
# -------- Funnel ---------
# =========================
def _session_flags(group: pd.DataFrame) -> pd.Series:
    """(Kept for compatibility; no longer used in main funnels except if you reuse it.)"""
    events = set(group["event_type"])
    return pd.Series(
        dict(
            has_view=("view" in events),
            has_cart=("cart" in events),
            has_purchase=("purchase" in events),
        )
    )


def funnel_by_session(df: pd.DataFrame) -> pd.DataFrame:
    """Fast: count sessions that have view/cart/purchase using distinct (session, event_type)."""
    if df.empty:
        return pd.DataFrame(
            [{
                "sessions_with_view": 0,
                "sessions_with_cart": 0,
                "sessions_with_purchase": 0,
                "view_to_cart_rate": 0.0,
                "cart_to_purchase_rate": 0.0,
            }]
        )

    tmp = df[["user_session", "event_type"]].drop_duplicates()

    sessions_with_view = tmp.loc[tmp["event_type"] == "view", "user_session"].nunique()
    sessions_with_cart = tmp.loc[tmp["event_type"] == "cart", "user_session"].nunique()
    sessions_with_purchase = tmp.loc[tmp["event_type"] == "purchase", "user_session"].nunique()

    view_to_cart_rate = (
        sessions_with_cart / sessions_with_view if sessions_with_view else 0.0
    )
    cart_to_purchase_rate = (
        sessions_with_purchase / sessions_with_cart if sessions_with_cart else 0.0
    )

    return pd.DataFrame(
        [{
            "sessions_with_view": int(sessions_with_view),
            "sessions_with_cart": int(sessions_with_cart),
            "sessions_with_purchase": int(sessions_with_purchase),
            "view_to_cart_rate": float(view_to_cart_rate),
            "cart_to_purchase_rate": float(cart_to_purchase_rate),
        }]
    )


def funnel_by_category(df: pd.DataFrame, top_n: int = 5) -> pd.DataFrame:
    """Fast category-level funnel using crosstab instead of groupby-apply."""
    if "category_code" not in df.columns or df.empty:
        return pd.DataFrame()

    tmp = df.dropna(subset=["category_code"])

    # pick top categories by purchase sessions
    purchases = tmp[tmp["event_type"] == "purchase"]
    if purchases.empty:
        return pd.DataFrame()

    top_cats = (
        purchases.groupby("category_code")["user_session"]
        .nunique()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    if not top_cats:
        return pd.DataFrame()

    tmp = tmp[tmp["category_code"].isin(top_cats)]

    # distinct (category, session, event_type)
    tmp = tmp[["category_code", "user_session", "event_type"]].drop_duplicates()

    # crosstab -> one row per (category, session) with columns event types
    ct = pd.crosstab(
        [tmp["category_code"], tmp["user_session"]],
        tmp["event_type"]
    ).reset_index()

    # ensure columns exist
    for col in ["view", "cart", "purchase"]:
        if col not in ct.columns:
            ct[col] = 0

    agg = (
        ct.groupby("category_code")[["view", "cart", "purchase"]]
        .sum()
        .reset_index()
        .rename(columns={"view": "has_view", "cart": "has_cart", "purchase": "has_purchase"})
    )

    agg["view_to_cart_rate"] = agg["has_cart"] / agg["has_view"].replace(0, pd.NA)
    agg["cart_to_purchase_rate"] = agg["has_purchase"] / agg["has_cart"].replace(0, pd.NA)

    # only fill NA on numeric columns
    num_cols = ["has_view", "has_cart", "has_purchase",
                "view_to_cart_rate", "cart_to_purchase_rate"]
    for col in num_cols:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)

    return agg


def funnel_by_price_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Fast price-bucket funnel using crosstab."""
    if "price" not in df.columns or df.empty:
        return pd.DataFrame()

    tmp = df.copy()
    tmp["price"] = tmp["price"].fillna(0)

    tmp["price_bucket"] = pd.cut(
        tmp["price"],
        bins=[-0.01, 50, 100, 200, 500, 1000, 10_000, 100_000],
        labels=["<=50", "50–100", "100–200", "200–500", "500–1000",
                "1000–10000", "10000+"],
    )

    tmp = tmp[["price_bucket", "user_session", "event_type"]].drop_duplicates()

    ct = pd.crosstab(
        [tmp["price_bucket"], tmp["user_session"]],
        tmp["event_type"]
    ).reset_index()

    for col in ["view", "cart", "purchase"]:
        if col not in ct.columns:
            ct[col] = 0

    agg = (
        ct.groupby("price_bucket")[["view", "cart", "purchase"]]
        .sum()
        .reset_index()
        .rename(columns={"view": "has_view", "cart": "has_cart", "purchase": "has_purchase"})
    )

    agg["view_to_cart_rate"] = agg["has_cart"] / agg["has_view"].replace(0, pd.NA)
    agg["cart_to_purchase_rate"] = agg["has_purchase"] / agg["has_cart"].replace(0, pd.NA)

    # only fill NA on numeric columns, not on price_bucket (categorical)
    num_cols = ["has_view", "has_cart", "has_purchase",
                "view_to_cart_rate", "cart_to_purchase_rate"]
    for col in num_cols:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)

    return agg


def funnel_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Fast hour-of-day funnel using crosstab (no groupby-apply)."""
    if "event_time" not in df.columns or df.empty:
        return pd.DataFrame()

    tmp = df[df["event_type"].isin(["view", "cart", "purchase"])].copy()
    tmp["event_time"] = pd.to_datetime(tmp["event_time"])
    tmp["hour"] = tmp["event_time"].dt.hour

    tmp = tmp[["hour", "user_session", "event_type"]].drop_duplicates()

    ct = pd.crosstab(
        [tmp["hour"], tmp["user_session"]],
        tmp["event_type"]
    ).reset_index()

    for col in ["view", "cart", "purchase"]:
        if col not in ct.columns:
            ct[col] = 0

    agg = (
        ct.groupby("hour")[["view", "cart", "purchase"]]
        .sum()
        .reset_index()
        .rename(columns={"view": "has_view", "cart": "has_cart", "purchase": "has_purchase"})
        .sort_values("hour")
    )

    agg["view_to_cart_rate"] = agg["has_cart"] / agg["has_view"].replace(0, pd.NA)
    agg["cart_to_purchase_rate"] = agg["has_purchase"] / agg["has_cart"].replace(0, pd.NA)

    # only fill NA on numeric columns
    num_cols = ["has_view", "has_cart", "has_purchase",
                "view_to_cart_rate", "cart_to_purchase_rate"]
    for col in num_cols:
        if col in agg.columns:
            agg[col] = agg[col].fillna(0.0)

    return agg


# =========================
# ------ Categories -------
# =========================
def category_revenue(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    purchases = df[df["event_type"] == "purchase"]
    if purchases.empty:
        return pd.DataFrame()

    return (
        purchases.groupby("category_code")["price"]
        .sum()
        .reset_index(name="revenue")
        .sort_values("revenue", ascending=False)
        .head(top_n)
    )


def category_price_revenue(df: pd.DataFrame, top_n=10) -> pd.DataFrame:
    purchases = df[df["event_type"] == "purchase"]

    if purchases.empty:
        return pd.DataFrame()

    agg = (
        purchases.groupby("category_code")
        .agg(revenue=("price", "sum"))
        .reset_index()
    )

    agg["margin_estimate"] = agg["revenue"] * 0.30
    return agg


def repeat_purchase_stats(df: pd.DataFrame) -> pd.DataFrame:
    purchases = df[df["event_type"] == "purchase"]
    if purchases.empty:
        return pd.DataFrame()

    counts = (
        purchases.groupby("user_id")["user_session"]
        .nunique()
        .reset_index(name="purchase_sessions")
    )

    counts["purchase_sessions_bucket"] = pd.cut(
        counts["purchase_sessions"],
        bins=[0, 1, 2, 3, 5, 100],
        labels=["1", "2", "3", "4–5", "6+"],
    )

    return (
        counts["purchase_sessions_bucket"]
        .value_counts()
        .sort_index()
        .reset_index(name="users")
        .rename(columns={"index": "purchase_sessions_bucket"})
    )


def time_to_purchase_distribution(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    tmp["event_time"] = pd.to_datetime(tmp["event_time"])

    views = (
        tmp[tmp["event_type"] == "view"]
        .groupby("user_session")["event_time"]
        .min()
        .rename("first_view_time")
    )

    purchases = (
        tmp[tmp["event_type"] == "purchase"]
        .groupby("user_session")["event_time"]
        .min()
        .rename("first_purchase_time")
    )

    merged = pd.concat([views, purchases], axis=1).dropna()

    merged["minutes_to_purchase"] = (
        merged["first_purchase_time"] - merged["first_view_time"]
    ).dt.total_seconds() / 60

    return merged[merged["minutes_to_purchase"] >= 0].reset_index()[["minutes_to_purchase"]]


def high_view_low_buy_categories(df: pd.DataFrame, min_views=200) -> pd.DataFrame:
    views = df[df["event_type"] == "view"]
    purchases = df[df["event_type"] == "purchase"]

    views_cnt = (
        views.groupby("category_code")["user_session"]
        .nunique()
        .reset_index(name="view_sessions")
    )

    buys_cnt = (
        purchases.groupby("category_code")["user_session"]
        .nunique()
        .reset_index(name="purchase_sessions")
        if not purchases.empty else
        pd.DataFrame(columns=["category_code", "purchase_sessions"])
    )

    merged = views_cnt.merge(buys_cnt, on="category_code", how="left").fillna(0)
    merged = merged[merged["view_sessions"] >= min_views]

    if merged.empty:
        return merged

    merged["conversion_rate"] = (
        merged["purchase_sessions"] / merged["view_sessions"]
    ).fillna(0.0)

    return merged.sort_values("conversion_rate").head(10)
