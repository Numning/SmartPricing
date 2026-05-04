"""
=============================================================================
📱 Smart Pricing — Interactive Smartphone Price Prediction Tool
=============================================================================
ITCS 227 — Introduction to Data Science — Capstone Project
Streamlit Web Application

Two modes:
    🛒 Customer Mode:  Check if a phone is overpriced / underpriced
    💼 Company Mode:   Predict the ideal selling price (brand-independent)

Run:
    python3 -m streamlit run streamlit_app.py
=============================================================================
"""

import os
import re
import warnings

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, ExtraTreesRegressor
)
from sklearn.metrics import r2_score, mean_absolute_error

warnings.filterwarnings("ignore")

RANDOM_STATE = 42

# ── Helper functions (same as main script) ───────────────────────────────────

def parse_weight(val):
    match = re.search(r"([\d.]+)", str(val))
    return float(match.group(1)) if match else np.nan

def parse_ram(val):
    match = re.search(r"(\d+)", str(val))
    return int(match.group(1)) if match else np.nan

def parse_camera(val):
    numbers = re.findall(r"(\d+)\s*MP", str(val), re.IGNORECASE)
    return max([int(n) for n in numbers]) if numbers else np.nan

def parse_camera_count(val):
    numbers = re.findall(r"(\d+)\s*MP", str(val), re.IGNORECASE)
    return len(numbers) if numbers else 0

def parse_battery(val):
    cleaned = str(val).replace(",", "")
    match = re.search(r"([\d.]+)", cleaned)
    return float(match.group(1)) if match else np.nan

def parse_screen(val):
    match = re.search(r"([\d.]+)", str(val))
    return float(match.group(1)) if match else np.nan

def parse_price(val):
    cleaned = str(val).replace(",", "")
    match = re.search(r"([\d.]+)", cleaned)
    return float(match.group(1)) if match else np.nan

def assign_tier(price):
    if price < 300:
        return "Budget"
    elif price < 700:
        return "Mid-Range"
    elif price < 1000:
        return "Premium"
    else:
        return "Ultra-Premium"


# ── Load & prepare data + models (cached) ────────────────────────────────────

@st.cache_resource
def load_data_and_models():
    """Load dataset, preprocess, and train models. Cached for performance."""

    DATA_PATH = os.path.join(
        os.path.expanduser("~"),
        ".cache/kagglehub/datasets/abdulmalik1518/mobiles-dataset-2025/versions/1",
        "Mobiles Dataset (2025).csv",
    )

    if not os.path.exists(DATA_PATH):
        import kagglehub
        dataset_path = kagglehub.dataset_download("abdulmalik1518/mobiles-dataset-2025")
        DATA_PATH = os.path.join(dataset_path, "Mobiles Dataset (2025).csv")

    df_raw = pd.read_csv(DATA_PATH, encoding="latin-1")
    df = df_raw.copy()

    # Parse specs
    df["Weight_g"] = df["Mobile Weight"].apply(parse_weight)
    df["RAM_GB"] = df["RAM"].apply(parse_ram)
    df["Front_Camera_MP"] = df["Front Camera"].apply(parse_camera)
    df["Back_Camera_MP"] = df["Back Camera"].apply(parse_camera)
    df["Back_Camera_Count"] = df["Back Camera"].apply(parse_camera_count)
    df["Battery_mAh"] = df["Battery Capacity"].apply(parse_battery)
    df["Screen_inches"] = df["Screen Size"].apply(parse_screen)

    price_cols_raw = {
        "Launched Price (Pakistan)": "Price_PKR",
        "Launched Price (India)": "Price_INR",
        "Launched Price (China)": "Price_CNY",
        "Launched Price (USA)": "Price_USD",
        "Launched Price (Dubai)": "Price_AED",
    }
    for raw_col, new_col in price_cols_raw.items():
        df[new_col] = df[raw_col].apply(parse_price)

    df["Year"] = df["Launched Year"]
    df["Company"] = df["Company Name"]

    # Tag tablets
    tablet_keywords = r"(?i)(Tab|Pad|pad|tab)"
    df["Is_Tablet"] = (
        df["Model Name"].str.contains(tablet_keywords, na=False) |
        (df["Weight_g"] > 400)
    )

    # Fix Nokia T21
    nokia_t21_mask = df["Model Name"].str.contains("T21", na=False) & (df["Company"] == "Nokia")
    if nokia_t21_mask.any():
        df.loc[nokia_t21_mask, "Price_USD"] = 162.0

    # Feature engineering
    df["Price_Per_GB_RAM"] = df["Price_USD"] / df["RAM_GB"]
    df["Total_Camera_MP"] = df["Front_Camera_MP"] + df["Back_Camera_MP"]

    # Base feature columns (without brand)
    feature_cols_no_brand = [
        "Weight_g", "RAM_GB", "Front_Camera_MP", "Back_Camera_MP",
        "Back_Camera_Count", "Battery_mAh", "Screen_inches", "Year"
    ]
    target_col = "Price_USD"

    df_all = df.dropna(subset=feature_cols_no_brand + [target_col]).copy()
    df_clean = df_all[~df_all["Is_Tablet"]].copy()
    df_clean = df_clean[df_clean["Price_USD"] <= 3000].copy()

    # Encode company
    le_company = LabelEncoder()
    df_clean["Company_Encoded"] = le_company.fit_transform(df_clean["Company"])

    # ── Features WITH brand (for Customer mode) ──
    features_with_brand = feature_cols_no_brand + ["Company_Encoded"]

    X_brand = df_clean[features_with_brand].values
    y = df_clean[target_col].values

    X_train_b, X_test_b, y_train, y_test = train_test_split(
        X_brand, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train Gradient Boosting (best model) WITH brand for Customer mode
    gb_model_brand = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, random_state=RANDOM_STATE
    )
    gb_model_brand.fit(X_train_b, y_train)
    y_pred_b = gb_model_brand.predict(X_test_b)
    r2_brand = r2_score(y_test, y_pred_b)
    mae_brand = mean_absolute_error(y_test, y_pred_b)

    # ── Features WITHOUT brand (for Company mode) ──
    X_no_brand = df_clean[feature_cols_no_brand].values

    X_train_nb, X_test_nb, y_train_nb, y_test_nb = train_test_split(
        X_no_brand, y, test_size=0.2, random_state=RANDOM_STATE
    )

    # Train Gradient Boosting WITHOUT brand for Company mode
    gb_model_no_brand = GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.05, max_depth=6,
        subsample=0.8, random_state=RANDOM_STATE
    )
    gb_model_no_brand.fit(X_train_nb, y_train_nb)
    y_pred_nb = gb_model_no_brand.predict(X_test_nb)
    r2_no_brand = r2_score(y_test_nb, y_pred_nb)
    mae_no_brand = mean_absolute_error(y_test_nb, y_pred_nb)

    # ── Classification ──
    tier_order = ["Budget", "Mid-Range", "Premium", "Ultra-Premium"]
    df_clean["Price_Tier"] = df_clean["Price_USD"].apply(assign_tier)
    le_tier = LabelEncoder()
    le_tier.fit(tier_order)
    df_clean["Price_Tier_Encoded"] = le_tier.transform(df_clean["Price_Tier"])

    # Classifier WITH brand
    X_cls_b = df_clean[features_with_brand].values
    y_cls = df_clean["Price_Tier_Encoded"].values
    rf_cls_brand = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_cls_brand.fit(X_cls_b, y_cls)

    # Classifier WITHOUT brand
    X_cls_nb = df_clean[feature_cols_no_brand].values
    rf_cls_no_brand = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_cls_no_brand.fit(X_cls_nb, y_cls)

    # Feature importances (with brand)
    importances_brand = pd.Series(
        gb_model_brand.feature_importances_, index=features_with_brand
    ).sort_values(ascending=True)

    # Feature importances (without brand)
    importances_no_brand = pd.Series(
        gb_model_no_brand.feature_importances_, index=feature_cols_no_brand
    ).sort_values(ascending=True)

    # Add predictions to df_clean
    all_preds = gb_model_brand.predict(df_clean[features_with_brand].values)
    df_clean["Predicted_USD"] = all_preds
    df_clean["Price_Diff_Pct"] = ((df_clean["Price_USD"] - df_clean["Predicted_USD"]) / df_clean["Predicted_USD"]) * 100

    return {
        "df_clean": df_clean,
        "gb_model_brand": gb_model_brand,
        "gb_model_no_brand": gb_model_no_brand,
        "rf_cls_brand": rf_cls_brand,
        "rf_cls_no_brand": rf_cls_no_brand,
        "le_company": le_company,
        "le_tier": le_tier,
        "feature_cols_no_brand": feature_cols_no_brand,
        "features_with_brand": features_with_brand,
        "tier_order": tier_order,
        "importances_brand": importances_brand,
        "importances_no_brand": importances_no_brand,
        "r2_brand": r2_brand,
        "mae_brand": mae_brand,
        "r2_no_brand": r2_no_brand,
        "mae_no_brand": mae_no_brand,
    }


# ── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="📱 Smart Pricing — Smartphone Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Load data ────────────────────────────────────────────────────────────────

data = load_data_and_models()
df_clean = data["df_clean"]
le_company = data["le_company"]
le_tier = data["le_tier"]
tier_order = data["tier_order"]

# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("📱 Smart Pricing")
st.sidebar.markdown("---")

mode = st.sidebar.radio(
    "**Select Mode:**",
    ["🛒 Customer — Is My Phone a Good Deal?",
     "💼 Company — What Should I Price This Phone?"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Model: Gradient Boosting**")

if "Customer" in mode:
    st.sidebar.markdown(f"- R² Score: `{data['r2_brand']:.4f}`")
    st.sidebar.markdown(f"- MAE: `${data['mae_brand']:,.0f}`")
    st.sidebar.caption("_(with brand feature)_")
else:
    st.sidebar.markdown(f"- R² Score: `{data['r2_no_brand']:.4f}`")
    st.sidebar.markdown(f"- MAE: `${data['mae_no_brand']:,.0f}`")
    st.sidebar.caption("_(brand-independent)_")

st.sidebar.markdown(f"- Dataset: `{len(df_clean)}` smartphones")
st.sidebar.markdown(f"- Brands: `{df_clean['Company'].nunique()}`")

st.sidebar.markdown("---")
st.sidebar.caption("ITCS 227 — Capstone Project")


# ── Customer Mode ────────────────────────────────────────────────────────────

if "Customer" in mode:
    st.title("🛒 Customer Mode — Is My Phone a Good Deal?")
    st.markdown(
        "Enter the phone's specifications and its **listed price**. "
        "We'll tell you if it's **overpriced**, **fairly priced**, or a **great deal**!"
    )
    st.markdown("---")

    brands = sorted(le_company.classes_.tolist())

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("📌 Brand", brands, index=brands.index("Samsung") if "Samsung" in brands else 0)
        ram = st.select_slider("💾 RAM (GB)", options=[1, 2, 3, 4, 6, 8, 12, 16], value=8)
        battery = st.slider("🔋 Battery (mAh)", 2000, 6500, 5000, step=100)

    with col2:
        back_cam = st.slider("📷 Back Camera (MP)", 2, 200, 50, step=1)
        front_cam = st.slider("🤳 Front Camera (MP)", 1, 60, 16, step=1)
        back_cam_count = st.select_slider("📷 Number of Back Cameras", options=[1, 2, 3, 4], value=2)

    with col3:
        screen = st.slider("📏 Screen Size (inches)", 5.0, 8.0, 6.6, step=0.1)
        weight = st.slider("⚖️ Weight (g)", 135, 295, 192, step=1)
        year = st.slider("📅 Launch Year", 2014, 2025, 2024, step=1)

    company_encoded = le_company.transform([brand])[0]

    features = np.array([[
        weight, ram, front_cam, back_cam,
        back_cam_count, battery, screen, year,
        company_encoded
    ]])

    st.markdown("---")
    actual_price = st.number_input(
        "💲 Listed / Actual Price (USD)",
        min_value=0, max_value=5000, value=499, step=10
    )

    if st.button("🔍 Analyze Price", type="primary", use_container_width=True):
        predicted = data["gb_model_brand"].predict(features)[0]
        diff_pct = ((actual_price - predicted) / predicted) * 100

        tier_encoded = data["rf_cls_brand"].predict(features)[0]
        tier_name = le_tier.inverse_transform([tier_encoded])[0]

        st.markdown("---")

        # Result cards
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("💲 Listed Price", f"${actual_price:,.0f}")
        col_b.metric("🤖 Fair Price (Predicted)", f"${predicted:,.0f}")
        col_c.metric("📊 Difference", f"{diff_pct:+.1f}%",
                      delta=f"${actual_price - predicted:+,.0f}",
                      delta_color="inverse")

        st.markdown("---")

        # Verdict
        if diff_pct > 25:
            st.error(f"🔴 **OVERPRICED** — This phone is **{diff_pct:.1f}% above** its fair market value. "
                     f"You're paying **${actual_price - predicted:,.0f} more** than expected for these specs. "
                     f"Consider waiting for a sale or looking at alternatives.")
        elif diff_pct > 10:
            st.warning(f"🟡 **SLIGHTLY OVERPRICED** — This phone is **{diff_pct:.1f}% above** its fair value. "
                       f"It's not a terrible deal, but you could find better value elsewhere.")
        elif diff_pct > -10:
            st.success(f"🟢 **FAIRLY PRICED** — This phone is within **±10%** of its predicted fair value. "
                       f"The price is reasonable for the specs offered.")
        elif diff_pct > -25:
            st.success(f"🟢 **GOOD DEAL** — This phone is **{abs(diff_pct):.1f}% below** its fair value. "
                      f"You're saving **${predicted - actual_price:,.0f}** compared to similar phones!")
        else:
            st.success(f"🟢 **EXCELLENT DEAL!** — This phone is **{abs(diff_pct):.1f}% below** fair value. "
                      f"You're saving **${predicted - actual_price:,.0f}**. Buy it before it's gone!")

        st.info(f"🏷️ **Predicted Price Tier:** {tier_name}")

        # Gauge chart
        st.markdown("### 📊 Price Position Gauge")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.barh(0, 100, color="#e0e0e0", height=0.5)
        ax.barh(0, 20, left=40, color="#4CAF50", height=0.5, alpha=0.3)
        position = max(0, min(100, 50 + diff_pct))
        marker_color = "#C44E52" if diff_pct > 10 else ("#4CAF50" if diff_pct < -10 else "#FF9800")
        ax.plot(position, 0, marker="v", markersize=20, color=marker_color, zorder=5)
        ax.text(position, -0.4, f"{diff_pct:+.1f}%", ha="center", fontsize=12, fontweight="bold", color=marker_color)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-0.8, 0.5)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xticklabels(["Great Deal\n(-50%)", "Good\n(-25%)", "Fair\n(0%)", "Overpriced\n(+25%)", "Very High\n(+50%)"])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        # Similar phones
        st.markdown("### 📱 Similar Phones in Dataset")
        similar = df_clean[
            (df_clean["RAM_GB"] == ram) &
            (df_clean["Company"] == brand)
        ][["Company", "Model Name", "RAM_GB", "Back_Camera_MP", "Battery_mAh",
           "Price_USD", "Predicted_USD", "Price_Diff_Pct"]].head(10)

        if len(similar) > 0:
            st.dataframe(similar.style.format({
                "Price_USD": "${:,.0f}",
                "Predicted_USD": "${:,.0f}",
                "Price_Diff_Pct": "{:+.1f}%"
            }), use_container_width=True)
        else:
            similar = df_clean[
                (df_clean["RAM_GB"] == ram)
            ][["Company", "Model Name", "RAM_GB", "Back_Camera_MP", "Battery_mAh",
               "Price_USD", "Predicted_USD", "Price_Diff_Pct"]].head(10)
            if len(similar) > 0:
                st.dataframe(similar.style.format({
                    "Price_USD": "${:,.0f}",
                    "Predicted_USD": "${:,.0f}",
                    "Price_Diff_Pct": "{:+.1f}%"
                }), use_container_width=True)
            else:
                st.info("No similar phones found in dataset with matching specs.")


# ── Company Mode (NO BRAND) ─────────────────────────────────────────────────

elif "Company" in mode:
    st.title("💼 Company Mode — What Should I Price This Phone?")
    st.markdown(
        "Enter your phone's specifications. We'll predict the **optimal selling price** "
        "based purely on hardware specs — **no brand bias**."
    )
    st.markdown("> 💡 *Brand is excluded from this prediction so the price reflects pure hardware value.*")
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        ram = st.select_slider("💾 RAM (GB)", options=[1, 2, 3, 4, 6, 8, 12, 16], value=8, key="co_ram")
        battery = st.slider("🔋 Battery (mAh)", 2000, 6500, 5000, step=100, key="co_bat")
        back_cam = st.slider("📷 Back Camera (MP)", 2, 200, 50, step=1, key="co_bcam")

    with col2:
        front_cam = st.slider("🤳 Front Camera (MP)", 1, 60, 16, step=1, key="co_fcam")
        back_cam_count = st.select_slider("📷 Number of Back Cameras", options=[1, 2, 3, 4], value=2, key="co_bcc")
        screen = st.slider("📏 Screen Size (inches)", 5.0, 8.0, 6.6, step=0.1, key="co_scr")

    with col3:
        weight = st.slider("⚖️ Weight (g)", 135, 295, 192, step=1, key="co_wt")
        year = st.slider("📅 Launch Year", 2014, 2025, 2024, step=1, key="co_yr")

    # Features WITHOUT brand
    features_nb = np.array([[
        weight, ram, front_cam, back_cam,
        back_cam_count, battery, screen, year
    ]])

    if st.button("💰 Predict Price", type="primary", use_container_width=True):
        predicted = data["gb_model_no_brand"].predict(features_nb)[0]
        tier_encoded = data["rf_cls_no_brand"].predict(features_nb)[0]
        tier_name = le_tier.inverse_transform([tier_encoded])[0]

        st.markdown("---")

        # Results
        col1, col2, col3 = st.columns(3)
        col1.metric("💰 Recommended Price", f"${predicted:,.0f}")
        col2.metric("🏷️ Price Tier", tier_name)
        col3.metric("📊 Price Range (±10%)", f"${predicted * 0.9:,.0f} – ${predicted * 1.1:,.0f}")

        st.markdown("---")

        # Pricing strategy suggestions
        st.markdown("### 💡 Pricing Strategy Recommendations")

        low_price = predicted * 0.85
        mid_price = predicted
        high_price = predicted * 1.15

        st.markdown(f"""
        | Strategy | Price | Description |
        |----------|-------|-------------|
        | 🟢 **Aggressive (Penetration)** | **${low_price:,.0f}** | Undercut market by 15% to gain market share quickly |
        | 🟡 **Market Rate** | **${mid_price:,.0f}** | Match the predicted fair market value |
        | 🔴 **Premium Positioning** | **${high_price:,.0f}** | Price 15% above market for premium brand perception |
        """)

        st.markdown("---")

        # Feature importance chart (without brand)
        st.markdown("### 📊 What Drives Smartphone Prices? (Spec-Based)")
        importances = data["importances_no_brand"]
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("viridis", len(importances))
        importances.plot(kind="barh", ax=ax, color=colors, edgecolor="white")
        ax.set_title("Feature Importance — Hardware Specs Only", fontweight="bold")
        ax.set_xlabel("Importance Score")
        for i, (val, name) in enumerate(zip(importances.values, importances.index)):
            ax.text(val + 0.003, i, f"{val:.3f}", va="center", fontsize=10)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        st.markdown("---")

        # Same-tier phones for reference
        st.markdown(f"### 📱 Other {tier_name} Phones for Reference")
        tier_phones = df_clean[df_clean["Price_Tier"] == tier_name][
            ["Company", "Model Name", "RAM_GB", "Back_Camera_MP",
             "Battery_mAh", "Screen_inches", "Price_USD"]
        ].sort_values("Price_USD", ascending=False).head(15)

        if len(tier_phones) > 0:
            st.dataframe(tier_phones.style.format({
                "Price_USD": "${:,.0f}",
            }), use_container_width=True)

        # Average specs in this tier
        st.markdown(f"### 📈 Your Phone vs. Average {tier_name} Specs")
        feature_cols_nb = data["feature_cols_no_brand"]
        tier_avg = df_clean[df_clean["Price_Tier"] == tier_name][feature_cols_nb].mean()

        comparison = pd.DataFrame({
            "Your Phone": [weight, ram, front_cam, back_cam, back_cam_count, battery, screen, year],
            f"Avg {tier_name}": tier_avg.values.round(1),
        }, index=feature_cols_nb)

        st.dataframe(comparison, use_container_width=True)

        # Market landscape
        st.markdown("### 🌐 Market Landscape — Where Does Your Phone Fit?")
        fig, ax = plt.subplots(figsize=(12, 6))
        for tier in tier_order:
            mask = df_clean["Price_Tier"] == tier
            ax.scatter(df_clean.loc[mask, "RAM_GB"], df_clean.loc[mask, "Price_USD"],
                       alpha=0.4, label=tier, s=40, edgecolors="white", linewidth=0.5)
        # Mark the predicted phone
        ax.scatter(ram, predicted, c="red", s=200, marker="*", zorder=5,
                   label=f"Your Phone (${predicted:,.0f})", edgecolors="black", linewidth=1.5)
        ax.set_xlabel("RAM (GB)")
        ax.set_ylabel("Price (USD)")
        ax.set_title("Market Landscape — RAM vs Price", fontweight="bold")
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
