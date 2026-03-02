"""
Zomathon PS1 — KPT Signal Enhancement: Synthetic Dataset Generator
====================================================================
Simulates Zomato-like order data incorporating three novel signals:

  1. BLE/Wi-Fi Crowd Sniffing  — proxy for total kitchen load (non-Zomato)
  2. Merchant Reliability Index (MRI) — dynamic trust score per merchant
  3. Scan-to-Dispatch Protocol — QR-scan as zero-bias ground truth

Dataset: ~230K orders | 80 merchants | 6 cities | Oct–Nov 2024
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

np.random.seed(42)

CITIES = {
    "Mumbai":    {"weight": 0.25, "base_kpt": 420},
    "Delhi":     {"weight": 0.22, "base_kpt": 400},
    "Bangalore": {"weight": 0.18, "base_kpt": 380},
    "Hyderabad": {"weight": 0.15, "base_kpt": 370},
    "Pune":      {"weight": 0.10, "base_kpt": 350},
    "Chennai":   {"weight": 0.10, "base_kpt": 360},
}
REST_TYPES = ["QSR", "Casual_Dining", "Cloud_Kitchen", "Fine_Dining"]
REST_ORDER_RATE = {"QSR": 80, "Casual_Dining": 40, "Cloud_Kitchen": 60, "Fine_Dining": 20}
N_MERCHANTS = 80
N_DAYS = 60
START_DATE = datetime(2024, 10, 1)


def rush_factor(hour, dow):
    is_wknd = dow >= 5
    if 12 <= hour <= 14: return 1.8 if is_wknd else 1.5
    if 19 <= hour <= 22: return 2.2 if is_wknd else 1.9
    if hour >= 23:       return 1.3
    if 8 <= hour <= 10:  return 1.2
    return 1.0


def generate_merchants():
    rows = []
    for i in range(N_MERCHANTS):
        city = np.random.choice(list(CITIES), p=[v["weight"] for v in CITIES.values()])
        # Bimodal MRI: 60% reliable, 40% unreliable
        mri = np.random.uniform(0.75, 0.95) if np.random.rand() < 0.6 else np.random.uniform(0.30, 0.65)
        rows.append({
            "merchant_id":      f"MX{i:04d}",
            "city":             city,
            "restaurant_type":  np.random.choice(REST_TYPES),
            "n_kitchen_staff":  np.random.randint(2, 12),
            "n_ovens":          np.random.randint(1, 6),
            "base_kpt_sec":     int(np.random.normal(CITIES[city]["base_kpt"], 50)),
            "true_mri":         round(mri, 4),
            "has_scan_dispatch": int(np.random.rand() < (0.8 if mri < 0.65 else 0.45)),
        })
    return pd.DataFrame(rows)


def simulate_all():
    merchants = generate_merchants()
    merchants.to_csv("/home/claude/merchants.csv", index=False)
    print(f"Merchants generated: {len(merchants)}")

    records = []
    hour_range = list(range(9, 23))
    hour_w = np.array([0.02, 0.03, 0.08, 0.12, 0.06, 0.04,
                       0.05, 0.10, 0.14, 0.07, 0.09, 0.08, 0.07, 0.05])
    hour_w /= hour_w.sum()

    for idx, m in merchants.iterrows():
        recent_true = []
        recent_for  = []

        for d in range(N_DAYS):
            date = START_DATE + timedelta(days=d)
            dow  = date.weekday()
            for o in range(np.random.poisson(REST_ORDER_RATE[m["restaurant_type"]])):
                hour = np.random.choice(hour_range, p=hour_w)
                rf   = rush_factor(hour, dow)

                # True KPT (physics of kitchen)
                queue_len = max(0, int(np.random.poisson(rf * 2.5)))
                true_kpt  = float(np.clip(m["base_kpt_sec"] * rf + queue_len * 12
                                          + np.random.normal(0, 35), 120, 2400))

                # BLE/Wi-Fi crowd signal
                base_dev  = {"QSR": 18, "Casual_Dining": 28, "Cloud_Kitchen": 4, "Fine_Dining": 20}[m["restaurant_type"]]
                ble       = max(0, int(np.random.poisson(base_dev * rf) + np.random.poisson(3)))
                ble_idx   = round(ble / 20.0, 3)

                # Rolling avg KPT
                avg_60m = np.mean(recent_true[-8:]) if len(recent_true) >= 3 else m["base_kpt_sec"]
                avg_7d  = m["base_kpt_sec"] * np.random.normal(1.0, 0.04)

                # System telemetry (BLE-corrected rolling avg)
                sys_tel = max(120.0, avg_60m * (1 + 0.18 * (ble_idx - 1.0)))

                # Merchant FOR signal
                rider_gap = np.random.uniform(20, 180)
                if m["has_scan_dispatch"]:
                    merchant_for = round(true_kpt + np.random.normal(0, 8), 1)
                    for_bias     = round(true_kpt - merchant_for, 1)
                else:
                    early = (1 - m["true_mri"]) * rider_gap * 0.75
                    noise = np.random.normal(0, 20 * (1 - m["true_mri"]))
                    merchant_for = round(true_kpt - early + noise, 1)
                    for_bias     = round(true_kpt - merchant_for, 1)

                # Observed MRI (system's running estimate)
                if len(recent_for) >= 20:
                    errs = [abs(f - t) for f, t in zip(recent_for[-10:], recent_true[-10:])]
                    observed_mri = round(max(0.05, 1 - np.mean(errs) / m["base_kpt_sec"]), 4)
                else:
                    observed_mri = 0.70

                # MRI De-biased KPT: alpha*FOR + (1-alpha)*system_telemetry
                alpha       = observed_mri
                debiased    = round(alpha * merchant_for + (1 - alpha) * sys_tel, 1)

                records.append({
                    "order_id":                   f"{m['merchant_id']}_D{d:03d}_O{o:04d}",
                    "merchant_id":                m["merchant_id"],
                    "city":                       m["city"],
                    "restaurant_type":            m["restaurant_type"],
                    "order_date":                 date.strftime("%Y-%m-%d"),
                    "hour":                       hour,
                    "day_of_week":                dow,
                    "is_weekend":                 int(dow >= 5),
                    "is_lunch_rush":              int(12 <= hour <= 14),
                    "is_dinner_rush":             int(19 <= hour <= 22),
                    "n_kitchen_staff":            m["n_kitchen_staff"],
                    "n_ovens":                    m["n_ovens"],
                    "queue_length":               queue_len,
                    "rush_factor":                round(rf, 2),
                    "avg_kpt_last_60min_sec":     round(avg_60m, 1),
                    "avg_kpt_last_7days_sec":     round(avg_7d, 1),
                    # NEW SIGNAL 1
                    "ble_wifi_device_count":      ble,
                    "ble_load_index":             ble_idx,
                    # NEW SIGNAL 2
                    "merchant_mri_observed":      observed_mri,
                    "merchant_mri_true":          m["true_mri"],
                    "merchant_reliability_band":  ("high" if m["true_mri"] >= 0.75 else
                                                   "medium" if m["true_mri"] >= 0.55 else "low"),
                    "merchant_for_kpt_sec":       merchant_for,
                    "for_bias_seconds":           for_bias,
                    "rider_eta_gap_sec":          round(rider_gap, 1),
                    # NEW SIGNAL 3
                    "has_scan_dispatch":          m["has_scan_dispatch"],
                    "scan_dispatch_kpt_sec":      (merchant_for if m["has_scan_dispatch"] else None),
                    # Derived
                    "system_telemetry_kpt_sec":   round(sys_tel, 1),
                    "mri_alpha":                  round(alpha, 4),
                    "debiased_kpt_sec":           debiased,
                    # Ground truth
                    "true_kpt_seconds":           round(true_kpt, 1),
                    "true_kpt_minutes":           round(true_kpt / 60, 2),
                })

                recent_true.append(true_kpt)
                recent_for.append(merchant_for)

    return pd.DataFrame(records), merchants


def print_summary(df):
    print(f"\n{'='*58}")
    print("  ZOMATHON PS1 — DATASET SUMMARY")
    print(f"{'='*58}")
    print(f"  Total orders:            {len(df):,}")
    print(f"  Date range:              Oct 1 – Nov 29, 2024")
    print(f"  Cities:                  {df['city'].nunique()}")
    print(f"  Merchants:               {df['merchant_id'].nunique()}")
    print(f"  Scan-Dispatch merchants: {df.drop_duplicates('merchant_id')['has_scan_dispatch'].mean()*100:.0f}%")
    print(f"\n  SIGNAL ERROR vs GROUND TRUTH (all orders)")
    print(f"  {'Signal':<38} {'MAE':>7} {'P90':>7}")
    print(f"  {'-'*54}")
    for label, col in [
        ("Raw Merchant FOR (current baseline)", "merchant_for_kpt_sec"),
        ("System Telemetry only (BLE-corrected)", "system_telemetry_kpt_sec"),
        ("MRI De-biased KPT (proposed)", "debiased_kpt_sec"),
    ]:
        err = (df[col] - df["true_kpt_seconds"]).abs()
        print(f"  {label:<38} {err.mean():>6.1f}s {err.quantile(0.9):>6.1f}s")

    print(f"\n  IMPROVEMENT — Dinner Rush + Low-MRI Merchants:")
    mask = (df["is_dinner_rush"] == 1) & (df["merchant_reliability_band"] == "low")
    sub  = df[mask]
    raw  = (sub["merchant_for_kpt_sec"] - sub["true_kpt_seconds"]).abs().mean()
    deb  = (sub["debiased_kpt_sec"] - sub["true_kpt_seconds"]).abs().mean()
    print(f"  Raw FOR MAE: {raw:.1f}s  →  De-biased MAE: {deb:.1f}s  ({(raw-deb)/raw*100:+.1f}%)")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    print("Generating Zomato KPT synthetic dataset...\n")
    df, merchants = simulate_all()
    print(f"Orders generated: {len(df):,}")

    df.to_csv("/home/claude/kpt_orders_full.csv", index=False)
    print("Saved: kpt_orders_full.csv")

    # Model-ready (no hidden eval columns)
    train_cols = [c for c in df.columns if c != "merchant_mri_true"]
    df[train_cols].to_csv("/home/claude/kpt_orders_model_ready.csv", index=False)
    print("Saved: kpt_orders_model_ready.csv")

    # MRI leaderboard
    mri = df.groupby("merchant_id").agg(
        city=("city","first"), restaurant_type=("restaurant_type","first"),
        true_mri=("merchant_mri_true","first"),
        observed_mri=("merchant_mri_observed","last"),
        n_orders=("order_id","count"),
        avg_for_bias_sec=("for_bias_seconds","mean"),
        has_scan_dispatch=("has_scan_dispatch","first"),
    ).reset_index()
    mri.to_csv("/home/claude/merchant_mri_leaderboard.csv", index=False)
    print("Saved: merchant_mri_leaderboard.csv")

    print_summary(df)
    print("All files saved. Upload CSVs to GitHub and link in your submission PDF.")
