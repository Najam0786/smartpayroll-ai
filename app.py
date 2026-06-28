"""
SmartPayroll AI — Streamlit Dashboard
Run: streamlit run app.py
"""

import os
import sys
import json
import joblib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartPayroll AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"], .stMarkdown, .stText {
    font-family: 'Inter', -apple-system, sans-serif !important;
}

/* Remove default top padding */
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; }
[data-testid="stAppViewContainer"] { background: #F4F7FB; }
[data-testid="stHeader"] { background: transparent; }

/* Header gradient card */
.hero {
    background: linear-gradient(135deg, #0A1628 0%, #1A3A5C 40%, #0F5F9E 100%);
    border-radius: 20px;
    padding: 2.8rem 3.5rem;
    color: white;
    margin-bottom: 1.8rem;
    box-shadow: 0 8px 32px rgba(10, 22, 40, 0.35);
    position: relative;
    overflow: hidden;
}
.hero::after {
    content: "";
    position: absolute;
    top: -60px; right: -60px;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
}
.hero h1 {
    font-size: 2.4rem; font-weight: 800;
    margin: 0; letter-spacing: -0.03em;
}
.hero .sub {
    font-size: 1.05rem; opacity: 0.78;
    margin: 0.4rem 0 1.2rem; font-weight: 300;
}
.pill {
    display: inline-block;
    background: rgba(255,255,255,0.12);
    border: 1px solid rgba(255,255,255,0.25);
    border-radius: 100px;
    padding: 4px 14px;
    font-size: 0.75rem;
    font-weight: 500;
    margin: 3px 4px 3px 0;
    letter-spacing: 0.02em;
}

/* KPI card */
.kpi {
    background: white;
    border-radius: 16px;
    padding: 1.4rem 1.6rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-top: 4px solid #0066CC;
    height: 100%;
    transition: transform 0.15s, box-shadow 0.15s;
}
.kpi:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }
.kpi-val  { font-size: 2.1rem; font-weight: 800; color: #0A1628; line-height: 1.1; }
.kpi-lbl  { font-size: 0.72rem; font-weight: 600; color: #7A8699; text-transform: uppercase; letter-spacing: 0.07em; margin-bottom: 0.3rem; }
.kpi-note { font-size: 0.78rem; color: #9CA8B8; margin-top: 0.4rem; }

/* Section divider */
.sec-hdr {
    font-size: 1.1rem; font-weight: 700; color: #0A1628;
    border-left: 4px solid #0066CC;
    padding-left: 0.8rem;
    margin: 1.8rem 0 1rem;
}

/* Profile + risk cards */
.card {
    background: white; border-radius: 16px;
    padding: 1.8rem; box-shadow: 0 2px 16px rgba(0,0,0,0.07);
}
.emp-id  { font-size: 1.5rem; font-weight: 800; color: #0A1628; }
.emp-sub { font-size: 0.9rem; color: #7A8699; margin-top: 0.2rem; }

.stat {
    background: #F4F7FB; border-radius: 10px;
    padding: 0.75rem 1rem; margin: 0.45rem 0;
    border-left: 3px solid #0066CC;
}
.stat-lbl { font-size: 0.68rem; color: #9CA8B8; text-transform: uppercase; letter-spacing: 0.07em; }
.stat-val { font-size: 1.05rem; font-weight: 600; color: #0A1628; margin-top: 0.1rem; }

/* Risk badges */
.badge { display: inline-block; border-radius: 100px; padding: 6px 18px; font-weight: 700; font-size: 0.88rem; letter-spacing: 0.04em; }
.badge-H { background: #FF3B3B; color: white; }
.badge-M { background: #FF9500; color: white; }
.badge-L { background: #30D158; color: white; }

/* Risk factor items */
.rfactor {
    background: #FFF3F3; border-radius: 8px;
    padding: 0.65rem 1rem; margin: 0.4rem 0;
    border-left: 3px solid #FF3B3B;
    font-size: 0.88rem; color: #1A1A2E;
}

/* Info banner */
.info-banner {
    border-radius: 12px; padding: 1rem 1.4rem; margin-bottom: 1.4rem;
    font-size: 0.9rem; border: 1px solid;
}
.info-blue   { background: #EFF6FF; border-color: #BFDBFE; color: #1E40AF; }
.info-amber  { background: #FFFBEB; border-color: #FDE68A; color: #92400E; }
.info-green  { background: #F0FDF4; border-color: #BBF7D0; color: #14532D; }

/* RAG answer bubble */
.chat-bubble {
    background: white; border-radius: 14px;
    padding: 1.5rem; margin: 0.8rem 0;
    box-shadow: 0 2px 16px rgba(0,0,0,0.07);
    border-left: 5px solid #0066CC;
    font-size: 0.94rem; line-height: 1.7;
    color: #1A1A2E;
}
.src-chip {
    display: inline-block; background: #EFF6FF;
    border: 1px solid #BFDBFE; color: #1D4ED8;
    border-radius: 100px; padding: 3px 11px;
    font-size: 0.73rem; font-weight: 500; margin: 6px 4px 0 0;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] { gap: 6px; }
.stTabs [data-baseweb="tab"] {
    background: white; border-radius: 10px;
    padding: 10px 22px;
    border: 1px solid #E2E8F0;
    font-weight: 500; font-size: 0.9rem;
    color: #4A5568;
}
.stTabs [aria-selected="true"] {
    background: #0A1628 !important;
    color: white !important;
    border-color: #0A1628 !important;
}

/* Streamlit element overrides */
div[data-testid="stNumberInput"] label { font-weight: 600; color: #0A1628; }
div[data-testid="stSelectbox"] label   { font-weight: 600; color: #0A1628; }
div[data-testid="stTextArea"] label    { font-weight: 600; color: #0A1628; }

.stButton > button[kind="primary"] {
    background: #0A1628 !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    letter-spacing: 0.02em !important;
    padding: 0.55rem 1.4rem !important;
}
.stButton > button[kind="primary"]:hover {
    background: #0F2847 !important;
    box-shadow: 0 4px 12px rgba(10,22,40,0.3) !important;
}

/* Footer */
.footer {
    text-align: center; color: #9CA8B8;
    font-size: 0.78rem; margin-top: 3rem;
    padding: 1.5rem; border-top: 1px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ──────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_hr():
    return pd.read_parquet(ROOT / "data" / "processed" / "hr_silver.parquet")


@st.cache_data(show_spinner=False)
def load_payroll():
    return pd.read_parquet(ROOT / "data" / "synthetic" / "payroll.parquet")


@st.cache_resource(show_spinner=False)
def load_iso_model():
    base = ROOT / "data" / "processed"
    model    = joblib.load(base / "isolation_forest.joblib")
    scaler   = joblib.load(base / "isolation_forest_scaler.joblib")
    features = joblib.load(base / "isolation_forest_features.joblib")
    return model, scaler, features


# ── Business logic (inline — no import chain risk) ──────────────────────────

TAX_BANDS = {
    "ES": (0.10, 0.47), "BE": (0.25, 0.50),
    "DE": (0.14, 0.45), "NL": (0.19, 0.495), "FR": (0.11, 0.45),
}

def rule_flags(row: pd.Series) -> list[tuple[str, str]]:
    flags, gross, net = [], float(row.get("gross_pay", 0)), float(row.get("net_pay", 0))
    tax, pension, country = float(row.get("income_tax", 0)), float(row.get("pension", 0)), str(row.get("country", "ES"))
    if net <= 0:
        flags.append(("ZERO_OR_NEGATIVE_NET_PAY", "CRITICAL"))
    if gross > 0 and net > gross:
        flags.append(("NET_EXCEEDS_GROSS", "CRITICAL"))
    if pension == 0 and gross > 1500:
        flags.append(("MISSING_PENSION", "CRITICAL"))
    if gross > 0 and tax > 0:
        rate, band = tax / gross, TAX_BANDS.get(country, (0.10, 0.55))
        if rate > band[1]:
            flags.append(("TAX_RATE_TOO_HIGH", "CRITICAL"))
        elif rate < band[0]:
            flags.append(("TAX_RATE_TOO_LOW", "WARNING"))
    if gross > 0 and net > 0 and (1 - net / gross) > 0.80:
        flags.append(("EXCESSIVE_DEDUCTIONS", "WARNING"))
    return flags


@st.cache_data(show_spinner=False)
def detect_period(period: str) -> pd.DataFrame:
    """Run two-layer anomaly detection on one pay period (cached per period)."""
    payroll = load_payroll()
    df = payroll[payroll["pay_period"] == period].copy().reset_index(drop=True)

    # Layer 1: Rules
    all_flags = df.apply(rule_flags, axis=1)
    df["rule_flags"]  = all_flags.apply(lambda f: " | ".join(x[0] for x in f) if f else "")
    df["rule_sev"]    = all_flags.apply(
        lambda f: "CRITICAL" if any(x[1] == "CRITICAL" for x in f)
        else ("WARNING" if f else "CLEAN")
    )

    # Layer 2: Isolation Forest
    model, scaler, feat_cols = load_iso_model()
    df2 = df.copy()
    df2["net_to_gross"]    = df2["net_pay"] / (df2["gross_pay"] + 1e-6)
    df2["tax_rate"]        = df2["income_tax"] / (df2["gross_pay"] + 1e-6)
    df2["deduction_rate"]  = (df2["income_tax"] + df2["social_security"] + df2["pension"]) / (df2["gross_pay"] + 1e-6)
    avail = [c for c in feat_cols if c in df2.columns]
    X_scaled = scaler.transform(df2[avail].fillna(0).values)
    df["iso_score"] = model.decision_function(X_scaled).round(4)
    df["iso_flag"]  = model.predict(X_scaled) == -1

    # Combined severity
    def combined_sev(row):
        if row["rule_sev"] == "CRITICAL":
            return "CRITICAL"
        if row["rule_sev"] == "WARNING" or row["iso_flag"]:
            return "WARNING"
        return "CLEAN"

    df["detected_severity"] = df.apply(combined_sev, axis=1)
    df["detected_by"]       = df.apply(
        lambda r: "RULE+MODEL" if r["rule_sev"] != "CLEAN" and r["iso_flag"]
        else ("RULE" if r["rule_sev"] != "CLEAN" else ("MODEL" if r["iso_flag"] else "—")),
        axis=1,
    )
    return df


def attrition_risk(row: pd.Series) -> dict:
    score, factors = 0, []
    if row.get("OverTime", 0) == 1:
        factors.append("Working overtime — 2.9× higher attrition risk")
        score += 3
    if row.get("JobSatisfaction", 4) <= 2:
        factors.append(f"Low job satisfaction: {int(row['JobSatisfaction'])}/4")
        score += 2
    if row.get("YearsAtCompany", 10) < 3:
        factors.append(f"Short tenure: {int(row['YearsAtCompany'])} years")
        score += 2
    if row.get("DistanceFromHome", 0) > 20:
        factors.append(f"Long commute: {int(row['DistanceFromHome'])} km")
        score += 1
    if row.get("MonthlyIncome", 9999) < 3000:
        factors.append(f"Below median salary: ${row['MonthlyIncome']:,.0f}")
        score += 2
    level = "HIGH" if score >= 5 else "MEDIUM" if score >= 3 else "LOW"
    recs  = {
        "HIGH":   "Immediate retention conversation recommended",
        "MEDIUM": "Monitor closely — discuss in next 1:1",
        "LOW":    "No immediate action required",
    }
    return {"level": level, "score": score, "factors": factors, "rec": recs[level]}


CHART_FONT   = dict(family="Inter, sans-serif", size=12)
CHART_LAYOUT = dict(
    plot_bgcolor="white", paper_bgcolor="white",
    font=CHART_FONT, margin=dict(t=52, b=36, l=40, r=20),
    title_font=dict(size=14, color="#0A1628"),
)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <h1>SmartPayroll AI</h1>
  <div class="sub">Enterprise HR Intelligence &amp; Payroll Analytics Platform</div>
  <div>
    <span class="pill">XGBoost + SMOTE</span>
    <span class="pill">Isolation Forest</span>
    <span class="pill">RAG · Phi-4-mini</span>
    <span class="pill">LangGraph Agents</span>
    <span class="pill">AUC 0.772</span>
    <span class="pill">Recall 94.4 %</span>
    <span class="pill">1,470 Employees</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "  Executive Dashboard  ",
    "  Employee Risk Profiler  ",
    "  Payroll Anomaly Scanner  ",
    "  HR Policy Q&A  ",
])


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — EXECUTIVE DASHBOARD
# ───────────────────────────────────────────────────────────────────────────────
with tab1:
    hr      = load_hr()
    payroll = load_payroll()

    attr_rate  = hr["Attrition"].mean()
    avg_income = hr["MonthlyIncome"].mean()
    n_anomaly  = int(payroll["is_anomaly"].sum())
    n_records  = len(payroll)

    # ── KPI row ──────────────────────────────────────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    kpis = [
        ("1,470",         "Total Employees",     "#0066CC", "IBM HR Analytics dataset"),
        (f"{attr_rate:.1%}", "Attrition Rate",   "#FF3B3B", "Actual leavers in dataset"),
        (f"${avg_income:,.0f}", "Avg Monthly Salary", "#0066CC", "Across all departments"),
        (f"{n_anomaly:,}", "Payroll Anomalies",  "#FF9500", f"of {n_records:,} records ({n_anomaly/n_records:.1%})"),
        ("0.772",          "XGBoost AUC",        "#30D158", "vs 0.612 LR baseline"),
    ]
    for col, (val, lbl, color, note) in zip([k1, k2, k3, k4, k5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-top-color:{color}">
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-val" style="color:{color}">{val}</div>
              <div class="kpi-note">{note}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Row A: Attrition by Department + Salary Distribution ─────────────────
    ca1, ca2 = st.columns(2)

    with ca1:
        dept_stats = (
            hr.groupby("Department")["Attrition"]
            .agg(rate="mean", left="sum", total="count")
            .reset_index()
        )
        dept_stats["pct"] = (dept_stats["rate"] * 100).round(1)
        clrs = ["#FF3B3B", "#FF9500", "#0066CC"]

        fig = go.Figure()
        for i, r in dept_stats.iterrows():
            fig.add_trace(go.Bar(
                x=[r["Department"]], y=[r["pct"]],
                marker_color=clrs[i % 3],
                text=[f"{r['pct']}%"], textposition="outside",
                showlegend=False,
                customdata=[[r["left"], r["total"]]],
                hovertemplate="<b>%{x}</b><br>Attrition: %{y}%<br>Left: %{customdata[0]} of %{customdata[1]}<extra></extra>",
            ))
        fig.update_layout(
            title="Attrition Rate by Department",
            yaxis=dict(title="Attrition Rate (%)", gridcolor="#F1F5F9"),
            **CHART_LAYOUT, height=360,
        )
        st.plotly_chart(fig, use_container_width=True)

    with ca2:
        fig2 = px.box(
            hr, x="Department", y="MonthlyIncome", color="Department",
            color_discrete_sequence=["#FF3B3B", "#0066CC", "#30D158"],
            title="Monthly Salary Distribution by Department",
            labels={"MonthlyIncome": "Monthly Salary ($)", "Department": ""},
            points="outliers",
        )
        fig2.update_layout(**CHART_LAYOUT, height=360, showlegend=False,
                           yaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig2, use_container_width=True)

    # ── Row B: Overtime Impact + Satisfaction Heatmap ────────────────────────
    cb1, cb2 = st.columns(2)

    with cb1:
        ot = hr.groupby("OverTime")["Attrition"].mean().reset_index()
        ot["label"] = ot["OverTime"].map({0: "No Overtime", 1: "Overtime"})
        ot["pct"]   = (ot["Attrition"] * 100).round(1)

        fig3 = go.Figure(go.Pie(
            labels=ot["label"], values=ot["pct"],
            hole=0.58,
            marker_colors=["#30D158", "#FF3B3B"],
            textinfo="label+percent", textfont_size=12,
            hovertemplate="<b>%{label}</b><br>Attrition: %{value}%<extra></extra>",
        ))
        fig3.update_layout(
            title="Attrition Rate: Overtime vs No Overtime",
            **CHART_LAYOUT, height=360,
            annotations=[dict(text="Overtime<br>Impact", x=0.5, y=0.5,
                               font=dict(size=12, color="#0A1628"), showarrow=False)],
        )
        st.plotly_chart(fig3, use_container_width=True)

    with cb2:
        pivot = (
            hr.groupby(["JobSatisfaction", "WorkLifeBalance"])["Attrition"]
            .mean().unstack(fill_value=0) * 100
        ).round(1)

        fig4 = go.Figure(go.Heatmap(
            z=pivot.values,
            x=[f"WLB {c}" for c in pivot.columns],
            y=[f"Satisfaction {r}" for r in pivot.index],
            colorscale=[[0, "#30D158"], [0.4, "#FF9500"], [1, "#FF3B3B"]],
            text=[[f"{v:.0f}%" for v in row] for row in pivot.values],
            texttemplate="%{text}", textfont={"size": 12},
            colorbar=dict(title="Attrition %", ticksuffix="%", len=0.85),
            hovertemplate="<b>%{y} × %{x}</b><br>Attrition: %{z}%<extra></extra>",
        ))
        fig4.update_layout(
            title="Attrition: Job Satisfaction × Work-Life Balance",
            **CHART_LAYOUT, height=360,
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── Row C: Payroll anomaly charts ─────────────────────────────────────────
    cc1, cc2 = st.columns(2)

    with cc1:
        anom_types = (
            payroll[payroll["is_anomaly"]]["anomaly_type"]
            .value_counts().reset_index()
        )
        anom_types.columns = ["Type", "Count"]
        fig5 = px.bar(
            anom_types, x="Count", y="Type", orientation="h",
            color="Count", color_continuous_scale=["#FF9500", "#FF3B3B"],
            title="Injected Payroll Anomaly Types (6-Month Dataset)",
            labels={"Count": "Records", "Type": ""},
        )
        fig5.update_layout(**CHART_LAYOUT, height=320,
                           coloraxis_showscale=False,
                           xaxis=dict(gridcolor="#F1F5F9"))
        st.plotly_chart(fig5, use_container_width=True)

    with cc2:
        dept_pay = (
            payroll.groupby("department")
            .agg(clean=("is_anomaly", lambda x: (~x).sum()),
                 anomalies=("is_anomaly", "sum"))
            .reset_index()
        )
        fig6 = go.Figure()
        fig6.add_trace(go.Bar(name="Clean",     x=dept_pay["department"], y=dept_pay["clean"],     marker_color="#30D158"))
        fig6.add_trace(go.Bar(name="Anomalies", x=dept_pay["department"], y=dept_pay["anomalies"], marker_color="#FF3B3B"))
        fig6.update_layout(
            barmode="stack",
            title="Payroll Records: Clean vs Anomalous by Department",
            yaxis=dict(title="Records", gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=-0.25),
            **CHART_LAYOUT, height=320,
        )
        st.plotly_chart(fig6, use_container_width=True)

    # ── Model performance summary ─────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>Model Performance Summary</div>", unsafe_allow_html=True)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    model_kpis = [
        ("0.772", "AUC-ROC",      "#0066CC", "XGBoost + SMOTE"),
        ("0.444", "F1 Score",     "#0066CC", "Minority class"),
        ("94.4%", "Recall",       "#30D158", "Catches most leavers"),
        ("30.3%", "Precision",    "#FF9500",  "Trade-off for recall"),
        ("84.0%", "Accuracy",     "#0066CC", "Overall"),
    ]
    for col, (val, lbl, color, note) in zip([mc1, mc2, mc3, mc4, mc5], model_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-top-color:{color}; padding: 1rem 1.2rem;">
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-val" style="color:{color}; font-size:1.6rem">{val}</div>
              <div class="kpi-note">{note}</div>
            </div>""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — EMPLOYEE RISK PROFILER
# ───────────────────────────────────────────────────────────────────────────────
with tab2:
    hr = load_hr()

    st.markdown("""
    <div class="info-banner info-blue">
      <b>Employee Risk Profiler</b> — Enter an employee ID (1–1,470) to view their full
      profile, attrition risk score, and the key risk drivers identified by the production
      rule engine. Compare the employee against their department benchmarks.
    </div>""", unsafe_allow_html=True)

    ci1, ci2 = st.columns([3, 1])
    with ci1:
        emp_id = st.number_input(
            "Employee ID", min_value=1, max_value=len(hr), value=5, step=1,
            help="Enter any integer from 1 to 1,470",
        )
    with ci2:
        st.markdown("<br>", unsafe_allow_html=True)
        go_btn = st.button("Analyze", type="primary", use_container_width=True)

    row  = hr.iloc[int(emp_id) - 1]
    risk = attrition_risk(row)
    dept = str(row["Department"])
    dept_df = hr[hr["Department"] == dept]

    color_map = {"HIGH": "#FF3B3B", "MEDIUM": "#FF9500", "LOW": "#30D158"}
    badge_map = {"HIGH": "badge-H", "MEDIUM": "badge-M", "LOW": "badge-L"}
    risk_color = color_map[risk["level"]]
    risk_badge = badge_map[risk["level"]]

    cp1, cp2 = st.columns([1, 1])

    with cp1:
        st.markdown(f"""
        <div class="card">
          <div class="emp-id">Employee #{int(emp_id):,}</div>
          <div class="emp-sub">{row['JobRole']} &nbsp;·&nbsp; {dept}</div>
          <br>
          <div class="stat">
            <div class="stat-lbl">Monthly Salary</div>
            <div class="stat-val">${row['MonthlyIncome']:,.0f}</div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Age / Tenure</div>
            <div class="stat-val">{int(row['Age'])} yrs old &nbsp;·&nbsp; {int(row['YearsAtCompany'])} yrs at company</div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Total Career Experience</div>
            <div class="stat-val">{int(row['TotalWorkingYears'])} years total</div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Distance from Home</div>
            <div class="stat-val">{int(row['DistanceFromHome'])} km</div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Overtime</div>
            <div class="stat-val" style="color:{'#FF3B3B' if row['OverTime']==1 else '#30D158'}">
              {"Yes — actively working OT" if row['OverTime']==1 else "No"}
            </div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Composite Satisfaction Score</div>
            <div class="stat-val">{row['SatisfactionScore']:.2f} / 4.00</div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Job Satisfaction</div>
            <div class="stat-val">{int(row['JobSatisfaction'])} / 4</div>
          </div>
          <div class="stat">
            <div class="stat-lbl">Employee Status</div>
            <div class="stat-val" style="color:{'#FF3B3B' if row['Attrition']==1 else '#30D158'}">
              {"Left Company" if row['Attrition']==1 else "Active Employee"}
            </div>
          </div>
        </div>""", unsafe_allow_html=True)

    with cp2:
        st.markdown(f"""
        <div class="card" style="border-top: 4px solid {risk_color}">
          <div class="stat-lbl">Attrition Risk Assessment</div>
          <div style="margin: 1rem 0 0.4rem;">
            <span class="badge {risk_badge}">{risk['level']} RISK</span>
            <span style="color:#9CA8B8; margin-left:0.8rem; font-size:0.85rem;">
              Score: {risk['score']} / 10
            </span>
          </div>
        """, unsafe_allow_html=True)

        # Score gauge bar
        pct = min(risk["score"] / 10 * 100, 100)
        st.markdown(f"""
          <div style="background:#F1F5F9; border-radius:100px; height:8px; margin:0.8rem 0 1.2rem; overflow:hidden;">
            <div style="width:{pct}%; height:100%; background:{risk_color};
                        border-radius:100px; transition:width 0.5s;"></div>
          </div>
          <div style="background:#F4F7FB; border-radius:10px; padding:1rem; margin-bottom:1rem;">
            <div class="stat-lbl">Recommendation</div>
            <div style="font-size:0.92rem; font-weight:600; color:{risk_color}; margin-top:0.3rem;">
              {risk['rec']}
            </div>
          </div>
        """, unsafe_allow_html=True)

        if risk["factors"]:
            st.markdown("<div class='stat-lbl' style='margin-bottom:0.5rem;'>Risk Factors Detected</div>", unsafe_allow_html=True)
            for f in risk["factors"]:
                st.markdown(f'<div class="rfactor">⚠&nbsp; {f}</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-banner info-green" style="margin:0">
              No significant risk factors detected for this employee.
            </div>""", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    # ── Department benchmark chart ────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>Department Benchmarks — " + dept + "</div>", unsafe_allow_html=True)

    bc1, bc2 = st.columns(2)

    with bc1:
        fig_sal = go.Figure()
        fig_sal.add_trace(go.Histogram(
            x=dept_df["MonthlyIncome"], name=dept,
            marker_color="#0066CC", opacity=0.7,
            nbinsx=30, bingroup=1,
        ))
        fig_sal.add_vline(
            x=float(row["MonthlyIncome"]),
            line_color="#FF3B3B", line_width=2.5, line_dash="dash",
            annotation_text=f"Emp #{int(emp_id)} (${row['MonthlyIncome']:,.0f})",
            annotation_font_color="#FF3B3B",
            annotation_position="top right",
        )
        fig_sal.update_layout(
            title=f"Salary Distribution — {dept}",
            xaxis_title="Monthly Salary ($)", yaxis_title="Employees",
            yaxis=dict(gridcolor="#F1F5F9"),
            **CHART_LAYOUT, height=300, showlegend=False,
        )
        st.plotly_chart(fig_sal, use_container_width=True)

    with bc2:
        metrics_data = {
            "Avg Salary":         (dept_df["MonthlyIncome"].mean(), row["MonthlyIncome"]),
            "Job Satisfaction":   (dept_df["JobSatisfaction"].mean(), float(row["JobSatisfaction"])),
            "Avg Tenure (yrs)":   (dept_df["YearsAtCompany"].mean(), float(row["YearsAtCompany"])),
            "Satisfaction Score": (dept_df["SatisfactionScore"].mean(), float(row["SatisfactionScore"])),
        }
        metrics_norm = {
            k: (a / max(a, e + 1e-9), e / max(a, e + 1e-9))
            for k, (a, e) in metrics_data.items()
        }
        cats = list(metrics_data.keys())
        dept_vals = [metrics_norm[k][0] for k in cats]
        emp_vals  = [metrics_norm[k][1] for k in cats]

        fig_rad = go.Figure()
        fig_rad.add_trace(go.Scatterpolar(
            r=dept_vals + [dept_vals[0]], theta=cats + [cats[0]],
            fill="toself", name=f"{dept} Avg",
            line_color="#0066CC", fillcolor="rgba(0,102,204,0.15)",
        ))
        fig_rad.add_trace(go.Scatterpolar(
            r=emp_vals + [emp_vals[0]], theta=cats + [cats[0]],
            fill="toself", name=f"Employee #{int(emp_id)}",
            line_color="#FF3B3B", fillcolor="rgba(255,59,59,0.15)",
        ))
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1.2])),
            title=f"Employee vs Department Average",
            showlegend=True,
            legend=dict(x=0.8, y=1.1),
            **CHART_LAYOUT, height=300,
        )
        st.plotly_chart(fig_rad, use_container_width=True)

    # Comparison table
    comp_rows = []
    for metric, (dept_val, emp_val) in metrics_data.items():
        vs = "Above avg" if emp_val > dept_val else ("Below avg" if emp_val < dept_val else "At avg")
        if metric == "Avg Salary":
            comp_rows.append({"Metric": metric, f"{dept} Avg": f"${dept_val:,.0f}", "This Employee": f"${emp_val:,.0f}", "Status": vs})
        elif "yrs" in metric.lower():
            comp_rows.append({"Metric": metric, f"{dept} Avg": f"{dept_val:.1f} yrs", "This Employee": f"{emp_val:.0f} yrs", "Status": vs})
        else:
            comp_rows.append({"Metric": metric, f"{dept} Avg": f"{dept_val:.2f}", "This Employee": f"{emp_val:.2f}", "Status": vs})
    comp_rows.append({
        "Metric": "Attrition Rate", f"{dept} Avg": f"{dept_df['Attrition'].mean():.1%}",
        "This Employee": "Left" if row["Attrition"] == 1 else "Active", "Status": "—",
    })
    comp_rows.append({
        "Metric": "Overtime Rate", f"{dept} Avg": f"{dept_df['OverTime'].mean():.1%}",
        "This Employee": "Yes" if row["OverTime"] == 1 else "No", "Status": "—",
    })
    st.dataframe(pd.DataFrame(comp_rows), use_container_width=True, hide_index=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — PAYROLL ANOMALY SCANNER
# ───────────────────────────────────────────────────────────────────────────────
with tab3:
    payroll = load_payroll()

    st.markdown("""
    <div class="info-banner info-amber">
      <b>Payroll Anomaly Scanner</b> — Two-layer detection: deterministic business rules
      (Layer 1, fast) + Isolation Forest ML model (Layer 2, catches subtle patterns).
      Select a pay period to scan the full employee batch.
    </div>""", unsafe_allow_html=True)

    periods = sorted(payroll["pay_period"].unique(), reverse=True)
    sf1, sf2 = st.columns([2, 2])
    with sf1:
        sel_period = st.selectbox("Pay Period", periods, key="period_sel")
    with sf2:
        sev_filter = st.selectbox("Filter Severity", ["All", "CRITICAL", "WARNING", "CLEAN"], key="sev_sel")

    with st.spinner("Running anomaly detection..."):
        results = detect_period(sel_period)

    total    = len(results)
    critical = int((results["detected_severity"] == "CRITICAL").sum())
    warning  = int((results["detected_severity"] == "WARNING").sum())
    clean    = int((results["detected_severity"] == "CLEAN").sum())
    gt_anom  = int(results["is_anomaly"].sum())
    detected = int((results["detected_severity"] != "CLEAN").sum())
    tp       = int((results["is_anomaly"] & (results["detected_severity"] != "CLEAN")).sum())
    recall   = tp / gt_anom if gt_anom > 0 else 0.0

    # KPI row
    sk1, sk2, sk3, sk4, sk5 = st.columns(5)
    scan_kpis = [
        (f"{total:,}",    "Total Records",    "#0066CC", sel_period),
        (f"{clean:,}",    "Clean",            "#30D158", f"{clean/total:.1%} of batch"),
        (f"{warning:,}",  "Warnings",         "#FF9500",  f"{warning/total:.1%} of batch"),
        (f"{critical:,}", "Critical",         "#FF3B3B", f"{critical/total:.1%} of batch"),
        (f"{recall:.1%}", "Detection Recall", "#30D158" if recall > 0.8 else "#FF9500", f"{tp} of {gt_anom} true anomalies"),
    ]
    for col, (val, lbl, color, note) in zip([sk1, sk2, sk3, sk4, sk5], scan_kpis):
        with col:
            st.markdown(f"""
            <div class="kpi" style="border-top-color:{color}; padding:1.1rem 1.3rem; margin-bottom:1rem;">
              <div class="kpi-lbl">{lbl}</div>
              <div class="kpi-val" style="color:{color}; font-size:1.8rem">{val}</div>
              <div class="kpi-note">{note}</div>
            </div>""", unsafe_allow_html=True)

    # Charts
    sc1, sc2 = st.columns(2)

    with sc1:
        sev_counts = results["detected_severity"].value_counts()
        _sev_color_map = {"CRITICAL": "#FF3B3B", "WARNING": "#FF9500", "CLEAN": "#30D158"}
        _sev_colors = [_sev_color_map.get(lbl, "#999999") for lbl in sev_counts.index.tolist()]
        fig_pie = go.Figure(go.Pie(
            labels=sev_counts.index.tolist(),
            values=sev_counts.values.tolist(),
            hole=0.58,
            marker_colors=_sev_colors,
            textinfo="label+percent", textfont_size=12,
            hovertemplate="<b>%{label}</b><br>Records: %{value}<extra></extra>",
        ))
        fig_pie.update_layout(
            title=f"Severity Breakdown — {sel_period}",
            **CHART_LAYOUT, height=320,
            annotations=[dict(text="Detection<br>Results", x=0.5, y=0.5,
                               font=dict(size=11, color="#0A1628"), showarrow=False)],
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with sc2:
        by_dept = (
            results.groupby("department")["detected_severity"]
            .value_counts().unstack(fill_value=0).reset_index()
        )
        fig_dept = go.Figure()
        for sev, color in [("CLEAN", "#30D158"), ("WARNING", "#FF9500"), ("CRITICAL", "#FF3B3B")]:
            if sev in by_dept.columns:
                fig_dept.add_trace(go.Bar(
                    name=sev, x=by_dept["department"], y=by_dept[sev],
                    marker_color=color,
                ))
        fig_dept.update_layout(
            barmode="stack",
            title=f"Detection Results by Department — {sel_period}",
            yaxis=dict(title="Records", gridcolor="#F1F5F9"),
            legend=dict(orientation="h", y=-0.25),
            **CHART_LAYOUT, height=320,
        )
        st.plotly_chart(fig_dept, use_container_width=True)

    # Records table
    st.markdown("<div class='sec-hdr'>Record Detail</div>", unsafe_allow_html=True)

    display = results.copy()
    if sev_filter != "All":
        display = display[display["detected_severity"] == sev_filter]

    show_cols = [
        "record_id", "employee_id", "department", "country",
        "gross_pay", "income_tax", "pension", "net_pay",
        "detected_severity", "detected_by", "rule_flags",
        "iso_score", "is_anomaly", "anomaly_type",
    ]
    avail_cols = [c for c in show_cols if c in display.columns]

    def _color_sev(val):
        if val == "CRITICAL":
            return "background-color:#FFF0F0; color:#CC0000; font-weight:600"
        if val == "WARNING":
            return "background-color:#FFF8EE; color:#CC7700; font-weight:600"
        return "color:#008060; font-weight:500"

    styled = (
        display[avail_cols].head(300)
        .style
        .applymap(_color_sev, subset=["detected_severity"])
        .format({
            "gross_pay":   "${:,.2f}",
            "income_tax":  "${:,.2f}",
            "pension":     "${:,.2f}",
            "net_pay":     "${:,.2f}",
            "iso_score":   "{:.4f}",
        }, na_rep="—")
    )
    st.dataframe(styled, use_container_width=True, height=420)

    with st.expander("How Two-Layer Detection Works"):
        st.markdown("""
        **Layer 1 — Deterministic Rules** (always runs, zero ML cost)
        - Zero or negative net pay → `CRITICAL`
        - Net pay exceeds gross pay → `CRITICAL`
        - Missing pension on salary > €1,500 → `CRITICAL`
        - Tax rate outside statutory band (ES/BE/DE/NL/FR) → `CRITICAL` or `WARNING`
        - Total deductions > 80 % of gross → `WARNING`

        **Layer 2 — Isolation Forest** (catches subtle statistical outliers)
        - Trained on 6 months of synthetic payroll data
        - 200 estimators, 2 % contamination, 9 features
        - Flags records that are hard to isolate in feature space

        **Combined**: Rule flags take severity precedence. If rules are clean but the model
        flags a record, it is marked `WARNING`. This gives high recall while keeping
        CRITICAL flags precise.
        """)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 4 — HR POLICY Q&A
# ───────────────────────────────────────────────────────────────────────────────
with tab4:
    has_azure = bool(os.getenv("AZURE_PROJECT_ENDPOINT") and os.getenv("AZURE_API_KEY"))

    st.markdown("""
    <div class="info-banner info-blue">
      <b>HR Policy Q&A (RAG)</b> — Ask questions about HR policies in plain English.
      The system embeds your question with <code>text-embedding-3-small</code>,
      retrieves the most relevant policy chunks using cosine similarity,
      and generates a grounded answer with <code>Phi-4-mini-instruct</code>.
    </div>""", unsafe_allow_html=True)

    EXAMPLES = [
        "What is the annual leave entitlement in Spain?",
        "How is overtime compensated at SmartPayroll?",
        "What notice period is required for taking 5 days leave?",
        "What are the standard working hours in Germany?",
    ]

    st.markdown("**Try an example question:**")
    ex_cols = st.columns(len(EXAMPLES))
    selected_q = st.session_state.get("selected_q", "")
    for col, q in zip(ex_cols, EXAMPLES):
        with col:
            if st.button(q, use_container_width=True, key=f"eq_{q[:15]}"):
                st.session_state["selected_q"] = q
                selected_q = q

    question = st.text_area(
        "Your question",
        value=selected_q,
        placeholder="e.g. What is the annual leave entitlement for employees in Spain?",
        height=90,
    )

    ask_btn = st.button("Ask HR Policy AI", type="primary")

    if ask_btn and question.strip():
        if not has_azure:
            st.markdown("""
            <div class="info-banner info-amber">
              Azure API credentials not configured. Set <code>AZURE_PROJECT_ENDPOINT</code>
              and <code>AZURE_API_KEY</code> in your <code>.env</code> file to enable live RAG.
              Showing a simulated demo response below.
            </div>""", unsafe_allow_html=True)

            demo_map = {
                "leave":    "Based on the SmartPayroll HR Policy (leave_policy_smartpayroll_ES), employees in Spain are entitled to **22 working days** of paid annual leave per calendar year, in addition to national and regional public holidays. Leave requests must be submitted at least **15 days in advance** for periods exceeding 5 consecutive days.\n\n*Note: Verify with your HR team before taking action.*",
                "overtime": "Per the SmartPayroll Overtime Policy, overtime is compensated at **125% of the standard hourly rate** for the first 4 hours beyond the standard week, and **150%** thereafter. Employees may alternatively opt for compensatory time off at the same multiples.\n\n*Note: Verify with your HR team before taking action.*",
                "notice":   "The SmartPayroll Leave Policy requires employees to provide a minimum of **15 calendar days' notice** for leave periods of 5 or more consecutive days. For shorter periods, **48 hours' notice** is required unless the absence is due to illness or emergency.\n\n*Note: Verify with your HR team before taking action.*",
                "hours":    "Standard working hours in Germany under the SmartPayroll policy are **38.5 hours per week** across 5 days (Monday–Friday). Flexible working arrangements and remote work up to 2 days per week may be agreed with the line manager.\n\n*Note: Verify with your HR team before taking action.*",
            }
            q_lower = question.lower()
            answer = next((v for k, v in demo_map.items() if k in q_lower),
                          "Based on the available SmartPayroll policy documents, I was unable to find a specific answer to your question. Please contact your HR team directly for clarification.\n\n*Note: Verify with your HR team before taking action.*")
            sources = ["leave_policy_smartpayroll", "overtime_policy_smartpayroll"]

            st.markdown(f"""
            <div class="chat-bubble">
              <div style="font-size:0.8rem; color:#7A8699; margin-bottom:0.8rem; font-weight:500;">
                DEMO RESPONSE (Azure API not configured)
              </div>
              <b>Q:</b> {question}<br><br>
              <b>A:</b> {answer.replace(chr(10), '<br>')}
              <br><br>
              {''.join(f'<span class="src-chip">{s}</span>' for s in sources)}
            </div>""", unsafe_allow_html=True)
        else:
            with st.spinner("Searching policy documents and generating answer..."):
                try:
                    from src.rag.document_processor import process_policy_documents
                    from src.rag.chain import answer_question

                    chunks = process_policy_documents()
                    result = answer_question(question, chunks)

                    st.markdown(f"""
                    <div class="chat-bubble">
                      <b>Q:</b> {question}<br><br>
                      <b>A:</b> {result['answer'].replace(chr(10), '<br>')}
                      <br><br>
                      {''.join(f'<span class="src-chip">{s}</span>' for s in result['sources'])}
                    </div>""", unsafe_allow_html=True)

                    st.caption(
                        f"Retrieved {result['chunks_used']} policy chunks · "
                        f"Top similarity: {result['top_score']}"
                    )
                except Exception as exc:
                    st.error(f"RAG pipeline error: {exc}")

    elif ask_btn:
        st.warning("Please enter a question first.")

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("How the RAG pipeline works"):
        rc1, rc2 = st.columns([1, 1])
        with rc1:
            st.markdown("""
            **Step 1 — Document Processing**
            HR policy Markdown files are split into 500-character chunks
            with 50-character overlap to preserve context at boundaries.

            **Step 2 — Embedding**
            Each chunk is converted to a 1,536-dimension vector using
            Azure's `text-embedding-3-small` model.

            **Step 3 — Retrieval**
            Your question is embedded with the same model. Cosine similarity
            is computed against all stored chunks. Top-3 are returned.

            **Step 4 — Generation**
            The top chunks are injected as context into a `Phi-4-mini-instruct`
            prompt. The model is constrained to cite sources and admit gaps.
            """)
        with rc2:
            rag_steps = ["Question", "Embed Query", "Cosine Similarity Search",
                         "Retrieve Top-3 Chunks", "Phi-4-mini Generation", "Grounded Answer"]
            fig_flow = go.Figure(go.Funnel(
                y=rag_steps,
                x=[100, 95, 85, 70, 65, 60],
                marker_color=["#0066CC", "#1A7AD6", "#3494E6", "#4DAAEF",
                               "#30D158", "#25A244"],
                textinfo="label",
            ))
            fig_flow.update_layout(
                title="RAG Pipeline Flow",
                **CHART_LAYOUT, height=340,
            )
            st.plotly_chart(fig_flow, use_container_width=True)


# ── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  SmartPayroll AI &nbsp;·&nbsp; Python · Pandas · XGBoost · Isolation Forest · LangGraph · Azure AI
  &nbsp;·&nbsp;
  <a href="https://github.com/Najam0786/smartpayroll-ai" style="color:#0066CC; text-decoration:none;">
    GitHub Repository
  </a>
</div>
""", unsafe_allow_html=True)
