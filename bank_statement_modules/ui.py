# bank_statement_modules/ui.py
import pandas as pd
import streamlit as st

streamlit_css = """
<style>
/* === Fonts & base === */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@400;600;700&display=swap');

:root{
  --bg-dark: #0f1724;
  --panel: rgba(24, 26, 32, 0.72);
  --glass-border: rgba(255,255,255,0.06);
  --primary: #6C7CFF;
  --accent: #7EE7B9;
  --muted: #94a3b8;
  --positive: #10b981;
  --negative: #ef4444;
  --card-radius: 12px;
}

/* App shell */
.stApp {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', Roboto, 'Helvetica Neue', Arial;
  background: linear-gradient(180deg, #071025 0%, #0f1724 100%) fixed !important;
  color: #dbeafe;
}

/* Main container padding */
.main .block-container {
  padding: 28px 40px !important;
  max-width: 1200px;
}

/* Title */
.stMarkdown h1 {
  font-family: 'Poppins', Inter, sans-serif;
  font-weight: 700;
  font-size: 2.1rem;
  letter-spacing: -0.02em;
  margin-bottom: 0.5rem;
  background: linear-gradient(90deg, var(--primary), #8b5cf6);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}

/* Step wizard */
.step-wizard {
  display:flex;
  gap: 12px;
  align-items:center;
  margin: 14px 0 20px 0;
}
.step {
  display:flex;
  align-items:center;
  gap:10px;
  padding: 8px 12px;
  border-radius: 999px;
  background: rgba(255,255,255,0.02);
  border: 1px solid transparent;
  transition: all .25s ease;
  color: var(--muted);
  font-weight:600;
  font-size: 0.95rem;
}
.step .circle {
  width:28px;
  height:28px;
  border-radius:50%;
  display:inline-flex;
  align-items:center;
  justify-content:center;
  font-weight:700;
  color: white;
  background: rgba(255,255,255,0.06);
  font-size:0.85rem;
}
.step.completed {
  color: #e6fef3;
  background: linear-gradient(90deg, rgba(108,124,255,0.12), rgba(142,92,255,0.06));
  border-color: rgba(108,124,255,0.14);
}
.step.completed .circle { background: linear-gradient(90deg, var(--primary), #8b5cf6); box-shadow: 0 6px 18px rgba(108,124,255,0.14); }
.step.current { color: white; border-color: rgba(255,255,255,0.08); background: rgba(255,255,255,0.02); }
.step.pending { opacity: 0.7; }

/* Left column controls card */
.controls-card {
  background: var(--panel);
  border-radius: var(--card-radius);
  padding: 18px;
  border: 1px solid var(--glass-border);
  box-shadow: 0 8px 30px rgba(2,6,23,0.6);
}

/* Metrics row (cards) */
.metrics-row {
  display:flex;
  gap: 16px;
  margin: 18px 0;
  flex-wrap:wrap;
}
.metric {
  flex:1;
  min-width: 180px;
  background: rgba(255,255,255,0.02);
  padding: 18px;
  border-radius: 12px;
  border: 1px solid var(--glass-border);
}
.metric .label { color: var(--muted); font-size:0.85rem; margin-bottom:6px; font-weight:600; }
.metric .value { font-size:1.6rem; font-weight:700; line-height:1; margin-top:2px; }

/* colored metric values (use with inner HTML) */
.deposit-metric { color: var(--positive); font-weight:800; font-size:1.3rem; }
.withdrawal-metric { color: var(--negative); font-weight:800; font-size:1.3rem; }
.ratio-metric { color: #ffd166; font-weight:700; font-size:1.1rem; }

/* Progress bar styling */
.stProgress > div > div > div > div {
  border-radius: 8px !important;
  background: linear-gradient(90deg, rgba(108,124,255,0.9), rgba(126,231,185,0.9)) !important;
  box-shadow: 0 6px 20px rgba(41, 56, 115, 0.25) !important;
}

/* Dataframe zebra stripes + table tuning */
.stDataFrame table {
  border-collapse: collapse;
  width: 100%;
}
.stDataFrame table tbody tr:nth-child(odd) td { background: rgba(255,255,255,0.02); }
.stDataFrame table tbody tr:nth-child(even) td { background: rgba(255,255,255,0.01); }
.stDataFrame table td, .stDataFrame table th {
  padding: 10px 12px !important;
  border-bottom: 1px solid rgba(255,255,255,0.03);
  font-family: 'JetBrains Mono', monospace;
  font-size: 0.92rem;
}

/* Download buttons (small) */
.stDownloadButton > button {
  border-radius: 10px !important;
  padding: 8px 12px !important;
  font-weight: 700 !important;
}

/* File uploader styling (target container) */
.stFileUploader > div > div {
  background: linear-gradient(180deg, rgba(255,255,255,0.02), rgba(255,255,255,0.01)) !important;
  border-radius: 12px !important;
  padding: 18px !important;
  border: 1px dashed var(--glass-border) !important;
}

/* Minor element spacing */
.small-muted { color: var(--muted); font-size:0.9rem; }

/* Responsive tweak */
@media (max-width: 900px){
  .metrics-row { flex-direction: column; }
}
</style>
"""


def render_metrics_block(df: pd.DataFrame):
    """Render metric cards with colored values."""
    df = df.copy()
    w_col = "withdrawal_dr" if "withdrawal_dr" in df.columns else "dr" if "dr" in df.columns else None
    d_col = "deposit_cr" if "deposit_cr" in df.columns else "cr" if "cr" in df.columns else None

    total_withdrawals = pd.to_numeric(df[w_col], errors="coerce").fillna(0).sum() if w_col else 0
    num_withdrawals = (pd.to_numeric(df[w_col], errors="coerce").fillna(0) > 0).sum() if w_col else 0
    total_deposits = pd.to_numeric(df[d_col], errors="coerce").fillna(0).sum() if d_col else 0
    num_deposits = (pd.to_numeric(df[d_col], errors="coerce").fillna(0) > 0).sum() if d_col else 0
    total_txns = len(df)
    wd_ratio = f"{num_withdrawals}/{num_deposits}" if num_deposits > 0 else "N/A"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", total_txns)

    c2.markdown(
        f"<div style='color:#ef4444; font-size:1.5rem; font-weight:700;'>₹{total_withdrawals:,.2f}</div>"
        f"<div style='color:#ef4444; font-size:0.9rem;'>{num_withdrawals} txns</div>",
        unsafe_allow_html=True,
    )
    c2.caption("Total Withdrawals")

    c3.markdown(
        f"<div style='color:#10b981; font-size:1.5rem; font-weight:700;'>₹{total_deposits:,.2f}</div>"
        f"<div style='color:#10b981; font-size:0.9rem;'>{num_deposits} txns</div>",
        unsafe_allow_html=True,
    )
    c3.caption("Total Deposits")

    c4.markdown(f"<div style='color:#eab308; font-size:1.5rem; font-weight:700;'>{wd_ratio}</div>", unsafe_allow_html=True)
    c4.caption("W / D (count)")

