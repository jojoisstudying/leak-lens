"""
UMKM Business Decision Copilot — Streamlit dashboard for Indonesian SMB sales analytics.
"""

from __future__ import annotations

import html
import io
import json
import os
import random
import re
import time
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from dotenv import load_dotenv

APP_DIR = Path(__file__).resolve().parent
# Wajib file `.env` (bukan `.env.example`) — load dari folder app.py agar cwd Streamlit tidak masalah.
load_dotenv(APP_DIR / ".env", override=True)

# ---------------------------------------------------------------------------
# LLM — server-side only (.env atau variabel inline di bawah). Jangan taruh di UI Streamlit.
# ---------------------------------------------------------------------------
_GROQ_API_KEY_INLINE = ""
_OPENROUTER_API_KEY_INLINE = ""
_OPENAI_API_KEY_INLINE = ""
_ANTHROPIC_API_KEY_INLINE = ""

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openrouter").strip().lower().split("#")[0].strip()
LLM_MODEL_RAW = os.getenv("LLM_MODEL", "").strip().split("#")[0].strip()

DEFAULT_LLM_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "openrouter": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "openai": "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-20241022",
}

# Slug lama / salah ejaan → ID resmi di OpenRouter
OPENROUTER_MODEL_ALIASES: dict[str, str] = {
    "nvidia/nemotron-3-ultra:free": "nvidia/nemotron-3-ultra-550b-a55b:free",
    "nvidia/nemotron-3-ultra-free": "nvidia/nemotron-3-ultra-550b-a55b:free",
}

OPENROUTER_APP_TITLE = os.getenv("OPENROUTER_APP_TITLE", "UMKM Business Decision Copilot")
OPENROUTER_APP_URL = os.getenv("OPENROUTER_APP_URL", "http://localhost:8501")

_ENV_PLACEHOLDER_MARKERS = (
    "your_",
    "_here",
    "changeme",
    "xxx",
    "your_openrouter_api_key_here",
    "your_groq_api_key_here",
    "your_openai_api_key_here",
    "your_anthropic_api_key_here",
)


def _is_placeholder_secret(value: str) -> bool:
    v = value.strip().lower()
    if not v:
        return True
    return any(m in v for m in _ENV_PLACEHOLDER_MARKERS)


def _effective_llm_model(provider: str) -> str:
    model = LLM_MODEL_RAW
    if not model:
        return DEFAULT_LLM_MODELS.get(provider, DEFAULT_LLM_MODELS["openrouter"])
    if provider == "groq" and ("/" in model or model.startswith("nvidia")):
        return DEFAULT_LLM_MODELS["groq"]
    if provider == "openrouter":
        return OPENROUTER_MODEL_ALIASES.get(model, model)
    return model


def _resolve_provider_api_key(provider: str) -> str:
    mapping = {
        "groq": ("GROQ_API_KEY", _GROQ_API_KEY_INLINE, "Groq"),
        "openrouter": ("OPENROUTER_API_KEY", _OPENROUTER_API_KEY_INLINE, "OpenRouter"),
        "openai": ("OPENAI_API_KEY", _OPENAI_API_KEY_INLINE, "OpenAI"),
        "anthropic": ("ANTHROPIC_API_KEY", _ANTHROPIC_API_KEY_INLINE, "Anthropic"),
    }
    if provider not in mapping:
        raise ValueError(
            f"LLM_PROVIDER '{provider}' tidak dikenal. Gunakan: groq, openrouter, openai, anthropic."
        )
    env_name, inline, label = mapping[provider]
    key = os.getenv(env_name, "").strip() or inline.strip()
    if _is_placeholder_secret(key):
        raise ValueError(
            f"{label} belum dikonfigurasi. Buat file `{APP_DIR / '.env'}` (salin dari `.env.example`), "
            f"isi {env_name}=..., dan set LLM_PROVIDER={provider}. "
            f"Mengedit `.env.example` saja tidak cukup — aplikasi hanya membaca `.env`."
        )
    return key


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="UMKM Business Decision Copilot",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    #MainMenu, footer, header { visibility: hidden; }

    :root {
        --accent: #14b8a6;
        --accent-dark: #0f766e;
        --accent-soft: rgba(20, 184, 166, 0.12);
        --bg: #0b1220;
        --surface: #141d30;
        --surface-2: #1b2540;
        --border: #26314d;
        --text: #f1f5f9;
        --text-dim: #94a3b8;
        --text-faint: #64748b;
    }

    html, body, [class*="css"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp { background: var(--bg); color: var(--text); }
    [data-testid="stAppViewContainer"] .main { background: var(--bg); }
    .main .block-container { padding-top: 2rem !important; max-width: 1180px; }
    .main .stMarkdown p, .main .stMarkdown li, .main label, .main span { color: var(--text-dim); }

    /* Header dengan aksen gradient */
    .dash-header {
        margin-bottom: 1.75rem;
        padding-bottom: 1.25rem;
        border-bottom: 1px solid var(--border);
        position: relative;
    }
    .dash-header::before {
        content: "";
        position: absolute;
        top: -2rem; left: -2rem;
        width: 64px; height: 4px;
        border-radius: 4px;
        background: linear-gradient(90deg, var(--accent), #6366f1);
    }
    .dash-title {
        color: var(--text) !important;
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0.5rem 0 0.4rem 0;
        line-height: 1.25;
    }
    .dash-caption { color: var(--text-dim) !important; font-size: 0.95rem; margin: 0 0 0.25rem 0; line-height: 1.55; }
    .dash-subtitle { color: var(--text-faint) !important; font-size: 0.85rem; margin: 0; line-height: 1.45; }

    /* Navigasi tab */
    div[data-testid="stRadio"] > div {
        flex-direction: row;
        background-color: var(--surface);
        padding: 4px;
        border-radius: 10px;
        width: fit-content;
        border: 1px solid var(--border);
    }
    div[data-testid="stRadio"] label {
        background-color: transparent;
        padding: 7px 18px;
        border-radius: 7px;
        font-weight: 600;
        font-size: 0.875rem;
        color: var(--text-dim) !important;
        transition: all 0.15s ease;
    }
    div[data-testid="stRadio"] label:hover { color: var(--text) !important; }
    div[data-testid="stRadio"] label[data-checked="true"] {
        background: linear-gradient(135deg, var(--accent-dark), var(--accent));
        color: #ffffff !important;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.35);
    }

    section[data-testid="stSidebar"] {
        background-color: var(--surface);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] p { color: var(--text) !important; }

    /* Dropzone upload */
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
        background: var(--surface-2);
        border: 1.5px dashed var(--border);
        border-radius: 10px;
        transition: border-color 0.15s ease;
    }
    section[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent); }

    /* Headings & KPI Cards */
    .section-heading { color: var(--text) !important; font-size: 1.2rem; font-weight: 700; letter-spacing: -0.01em; margin: 0.5rem 0 1rem 0; }
    .kpi-card {
        background: linear-gradient(160deg, var(--surface-2), var(--surface));
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.25);
        min-height: 92px;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .kpi-card:hover { transform: translateY(-2px); border-color: var(--accent); }
    .kpi-label { color: var(--text-dim) !important; font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.03em; margin-bottom: 0.4rem; }
    .kpi-value { color: var(--text) !important; font-size: 1.45rem; font-weight: 800; line-height: 1.2; letter-spacing: -0.02em; }
    .kpi-subtitle { color: var(--accent) !important; font-size: 0.75rem; font-weight: 600; margin-top: 0.4rem; }

    /* LLM Panel & Component Styling */
    .llm-insight-panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-left: 3px solid var(--accent);
        padding: 26px 28px;
        border-radius: 12px;
        margin: 1rem 0 0.5rem 0;
        color: var(--text) !important;
        line-height: 1.65;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    .llm-insight-panel h2, .llm-insight-panel h3 { color: var(--text) !important; margin-top: 1.25rem; margin-bottom: 0.5rem; font-weight: 700; }
    .llm-insight-panel p, .llm-insight-panel li { color: var(--text-dim) !important; }

    .saas-details { background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 0.5rem 0.75rem; margin: 0.75rem 0; }
    .saas-details summary { cursor: pointer; font-weight: 600; color: var(--text) !important; padding: 0.4rem 0.2rem; }
    .saas-json { background: var(--bg); border: 1px solid var(--border); border-radius: 8px; padding: 12px; font-size: 0.8rem; color: var(--text-dim) !important; overflow-x: auto; margin: 0.5rem 0 0.25rem 0; white-space: pre-wrap; }

    .stButton > button {
        border-radius: 8px !important;
        font-weight: 600 !important;
        background: linear-gradient(135deg, var(--accent-dark), var(--accent)) !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 2px 8px rgba(20, 184, 166, 0.25);
        transition: transform 0.12s ease, box-shadow 0.12s ease;
    }
    .stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 14px rgba(20, 184, 166, 0.4); }
    .stButton > button:active { transform: translateY(0); }
    div[data-testid="stDownloadButton"] > button {
        background: var(--surface-2) !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }
    div[data-testid="stDownloadButton"] > button:hover { border-color: var(--accent) !important; }

    /* Landing Page */
    .landing-wrap { max-width: 760px; margin: 3rem auto 1rem auto; text-align: center; }
    .landing-badge {
        display: inline-block; background: var(--accent-soft); color: var(--accent);
        border: 1px solid rgba(20,184,166,0.35); padding: 5px 14px; border-radius: 999px;
        font-size: 0.78rem; font-weight: 600; margin-bottom: 1.25rem;
    }
    .landing-title { font-size: 2.4rem; font-weight: 800; letter-spacing: -0.03em; color: var(--text); line-height: 1.15; margin-bottom: 1rem; }
    .landing-title span { background: linear-gradient(90deg, var(--accent), #6366f1); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .landing-subtitle { font-size: 1.05rem; color: var(--text-dim); line-height: 1.65; max-width: 560px; margin: 0 auto 2rem auto; }
    .feature-grid { display: flex; gap: 16px; justify-content: center; flex-wrap: wrap; margin-top: 2.5rem; }
    .feature-card {
        background: linear-gradient(160deg, var(--surface-2), var(--surface));
        border: 1px solid var(--border); border-radius: 14px; padding: 20px; width: 220px; text-align: left;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .feature-card:hover { transform: translateY(-3px); border-color: var(--accent); }
    .feature-card .icon { font-size: 1.4rem; margin-bottom: 8px; }
    .feature-card .title { font-weight: 700; color: var(--text); font-size: 0.92rem; margin-bottom: 4px; }
    .feature-card .desc { color: var(--text-faint); font-size: 0.8rem; line-height: 1.5; }
    @media (max-width: 640px) {
        .landing-title { font-size: 1.7rem; }
        .landing-subtitle { font-size: 0.9rem; }
    }
    /* Responsif untuk layar sempit / mobile */
    @media (max-width: 640px) {
        .main .block-container { padding-left: 0.75rem !important; padding-right: 0.75rem !important; }
        .dash-title { font-size: 1.35rem; }
        .dash-caption, .dash-subtitle { font-size: 0.8rem; }
        div[data-testid="stRadio"] > div { flex-wrap: wrap; width: 100%; }
        div[data-testid="stRadio"] label { padding: 6px 10px; font-size: 0.8rem; }
        .kpi-card { min-height: 72px; padding: 10px 12px; }
        .kpi-value { font-size: 1.1rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

ECOMM_KEYWORDS = (
    "gofood",
    "grab",
    "shopee",
    "tokopedia",
    "tiktok",
    "bukalapak",
    "delivery",
    "online",
    "marketplace",
    "ecommerce",
    "e-commerce",
)
DEFAULT_PLATFORM_FEE = 0.20

COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "date": ("date", "tanggal", "tgl", "order_date", "transaction_date", "waktu", "datetime"),
    "price": ("price", "harga", "unit_price", "harga_satuan", "selling_price"),
    "qty": ("qty", "quantity", "jumlah", "kuantitas", "qnty", "pcs"),
    "discount": ("discount", "diskon", "potongan", "discount_amount"),
    "sku": ("sku", "product", "produk", "item", "nama_produk", "product_name", "item_name"),
    "channel": (
        "channel",
        "outlet",
        "platform",
        "source",
        "kanal",
        "toko",
        "store",
        "metode_pembayaran",
        "payment_method",
        "pembayaran",
        "wilayah",
        "region",
        "area",
    ),
    "revenue": (
        "revenue",
        "sales",
        "total_sales",
        "total_penjualan",
        "pendapatan",
        "omzet",
        "gross_sales",
        "nilai_transaksi",
        "total_harga",
    ),
    "order_id": ("order_id", "order", "invoice", "no_order", "id_order", "transaction_id", "transaksi_id", "id_transaksi"),
}

_HEADER_EXTRA_CELLS = frozenset(
    {
        "transaksi_id",
        "pelanggan_id",
        "kategori_produk",
        "metode_pembayaran",
        "total_penjualan",
        "harga_satuan",
        "nama_produk",
        "wilayah",
    }
)

_FOOTER_LABELS = frozenset({"total", "jumlah", "subtotal", "grandtotal", "grand_total", "summary", "ringkasan"})


def _normalize_col(name: str) -> str:
    s = str(name).strip().lower()
    s = re.sub(r"[\s\-/]+", "_", s)
    s = re.sub(r"[^\w]", "", s)
    return s


def _find_column(columns: list[str], role: str) -> str | None:
    normalized = {_normalize_col(c): c for c in columns}
    for alias in COLUMN_ALIASES.get(role, ()):
        key = _normalize_col(alias)
        if key in normalized:
            return normalized[key]
    for col in columns:
        nc = _normalize_col(col)
        for alias in COLUMN_ALIASES.get(role, ()):
            if _normalize_col(alias) in nc or nc in _normalize_col(alias):
                return col
    return None


def _all_header_tokens() -> set[str]:
    tokens: set[str] = set(_HEADER_EXTRA_CELLS)
    for aliases in COLUMN_ALIASES.values():
        for alias in aliases:
            tokens.add(_normalize_col(alias))
    return tokens


def _cell_looks_like_header(cell: Any) -> bool:
    nc = _normalize_col(str(cell))
    if not nc or nc in {"nan", "none"}:
        return False
    tokens = _all_header_tokens()
    if nc in tokens:
        return True
    return any(len(t) >= 4 and (t in nc or nc in t) for t in tokens)


def _score_header_row(row: pd.Series) -> int:
    return sum(1 for v in row if _cell_looks_like_header(v))


def _needs_header_promotion(df: pd.DataFrame) -> bool:
    if df.empty:
        return False
    col_names = [str(c).lower() for c in df.columns]
    unnamed = sum(1 for c in col_names if c.startswith("unnamed") or "unnamed:" in c)
    if unnamed >= max(1, len(col_names) // 3):
        return True
    mapped = sum(1 for role in ("date", "price", "qty", "revenue", "order_id") if _find_column(list(df.columns), role))
    return mapped < 2


def _promote_header_row(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    best_idx = -1
    best_score = 0
    scan_limit = min(25, len(df))
    for i in range(scan_limit):
        score = _score_header_row(df.iloc[i])
        if score > best_score:
            best_score = score
            best_idx = i
    if best_idx < 0 or best_score < 3:
        return df

    header = [_normalize_col(v) if not (isinstance(v, float) and np.isnan(v)) else f"col_{j}" for j, v in enumerate(df.iloc[best_idx])]
    for j, name in enumerate(header):
        if not name or name == "nan":
            header[j] = f"col_{j}"

    body = df.iloc[best_idx + 1 :].copy()
    body.columns = header
    return body.reset_index(drop=True)


def _strip_footer_and_empty_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df.copy()
    first_col = out.columns[0]
    lead = out[first_col].astype(str).str.strip().str.lower()
    out = out[~lead.isin(_FOOTER_LABELS)]
    out = out.replace(r"^\s*$", np.nan, regex=True)
    out = out.dropna(how="all")
    return out.reset_index(drop=True)


def parse_currency(value: Any) -> float:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    s = str(value).strip()
    if not s or s.lower() in ("nan", "none", "-", ""):
        return 0.0
    s = re.sub(r"(?i)rp\.?\s*", "", s)
    s = s.replace("\u00a0", "").replace(" ", "")
    if re.search(r",\d{1,2}$", s):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    s = re.sub(r"[^\d.\-]", "", s)
    try:
        return float(s) if s else 0.0
    except ValueError:
        return 0.0


def parse_dates_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = pd.to_datetime(series, errors="coerce", dayfirst=True, utc=False)
    if parsed.notna().mean() < 0.3:
        parsed = pd.to_datetime(series, errors="coerce", dayfirst=False, utc=False)
    return parsed


def detect_platform_fee(channel_value: Any) -> float:
    if channel_value is None or (isinstance(channel_value, float) and np.isnan(channel_value)):
        return 0.0
    text = str(channel_value).lower()
    for kw in ECOMM_KEYWORDS:
        if kw in text:
            return DEFAULT_PLATFORM_FEE
    return 0.0


def channel_needs_fee(channel_value: Any) -> bool:
    return detect_platform_fee(channel_value) > 0


def generate_sample_sales_csv(seed: int = 42) -> bytes:
    """Bikin CSV sintetis warung/UMKM F&B untuk demo — dilewatkan lewat pipeline cleaning yang sama."""
    rng = random.Random(seed)
    skus = ["Nasi Goreng", "Ayam Geprek", "Es Teh Manis", "Mie Ayam", "Kopi Susu", "Cireng", "Bakso Urat"]
    channels = ["Tunai", "GoFood", "ShopeeFood", "GrabFood", "QRIS"]
    rows = []
    start = date(2025, 8, 1)
    for d in range(45):
        cur = pd.Timestamp(start) + pd.Timedelta(days=d)
        n_orders = rng.randint(15, 40)
        for _ in range(n_orders):
            sku = rng.choices(skus, weights=[25, 20, 18, 15, 12, 6, 4])[0]
            price = {"Nasi Goreng": 18000, "Ayam Geprek": 16000, "Es Teh Manis": 5000, "Mie Ayam": 15000,
                      "Kopi Susu": 12000, "Cireng": 8000, "Bakso Urat": 17000}[sku]
            qty = rng.choices([1, 2, 3], weights=[70, 25, 5])[0]
            channel = rng.choices(channels, weights=[35, 25, 20, 15, 5])[0]
            hour = rng.choices(range(9, 22), weights=[2,3,4,6,10,8,5,4,6,9,10,7,3])[0]
            ts = cur + pd.Timedelta(hours=int(hour), minutes=rng.randint(0, 59))
            discount = rng.choice([0, 0, 0, 1000, 2000])
            rows.append({
                "tanggal": ts.strftime("%Y-%m-%d %H:%M"),
                "produk": sku,
                "harga_satuan": price,
                "jumlah": qty,
                "diskon": discount,
                "channel": channel,
            })
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")


@st.cache_data(show_spinner="Membersihkan & memproses data...", ttl=3600, max_entries=20)
def load_and_clean_sales(file_bytes: bytes, file_name: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    meta: dict[str, Any] = {"warnings": [], "columns_mapped": {}, "rows_in": 0, "rows_out": 0}

    try:
        if file_name.lower().endswith(".csv"):
            buffer = io.BytesIO(file_bytes)
            df = None
            for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
                try:
                    buffer.seek(0)
                    df = pd.read_csv(buffer, encoding=enc)
                    break
                except UnicodeDecodeError:
                    continue
            if df is None:
                buffer.seek(0)
                df = pd.read_csv(buffer, encoding="utf-8", errors="replace")
            if _needs_header_promotion(df):
                buffer.seek(0)
                df = pd.read_csv(buffer, header=None, encoding=enc if df is not None else "utf-8")
                df = _promote_header_row(df)
        elif file_name.lower().endswith((".xlsx", ".xls")):
            buffer = io.BytesIO(file_bytes)
            df = pd.read_excel(buffer, engine="openpyxl")
            if _needs_header_promotion(df):
                buffer.seek(0)
                df = pd.read_excel(buffer, header=None, engine="openpyxl")
                df = _promote_header_row(df)
        else:
            raise ValueError("Format file harus .csv atau .xlsx")
    except Exception as exc:
        raise ValueError(f"Gagal membaca file: {exc}") from exc

    if df.empty:
        raise ValueError("File kosong — tidak ada baris data.")

    meta["rows_in"] = len(df)
    df = df.copy()
    if _needs_header_promotion(df):
        df = _promote_header_row(df)
    df = _strip_footer_and_empty_rows(df)
    if df.empty:
        raise ValueError("Tidak ada baris transaksi setelah deteksi header / hapus baris summary.")

    df.columns = [_normalize_col(c) for c in df.columns]
    # dedupe column names
    seen: dict[str, int] = {}
    new_cols = []
    for c in df.columns:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols

    orig_roles = {}
    for role in COLUMN_ALIASES:
        found = _find_column(list(df.columns), role)
        if found:
            orig_roles[role] = found
    meta["columns_mapped"] = orig_roles

    date_col = orig_roles.get("date")
    price_col = orig_roles.get("price")
    qty_col = orig_roles.get("qty")
    discount_col = orig_roles.get("discount")
    sku_col = orig_roles.get("sku")
    channel_col = orig_roles.get("channel")
    revenue_col = orig_roles.get("revenue")

    if date_col:
        df["date_parsed"] = parse_dates_series(df[date_col])
    else:
        df["date_parsed"] = pd.NaT
        meta["warnings"].append("Kolom tanggal tidak terdeteksi — filter tanggal & tren harian terbatas.")

    for col_key, std_name in (
        (price_col, "price_clean"),
        (qty_col, "qty_clean"),
        (discount_col, "discount_clean"),
        (revenue_col, "revenue_clean"),
    ):
        if col_key and col_key in df.columns:
            if pd.api.types.is_numeric_dtype(df[col_key]):
                df[std_name] = pd.to_numeric(df[col_key], errors="coerce").fillna(0.0)
            else:
                df[std_name] = df[col_key].map(parse_currency)
        else:
            df[std_name] = 0.0

    if sku_col:
        df["sku_label"] = df[sku_col].astype(str).str.strip()
        df.loc[df["sku_label"].isin(["nan", "None", ""]), "sku_label"] = "Unknown SKU"
    else:
        df["sku_label"] = "Unknown SKU"
        meta["warnings"].append("Kolom SKU/produk tidak terdeteksi — analisis Pareto menggunakan label generik.")

    if channel_col:
        df["channel_label"] = df[channel_col].astype(str).str.strip()
    else:
        df["channel_label"] = "Unknown"
        meta["warnings"].append("Kolom outlet/channel tidak terdeteksi — biaya platform default 0% kecuali terdeteksi dari teks.")

    df["platform_fee_rate"] = df["channel_label"].map(detect_platform_fee)

    has_price_qty = (df["price_clean"] > 0).any() and (df["qty_clean"] > 0).any()
    has_revenue = (df["revenue_clean"] > 0).any()
    if has_price_qty:
        df["gross_sales"] = df["price_clean"] * df["qty_clean"]
        if has_revenue:
            rev_mask = df["revenue_clean"] > 0
            df.loc[rev_mask, "gross_sales"] = df.loc[rev_mask, "revenue_clean"]
        elif revenue_col:
            meta["warnings"].append(
                f"Kolom `{revenue_col}` kosong/tidak valid — gross sales dihitung dari harga × jumlah."
            )
    elif has_revenue:
        df["gross_sales"] = df["revenue_clean"]
        if not qty_col or (df["qty_clean"] == 0).all():
            df["qty_clean"] = 1.0
        meta["warnings"].append("Menggunakan kolom revenue/total sebagai gross sales (price×qty tidak lengkap).")
    else:
        df["gross_sales"] = 0.0
        meta["warnings"].append("Tidak dapat menghitung gross sales — periksa kolom harga, qty, atau revenue.")

    df["discount_clean"] = df["discount_clean"].clip(lower=0)
    df["platform_fee_amount"] = df["gross_sales"] * df["platform_fee_rate"]
    df["net_sales"] = (df["gross_sales"] - df["discount_clean"] - df["platform_fee_amount"]).clip(lower=0)

    df["hour"] = pd.NA
    if date_col and df["date_parsed"].notna().any():
        df["hour"] = df["date_parsed"].dt.hour
    elif date_col:
        for val in df[date_col].dropna().head(50):
            ts = pd.to_datetime(str(val), errors="coerce")
            if pd.notna(ts):
                df.loc[df[date_col] == val, "hour"] = ts.hour

    df["order_key"] = (
        df[orig_roles["order_id"]].astype(str)
        if orig_roles.get("order_id")
        else df.index.astype(str) + "_" + df["date_parsed"].astype(str)
    )

    meta["rows_out"] = len(df)
    return df, meta


def compute_analytics_summary(df: pd.DataFrame) -> dict[str, Any]:
    if df.empty:
        return {"error": "Data kosong setelah filter."}

    total_gross = float(df["gross_sales"].sum())
    total_net = float(df["net_sales"].sum())
    total_discount = float(df["discount_clean"].sum())
    total_fees = float(df["platform_fee_amount"].sum())
    n_orders = int(df["order_key"].nunique())
    n_lines = int(len(df))
    aov_gross = total_gross / n_orders if n_orders else 0.0
    aov_net = total_net / n_orders if n_orders else 0.0

    daily = (
        df.groupby(df["date_parsed"].dt.date, dropna=True)
        .agg(gross=("gross_sales", "sum"), net=("net_sales", "sum"))
        .reset_index()
    )
    if not daily.empty:
        date_col = daily.columns[0]
        daily = daily.rename(columns={date_col: "date"})
        daily["date"] = daily["date"].astype(str)
        daily_trend = daily.tail(14).to_dict(orient="records")
    else:
        daily_trend = []

    wow = _week_over_week(df)

    sku_rev = df.groupby("sku_label", as_index=False)["net_sales"].sum().sort_values("net_sales", ascending=False)
    top_skus = sku_rev.head(10).to_dict(orient="records")
    pareto_info = _pareto_skus(sku_rev)

    channel_mix = (
        df.groupby("channel_label", as_index=False)
        .agg(net_sales=("net_sales", "sum"), orders=("order_key", "nunique"))
        .sort_values("net_sales", ascending=False)
        .head(10)
        .to_dict(orient="records")
    )

    peak = _peak_hours_table(df)

    date_min = df["date_parsed"].min()
    date_max = df["date_parsed"].max()
    period = {
        "from": str(date_min.date()) if pd.notna(date_min) else None,
        "to": str(date_max.date()) if pd.notna(date_max) else None,
    }

    return {
        "period": period,
        "wow_comparison": wow,
        "totals": {
            "gross_sales_idr": round(total_gross, 0),
            "net_sales_idr": round(total_net, 0),
            "discounts_idr": round(total_discount, 0),
            "platform_fees_idr": round(total_fees, 0),
            "orders": n_orders,
            "line_items": n_lines,
            "aov_gross_idr": round(aov_gross, 0),
            "aov_net_idr": round(aov_net, 0),
        },
        "daily_trend_last_14d": daily_trend,
        "top_10_skus_by_net_revenue": top_skus,
        "pareto_sku": pareto_info,
        "channel_mix_top": channel_mix,
        "peak_hours": peak,
    }


def _week_over_week(df: pd.DataFrame) -> dict[str, Any] | None:
    sub = df.dropna(subset=["date_parsed"])
    if sub.empty:
        return None
    max_date = sub["date_parsed"].max().normalize()
    last7_start = max_date - pd.Timedelta(days=6)
    prev7_start = max_date - pd.Timedelta(days=13)
    prev7_end = max_date - pd.Timedelta(days=7)
    if sub["date_parsed"].min().normalize() > prev7_start:
        return None  # data belum cukup 14 hari untuk dibandingkan
    last7 = sub[sub["date_parsed"] >= last7_start]["net_sales"].sum()
    prev7 = sub[(sub["date_parsed"] >= prev7_start) & (sub["date_parsed"] <= prev7_end)]["net_sales"].sum()
    growth_pct = ((last7 - prev7) / prev7 * 100) if prev7 > 0 else None
    return {
        "last_7d_net_sales_idr": round(float(last7), 0),
        "prev_7d_net_sales_idr": round(float(prev7), 0),
        "growth_pct": round(growth_pct, 1) if growth_pct is not None else None,
    }


def _pareto_skus(sku_rev: pd.DataFrame) -> dict[str, Any]:
    if sku_rev.empty or sku_rev["net_sales"].sum() <= 0:
        return {"top_20pct_sku_count": 0, "revenue_share_pct": 0, "sku_names_sample": []}
    sku_rev = sku_rev.sort_values("net_sales", ascending=False).reset_index(drop=True)
    total = sku_rev["net_sales"].sum()
    sku_rev["cum_share"] = sku_rev["net_sales"].cumsum() / total
    top_n = max(1, int(np.ceil(len(sku_rev) * 0.2)))
    top_slice = sku_rev.head(top_n)
    share = float(top_slice["net_sales"].sum() / total * 100)
    return {
        "top_20pct_sku_count": top_n,
        "total_skus": len(sku_rev),
        "revenue_share_pct": round(share, 1),
        "sku_names_sample": top_slice["sku_label"].head(5).tolist(),
    }


def _peak_hours_table(df: pd.DataFrame) -> list[dict[str, Any]]:
    sub = df.dropna(subset=["hour"]).copy()
    if sub.empty:
        return []
    sub["hour"] = sub["hour"].astype(int)
    agg = (
        sub.groupby("hour")
        .agg(volume=("order_key", "nunique"), net_sales=("net_sales", "sum"), gross_sales=("gross_sales", "sum"))
        .reset_index()
    )
    agg["aov_net"] = np.where(agg["volume"] > 0, agg["net_sales"] / agg["volume"], 0)
    return agg.sort_values("hour").to_dict(orient="records")


def build_insights_prompt(summary: dict[str, Any]) -> str:
    payload = json.dumps(summary, ensure_ascii=False, indent=2)
    return f"""Anda adalah konsultan bisnis untuk UMKM Indonesia (F&B, retail, omnichannel).
Analisis metrik penjualan berikut (JSON) dan berikan rekomendasi praktis dalam Bahasa Indonesia.

DATA METRIK:
{payload}

ATURAN OUTPUT (WAJIB — gunakan markdown persis struktur ini):

## 3 Diagnosa Utama Bisnis
1. **🔴 Merah (Masalah Kritis):** ...
2. **🟡 Kuning (Peringatan):** ...
3. **🟢 Hijau (Peluang):** ...

## 3 Action Plan Besok (Lakukan Ini, Bukan Cuma Teori)
1. ...
2. ...
3. ...

## Snapshot Metrik Kunci
- Gunakan bullet pendek dengan angka dari data (IDR, AOV, channel, jam puncak, Pareto SKU).
- Setiap poin harus actionable untuk pemilik UMKM.

Jangan menambahkan section lain. Angka harus konsisten dengan JSON. Jika data tanggal/jam kurang, sebutkan keterbatasannya secara jujur."""


def generate_fallback_insight(summary: dict[str, Any]) -> str:
    """Insight berbasis aturan dari data asli — dipakai saat LLM gagal/limit, supaya demo tetap jalan."""
    t = summary.get("totals", {})
    pareto = summary.get("pareto_sku", {})
    peak = summary.get("peak_hours", [])
    channels = summary.get("channel_mix_top", [])
    top_skus = summary.get("top_10_skus_by_net_revenue", [])

    fee_ratio = (t.get("platform_fees_idr", 0) / t.get("gross_sales_idr", 1)) if t.get("gross_sales_idr") else 0
    best_hour = max(peak, key=lambda p: p.get("net_sales", 0)) if peak else None
    top_channel = channels[0] if channels else None
    top_sku = top_skus[0] if top_skus else None

    merah = (
        f"Biaya platform menggerus **{fee_ratio*100:.1f}%** dari gross sales — pertimbangkan dorong transaksi Tunai/QRIS."
        if fee_ratio > 0.12 else
        f"Konsentrasi revenue tinggi: **{pareto.get('top_20pct_sku_count', 0)} produk** menyumbang {pareto.get('revenue_share_pct', 0)}% penjualan — risiko jika stok/produk itu bermasalah."
    )
    kuning = (
        f"Channel **{top_channel['channel_label']}** mendominasi ({format_idr(top_channel['net_sales'])}) — cek apakah channel lain kurang dioptimalkan."
        if top_channel else "Data channel belum cukup untuk diagnosa mendalam."
    )
    hijau = (
        f"Jam **{int(best_hour['hour']):02d}:00** adalah jam puncak (net {format_idr(best_hour['net_sales'])}) — peluang promo terarah di jam ini."
        if best_hour else "Peluang: lengkapi data jam transaksi untuk analisis jam puncak."
    )
    return f"""## 3 Diagnosa Utama Bisnis
1. **🔴 Merah (Masalah Kritis):** {merah}
2. **🟡 Kuning (Peringatan):** {kuning}
3. **🟢 Hijau (Peluang):** {hijau}

## 3 Action Plan Besok (Lakukan Ini, Bukan Cuma Teori)
1. Review margin produk **{top_sku['sku_label'] if top_sku else '-'}** — pastikan harga jual masih untung setelah diskon & biaya platform.
2. Siapkan stok & staf ekstra menjelang jam puncak yang teridentifikasi di atas.
3. Uji promo kecil di channel selain **{top_channel['channel_label'] if top_channel else '-'}** untuk diversifikasi revenue.

## Snapshot Metrik Kunci
- Net Sales: {format_idr(t.get('net_sales_idr', 0))} dari {t.get('orders', 0)} order (AOV {format_idr(t.get('aov_net_idr', 0))})
- Biaya Platform: {format_idr(t.get('platform_fees_idr', 0))} ({fee_ratio*100:.1f}% dari gross)
- Top SKU: {top_sku['sku_label'] if top_sku else '-'}

*Catatan: ini ringkasan otomatis berbasis aturan (layanan AI sedang tidak tersedia).*"""


def _chat_completions_request(
    url: str,
    headers: dict[str, str],
    model: str,
    prompt: str,
    provider_label: str,
    max_tokens: int = 2048,
) -> str:
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "Anda analis bisnis UMKM Indonesia. Jawab dalam Bahasa Indonesia."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.4,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, json=body, timeout=120)
    if resp.status_code >= 400:
        try:
            err = resp.json()
        except Exception:
            err = resp.text
        raise RuntimeError(f"{provider_label} API error ({resp.status_code}): {err}")
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()


def call_llm_insights(prompt: str) -> str:
    provider = LLM_PROVIDER
    model = _effective_llm_model(provider)
    api_key = _resolve_provider_api_key(provider)

    if provider == "anthropic":
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        body = {
            "model": model,
            "max_tokens": 2048,
            "system": "Anda analis bisnis UMKM Indonesia. Jawab dalam Bahasa Indonesia.",
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = requests.post(url, headers=headers, json=body, timeout=120)
        if resp.status_code >= 400:
            try:
                err = resp.json()
            except Exception:
                err = resp.text
            raise RuntimeError(f"Anthropic API error ({resp.status_code}): {err}")
        data = resp.json()
        blocks = data.get("content", [])
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        return "\n".join(texts).strip()

    if provider == "groq":
        url = "https://api.groq.com/openai/v1/chat/completions"
    elif provider == "openrouter":
        url = "https://openrouter.ai/api/v1/chat/completions"
    else:
        url = "https://api.openai.com/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    if provider == "openrouter":
        headers["HTTP-Referer"] = OPENROUTER_APP_URL
        headers["X-Title"] = OPENROUTER_APP_TITLE

    return _chat_completions_request(url, headers, model, prompt, provider.capitalize())


def apply_fee_overrides(df: pd.DataFrame, overrides: dict[str, float]) -> pd.DataFrame:
    """Timpa platform_fee_rate per channel_label sesuai input user, lalu hitung ulang net_sales."""
    if not overrides:
        return df
    out = df.copy()
    for ch, rate in overrides.items():
        mask = out["channel_label"] == ch
        out.loc[mask, "platform_fee_rate"] = rate
    out["platform_fee_amount"] = out["gross_sales"] * out["platform_fee_rate"]
    out["net_sales"] = (out["gross_sales"] - out["discount_clean"] - out["platform_fee_amount"]).clip(lower=0)
    return out


def apply_filters(
    df: pd.DataFrame,
    date_range: tuple[date | None, date | None] | None,
    channels: list[str] | None,
) -> pd.DataFrame:
    out = df.copy()
    if date_range and date_range[0] and date_range[1] and out["date_parsed"].notna().any():
        start = pd.Timestamp(date_range[0])
        end = pd.Timestamp(date_range[1]) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
        mask = (out["date_parsed"] >= start) & (out["date_parsed"] <= end)
        out = out.loc[mask]
    if channels:
        out = out[out["channel_label"].isin(channels)]
    return out


def _style_chart(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0", family="Inter, -apple-system, sans-serif"),
        title_font=dict(color="#f8fafc", size=15),
        colorway=["#14b8a6", "#6366f1", "#f59e0b", "#f43f5e", "#38bdf8", "#a78bfa"],
        margin=dict(t=48, l=8, r=8, b=8),
    )
    return fig


def chart_daily_revenue(df: pd.DataFrame) -> go.Figure:
    sub = df.dropna(subset=["date_parsed"]).copy()
    if sub.empty:
        fig = go.Figure()
        fig.update_layout(title="Daily Revenue Trend — data tanggal tidak tersedia")
        return _style_chart(fig)
    daily = sub.groupby(sub["date_parsed"].dt.date).agg(gross=("gross_sales", "sum"), net=("net_sales", "sum")).reset_index()
    daily.columns = ["date", "Gross Revenue", "Net Revenue"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["Gross Revenue"], name="Gross Revenue", mode="lines+markers"))
    fig.add_trace(go.Scatter(x=daily["date"], y=daily["Net Revenue"], name="Net Revenue", mode="lines+markers"))
    fig.update_layout(
        title="Daily Revenue Trend (Gross vs Net)",
        xaxis_title="Tanggal",
        yaxis_title="IDR",
        hovermode="x unified",
        height=420,
    )
    return _style_chart(fig)


def chart_top_skus(df: pd.DataFrame) -> go.Figure:
    sku = df.groupby("sku_label", as_index=False)["net_sales"].sum().sort_values("net_sales", ascending=False).head(10)
    if sku.empty:
        fig = go.Figure()
        fig.update_layout(title="Top 10 SKUs — tidak ada data revenue")
        return _style_chart(fig)
    fig = px.bar(sku, x="net_sales", y="sku_label", orientation="h", title="Top 10 SKUs by Net Revenue Contribution")
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, xaxis_title="Net Sales (IDR)", yaxis_title="SKU", height=420)
    return _style_chart(fig)


def chart_peak_hours(df: pd.DataFrame) -> go.Figure:
    peak = _peak_hours_table(df)
    if not peak:
        fig = go.Figure()
        fig.update_layout(title="Peak Sales Hours — kolom jam tidak tersedia")
        return _style_chart(fig)
    ph = pd.DataFrame(peak)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=ph["hour"], y=ph["net_sales"], name="Net Sales", marker_color="#636EFA"))
    fig.add_trace(go.Scatter(x=ph["hour"], y=ph["aov_net"], name="AOV (Net)", yaxis="y2", mode="lines+markers", line=dict(color="#EF553B")))
    fig.update_layout(
        title="Peak Sales Hours — Volume (Net Sales) & AOV",
        xaxis_title="Jam (0–23)",
        yaxis_title="Net Sales (IDR)",
        yaxis2=dict(title="AOV Net (IDR)", overlaying="y", side="right"),
        barmode="overlay",
        height=420,
    )
    return _style_chart(fig)


def format_idr(x: float) -> str:
    try:
        return f"Rp {x:,.0f}".replace(",", ".")
    except Exception:
        return str(x)


def render_kpi_card(title: str, value: str, subtitle: str = "") -> str:
    sub_html = f'<div class="kpi-subtitle">{html.escape(subtitle)}</div>' if subtitle else ""
    return f"""
<div class="kpi-card">
  <div class="kpi-label">{html.escape(title)}</div>
  <div class="kpi-value">{html.escape(value)}</div>
  {sub_html}
</div>
"""


def render_kpi_grid_from_totals(t: dict[str, Any], wow: dict[str, Any] | None = None) -> None:
    wow_sub = ""
    if wow and wow.get("growth_pct") is not None:
        g = wow["growth_pct"]
        arrow = "▲" if g >= 0 else "▼"
        wow_sub = f"{arrow} {abs(g):.1f}% vs 7 hari sebelumnya"
    cards = [
        ("Net Sales", format_idr(t.get("net_sales_idr", 0)), wow_sub),
        ("Orders", f"{t.get('orders', 0):,}".replace(",", "."), ""),
        ("AOV (Net)", format_idr(t.get("aov_net_idr", 0)), ""),
        ("Platform Fees", format_idr(t.get("platform_fees_idr", 0)), ""),
    ]
    cols = st.columns(4)
    for col, (title, value, subtitle) in zip(cols, cards):
        with col:
            st.markdown(render_kpi_card(title, value, subtitle), unsafe_allow_html=True)


def render_dashboard_header() -> None:
    st.markdown(
        """
<div class="dash-header">
  <h1 class="dash-title">🇮🇩 UMKM Business Decision Copilot</h1>
  <p class="dash-caption">Upload penjualan CSV/XLSX → bersihkan otomatis → visualisasi → insight LLM untuk keputusan bisnis UMKM.</p>
  <p class="dash-subtitle">Ringkasan KPI, grafik penjualan, dan rekomendasi AI untuk keputusan operasional harian.</p>
</div>
        """,
        unsafe_allow_html=True,
    )


def render_section_heading(text: str) -> None:
    st.markdown(f'<h2 class="section-heading">{html.escape(text)}</h2>', unsafe_allow_html=True)


def render_json_details(title: str, data: Any) -> None:
    payload = json.dumps(data, ensure_ascii=False, indent=2, default=str)
    st.markdown(
        f"""
<details class="saas-details">
  <summary>{html.escape(title)}</summary>
  <pre class="saas-json">{html.escape(payload)}</pre>
</details>
        """,
        unsafe_allow_html=True,
    )


def _markdown_to_html(text: str) -> str:
    """Converter markdown ringan, tanpa dependency eksternal (supaya tidak pernah gagal/fallback mentah)."""
    lines = text.split("\n")
    out: list[str] = []
    list_tag: str | None = None

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            out.append(f"</{list_tag}>")
            list_tag = None

    def inline(s: str) -> str:
        s = html.escape(s)
        s = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)
        s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<em>\1</em>", s)
        return s

    for raw in lines:
        line = raw.strip()
        if not line:
            close_list()
            continue
        m = re.match(r"^(#{2,3})\s+(.*)", line)
        if m:
            close_list()
            tag = "h2" if len(m.group(1)) == 2 else "h3"
            out.append(f"<{tag}>{inline(m.group(2))}</{tag}>")
            continue
        m = re.match(r"^\d+\.\s+(.*)", line)
        if m:
            if list_tag != "ol":
                close_list()
                out.append("<ol>")
                list_tag = "ol"
            out.append(f"<li>{inline(m.group(1))}</li>")
            continue
        m = re.match(r"^[-*]\s+(.*)", line)
        if m:
            if list_tag != "ul":
                close_list()
                out.append("<ul>")
                list_tag = "ul"
            out.append(f"<li>{inline(m.group(1))}</li>")
            continue
        close_list()
        out.append(f"<p>{inline(line)}</p>")
    close_list()
    return "\n".join(out)


def build_printable_report_html(insight_text: str, totals: dict[str, Any]) -> str:
    body = _markdown_to_html(insight_text)
    generated = date.today().strftime("%d %B %Y")
    return f"""<!DOCTYPE html>
<html lang="id"><head><meta charset="utf-8">
<title>Laporan Insight Bisnis — {generated}</title>
<style>
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; color:#1e293b; max-width:800px; margin:32px auto; padding:0 24px; }}
  h1 {{ font-size:1.4rem; border-bottom:2px solid #0f766e; padding-bottom:8px; }}
  h2, h3 {{ color:#0f172a; margin-top:20px; }}
  p, li {{ color:#334155; }}
  strong {{ color:#0f172a; }}
  .meta {{ color:#64748b; font-size:0.85rem; margin-bottom:24px; }}
  .kpi {{ display:flex; gap:16px; flex-wrap:wrap; margin-bottom:24px; }}
  .kpi div {{ border:1px solid #cbd5e1; border-radius:8px; padding:10px 16px; }}
  @media print {{ body {{ margin:0; }} }}
</style></head>
<body>
  <h1>🇮🇩 Laporan Insight Bisnis — UMKM Business Decision Copilot</h1>
  <p class="meta">Dibuat otomatis pada {generated}</p>
  <div class="kpi">
    <div><b>Net Sales</b><br>{format_idr(totals.get('net_sales_idr', 0))}</div>
    <div><b>Orders</b><br>{totals.get('orders', 0)}</div>
    <div><b>AOV</b><br>{format_idr(totals.get('aov_net_idr', 0))}</div>
  </div>
  {body}
  <p class="meta">Tekan Ctrl/Cmd+P pada file ini lalu pilih "Save as PDF" untuk menyimpan sebagai PDF.</p>
</body></html>"""


def render_llm_insight_panel(text: str) -> None:
    body = _markdown_to_html(text)
    st.markdown(f'<div class="llm-insight-panel">{body}</div>', unsafe_allow_html=True)


def render_landing_page() -> None:
    st.markdown(
        """
<div class="landing-wrap">
  <span class="landing-badge">🇮🇩 Dibuat untuk UMKM Indonesia</span>
  <h1 class="landing-title">Data penjualan berantakan?<br><span>Biar AI yang urus.</span></h1>
  <p class="landing-subtitle">
    Upload CSV/XLSX dari kasir, GoFood, Shopee, atau mana pun — sistem otomatis membersihkan data,
    hitung KPI, dan kasih rekomendasi bisnis yang bisa dieksekusi besok pagi.
  </p>
</div>
<div class="feature-grid">
  <div class="feature-card"><div class="icon">🧹</div><div class="title">Auto-cleaning</div>
    <div class="desc">Deteksi kolom & header otomatis, walau format file berantakan.</div></div>
  <div class="feature-card"><div class="icon">📊</div><div class="title">KPI Instan</div>
    <div class="desc">Net sales, AOV, biaya platform, tren harian dalam sekali lihat.</div></div>
  <div class="feature-card"><div class="icon">✨</div><div class="title">Insight AI</div>
    <div class="desc">Diagnosa & action plan konkret, bukan cuma angka mentah.</div></div>
</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    _, mid, _ = st.columns([1, 1, 1])
    with mid:
        if st.button("🚀 Mulai Analisis", type="primary", use_container_width=True):
            st.session_state["page"] = "dashboard"
            st.rerun()
    st.markdown(
        '<p style="text-align:center;color:#64748b;font-size:0.8rem;margin-top:0.75rem;">Gratis dicoba — tidak perlu daftar akun.</p>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
if st.session_state.get("page", "landing") == "landing":
    render_landing_page()
    st.stop()

render_dashboard_header()

with st.sidebar:
    if st.button("← Beranda", use_container_width=False):
        st.session_state["page"] = "landing"
        st.rerun()
    st.header("Data & Konfigurasi")
    try:
        _resolve_provider_api_key(LLM_PROVIDER)
        st.success(f"AI siap ✅ ({LLM_PROVIDER})", icon="✅")
    except Exception:
        st.warning(f"AI belum dikonfigurasi ⚠️ ({LLM_PROVIDER}) — insight akan pakai mode fallback.", icon="⚠️")
    uploaded = st.file_uploader("Upload file penjualan", type=["csv", "xlsx"])
    use_sample = st.button("🎯 Coba dengan Data Contoh", use_container_width=True)

df_clean: pd.DataFrame | None = None
meta: dict[str, Any] = {}
filtered: pd.DataFrame | None = None

file_bytes = None
file_name = None
if use_sample:
    file_bytes = generate_sample_sales_csv()
    file_name = "sample_umkm_sales.csv"
elif uploaded is not None:
    file_bytes = uploaded.getvalue()
    file_name = uploaded.name

# Reset insight AI kalau file berganti (identitas = nama + ukuran)
file_identity = f"{file_name}:{len(file_bytes)}" if file_bytes else None
if file_identity and st.session_state.get("_active_file") != file_identity:
    st.session_state["_active_file"] = file_identity
    st.session_state.pop("llm_insight", None)
    st.session_state["_llm_calls"] = 0

if file_bytes is not None:
    try:
        df_clean, meta = load_and_clean_sales(file_bytes, file_name)
    except ValueError as err:
        st.error(f"⚠️ File tidak bisa diproses: {err}")
        st.stop()
    except Exception:
        st.error("⚠️ Terjadi kendala saat memproses file. Coba periksa format kolom, atau gunakan Data Contoh.")
        st.stop()

    if meta.get("warnings"):
        for w in meta["warnings"]:
            st.sidebar.warning(w)

    st.sidebar.divider()
    st.sidebar.subheader("Filter")
    date_filter_enabled = df_clean["date_parsed"].notna().any()
    dr: tuple[date | None, date | None] | None = None
    if date_filter_enabled:
        dmin = df_clean["date_parsed"].min().date()
        dmax = df_clean["date_parsed"].max().date()
        dr = st.sidebar.date_input("Rentang tanggal", value=(dmin, dmax), min_value=dmin, max_value=dmax)
        if isinstance(dr, date):
            dr = (dr, dr)
    else:
        st.sidebar.info("Filter tanggal nonaktif — kolom tanggal tidak valid.")

    channels_all = sorted(df_clean["channel_label"].dropna().unique().tolist())
    if len(channels_all) > 1 or (len(channels_all) == 1 and channels_all[0] != "Unknown"):
        selected_channels = st.sidebar.multiselect("Outlet / Channel", channels_all, default=channels_all)
    else:
        selected_channels = channels_all

    with st.sidebar.expander("⚙️ Biaya Platform per Channel"):
        st.caption(
            "Persentase komisi yang dipotong platform online dari tiap transaksi "
            "(mis. GoFood/Shopee ambil ~15-20%). Ini mengurangi Net Sales — geser kalau komisi asli beda."
        )
        fee_overrides: dict[str, float] = {}
        ecomm_channels = [c for c in channels_all if channel_needs_fee(c)]
        if ecomm_channels:
            for ch in ecomm_channels:
                pct = st.slider(ch, 0, 40, 20, step=1, key=f"fee_{ch}")
                fee_overrides[ch] = pct / 100.0
        else:
            st.caption("Tidak ada channel online terdeteksi.")

    df_clean = apply_fee_overrides(df_clean, fee_overrides)
    filtered = apply_filters(df_clean, dr if date_filter_enabled else None, selected_channels if selected_channels else None)
    summary = compute_analytics_summary(filtered)

    t = summary.get("totals", {})
    render_kpi_grid_from_totals(t, summary.get("wow_comparison"))
    st.write("")
    nav = st.radio(
        "Menu Navigation",
        ["🚨 Executive Decisions", "📊 Visual Analytics", "🧹 Cleaned Data Preview"],
        horizontal=True,
        label_visibility="collapsed",
    )
    st.write("")

    if nav == "🚨 Executive Decisions":
        col_main, col_side = st.columns([2, 1], gap="large")
        with col_main:
            render_section_heading("Actionable Business Insights")
            generate = st.button("✨ Generate Insight LLM", type="primary", use_container_width=True)
            st.caption(
                f"Via **{LLM_PROVIDER}** (`{_effective_llm_model(LLM_PROVIDER)}`) · "
                "🔒 hanya ringkasan angka yang dikirim ke AI, bukan data transaksi mentah."
            )

            if generate:
                last_call = st.session_state.get("_last_llm_ts", 0)
                calls_used = st.session_state.get("_llm_calls", 0)
                if time.time() - last_call < 15:
                    st.warning("Mohon tunggu ±15 detik sebelum generate lagi.")
                elif calls_used >= 8:
                    st.info("Batas generate untuk sesi demo ini tercapai. Hubungi kami untuk akses penuh.")
                else:
                    prompt = build_insights_prompt(summary)
                    with st.spinner(f"Memanggil {LLM_PROVIDER}..."):
                        try:
                            text = call_llm_insights(prompt)
                            st.session_state["llm_insight"] = text
                        except Exception:
                            st.session_state["llm_insight"] = generate_fallback_insight(summary)
                            st.info("ℹ️ Layanan AI sedang tidak tersedia — menampilkan ringkasan otomatis berbasis data Anda.")
                    st.session_state["_last_llm_ts"] = time.time()
                    st.session_state["_llm_calls"] = calls_used + 1

            if st.session_state.get("llm_insight"):
                render_llm_insight_panel(st.session_state["llm_insight"])
                dl_col1, dl_col2 = st.columns(2)
                with dl_col1:
                    st.download_button(
                        "⬇️ Markdown",
                        data=st.session_state["llm_insight"].encode("utf-8"),
                        file_name="insight_bisnis.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with dl_col2:
                    report_html = build_printable_report_html(st.session_state["llm_insight"], t)
                    st.download_button(
                        "⬇️ Laporan (HTML → PDF)",
                        data=report_html.encode("utf-8"),
                        file_name="laporan_insight.html",
                        mime="text/html",
                        use_container_width=True,
                    )
            else:
                st.info("Klik **Generate Insight LLM** untuk melihat Diagnosa, Action Plan, dan Snapshot Metrik.")

        with col_side:
            render_section_heading("Ringkasan Cepat")
            pareto = summary.get("pareto_sku", {})
            top_skus = summary.get("top_10_skus_by_net_revenue", [])
            channels_mix = summary.get("channel_mix_top", [])
            snapshot_html = f"""
<div style="background:#141d30;border:1px solid #26314d;border-radius:12px;padding:16px 18px;font-size:0.85rem;color:#cbd5e1;line-height:1.9;">
  <b style="color:#f1f5f9;">🏆 Top SKU</b><br>{html.escape(top_skus[0]['sku_label']) if top_skus else '-'}<br><br>
  <b style="color:#f1f5f9;">📡 Channel Terbesar</b><br>{html.escape(channels_mix[0]['channel_label']) if channels_mix else '-'}<br><br>
  <b style="color:#f1f5f9;">📦 Konsentrasi SKU (Pareto)</b><br>{pareto.get('top_20pct_sku_count', 0)} produk = {pareto.get('revenue_share_pct', 0)}% revenue
</div>
"""
            st.markdown(snapshot_html, unsafe_allow_html=True)
            render_json_details("Lihat JSON metrik untuk LLM", summary)

    elif nav == "📊 Visual Analytics":
        if filtered.empty:
            st.warning("Tidak ada data setelah filter.")
        else:
            st.plotly_chart(chart_daily_revenue(filtered), use_container_width=True)
            left, right = st.columns(2, gap="medium")
            with left:
                st.plotly_chart(chart_top_skus(filtered), use_container_width=True)
            with right:
                st.plotly_chart(chart_peak_hours(filtered), use_container_width=True)

            with st.expander("📋 Lihat tabel Peak Hours"):
                peak_df = pd.DataFrame(summary.get("peak_hours", []))
                if not peak_df.empty:
                    st.dataframe(peak_df, use_container_width=True, hide_index=True)
                else:
                    st.caption("Tabel jam puncak tidak tersedia.")

    else:
        col_main, col_side = st.columns([3, 1], gap="large")
        with col_main:
            render_section_heading("Data setelah pembersihan")
            display_cols = [
                c
                for c in [
                    "date_parsed",
                    "sku_label",
                    "channel_label",
                    "price_clean",
                    "qty_clean",
                    "discount_clean",
                    "gross_sales",
                    "platform_fee_rate",
                    "platform_fee_amount",
                    "net_sales",
                    "hour",
                    "order_key",
                ]
                if c in filtered.columns
            ]
            st.dataframe(filtered[display_cols], use_container_width=True, height=440)
        with col_side:
            render_section_heading("Export & Info")
            csv_bytes = filtered.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ Export CSV Bersih",
                data=csv_bytes,
                file_name="umkm_sales_cleaned.csv",
                mime="text/csv",
                type="primary",
                use_container_width=True,
            )
            st.caption(f"{meta.get('rows_in', 0)} baris masuk → {meta.get('rows_out', 0)} baris valid setelah dibersihkan.")
            render_json_details("Mapping kolom terdeteksi", meta.get("columns_mapped", {}))
else:
    st.markdown(
        """
<div style="background:linear-gradient(160deg,#1b2540,#141d30);border:1px solid #26314d;border-left:3px solid #14b8a6;
            border-radius:14px;padding:32px 36px;margin-top:0.5rem;box-shadow:0 8px 24px rgba(0,0,0,0.25);">
  <p style="color:#f1f5f9;font-size:1.1rem;font-weight:700;letter-spacing:-0.01em;margin:0 0 0.6rem 0;">
    Ubah data penjualan mentah jadi keputusan bisnis dalam &lt; 1 menit.
  </p>
  <p style="color:#94a3b8;font-size:0.9rem;line-height:1.65;margin:0;">
    Upload CSV/XLSX penjualan Anda (Excel kasir, laporan GoFood/Shopee, dll) di sidebar kiri —
    sistem otomatis membersihkan data, menghitung KPI, dan memberi rekomendasi AI yang bisa
    langsung dieksekusi besok. Belum punya file? Klik <b style="color:#14b8a6;">🎯 Coba dengan Data Contoh</b> di sidebar.
  </p>
</div>
        """,
        unsafe_allow_html=True,
    )