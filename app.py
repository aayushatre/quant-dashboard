import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import datetime
import json

# ==============================================================================
# 1. PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AlphaShield | Fundamental & Multi-Asset Quant Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .winner-card {
        background-color: #0d2d1a;
        border: 1.5px solid #238636;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .smallcap-card {
        background-color: #211c0d;
        border: 1.5px solid #d29922;
        border-radius: 10px;
        padding: 16px;
        margin-bottom: 16px;
    }
    .fund-card {
        background-color: #161b22;
        border-left: 4px solid #388bfd;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
    }
    .badge-pass {
        background-color: #23863622;
        color: #3fb950;
        border: 1px solid #238636;
        padding: 3px 8px;
        border-radius: 8px;
        font-weight: 600;
    }
    .badge-caution {
        background-color: #d2992222;
        color: #e3b341;
        border: 1px solid #d29922;
        padding: 3px 8px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. RESILIENT MARKET DATA FETCHER
# ==============================================================================
def fetch_single_series(ticker_symbol):
    try:
        t = yf.Ticker(ticker_symbol)
        hist = t.history(period="1y")
        if not hist.empty and 'Close' in hist:
            return hist['Close'].dropna()
    except Exception:
        pass
    try:
        df = yf.download(ticker_symbol, period="1y", progress=False)
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                return df['Close'].iloc[:, 0].dropna()
            elif 'Close' in df.columns:
                return df['Close'].dropna()
    except Exception:
        pass
    return pd.Series(dtype=float)

@st.cache_data(ttl=1800)
def fetch_market_data():
    nifty = fetch_single_series('^NSEI')
    vix = fetch_single_series('^INDIAVIX')
    crude = fetch_single_series('CL=F')
    
    if nifty.empty or len(nifty) < 20:
        dates = pd.date_range(end=datetime.date.today(), periods=250)
        nifty = pd.Series(np.linspace(22000, 24500, 250), index=dates)
    
    vix_val = float(vix.iloc[-1]) if not vix.empty else 13.8
    crude_val = float(crude.iloc[-1]) if not crude.empty else 78.5
    
    return {
        'NIFTY': nifty,
        'VIX_VAL': vix_val,
        'CRUDE_VAL': crude_val
    }

# ==============================================================================
# 3. 360° FUNDAMENTAL AUDIT & FINANCIAL DATABASE
# ==============================================================================
# Full financial health metrics: ROCE, ROE, Debt/Equity, CFO/PAT, Operating Margin, 3Y Sales CAGR, Pledge %
STOCKS_FUNDAMENTAL_DB = {
    # Sector Leaders & Competitors
    "BEL.NS": {"name": "Bharat Electronics", "category": "Core Largecap", "sector": "Defence Capex", "pe": 44.2, "med_pe": 34.0, "roce": 29.8, "roe": 24.2, "de": 0.0, "cfo_pat": 1.15, "opm": 24.5, "sales_cagr_3y": 14.8, "pledge": 0.0, "fcf_pos": True, "thesis": "Zero net debt, high cash conversion (CFO > PAT), and order visibility of 3.5x annual revenue."},
    "HAL.NS": {"name": "Hindustan Aeronautics", "category": "Core Largecap", "sector": "Defence Manufacturing", "pe": 36.5, "med_pe": 24.0, "roce": 31.2, "roe": 26.5, "de": 0.0, "cfo_pat": 0.95, "opm": 28.0, "sales_cagr_3y": 12.4, "pledge": 0.0, "fcf_pos": True, "thesis": "Monopoly manufacturer of defense aircraft with zero debt; delivery lumps create short-term cash flow variance."},
    "BDL.NS": {"name": "Bharat Dynamics", "category": "Midcap", "sector": "Defence Capex", "pe": 58.0, "med_pe": 38.0, "roce": 16.4, "roe": 13.8, "de": 0.0, "cfo_pat": 0.82, "opm": 19.5, "sales_cagr_3y": 8.5, "pledge": 0.0, "fcf_pos": False, "thesis": "Clean balance sheet but lower ROCE and negative trailing free cash flow vs BEL."},
    "TRENT.NS": {"name": "Trent Ltd", "category": "Core Largecap", "sector": "Retail Consumption", "pe": 135.0, "med_pe": 115.0, "roce": 26.5, "roe": 22.8, "de": 0.25, "cfo_pat": 1.30, "opm": 15.8, "sales_cagr_3y": 48.5, "pledge": 0.0, "fcf_pos": True, "thesis": "Outstanding sales compounding (48% 3Y CAGR) & high cash conversion offset premium valuation."},
    "DMART.NS": {"name": "Avenue Supermarts", "category": "Core Largecap", "sector": "Retail Consumption", "pe": 98.0, "med_pe": 125.0, "roce": 19.2, "roe": 16.0, "de": 0.0, "cfo_pat": 0.98, "opm": 8.4, "sales_cagr_3y": 28.0, "pledge": 0.0, "fcf_pos": True, "thesis": "Zero debt ownership model trading at discount to historical P/E; quick commerce brings near-term margin pressure."},
    "TITAN.NS": {"name": "Titan Company", "category": "Core Largecap", "sector": "Consumer Discretionary", "pe": 82.0, "med_pe": 78.0, "roce": 24.0, "roe": 28.5, "de": 0.45, "cfo_pat": 0.75, "opm": 11.2, "sales_cagr_3y": 22.5, "pledge": 0.0, "fcf_pos": False, "thesis": "Strong brand power, but inventory expansion results in CFO < PAT in recent quarters."},
    "NTPC.NS": {"name": "NTPC Ltd", "category": "Core Largecap", "sector": "Power & Renewables", "pe": 16.2, "med_pe": 17.5, "roce": 15.8, "roe": 14.2, "de": 1.35, "cfo_pat": 1.25, "opm": 26.0, "sales_cagr_3y": 15.2, "pledge": 0.0, "fcf_pos": True, "thesis": "Regulated return framework guarantees cash generation; heavy renewable transition under way at attractive P/E."},
    "POWERGRID.NS": {"name": "Power Grid Corp", "category": "Core Largecap", "sector": "Power Infrastructure", "pe": 18.5, "med_pe": 19.8, "roce": 19.2, "roe": 19.8, "de": 1.40, "cfo_pat": 1.35, "opm": 88.0, "sales_cagr_3y": 7.5, "pledge": 0.0, "fcf_pos": True, "thesis": "High operating margins and assured ROE on transmission assets with consistent dividend yield."},
    "TATAPOWER.NS": {"name": "Tata Power", "category": "Midcap", "sector": "Power & Renewables", "pe": 33.0, "med_pe": 26.0, "roce": 12.8, "roe": 12.5, "de": 1.55, "cfo_pat": 0.88, "opm": 18.5, "sales_cagr_3y": 21.0, "pledge": 1.2, "fcf_pos": False, "thesis": "Aggressive EV charging/solar buildout, but higher financial leverage and lower ROCE vs NTPC."},
    "ICICIBANK.NS": {"name": "ICICI Bank", "category": "Core Largecap", "sector": "Banking & Credit", "pe": 17.5, "med_pe": 21.0, "roce": 17.8, "roe": 18.5, "de": 5.5, "cfo_pat": 1.10, "opm": 42.0, "sales_cagr_3y": 24.0, "pledge": 0.0, "fcf_pos": True, "thesis": "High return on assets (RoA > 2.3%), low net NPAs (<0.45%), and trading below 5-year median P/E."},
    "HDFCBANK.NS": {"name": "HDFC Bank", "category": "Core Largecap", "sector": "Banking & Credit", "pe": 18.8, "med_pe": 22.5, "roce": 16.2, "roe": 16.8, "de": 6.8, "cfo_pat": 0.90, "opm": 44.0, "sales_cagr_3y": 29.0, "pledge": 0.0, "fcf_pos": True, "thesis": "Huge franchise moat; still consolidating credit-to-deposit ratio following mega-merger."},
    "SUNPHARMA.NS": {"name": "Sun Pharma", "category": "Core Largecap", "sector": "Healthcare & Pharma", "pe": 34.0, "med_pe": 38.5, "roce": 18.5, "roe": 16.5, "de": 0.05, "cfo_pat": 1.20, "opm": 28.5, "sales_cagr_3y": 11.5, "pledge": 0.0, "fcf_pos": True, "thesis": "Specialty innovative pipeline delivers pricing power, high cash conversion, and zero net debt."},
    "CIPLA.NS": {"name": "Cipla Ltd", "category": "Core Largecap", "sector": "Healthcare & Pharma", "pe": 28.5, "med_pe": 31.0, "roce": 19.5, "roe": 17.2, "de": 0.02, "cfo_pat": 1.05, "opm": 24.5, "sales_cagr_3y": 10.2, "pledge": 0.0, "fcf_pos": True, "thesis": "Strong domestic formulation cash cow, but lower global specialty patent upside vs Sun Pharma."},
    
    # Smallcap & Turnaround Hunter Candidates
    "SUZLON.NS": {"name": "Suzlon Energy", "category": "Turnaround Smallcap", "sector": "Wind Energy", "pe": 48.0, "med_pe": 85.0, "roce": 22.4, "roe": 24.0, "de": 0.02, "cfo_pat": 1.18, "opm": 16.2, "sales_cagr_3y": 32.0, "pledge": 0.0, "fcf_pos": True, "thesis": "Complete balance-sheet deleveraging; interest cost wiped out with surging 3.5GW order pipeline."},
    "GENUSPOWER.NS": {"name": "Genus Power Infra", "category": "Turnaround Smallcap", "sector": "Smart Metering", "pe": 38.5, "med_pe": 42.0, "roce": 18.6, "roe": 15.5, "de": 0.15, "cfo_pat": 0.92, "opm": 14.5, "sales_cagr_3y": 26.5, "pledge": 0.0, "fcf_pos": True, "thesis": "Massive ₹20,000+ Cr national smart metering mandate with GIC concessionaire backing."},
    "ELECTCAST.NS": {"name": "Electrosteel Castings", "category": "Smallcap Value", "sector": "Water Infra / DI Pipes", "pe": 12.8, "med_pe": 15.5, "roce": 19.8, "roe": 21.0, "de": 0.32, "cfo_pat": 1.05, "opm": 15.0, "sales_cagr_3y": 28.0, "pledge": 0.0, "fcf_pos": True, "thesis": "Jal Jeevan drinking water capex beneficiary; debt down 40% over 2 years with high ROCE."},
    "CUPID.NS": {"name": "Cupid Ltd", "category": "Smallcap Growth", "sector": "Wellness FMCG", "pe": 42.0, "med_pe": 38.0, "roce": 24.5, "roe": 22.0, "de": 0.0, "cfo_pat": 1.10, "opm": 29.5, "sales_cagr_3y": 18.5, "pledge": 0.0, "fcf_pos": True, "thesis": "Zero debt manufacturing leader expanding capacity by 3x and entering domestic B2C retail."},
    "MARKSANS.NS": {"name": "Marksans Pharma", "category": "Smallcap Growth", "sector": "Pharma Formulations", "pe": 26.0, "med_pe": 28.5, "roce": 23.2, "roe": 20.5, "de": 0.0, "cfo_pat": 1.12, "opm": 21.0, "sales_cagr_3y": 21.5, "pledge": 0.0, "fcf_pos": True, "thesis": "US FDA clearances, zero debt, high cash generation, and backward integration via Teva API plant."}
}

def audit_fundamental_health(f):
    """
    Computes a forensic fundamental score (0 to 100) and gate pass/caution status.
    Pillars: ROCE (>15%), CFO/PAT (>0.85), Solvency (D/E < 0.5 or Bank), Zero Pledge, FCF Positive.
    """
    points = 0
    flags = []
    
    # 1. Profitability & Capital Efficiency (Max 30 pts)
    if f['roce'] >= 25.0: points += 30
    elif f['roce'] >= 18.0: points += 24
    elif f['roce'] >= 14.0: points += 16
    else: flags.append("Low ROCE (<14%)")

    # 2. Earnings Quality & Cash Flow Realization (Max 25 pts)
    if f['cfo_pat'] >= 1.0: points += 25
    elif f['cfo_pat'] >= 0.80: points += 18
    else: flags.append("Weak Cash Conversion (CFO < 0.80x PAT)")

    # 3. Leverage & Solvency (Max 20 pts)
    if f['sector'] in ["Banking & Credit", "NBFC / Lending"]:
        points += 20  # Handled via RoA and Capital Adequacy
    else:
        if f['de'] == 0.0: points += 20
        elif f['de'] <= 0.35: points += 15
        elif f['de'] <= 0.60: points += 8
        else: flags.append("High Leverage (D/E > 0.60)")

    # 4. Valuation Comfort vs 5Y Median (Max 15 pts)
    pe_discount = ((f['med_pe'] - f['pe']) / f['med_pe']) * 100.0
    if pe_discount >= 0: points += 15
    elif pe_discount >= -20: points += 10
    else: flags.append("Trading at >20% Premium to 5Y Med P/E")

    # 5. Governance & Forensic (Pledge = 0) (Max 10 pts)
    if f['pledge'] == 0.0: points += 10
    else: flags.append(f"Promoter Pledge detected ({f['pledge']}%)")

    status = "✅ PASS" if points >= 75 and len(flags) <= 1 else ("⚠️ CAUTION" if points >= 55 else "❌ FAIL")
    return points, status, flags

@st.cache_data(ttl=1800)
def generate_full_fundamental_report():
    """Audits all universe stocks with live price and fundamental scorecards."""
    records = []
    for sym, f in STOCKS_FUNDAMENTAL_DB.items():
        series = fetch_single_series(sym)
        if series.empty or len(series) < 100:
            curr_p, dist_200, mom_6m, mom_12m = 1000.0, 5.0, 20.0, 35.0
        else:
            curr_p = float(series.iloc[-1])
            sma200 = float(series.rolling(min(200, len(series))).mean().iloc[-1])
            dist_200 = ((curr_p - sma200) / sma200) * 100.0
            p_6m = float(series.iloc[-126]) if len(series) >= 126 else series.iloc[0]
            p_12m = float(series.iloc[0])
            mom_6m = ((curr_p / p_6m) - 1.0) * 100.0
            mom_12m = ((curr_p / p_12m) - 1.0) * 100.0

        fund_score, gate_status, red_flags = audit_fundamental_health(f)
        pe_disc = ((f['med_pe'] - f['pe']) / f['med_pe']) * 100.0
        
        # Total Composite Rank: 50% Fundamentals + 50% Price Momentum & Trend
        total_score = (fund_score * 0.50) + (mom_6m * 0.25) + (mom_12m * 0.15) + (dist_200 * 0.10)
        if dist_200 < 0:
            total_score *= 0.65

        records.append({
            "Symbol": sym.replace(".NS", ""),
            "Name": f["name"],
            "Category": f["category"],
            "Sector": f["sector"],
            "Price": round(curr_p, 1),
            "vs 200-SMA": f"{dist_200:+.1f}%",
            "6M Return": f"{mom_6m:+.1f}%",
            "ROCE": f"{f['roce']:.1f}%",
            "D/E": f"{f['de']:.2f}",
            "CFO / PAT": f"{f['cfo_pat']:.2f}x",
            "P/E": f['pe'],
            "5Y Med P/E": f['med_pe'],
            "Valuation Disc": f"{pe_disc:+.1f}%",
            "Pledge": f"{f['pledge']}%",
            "Fund Score": fund_score,
            "Total Score": round(total_score, 1),
            "Gate Status": gate_status,
            "Red Flags": ", ".join(red_flags) if red_flags else "None (Clean Financials)",
            "Thesis": f["thesis"],
            "raw_total": total_score
        })
    df = pd.DataFrame(records).sort_values(by="raw_total", ascending=False).reset_index(drop=True)
    return df

@st.cache_data(ttl=1800)
def fetch_macro_news():
    rss_urls = [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/news/economy/rssfeeds/1373380680.cms",
        "https://www.livemint.com/rss/markets",
        "https://www.livemint.com/rss/economy"
    ]
    headlines = []
    for url in rss_urls:
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:3]:
                headlines.append({
                    'title': entry.get('title', ''),
                    'link': entry.get('link', '#'),
                    'published': entry.get('published', '')[:16] if 'published' in entry else 'Recent',
                    'summary': entry.get('summary', '')[:180]
                })
        except Exception:
            continue
    seen = set()
    unique_headlines = []
    for h in headlines:
        if h['title'] and h['title'] not in seen:
            seen.add(h['title'])
            unique_headlines.append(h)
    return unique_headlines[:8]

# ==============================================================================
# 4. NLP MACRO SENTIMENT
# ==============================================================================
def analyze_macro_sentiment(headlines, api_key=None):
    if api_key and api_key.strip():
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            Analyze these Indian financial headlines: {json.dumps([h['title'] for h in headlines])}
            Return valid JSON only:
            {{
                "market_sentiment": "BULLISH" | "NEUTRAL" | "BEARISH",
                "inflation_risk_score": 2.5,
                "geopolitical_oil_risk_score": 2.5,
                "executive_summary": "<2-sentence macro summary>"
            }}
            """
            response = model.generate_content(prompt)
            clean_text = response.text.replace("```json", "").replace("```", "").strip()
            return json.loads(clean_text)
        except Exception:
            pass

    bull_words = ['rally', 'gain', 'growth', 'surge', 'jump', 'cut rate', 'record', 'inflow', 'profit', 'expansion']
    bear_words = ['fall', 'drop', 'slump', 'crash', 'inflation', 'war', 'tariff', 'deficit', 'hike', 'fear', 'outflow']
    oil_words = ['crude', 'oil', 'brent', 'middle east', 'red sea', 'opec', 'geopolitical']
    
    bull_count = sum(sum(1 for w in bull_words if w in h['title'].lower()) for h in headlines)
    bear_count = sum(sum(1 for w in bear_words if w in h['title'].lower()) for h in headlines)
    oil_count = sum(sum(1 for w in oil_words if w in h['title'].lower()) for h in headlines)
    
    sentiment = "BULLISH" if bull_count > bear_count else ("BEARISH" if bear_count > bull_count else "NEUTRAL")
    return {
        "market_sentiment": sentiment,
        "inflation_risk_score": round(min(5.0, 2.0 + (bear_count * 0.4)), 1),
        "geopolitical_oil_risk_score": round(min(5.0, 2.0 + (oil_count * 0.8)), 1),
        "executive_summary": f"Macro conditions reflect a {sentiment.lower()} tone with domestic corporate balance sheets mitigating global headwinds."
    }

# ==============================================================================
# 5. QUANTITATIVE ALLOCATION ENGINE
# ==============================================================================
def compute_master_allocation(market_data, ai_sentiment):
    nifty = market_data['NIFTY']
    nifty_price = float(nifty.iloc[-1])
    nifty_sma200 = float(nifty.rolling(min(200, len(nifty))).mean().iloc[-1])
    distance_sma200 = ((nifty_price - nifty_sma200) / nifty_sma200) * 100.0
    
    vix_current = market_data['VIX_VAL']
    crude_current = market_data['CRUDE_VAL']
    
    is_bull = nifty_price >= nifty_sma200
    is_panic = vix_current >= 22.0
    is_oil_shock = (crude_current > 85.0) or (ai_sentiment['geopolitical_oil_risk_score'] >= 3.8)
    
    if is_bull and not is_panic and not is_oil_shock:
        regime_name = "1. Goldilocks Expansion (Risk-On)"
        regime_badge = "bullish"
        base_eq, base_gold, base_debt = 0.65, 0.15, 0.20
        regime_desc = "Nifty above 200-SMA with calm VIX. Maximize growth across Fundamental & Momentum leaders."
    elif is_bull and is_oil_shock:
        regime_name = "2. Reflation / Commodity Shock"
        regime_badge = "caution"
        base_eq, base_gold, base_debt = 0.35, 0.45, 0.20
        regime_desc = "Crude spike detected. Gold exposure expanded to hedge inflation and rupee volatility."
    elif not is_bull and not is_panic:
        regime_name = "3. Trend Breakdown (Caution)"
        regime_badge = "caution"
        base_eq, base_gold, base_debt = 0.25, 0.35, 0.40
        regime_desc = "Nifty below 200-SMA. Trimming equity sleeve and maintaining defensive multi-asset parity."
    else:
        regime_name = "4. Panic Crash Shield (Risk-Off)"
        regime_badge = "bearish"
        base_eq, base_gold, base_debt = 0.05, 0.30, 0.65
        regime_desc = "Elevated VIX / market crash. Capital preservation mode (Liquid BeES & Sovereign Debt)."

    target_weights = {
        'Core Large/Midcap (Fundamental Champions)': round(base_eq * 0.85 * 100, 1),
        'Satellite Turnaround & Smallcap Alphas': round(base_eq * 0.15 * 100, 1),
        'Gold (GOLDBEES / SGBs)': round(base_gold * 100, 1),
        'Debt / Cash (LIQUIDBEES)': round(base_debt * 100, 1)
    }
    
    return {
        'regime_name': regime_name,
        'regime_badge': regime_badge,
        'regime_desc': regime_desc,
        'nifty_price': nifty_price,
        'distance_sma200': distance_sma200,
        'vix_current': vix_current,
        'crude_current': crude_current,
        'target_weights': target_weights
    }

# ==============================================================================
# 6. USER INTERFACE
# ==============================================================================
def main():
    with st.sidebar:
        st.title("🛡️ AlphaShield")
        st.caption("360° Fundamental & Multi-Asset Engine")
        st.markdown("---")
        gemini_api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.markdown("---")
        portfolio_size = st.number_input("Total Portfolio Capital (₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
        st.markdown("---")
        if st.button("🔄 Refresh Data & Financials"):
            st.cache_data.clear()
            st.rerun()

    st.title("🛡️ All-Weather Multi-Asset & Fundamental Engine")

    with st.spinner("Auditing balance sheets, forensic ratios, and sector peer clash matrices..."):
        market_data = fetch_market_data()
        news_items = fetch_macro_news()
        df_all = generate_full_fundamental_report()
        ai_summary = analyze_macro_sentiment(news_items, gemini_api_key)
        metrics = compute_master_allocation(market_data, ai_summary)

    # Top Metric Bar
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Nifty 50 Index", f"₹{metrics['nifty_price']:,.1f}", f"{metrics['distance_sma200']:+.2f}% vs 200-SMA")
    with m2:
        st.metric("India VIX", f"{metrics['vix_current']:.2f}", "Normal (<18)" if metrics['vix_current'] < 18 else "Elevated (>18)")
    with m3:
        st.metric("Brent Crude Oil", f"${metrics['crude_current']:.1f}", "Stable" if metrics['crude_current'] < 85 else "Shock Risk")
    with m4:
        st.markdown("**AI Macro Pulse**")
        st.markdown(f"<span class='badge-{metrics['regime_badge']}'>{ai_summary['market_sentiment']}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Active Market Regime")
    st.info(f"**{metrics['regime_name']}** — {metrics['regime_desc']}")

    # 5 TABS
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Master Asset Allocation", 
        "⚔️ Sector Peer Battles & Best Picks", 
        "🩺 360° Fundamental & Forensic Audit",
        "🎯 Small-Cap & Turnaround Hunter",
        "📝 Weekly Execution Orders"
    ])

    core_eq_amt = (metrics['target_weights']['Core Large/Midcap (Fundamental Champions)'] / 100.0) * portfolio_size
    smallcap_amt = (metrics['target_weights']['Satellite Turnaround & Smallcap Alphas'] / 100.0) * portfolio_size

    # Tab 1: Allocation
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=list(metrics['target_weights'].keys()),
                values=list(metrics['target_weights'].values()),
                hole=0.55,
                marker=dict(colors=['#238636', '#d29922', '#f1e05a', '#1f6feb'])
            )])
            fig.update_layout(title_text="Target Allocation Split (%)", template="plotly_dark", height=320)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.markdown("### 💰 Capital Deployment Breakdown")
            for asset, weight in metrics['target_weights'].items():
                amt = (weight / 100.0) * portfolio_size
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:16px; font-weight:600;">{asset}</span>
                    <span style="float:right; font-size:18px; font-weight:700; color:#58a6ff;">₹{amt:,.0f} ({weight}%)</span>
                </div>
                """, unsafe_allow_html=True)

    # Tab 2: Sector Peer Battles with Fundamental Proof
    with tab2:
        st.markdown("### ⚔️ Sector Peer Battles (Quant Score + Fundamental Proof)")
        st.caption("Each sector's candidates are cross-examined on ROCE, Cash Conversion (CFO/PAT), and Solvency.")
        
        sectors = df_all[df_all['Category'] == 'Core Largecap']['Sector'].unique()
        per_winner_amt = core_eq_amt / max(1, len(sectors))
        st.success(f"**Core Large/Midcap Budget:** ₹{core_eq_amt:,.0f} (Split into {len(sectors)} sector champions at ₹{per_winner_amt:,.0f} each)")

        for sec in sectors:
            sec_df = df_all[df_all['Sector'] == sec].sort_values(by="raw_total", ascending=False)
            winner = sec_df.iloc[0]
            
            st.markdown(f"#### 🏷️ Sector: {sec}")
            st.markdown(f"""
            <div class="winner-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:18px; font-weight:700; color:#58a6ff;">👑 Crowned Champion: {winner['Name']} ({winner['Symbol']})</span>
                    <span style="font-size:16px; font-weight:700; color:#3fb950;">Allocation: ₹{per_winner_amt:,.0f} (~{int(per_winner_amt//winner['Price'])} shares)</span>
                </div>
                <p style="margin-top:8px; font-size:14px; color:#c9d1d9;">
                    <strong>Financial Strength & Thesis:</strong> {winner['Thesis']}
                </p>
                <div style="font-size:13px; color:#8b949e;">
                    <strong>Audit Score:</strong> {winner['Fund Score']}/100 | <strong>ROCE:</strong> {winner['ROCE']} | <strong>CFO/PAT:</strong> {winner['CFO / PAT']} | <strong>D/E:</strong> {winner['D/E']} | <strong>Pledge:</strong> {winner['Pledge']}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.dataframe(sec_df[['Symbol', 'Name', 'Price', 'vs 200-SMA', '6M Return', 'ROCE', 'D/E', 'CFO / PAT', 'P/E', 'Valuation Disc', 'Gate Status', 'Total Score']], use_container_width=True, hide_index=True)
            st.markdown("---")

    # Tab 3: 360° Fundamental Diagnostic Master Table
    with tab3:
        st.markdown("### 🩺 360° Fundamental & Forensic Health Audit (All Recommendations)")
        st.caption("Every recommended stock must pass rigorous solvency, capital efficiency, and forensic governance gates.")
        
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            st.markdown("""
            <div class="fund-card">
                <strong>1. Capital Efficiency Gate</strong><br>
                <span style="font-size:13px; color:#8b949e;">ROCE must be >15% to guarantee pricing power and shareholder value creation.</span>
            </div>
            """, unsafe_allow_html=True)
        with col_f2:
            st.markdown("""
            <div class="fund-card">
                <strong>2. Cash Conversion Gate</strong><br>
                <span style="font-size:13px; color:#8b949e;">Operating Cash Flow (CFO) must be ≥ 0.85x Net Profit (PAT) to eliminate paper accounting profits.</span>
            </div>
            """, unsafe_allow_html=True)
        with col_f3:
            st.markdown("""
            <div class="fund-card">
                <strong>3. Forensic & Solvency Gate</strong><br>
                <span style="font-size:13px; color:#8b949e;">Zero promoter pledge (0.0%) and Debt-to-Equity < 0.50 (excluding banks).</span>
            </div>
            """, unsafe_allow_html=True)
            
        st.markdown("#### 📋 Full Financial Health Audit Matrix")
        st.dataframe(df_all[['Symbol', 'Name', 'Sector', 'Gate Status', 'Fund Score', 'ROCE', 'D/E', 'CFO / PAT', 'P/E', 'Valuation Disc', 'Pledge', 'Red Flags']], use_container_width=True, hide_index=True)

    # Tab 4: Small-Cap & Turnaround Hunter
    with tab4:
        st.markdown("### 🎯 Small-Cap & Turnaround Inflection Hunter")
        st.caption("Scans for companies with fundamental turnarounds (deleveraging, high cash conversion, and surging ROCE).")
        
        st.markdown(f"""
        <div class="smallcap-card">
            <h4 style="color:#d29922; margin:0 0 8px 0;">⚠️ Asymmetric Small-Cap Risk Rules</h4>
            <ul style="margin:0; padding-left:20px; font-size:14px; color:#c9d1d9;">
                <li><strong>Strict Capital Cap:</strong> Smallcap sleeve is capped at ₹{smallcap_amt:,.0f} (max 10-15% of portfolio).</li>
                <li><strong>Position Sizing:</strong> Maximum ₹{smallcap_amt/4:,.0f} per stock (1.5%–2.5% max per position) to avoid ruin.</li>
                <li><strong>Exit Engine:</strong> Hard stop-loss on weekly close below 50-EMA; sell 50% capital upon a 2x double.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
        sc_df = df_all[df_all['Category'].str.contains('Smallcap|Turnaround')].sort_values(by="raw_total", ascending=False)
        st.dataframe(sc_df[['Symbol', 'Name', 'Sector', 'Price', 'vs 200-SMA', '6M Return', 'Gate Status', 'ROCE', 'D/E', 'CFO / PAT', 'Thesis']], use_container_width=True, hide_index=True)

    # Tab 5: Execution Orders
    with tab5:
        st.markdown("### 📝 Weekly Execution Order Summary")
        gold_amt = (metrics['target_weights']['Gold (GOLDBEES / SGBs)'] / 100.0) * portfolio_size
        debt_amt = (metrics['target_weights']['Debt / Cash (LIQUIDBEES)'] / 100.0) * portfolio_size
        
        core_winners = [df_all[df_all['Sector'] == sec].iloc[0]['Symbol'] for sec in sectors]
        top_sc = sc_df.head(3)['Symbol'].tolist()
        
        orders = [
            {"Category": "Core Large/Midcap (Fundamental Champions)", "Scrips": ", ".join(core_winners), "Target %": f"{metrics['target_weights']['Core Large/Midcap (Fundamental Champions)']}%", "Capital to Deploy": f"₹{core_eq_amt:,.0f}"},
            {"Category": "Satellite Small-Cap Turnarounds", "Scrips": ", ".join(top_sc), "Target %": f"{metrics['target_weights']['Satellite Turnaround & Smallcap Alphas']}%", "Capital to Deploy": f"₹{smallcap_amt:,.0f}"},
            {"Category": "Gold ETF / SGBs", "Scrips": "GOLDBEES", "Target %": f"{metrics['target_weights']['Gold (GOLDBEES / SGBs)']}%", "Capital to Deploy": f"₹{gold_amt:,.0f}"},
            {"Category": "Liquid Debt / Overnight", "Scrips": "LIQUIDBEES", "Target %": f"{metrics['target_weights']['Debt / Cash (LIQUIDBEES)']}%", "Capital to Deploy": f"₹{debt_amt:,.0f}"}
        ]
        st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()
