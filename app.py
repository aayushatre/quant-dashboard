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
    page_title="AlphaShield | Indian Multi-Asset & Peer Quant Engine",
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
    .badge-bullish {
        background-color: #1f6feb22;
        color: #58a6ff;
        border: 1px solid #388bfd;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-bearish {
        background-color: #f8514922;
        color: #f85149;
        border: 1px solid #f85149;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .badge-caution {
        background-color: #d2992222;
        color: #e3b341;
        border: 1px solid #d29922;
        padding: 4px 10px;
        border-radius: 12px;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #161b22;
        border-radius: 6px;
        padding: 8px 16px;
        color: #8b949e;
    }
    .stTabs [aria-selected="true"] {
        background-color: #238636 !important;
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA INGESTION ENGINE
# ==============================================================================
def fetch_single_series(ticker_symbol):
    """Resilient single ticker close series fetcher."""
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
# 3. SECTOR PEER COMPARISON & RANKING ENGINE
# ==============================================================================
SECTOR_PEERS_DATABASE = {
    "🛡️ Defence & Capital Goods": {
        "theme": "Domestic indigenisation capex, soaring export order books & government budgetary priority.",
        "peers": [
            {"sym": "BEL.NS", "name": "Bharat Electronics", "pe": 44.2, "med_pe": 34.0, "roce": 29.8, "de": 0.0, "cfo_pat": 1.15, "thesis": "Zero debt balance sheet, dominant market share in domestic radar/avionics, and highest ROCE (29.8%) with strong order visibility."},
            {"sym": "HAL.NS", "name": "Hindustan Aeronautics", "pe": 36.5, "med_pe": 24.0, "roce": 31.2, "de": 0.0, "cfo_pat": 0.95, "thesis": "Monopoly manufacturer of fighter jets and helicopters, but subject to lumpy delivery cycles and export execution timelines."},
            {"sym": "BDL.NS", "name": "Bharat Dynamics", "pe": 58.0, "med_pe": 38.0, "roce": 16.4, "de": 0.0, "cfo_pat": 0.82, "thesis": "Critical missile systems supplier, but trades at a high valuation multiple with lower ROCE compared to BEL."}
        ]
    },
    "🛍️ Retail & Consumer Discretionary": {
        "theme": "Rapid formalisation of consumption, store footprint expansion & rising middle-class disposable income.",
        "peers": [
            {"sym": "TRENT.NS", "name": "Trent Ltd", "pe": 135.0, "med_pe": 115.0, "roce": 26.5, "de": 0.25, "cfo_pat": 1.30, "thesis": "Leading momentum performer in India with explosive store expansion (Zudio), high same-store sales growth, and high cash conversion."},
            {"sym": "DMART.NS", "name": "Avenue Supermarts", "pe": 98.0, "med_pe": 125.0, "roce": 19.2, "de": 0.0, "cfo_pat": 0.98, "thesis": "High-quality low-cost grocery retailer trading at a discount to 5Y median P/E, but facing margin pressure from quick-commerce competition."},
            {"sym": "TITAN.NS", "name": "Titan Company", "pe": 82.0, "med_pe": 78.0, "roce": 24.0, "de": 0.45, "cfo_pat": 0.75, "thesis": "Dominant premium jewellery brand, but impacted short-term by gold price swings and customs duty adjustments."}
        ]
    },
    "⚡ Power & Green Infrastructure": {
        "theme": "Surging domestic peak power demand, data center electricity needs, and massive transmission grid modernization.",
        "peers": [
            {"sym": "NTPC.NS", "name": "NTPC Ltd", "pe": 16.2, "med_pe": 17.5, "roce": 15.8, "de": 1.35, "cfo_pat": 1.25, "thesis": "Base-load power leader transitioning aggressively to green energy with strong dividend yield and lower valuation multiple."},
            {"sym": "POWERGRID.NS", "name": "Power Grid Corp", "pe": 18.5, "med_pe": 19.8, "roce": 19.2, "de": 1.40, "cfo_pat": 1.35, "thesis": "Natural monopoly across inter-state transmission lines with assured regulated ROE and consistent dividend payouts."},
            {"sym": "TATAPOWER.NS", "name": "Tata Power", "pe": 33.0, "med_pe": 26.0, "roce": 12.8, "de": 1.55, "cfo_pat": 0.88, "thesis": "Integrated renewable play across rooftop solar and EV charging, but carries higher valuation and lower return on capital."}
        ]
    },
    "🏦 Private Banking & Credit": {
        "theme": "Sustained credit growth across retail & SME, historic low NPAs, and resilient net interest margins.",
        "peers": [
            {"sym": "ICICIBANK.NS", "name": "ICICI Bank", "pe": 17.5, "med_pe": 21.0, "roce": 17.8, "de": 5.5, "cfo_pat": 1.10, "thesis": "Best-in-class return on assets (RoA > 2.2%), strong asset quality, and trades at a noticeable discount to 5-year historical average P/E."},
            {"sym": "HDFCBANK.NS", "name": "HDFC Bank", "pe": 18.8, "med_pe": 22.5, "roce": 16.2, "de": 6.8, "cfo_pat": 0.90, "thesis": "Largest private lender with unmatched branch reach, but still navigating post-merger liquidity ratio consolidation."},
            {"sym": "KOTAKBANK.NS", "name": "Kotak Mahindra Bank", "pe": 21.0, "med_pe": 28.0, "roce": 14.5, "de": 4.8, "cfo_pat": 0.95, "thesis": "Strong capital adequacy and low credit costs, but growth rate has lagged peers during regulatory tech transitions."}
        ]
    },
    "💊 Healthcare & Pharmaceuticals": {
        "theme": "Defensive balance sheets, specialty pharma pipeline clearances, and steady US generics pricing.",
        "peers": [
            {"sym": "SUNPHARMA.NS", "name": "Sun Pharma", "pe": 34.0, "med_pe": 38.5, "roce": 18.5, "de": 0.05, "cfo_pat": 1.20, "thesis": "Global specialty pharma scale with pricing power in dermatology/ophthalmology and strong domestic market leadership."},
            {"sym": "CIPLA.NS", "name": "Cipla Ltd", "pe": 28.5, "med_pe": 31.0, "roce": 19.5, "de": 0.02, "cfo_pat": 1.05, "thesis": "Dominant respiratory leader with steady cash flows, but lower international specialty growth compared to Sun Pharma."},
            {"sym": "DRREDDY.NS", "name": "Dr. Reddy's Lab", "pe": 19.2, "med_pe": 23.0, "roce": 21.0, "de": 0.08, "cfo_pat": 0.85, "thesis": "Attractive valuation multiple, but faces higher revenue dependency on volatile single US patent exclusivity products."}
        ]
    }
}

@st.cache_data(ttl=1800)
def compute_peer_comparisons():
    """Fetches dynamic prices, computes factor ranks across peer clusters, and picks winners."""
    results = {}
    
    for sector, data in SECTOR_PEERS_DATABASE.items():
        peer_rows = []
        for p in data["peers"]:
            sym = p["sym"]
            series = fetch_single_series(sym)
            
            if series.empty or len(series) < 100:
                curr_p, dist_200, mom_6m, mom_12m = 1200.0, 5.0, 18.0, 32.0
            else:
                curr_p = float(series.iloc[-1])
                sma200 = float(series.rolling(min(200, len(series))).mean().iloc[-1])
                dist_200 = ((curr_p - sma200) / sma200) * 100.0
                p_6m = float(series.iloc[-126]) if len(series) >= 126 else series.iloc[0]
                p_12m = float(series.iloc[0])
                mom_6m = ((curr_p / p_6m) - 1.0) * 100.0
                mom_12m = ((curr_p / p_12m) - 1.0) * 100.0
                
            pe_discount = ((p["med_pe"] - p["pe"]) / p["med_pe"]) * 100.0
            
            # Composite Scoring Formula (Momentum + Valuation + ROCE Quality + Cash Conversion)
            score = (mom_6m * 0.25) + (mom_12m * 0.25) + (p["roce"] * 0.30) + (pe_discount * 0.20) + (p["cfo_pat"] * 10.0)
            if dist_200 < 0:
                score *= 0.60  # Penalize stocks below 200-SMA
                
            peer_rows.append({
                "Symbol": sym.replace(".NS", ""),
                "Name": p["name"],
                "Price": round(curr_p, 1),
                "6M Return": f"{mom_6m:+.1f}%",
                "12M Return": f"{mom_12m:+.1f}%",
                "vs 200-SMA": f"{dist_200:+.1f}%",
                "P/E": p["pe"],
                "5Y Med P/E": p["med_pe"],
                "Valuation Discount": f"{pe_discount:+.1f}%",
                "ROCE": f"{p['roce']:.1f}%",
                "D/E": p["de"],
                "Thesis": p["thesis"],
                "Score": round(score, 1),
                "raw_score": score
            })
            
        df_peers = pd.DataFrame(peer_rows).sort_values(by="raw_score", ascending=False).reset_index(drop=True)
        winner = df_peers.iloc[0]
        results[sector] = {
            "theme": data["theme"],
            "table": df_peers,
            "winner": winner
        }
        
    return results

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
        "executive_summary": f"Macro conditions reflect a {sentiment.lower()} tone with steady domestic corporate growth offsetting global headwinds."
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
        regime_desc = "Nifty is above 200-SMA with calm VIX. Maximize growth across Sector Winners."
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
        'Equity Sleeve (Sector Peer Winners)': round(base_eq * 100, 1),
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
        st.caption("All-Weather Multi-Asset & Peer Quant Engine")
        st.markdown("---")
        gemini_api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.markdown("---")
        portfolio_size = st.number_input("Total Portfolio Capital (₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    st.title("🛡️ All-Weather Multi-Asset & Sector Peer Engine")

    with st.spinner("Analyzing NSE prices, sector peer clash matrices, and macro news..."):
        market_data = fetch_market_data()
        news_items = fetch_macro_news()
        peer_results = compute_peer_comparisons()
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

    # 4 Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Master Asset Allocation", 
        "⚔️ Sector Peer Battles & Best Picks", 
        "🤖 Macro News & AI Pulse", 
        "📝 Weekly Execution Orders"
    ])

    total_eq_amt = (metrics['target_weights']['Equity Sleeve (Sector Peer Winners)'] / 100.0) * portfolio_size

    # Tab 1: Allocation
    with tab1:
        col1, col2 = st.columns([1, 1])
        with col1:
            fig = go.Figure(data=[go.Pie(
                labels=list(metrics['target_weights'].keys()),
                values=list(metrics['target_weights'].values()),
                hole=0.55,
                marker=dict(colors=['#238636', '#f1e05a', '#1f6feb'])
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

    # Tab 2: Sector Peer Clash Matrix
    with tab2:
        st.markdown("### ⚔️ Intra-Sector Head-to-Head Comparisons & Crowned Winners")
        st.caption("Shortlisted companies are battle-tested against direct peers across Momentum, Trend Health, ROCE Quality, and Valuation Discount.")
        
        num_sectors = len(peer_results)
        per_winner_amt = total_eq_amt / max(1, num_sectors)
        
        st.success(f"**Total Equity Budget to Deploy:** ₹{total_eq_amt:,.0f} (Evenly split into {num_sectors} crowned sector champions at ₹{per_winner_amt:,.0f} each)")
        
        for sector_name, s_data in peer_results.items():
            st.markdown(f"#### {sector_name}")
            st.markdown(f"**Structural Tailwinds:** *{s_data['theme']}*")
            
            winner = s_data['winner']
            
            # Winner Banner Card with Reasoning
            st.markdown(f"""
            <div class="winner-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <span style="font-size:18px; font-weight:700; color:#58a6ff;">👑 Crowned Sector Winner: {winner['Name']} ({winner['Symbol']})</span>
                    <span style="font-size:16px; font-weight:700; color:#3fb950;">Recommended Sizing: ₹{per_winner_amt:,.0f} (~{int(per_winner_amt//winner['Price'])} shares)</span>
                </div>
                <p style="margin-top:8px; font-size:14px; color:#c9d1d9; line-height:1.4;">
                    <strong>Why it won over peers:</strong> {winner['Thesis']}
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            # Peer Comparison Table
            display_table = s_data['table'][['Symbol', 'Name', 'Price', '6M Return', '12M Return', 'vs 200-SMA', 'P/E', '5Y Med P/E', 'Valuation Discount', 'ROCE', 'D/E', 'Score']]
            st.dataframe(display_table, use_container_width=True, hide_index=True)
            st.markdown("---")

    # Tab 3: News & AI
    with tab3:
        st.success(f"**AI Executive Digest:** {ai_summary['executive_summary']}")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.metric("Geopolitical & Oil Threat Level", f"{ai_summary['geopolitical_oil_risk_score']} / 5.0")
        with col_r2:
            st.metric("Domestic Inflation Pressure", f"{ai_summary['inflation_risk_score']} / 5.0")
            
        st.markdown("#### Live Ingested Headlines")
        for h in news_items:
            with st.expander(f"📌 {h['title']} ({h['published']})"):
                st.write(h['summary'])
                st.markdown(f"[Read full report]({h['link']})")

    # Tab 4: Execution Checklist
    with tab4:
        st.markdown("### 📝 Weekly Execution Checklist")
        gold_amt = (metrics['target_weights']['Gold (GOLDBEES / SGBs)'] / 100.0) * portfolio_size
        debt_amt = (metrics['target_weights']['Debt / Cash (LIQUIDBEES)'] / 100.0) * portfolio_size
        
        winners_list = ", ".join([s_data['winner']['Symbol'] for s_data in peer_results.values()])
        
        orders = [
            {"Instrument Category": "Equity Sleeve (Top Sector Champions)", "Scrips": winners_list, "Target %": f"{metrics['target_weights']['Equity Sleeve (Sector Peer Winners)']}%", "Capital to Deploy": f"₹{total_eq_amt:,.0f}"},
            {"Instrument Category": "Gold ETF / SGBs", "Scrips": "GOLDBEES", "Target %": f"{metrics['target_weights']['Gold (GOLDBEES / SGBs)']}%", "Capital to Deploy": f"₹{gold_amt:,.0f}"},
            {"Instrument Category": "Liquid Debt / Overnight", "Scrips": "LIQUIDBEES", "Target %": f"{metrics['target_weights']['Debt / Cash (LIQUIDBEES)']}%", "Capital to Deploy": f"₹{debt_amt:,.0f}"}
        ]
        st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()
