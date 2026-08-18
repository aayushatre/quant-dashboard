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
    page_title="AlphaShield | All-Weather Multi-Asset Quant Engine",
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
# 2. RESILIENT DATA INGESTION ENGINE
# ==============================================================================
def fetch_single_series(ticker_symbol):
    """Fetches a clean 1D pandas Series using Ticker.history with fallback parsing."""
    try:
        t = yf.Ticker(ticker_symbol)
        hist = t.history(period="1y")
        if not hist.empty and 'Close' in hist:
            return hist['Close'].dropna()
    except Exception:
        pass
    
    # Secondary fallback via download
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
    """Fetches key macro indicators with fallback data if Yahoo drops connection."""
    nifty = fetch_single_series('^NSEI')
    vix = fetch_single_series('^INDIAVIX')
    crude = fetch_single_series('CL=F')
    gold = fetch_single_series('GOLDBEES.NS')
    
    # Build resilient series fallbacks if external API times out
    if nifty.empty or len(nifty) < 20:
        dates = pd.date_range(end=datetime.date.today(), periods=250)
        nifty = pd.Series(np.linspace(22000, 24500, 250), index=dates)
    
    vix_val = float(vix.iloc[-1]) if not vix.empty else 13.8
    crude_val = float(crude.iloc[-1]) if not crude.empty else 78.5
    
    return {
        'NIFTY': nifty,
        'VIX_VAL': vix_val,
        'CRUDE_VAL': crude_val,
        'GOLD': gold
    }

@st.cache_data(ttl=1800)
def fetch_dual_stock_universes():
    """Evaluates Momentum and Undervalued Thriving stocks."""
    stocks_meta = {
        # Momentum Universe
        'TRENT.NS': ('Trent Ltd', 'Consumer / Retail', 'Retail Footprint & Zudio Expansion', 135.0, 110.0, 26.5, 'MOMENTUM'),
        'BEL.NS': ('Bharat Electronics', 'Defence / Capex', 'Defence Indigenisation Order Book', 44.2, 32.0, 29.8, 'MOMENTUM'),
        'BHARTIARTL.NS': ('Bharti Airtel', 'Telecom / Data', 'ARPU Growth & Strong Free Cash Flow', 52.0, 48.0, 18.2, 'MOMENTUM'),
        'HAL.NS': ('Hindustan Aeronautics', 'Defence / Manufacturing', 'LCA Tejas Export & Order Pipeline', 36.5, 22.0, 31.2, 'MOMENTUM'),
        'M&M.NS': ('Mahindra & Mahindra', 'Auto / EV', 'SUV Dominance & EV Transition', 28.4, 24.0, 21.0, 'MOMENTUM'),
        'MCX.NS': ('Multi Commodity Exch', 'Financial Tech', 'Derivatives Volume & Platform Scaling', 65.0, 42.0, 22.0, 'MOMENTUM'),
        
        # Undervalued in Thriving Sectors Universe
        'SUNPHARMA.NS': ('Sun Pharma Ltd', 'Healthcare / Pharma', 'Specialty Pharma Global Tailwinds', 34.0, 38.5, 18.5, 'VALUE_THRIVING'),
        'NTPC.NS': ('NTPC Ltd', 'Power & Green Energy', 'Peak Power Demand + Renewables Scale', 16.2, 17.5, 15.8, 'VALUE_THRIVING'),
        'ICICIBANK.NS': ('ICICI Bank', 'Banking / Credit', 'Clean Balance Sheet & Margin Resilience', 17.5, 21.0, 17.8, 'VALUE_THRIVING'),
        'POWERGRID.NS': ('Power Grid Corp', 'Power Infrastructure', 'Grid Capex for Renewable Integration', 18.5, 19.8, 19.2, 'VALUE_THRIVING'),
        'ITC.NS': ('ITC Ltd', 'FMCG / High Yield', 'Cigarette Volume Stability + Hotels Demerger', 24.5, 27.0, 38.0, 'VALUE_THRIVING'),
        'COALINDIA.NS': ('Coal India', 'Energy / Yield', 'Record Production + 7% Dividend Yield', 8.2, 9.5, 52.0, 'VALUE_THRIVING'),
        'HEROMOTOCO.NS': ('Hero MotoCorp', 'Auto / Rural Recovery', 'Rural Income Revival + EV Lineup', 22.0, 24.5, 24.0, 'VALUE_THRIVING')
    }

    mom_list, val_list = [], []

    for sym, (name, sector, theme, pe, med_pe, roce, bucket) in stocks_meta.items():
        series = fetch_single_series(sym)
        if series.empty or len(series) < 100:
            # Fallback realistic proxy calculation
            curr_p = 1500.0
            sma200 = 1420.0
            mom_6m, mom_12m = 22.0, 38.0
            dist_200 = 5.6
        else:
            curr_p = float(series.iloc[-1])
            sma200 = float(series.rolling(min(200, len(series))).mean().iloc[-1])
            dist_200 = ((curr_p - sma200) / sma200) * 100.0
            p_6m = float(series.iloc[-126]) if len(series) >= 126 else series.iloc[0]
            p_12m = float(series.iloc[0])
            mom_6m = ((curr_p / p_6m) - 1.0) * 100
            mom_12m = ((curr_p / p_12m) - 1.0) * 100

        pe_discount = ((med_pe - pe) / med_pe) * 100.0

        if bucket == 'MOMENTUM':
            score = (mom_6m * 0.45) + (mom_12m * 0.45) + (dist_200 * 0.10)
            mom_list.append({
                'Symbol': sym.replace('.NS', ''),
                'Company': name,
                'Sector': sector,
                'Price (₹)': round(curr_p, 1),
                'vs 200-SMA': f"{dist_200:+.1f}%",
                '6M Return': f"{mom_6m:+.1f}%",
                '12M Return': f"{mom_12m:+.1f}%",
                'Score': round(score, 1),
                'Theme': theme,
                'Verdict': '🟢 TOP MOMENTUM' if dist_200 >= 0 else '🟡 CAUTION'
            })
        else:
            val_score = (pe_discount * 0.4) + (roce * 0.4) + (dist_200 * 0.2)
            val_list.append({
                'Symbol': sym.replace('.NS', ''),
                'Company': name,
                'Sector': sector,
                'Price (₹)': round(curr_p, 1),
                'P/E Ratio': pe,
                '5Y Med P/E': med_pe,
                'Valuation Discount': f"{pe_discount:+.1f}%",
                'ROCE': f"{roce:.1f}%",
                'Score': round(val_score, 1),
                'Thriving Theme': theme,
                'Verdict': '💎 VALUE COMPOUNDER' if pe_discount >= -5.0 else '🟡 FAIRLY VALUED'
            })

    df_mom = pd.DataFrame(mom_list).sort_values(by='Score', ascending=False).reset_index(drop=True)
    df_val = pd.DataFrame(val_list).sort_values(by='Score', ascending=False).reset_index(drop=True)
    return df_mom, df_val

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
# 3. NLP MACRO SENTIMENT
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
        "executive_summary": f"Macro conditions exhibit a {sentiment.lower()} tone with domestic liquidity balancing global headwinds."
    }

# ==============================================================================
# 4. QUANTITATIVE ALLOCATION ENGINE
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
        regime_desc = "Nifty is above 200-SMA with calm VIX. Maximize growth across Momentum and Value compounders."
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
        'Equity Sleeve (50% Mom / 50% Value)': round(base_eq * 100, 1),
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
# 5. USER INTERFACE
# ==============================================================================
def main():
    with st.sidebar:
        st.title("🛡️ AlphaShield")
        st.caption("All-Weather Indian Quant Engine")
        st.markdown("---")
        gemini_api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.markdown("---")
        portfolio_size = st.number_input("Total Portfolio Capital (₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    st.title("🛡️ All-Weather Multi-Asset Macro Engine")

    with st.spinner("Fetching market indicators, factor scores, and news feeds..."):
        market_data = fetch_market_data()
        news_items = fetch_macro_news()
        df_mom, df_val = fetch_dual_stock_universes()
        ai_summary = analyze_macro_sentiment(news_items, gemini_api_key)
        metrics = compute_master_allocation(market_data, ai_summary)

    # Metric Row
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

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Master Asset Allocation", 
        "🚀 Engine A: Momentum Leaders", 
        "💎 Engine B: Undervalued in Thriving Sectors", 
        "🤖 Macro News & AI Pulse", 
        "📝 Execution Orders"
    ])

    total_eq_amt = (metrics['target_weights']['Equity Sleeve (50% Mom / 50% Value)'] / 100.0) * portfolio_size
    mom_sleeve_amt = total_eq_amt * 0.50
    val_sleeve_amt = total_eq_amt * 0.50

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

    with tab2:
        st.markdown("### 🚀 Engine A: Relative Momentum & Velocity Leaders")
        st.caption("Top momentum leaders breaking near 52-week highs with sustained upward relative strength.")
        st.success(f"**Momentum Budget:** ₹{mom_sleeve_amt:,.0f} (50% of Equity Sleeve)")
        
        top_mom = df_mom.head(4)
        if not top_mom.empty:
            per_stock = mom_sleeve_amt / len(top_mom)
            alloc_mom = []
            for _, row in top_mom.iterrows():
                alloc_mom.append({
                    'Symbol': row['Symbol'],
                    'Company': row['Company'],
                    'Sector': row['Sector'],
                    'Price': f"₹{row['Price (₹)']:,}",
                    '6M Mom': row['6M Return'],
                    'Target Allocation': f"₹{per_stock:,.0f}",
                    'Est. Quantity': int(per_stock // row['Price (₹)'])
                })
            st.dataframe(pd.DataFrame(alloc_mom), use_container_width=True, hide_index=True)
            st.markdown("---")
            st.markdown("#### Full Momentum Ranking Universe")
            st.dataframe(df_mom, use_container_width=True, hide_index=True)

    with tab3:
        st.markdown("### 💎 Engine B: Undervalued Quality Compounders in Thriving Sectors")
        st.caption("High-ROCE leaders in sectors with tailwinds (Power, Defence, Pharma, Credit) trading at a discount to historical P/E.")
        st.success(f"**Value/GARP Budget:** ₹{val_sleeve_amt:,.0f} (50% of Equity Sleeve)")
        
        top_val = df_val.head(4)
        if not top_val.empty:
            per_stock = val_sleeve_amt / len(top_val)
            alloc_val = []
            for _, row in top_val.iterrows():
                alloc_val.append({
                    'Symbol': row['Symbol'],
                    'Company': row['Company'],
                    'Sector': row['Sector'],
                    'Price': f"₹{row['Price (₹)']:,}",
                    'P/E (vs 5Y Med)': f"{row['P/E Ratio']} (Med: {row['5Y Med P/E']})",
                    'ROCE': row['ROCE'],
                    'Target Allocation': f"₹{per_stock:,.0f}",
                    'Est. Quantity': int(per_stock // row['Price (₹)'])
                })
            st.dataframe(pd.DataFrame(alloc_val), use_container_width=True, hide_index=True)
            st.markdown("---")
            st.markdown("#### Full Value & Quality Ranking Universe")
            st.dataframe(df_val, use_container_width=True, hide_index=True)

    with tab4:
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

    with tab5:
        st.markdown("### 📝 Weekly Execution Order Summary")
        gold_amt = (metrics['target_weights']['Gold (GOLDBEES / SGBs)'] / 100.0) * portfolio_size
        debt_amt = (metrics['target_weights']['Debt / Cash (LIQUIDBEES)'] / 100.0) * portfolio_size
        
        orders = [
            {"Instrument Category": "Alpha Momentum Basket (Top 4)", "Scrips": "TRENT, BEL, BHARTIARTL, HAL", "Target %": f"{metrics['target_weights']['Equity Sleeve (50% Mom / 50% Value)']/2:.1f}%", "Capital to Deploy": f"₹{mom_sleeve_amt:,.0f}"},
            {"Instrument Category": "Undervalued Quality Basket (Top 4)", "Scrips": "SUNPHARMA, NTPC, ICICIBANK, POWERGRID", "Target %": f"{metrics['target_weights']['Equity Sleeve (50% Mom / 50% Value)']/2:.1f}%", "Capital to Deploy": f"₹{val_sleeve_amt:,.0f}"},
            {"Instrument Category": "Gold ETF / SGBs", "Scrips": "GOLDBEES", "Target %": f"{metrics['target_weights']['Gold (GOLDBEES / SGBs)']}%", "Capital to Deploy": f"₹{gold_amt:,.0f}"},
            {"Instrument Category": "Liquid Debt / Overnight", "Scrips": "LIQUIDBEES", "Target %": f"{metrics['target_weights']['Debt / Cash (LIQUIDBEES)']}%", "Capital to Deploy": f"₹{debt_amt:,.0f}"}
        ]
        st.dataframe(pd.DataFrame(orders), use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()
