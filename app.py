import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import feedparser
import datetime
import json

# ==============================================================================
# 1. STREAMLIT PAGE CONFIGURATION & STYLING
# ==============================================================================
st.set_page_config(
    page_title="AlphaShield | Indian Multi-Asset Quant Engine",
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
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# 2. DATA INGESTION ENGINE (NSE & MACRO DATA VIA YFINANCE)
# ==============================================================================
@st.cache_data(ttl=1800)
def fetch_market_data():
    tickers = {
        'NIFTY50': '^NSEI',
        'VIX': '^INDIAVIX',
        'CRUDE': 'CL=F',
        'EQUITY_ETF': 'NIFTYBEES.NS',
        'GOLD_ETF': 'GOLDBEES.NS',
        'LIQUID_ETF': 'LIQUIDBEES.NS'
    }
    end_dt = datetime.datetime.now()
    start_dt = end_dt - datetime.timedelta(days=400)
    data = {}
    for key, symbol in tickers.items():
        try:
            df = yf.download(symbol, start=start_dt, end=end_dt, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df = df['Close']
            else:
                df = df[['Close']]
            data[key] = df.dropna()
        except Exception:
            data[key] = pd.DataFrame()
    return data

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
            for entry in feed.entries[:4]:
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
    return unique_headlines[:10]

# ==============================================================================
# 3. AI & NLP SENTIMENT ENGINE (GEMINI API WITH FALLBACK)
# ==============================================================================
def analyze_macro_sentiment(headlines, api_key=None):
    if api_key and api_key.strip():
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            You are a Quantitative Macro Analyst. Analyze these top 10 Indian financial news headlines:
            {json.dumps([h['title'] for h in headlines])}

            Return a valid JSON object ONLY with the following exact keys:
            {{
                "market_sentiment": "BULLISH" | "NEUTRAL" | "BEARISH",
                "inflation_risk_score": <number between 1.0 and 5.0>,
                "geopolitical_oil_risk_score": <number between 1.0 and 5.0>,
                "executive_summary": "<concise 2-sentence summary of macro pulse>"
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
    oil_risk = min(5.0, 2.0 + (oil_count * 0.8))
    inflation_risk = min(5.0, 2.5 + (bear_count * 0.4))
    
    return {
        "market_sentiment": sentiment,
        "inflation_risk_score": round(inflation_risk, 1),
        "geopolitical_oil_risk_score": round(oil_risk, 1),
        "executive_summary": f"Macro conditions reflect a {sentiment.lower()} tone with domestic corporate strength balancing global cues."
    }

# ==============================================================================
# 4. QUANTITATIVE ALLOCATION ENGINE (REGIME + INVERSE VOLATILITY)
# ==============================================================================
def compute_master_allocation(data, ai_sentiment):
    nifty_series = data['NIFTY50'].iloc[:, 0] if not data['NIFTY50'].empty else pd.Series()
    vix_series = data['VIX'].iloc[:, 0] if not data['VIX'].empty else pd.Series()
    crude_series = data['CRUDE'].iloc[:, 0] if not data['CRUDE'].empty else pd.Series()
    
    if nifty_series.empty:
        return None
        
    nifty_price = float(nifty_series.iloc[-1])
    nifty_sma200 = float(nifty_series.rolling(200).mean().iloc[-1])
    distance_sma200 = ((nifty_price - nifty_sma200) / nifty_sma200) * 100.0
    
    vix_current = float(vix_series.iloc[-1]) if not vix_series.empty else 14.5
    crude_current = float(crude_series.iloc[-1]) if not crude_series.empty else 75.0
    
    eq_ret = nifty_series.pct_change().dropna().tail(60)
    gold_ret = data['GOLD_ETF'].iloc[:, 0].pct_change().dropna().tail(60) if not data['GOLD_ETF'].empty else eq_ret * 0.6
    
    eq_vol = float(eq_ret.std() * np.sqrt(252))
    gold_vol = float(gold_ret.std() * np.sqrt(252))
    debt_vol = 0.015
    
    inv_eq = 1.0 / max(eq_vol, 0.05)
    inv_gold = 1.0 / max(gold_vol, 0.05)
    inv_debt = 1.0 / debt_vol
    inv_total = inv_eq + inv_gold + inv_debt
    
    w_inv_eq = inv_eq / inv_total
    w_inv_gold = inv_gold / inv_total
    w_inv_debt = inv_debt / inv_total

    is_bull = nifty_price >= nifty_sma200
    is_panic = vix_current >= 22.0
    is_oil_shock = (crude_current > 85.0) or (ai_sentiment['geopolitical_oil_risk_score'] >= 3.8)
    
    if is_bull and not is_panic and not is_oil_shock:
        regime_name = "1. Goldilocks Expansion (Risk-On)"
        regime_badge = "bullish"
        base_eq, base_gold, base_debt = 0.65, 0.15, 0.20
        regime_desc = "Nifty in healthy structural uptrend above 200-SMA. Maximize equity compounding."
    elif is_bull and is_oil_shock:
        regime_name = "2. Reflation / Commodity Shock"
        regime_badge = "caution"
        base_eq, base_gold, base_debt = 0.40, 0.35, 0.25
        regime_desc = "Elevated crude oil / geopolitical risk. Expanding gold to hedge inflation & rupee."
    elif not is_bull and not is_panic:
        regime_name = "3. Correction / Caution"
        regime_badge = "caution"
        base_eq, base_gold, base_debt = 0.30, 0.35, 0.35
        regime_desc = "Nifty below 200-SMA. Maintain balanced multi-asset defensive posture."
    else:
        regime_name = "4. Crash Shield (Risk-Off)"
        regime_badge = "bearish"
        base_eq, base_gold, base_debt = 0.05, 0.30, 0.65
        regime_desc = "Market breakdown with elevated VIX. Capital preservation mode (Liquid / Overnight Debt)."

    final_eq = 0.60 * base_eq + 0.40 * w_inv_eq
    final_gold = 0.60 * base_gold + 0.40 * w_inv_gold
    final_debt = 0.60 * base_debt + 0.40 * w_inv_debt
    
    total = final_eq + final_gold + final_debt
    target_weights = {
        'Equity (NIFTYBEES)': round((final_eq / total) * 100, 1),
        'Gold (GOLDBEES)': round((final_gold / total) * 100, 1),
        'Debt / Cash (LIQUIDBEES)': round((final_debt / total) * 100, 1)
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
# 5. STREAMLIT UI
# ==============================================================================
def main():
    with st.sidebar:
        st.title("🛡️ AlphaShield System")
        st.caption("Indian Multi-Asset Macro Engine")
        st.markdown("---")
        gemini_api_key = st.text_input("Gemini API Key (Optional)", type="password")
        st.markdown("---")
        portfolio_size = st.number_input("Total Capital (₹)", min_value=10000, max_value=100000000, value=500000, step=25000)
        st.markdown("---")
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()

    st.title("🛡️ Indian Multi-Asset Macro-Regime Dashboard")

    with st.spinner("Fetching real-time NSE data and news feeds..."):
        market_data = fetch_market_data()
        news_items = fetch_macro_news()
        ai_summary = analyze_macro_sentiment(news_items, gemini_api_key)
        metrics = compute_master_allocation(market_data, ai_summary)

    if not metrics:
        st.error("Error retrieving market indicators.")
        return

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Nifty 50 Index", f"₹{metrics['nifty_price']:,.1f}", f"{metrics['distance_sma200']:+.2f}% vs 200-SMA")
    with m2:
        st.metric("India VIX", f"{metrics['vix_current']:.2f}", "Normal (<18)" if metrics['vix_current'] < 18 else "Elevated (>18)")
    with m3:
        st.metric("Brent Crude Oil", f"${metrics['crude_current']:.1f}", "Stable" if metrics['crude_current'] < 85 else "Shock Risk")
    with m4:
        st.markdown("**AI Sentiment**")
        st.markdown(f"<span class='badge-{metrics['regime_badge']}'>{ai_summary['market_sentiment']}</span>", unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("🎯 Active Market Regime")
    st.info(f"**{metrics['regime_name']}** — {metrics['regime_desc']}")

    tab1, tab2, tab3 = st.tabs(["📊 Target Capital Allocation", "🤖 Macro News & AI Pulse", "📝 Execution Orders"])

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
            st.markdown("### 💰 Capital Deployment")
            for asset, weight in metrics['target_weights'].items():
                amt = (weight / 100.0) * portfolio_size
                st.markdown(f"""
                <div class="metric-card">
                    <span style="font-size:16px; font-weight:600;">{asset}</span>
                    <span style="float:right; font-size:18px; font-weight:700; color:#58a6ff;">₹{amt:,.0f} ({weight}%)</span>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.success(f"**AI Macro Summary:** {ai_summary['executive_summary']}")
        st.markdown("#### Latest Financial News Feeds")
        for h in news_items:
            with st.expander(f"📌 {h['title']} ({h['published']})"):
                st.write(h['summary'])
                st.markdown(f"[Read source article]({h['link']})")

    with tab3:
        st.markdown("### 📝 Weekly Execution Checklist")
        orders = pd.DataFrame([
            {"Instrument": "Nifty 50 ETF", "Symbol": "NIFTYBEES", "Target %": f"{metrics['target_weights']['Equity (NIFTYBEES)']}%", "Rupees to Deploy": f"₹{(metrics['target_weights']['Equity (NIFTYBEES)']/100)*portfolio_size:,.0f}"},
            {"Instrument": "Gold BeES ETF", "Symbol": "GOLDBEES", "Target %": f"{metrics['target_weights']['Gold (GOLDBEES)']}%", "Rupees to Deploy": f"₹{(metrics['target_weights']['Gold (GOLDBEES)']/100)*portfolio_size:,.0f}"},
            {"Instrument": "Liquid BeES ETF", "Symbol": "LIQUIDBEES", "Target %": f"{metrics['target_weights']['Debt / Cash (LIQUIDBEES)']}%", "Rupees to Deploy": f"₹{(metrics['target_weights']['Debt / Cash (LIQUIDBEES)']/100)*portfolio_size:,.0f}"}
        ])
        st.dataframe(orders, use_container_width=True, hide_index=True)

if __name__ == '__main__':
    main()
