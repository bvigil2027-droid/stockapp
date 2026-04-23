"""
MarketIQ Platform - Flask Backend

Install:
    pip3 install flask yfinance requests beautifulsoup4 feedparser

Run:
    python3 app.py
"""

from flask import Flask, render_template, request, jsonify
import json, re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    import yfinance as yf
except ImportError:
    print("Run: pip3 install yfinance"); exit(1)

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Run: pip3 install requests beautifulsoup4"); exit(1)

try:
    import feedparser
except ImportError:
    print("Run: pip3 install feedparser"); exit(1)

app = Flask(__name__)
watchlist = []

# ── Constants ─────────────────────────────────────────────────────────────────

TOP_MOVERS = ["AAPL","MSFT","NVDA","TSLA","AMZN","GOOGL","META","AMD","NFLX",
              "BABA","JPM","BAC","GS","V","MA","XOM","CVX","WMT","JNJ","PFE",
              "ORCL","CRM","INTC","QCOM","DIS","UBER","SPOT","PLTR","COIN","SQ"]

MARKET_INDICES = {
    "S&P 500": "^GSPC",
    "NASDAQ":  "^IXIC",
    "DOW":     "^DJI",
    "VIX":     "^VIX",
}

NEWS_FEEDS = [
    ("Reuters",       "https://feeds.reuters.com/reuters/businessNews",                       "neutral"),
    ("Reuters Mkts",  "https://feeds.reuters.com/reuters/companyNews",                        "neutral"),
    ("CNBC",          "https://www.cnbc.com/id/100003114/device/rss/rss.html",                "neutral"),
    ("CNN Business",  "https://rss.cnn.com/rss/money_latest.rss",                            "left-center"),
    ("Fox Business",  "https://moxie.foxbusiness.com/google-manager/articles.xml",            "right-center"),
    ("MarketWatch",   "https://feeds.content.dowjones.io/public/rss/mw_realtimeheadlines",    "neutral"),
    ("Seeking Alpha", "https://seekingalpha.com/market_currents.xml",                         "neutral"),
    ("Yahoo Finance", "https://finance.yahoo.com/news/rssindex",                              "neutral"),
    ("Investopedia",  "https://www.investopedia.com/feedbuilder/feed/getfeed/?feedName=rss_headline", "neutral"),
]

SECTOR_PEERS = {
    "Technology":    ["AAPL","MSFT","GOOGL","META","NVDA","AMD","INTC","QCOM","ORCL","CRM"],
    "Consumer Cyclical": ["AMZN","TSLA","BABA","DIS","UBER","NFLX","SPOT"],
    "Financial Services": ["JPM","BAC","GS","V","MA","SQ","COIN"],
    "Energy":        ["XOM","CVX"],
    "Healthcare":    ["JNJ","PFE"],
}

BULLISH_WORDS = ["surge","soar","jump","gain","rally","beat","record","growth",
    "profit","upgrade","buy","strong","positive","rise","boost","exceed",
    "outperform","bullish","expand","partnership","deal","launch","optimistic",
    "recovery","dividend","revenue","high","milestone","breakout","upside","blowout"]

BEARISH_WORDS = ["fall","drop","decline","loss","miss","downgrade","sell","weak",
    "negative","cut","layoff","lawsuit","investigation","bearish","concern",
    "risk","warning","recall","debt","deficit","fraud","lower","disappoint",
    "reduce","contract","restructure","fine","crash","plunge","slump","fear","probe"]

# ── Data Fetchers ─────────────────────────────────────────────────────────────

def get_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="60d")
    if hist.empty: return None
    close, volume = hist["Close"], hist["Volume"]
    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss
    rsi   = (100 - 100 / (1 + rs)).iloc[-1]
    current   = close.iloc[-1]
    prev_day  = close.iloc[-2]
    week_ago  = close.iloc[-7]  if len(close) >= 7  else close.iloc[0]
    month_ago = close.iloc[-30] if len(close) >= 30 else close.iloc[0]
    avg_vol    = volume.mean()
    recent_vol = volume.iloc[-5:].mean()
    hist_1y  = stock.history(period="1y")
    high_52w = hist_1y["High"].max() if not hist_1y.empty else current
    low_52w  = hist_1y["Low"].min()  if not hist_1y.empty else current
    return {
        "current_price":     round(current, 2),
        "prev_close":        round(prev_day, 2),
        "sma20":             round(sma20, 2),
        "sma50":             round(sma50, 2),
        "rsi":               round(rsi, 1),
        "pct_1d":            round(((current - prev_day)  / prev_day)  * 100, 2),
        "pct_7d":            round(((current - week_ago)  / week_ago)  * 100, 2),
        "pct_30d":           round(((current - month_ago) / month_ago) * 100, 2),
        "volume_ratio":      round(recent_vol / avg_vol if avg_vol else 1.0, 2),
        "high_52w":          round(high_52w, 2),
        "low_52w":           round(low_52w, 2),
        "pct_from_52w_high": round(((current - high_52w) / high_52w) * 100, 1),
    }


def get_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "name":            info.get("longName", ticker),
        "sector":          info.get("sector", "N/A"),
        "industry":        info.get("industry", "N/A"),
        "pe_ratio":        info.get("trailingPE"),
        "forward_pe":      info.get("forwardPE"),
        "beta":            info.get("beta"),
        "analyst_target":  info.get("targetMeanPrice"),
        "recommendation":  info.get("recommendationKey", "N/A"),
        "revenue_growth":  info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "market_cap":      info.get("marketCap"),
        "dividend_yield":  info.get("dividendYield"),
        "profit_margins":  info.get("profitMargins"),
        "short_ratio":     info.get("shortRatio"),
        "gross_margins":   info.get("grossMargins"),
    }


def get_price_history(ticker, period="3mo"):
    stock = yf.Ticker(ticker)
    hist  = stock.history(period=period)
    if hist.empty: return []
    return [{"date": d.strftime("%Y-%m-%d"), "open": round(r["Open"],2),
             "high": round(r["High"],2), "low": round(r["Low"],2),
             "close": round(r["Close"],2), "volume": int(r["Volume"])}
            for d, r in hist.iterrows()]


# ── News Fetchers ─────────────────────────────────────────────────────────────

def fetch_feed(source_name, url, bias, ticker="", company_name=""):
    articles = []
    try:
        feed = feedparser.parse(url)
        keywords = []
        if ticker: keywords.append(ticker.upper())
        if company_name:
            first = company_name.split()[0]
            if len(first) > 4: keywords.append(first.lower())

        for entry in feed.entries[:40]:
            title   = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()
            link    = entry.get("link", "")
            published = entry.get("published", "")

            if not title or len(title) < 15: continue

            if keywords:
                combined = (title + " " + summary).lower()
                if not any(kw.lower() in combined for kw in keywords):
                    continue

            articles.append({
                "title":     title,
                "summary":   BeautifulSoup(summary, "html.parser").get_text()[:400] if summary else "",
                "source":    source_name,
                "bias":      bias,
                "link":      link,
                "published": published[:16] if published else "",
            })
        return articles
    except Exception:
        return []


def get_news_multisource(ticker="", company_name="", limit=30):
    all_articles = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(fetch_feed, name, url, bias, ticker, company_name): name
                   for name, url, bias in NEWS_FEEDS}
        for f in as_completed(futures):
            try: all_articles.extend(f.result())
            except: pass

    seen, deduped = set(), []
    for a in all_articles:
        key = a["title"][:60].lower()
        if key not in seen:
            seen.add(key)
            deduped.append(a)
    return deduped[:limit]


def get_ticker_news(ticker, company_name="", limit=20):
    articles = get_news_multisource(ticker=ticker, company_name=company_name, limit=limit)
    if not articles:
        for n in (yf.Ticker(ticker).news or [])[:limit]:
            title = n.get("title", "")
            if title:
                articles.append({
                    "title":   title, "summary": "",
                    "source":  n.get("publisher", "Yahoo Finance"),
                    "bias":    "neutral",
                    "link":    n.get("link", ""),
                    "published": datetime.fromtimestamp(n.get("providerPublishTime",0)).strftime("%Y-%m-%d"),
                })
    return articles


def get_general_market_news(limit=20):
    return get_news_multisource(ticker="", company_name="", limit=limit)


# ── Scoring Engine ────────────────────────────────────────────────────────────

def score_news_articles(articles):
    bull, bear = 0, 0
    matched = {"bullish": [], "bearish": []}
    for a in articles:
        lower = (a["title"] + " " + a.get("summary","")).lower()
        for w in BULLISH_WORDS:
            if w in lower: bull += 1; matched["bullish"].append(w)
        for w in BEARISH_WORDS:
            if w in lower: bear += 1; matched["bearish"].append(w)
    total = bull + bear
    if total == 0: score, label = 0, "NEUTRAL"
    else:
        score = round(((bull - bear) / total) * 100, 1)
        label = "POSITIVE" if score > 20 else "NEGATIVE" if score < -20 else "MIXED"
    return {"sentiment_label": label, "sentiment_score": score,
            "bullish_hits": bull, "bearish_hits": bear,
            "top_bullish": list(set(matched["bullish"]))[:5],
            "top_bearish": list(set(matched["bearish"]))[:5]}


def score_technicals(t):
    score, bull, bear = 0, [], []
    if t["current_price"] > t["sma20"]: score += 15; bull.append(f"Price above SMA20 (${t['sma20']})")
    else: score -= 15; bear.append(f"Price below SMA20 (${t['sma20']})")
    if t["current_price"] > t["sma50"]: score += 15; bull.append(f"Price above SMA50 (${t['sma50']})")
    else: score -= 15; bear.append(f"Price below SMA50 (${t['sma50']})")
    rsi = t["rsi"]
    if rsi < 30: score += 20; bull.append(f"RSI oversold at {rsi}")
    elif rsi > 70: score -= 20; bear.append(f"RSI overbought at {rsi}")
    else: score += 5; bull.append(f"RSI neutral at {rsi}")
    if t["pct_7d"] > 3: score += 10; bull.append(f"Strong 7-day momentum (+{t['pct_7d']}%)")
    elif t["pct_7d"] < -3: score -= 10; bear.append(f"Weak 7-day momentum ({t['pct_7d']}%)")
    if t["pct_30d"] > 5: score += 10; bull.append(f"Strong 30-day trend (+{t['pct_30d']}%)")
    elif t["pct_30d"] < -5: score -= 10; bear.append(f"Weak 30-day trend ({t['pct_30d']}%)")
    if t["volume_ratio"] > 1.5:
        if t["pct_1d"] > 0: score += 10; bull.append(f"High volume on up day ({t['volume_ratio']}x avg)")
        else: score -= 10; bear.append(f"High volume on down day ({t['volume_ratio']}x avg)")
    pfh = t["pct_from_52w_high"]
    if pfh > -5: score += 5; bull.append(f"Near 52-week high ({pfh}%)")
    elif pfh < -30: score += 8; bull.append(f"Far below 52-week high — potential upside ({pfh}%)")
    return score, bull, bear


def score_fundamentals(f):
    score, bull, bear = 0, [], []
    rec = (f.get("recommendation") or "").lower()
    if rec in ("strong_buy","buy"): score += 20; bull.append(f"Analyst rating: {rec.replace('_',' ').title()}")
    elif rec in ("underperform","sell","strong_sell"): score -= 20; bear.append(f"Analyst rating: {rec.replace('_',' ').title()}")
    pe = f.get("pe_ratio")
    if pe and pe < 15: score += 10; bull.append(f"Low P/E ({round(pe,1)}) — undervalued")
    elif pe and pe > 50: score -= 5; bear.append(f"High P/E ({round(pe,1)}) — priced for perfection")
    rg = f.get("revenue_growth")
    if rg and rg > 0.1: score += 10; bull.append(f"Strong revenue growth ({round(rg*100,1)}%)")
    elif rg and rg < 0: score -= 10; bear.append(f"Negative revenue growth ({round(rg*100,1)}%)")
    eg = f.get("earnings_growth")
    if eg and eg > 0.1: score += 10; bull.append(f"Strong earnings growth ({round(eg*100,1)}%)")
    elif eg and eg < 0: score -= 10; bear.append(f"Negative earnings growth ({round(eg*100,1)}%)")
    beta = f.get("beta")
    if beta and beta > 2: bear.append(f"High beta ({round(beta,2)}) — high volatility")
    elif beta and beta < 0.5: bull.append(f"Low beta ({round(beta,2)}) — stable")
    sr = f.get("short_ratio")
    if sr and sr > 5: bear.append(f"High short ratio ({round(sr,1)}) — significant short interest")
    return score, bull, bear


def compute_prediction(tech_score, fund_score, news_score_val):
    total   = (tech_score * 0.5) + (fund_score * 0.3) + (news_score_val * 0.2)
    clamped = max(-80, min(80, total))
    up_prob = round(50 + (clamped * 0.4))
    dn_prob = 100 - up_prob
    if up_prob >= 65:   verdict = "BULLISH"; confidence = "HIGH" if up_prob >= 75 else "MEDIUM"
    elif up_prob <= 35: verdict = "BEARISH"; confidence = "HIGH" if up_prob <= 25 else "MEDIUM"
    else:               verdict = "NEUTRAL"; confidence = "LOW"
    return up_prob, dn_prob, verdict, confidence


# ── Deep AI Analysis ──────────────────────────────────────────────────────────

def classify_event_type(title):
    t = title.lower()
    if any(w in t for w in ["earnings","revenue","profit","eps","quarterly","q1","q2","q3","q4"]): return "Earnings"
    if any(w in t for w in ["regulation","sec","ftc","doj","probe","lawsuit","fine","ban"]): return "Regulatory"
    if any(w in t for w in ["launch","release","announce","introduce","unveil","new product"]): return "Product Launch"
    if any(w in t for w in ["fed","interest rate","inflation","gdp","recession","macro","economy"]): return "Macro"
    if any(w in t for w in ["acquire","merger","acquisition","deal","buyout","takeover"]): return "M&A"
    if any(w in t for w in ["upgrade","downgrade","target","analyst","rating","price target"]): return "Analyst Action"
    if any(w in t for w in ["hire","fire","ceo","cfo","executive","resign","appoint"]): return "Leadership"
    return "Market News"


def get_related_entities(ticker, sector):
    peers = []
    for s, tickers in SECTOR_PEERS.items():
        if ticker in tickers:
            peers = [t for t in tickers if t != ticker][:4]
            break
    return peers


def deep_analysis(ticker, fund, tech, articles, news_result):
    """Generate a structured deep analysis following the AI prompt instructions."""
    name    = fund.get("name", ticker)
    sector  = fund.get("sector", "N/A")
    ts, _, _ = score_technicals(tech)
    fs, _, _ = score_fundamentals(fund)
    up, dn, verdict, conf = compute_prediction(ts, fs, news_result["sentiment_score"])

    # Event classification
    event_types = {}
    for a in articles[:10]:
        et = classify_event_type(a["title"])
        event_types[et] = event_types.get(et, 0) + 1
    top_events = sorted(event_types.items(), key=lambda x: x[1], reverse=True)[:3]

    # Related entities
    peers = get_related_entities(ticker, sector)

    # Sentiment label
    sent = news_result["sentiment_label"]

    # Time horizons based on scores
    price   = tech["current_price"]
    rsi_val = tech["rsi"]
    pct_7d  = tech["pct_7d"]
    pct_30d = tech["pct_30d"]
    target  = fund.get("analyst_target")
    upside  = round(((target - price) / price) * 100, 1) if target else None

    # Short term
    if pct_7d > 3 and sent == "POSITIVE":
        short_term = "Momentum is strong. Positive news flow may sustain upward pressure in the next 1–5 days."
        short_signal = "BULLISH"
    elif pct_7d < -3 and sent == "NEGATIVE":
        short_term = "Selling pressure is elevated. Negative sentiment may continue driving price lower short-term."
        short_signal = "BEARISH"
    elif rsi_val > 70:
        short_term = "Technically overbought. A short-term pullback or consolidation is likely even if fundamentals are strong."
        short_signal = "NEUTRAL"
    elif rsi_val < 30:
        short_term = "Technically oversold. A short-term bounce is likely regardless of broader trend."
        short_signal = "BULLISH"
    else:
        short_term = "No strong short-term directional signal. Price action is consolidating."
        short_signal = "NEUTRAL"

    # Medium term
    rg = fund.get("revenue_growth") or 0
    if rg > 0.1 and up > 60:
        med_term = "Fundamental growth trajectory supports a continued uptrend over the next 1–4 weeks if macro conditions hold."
        med_signal = "BULLISH"
    elif rg < 0 and up < 40:
        med_term = "Declining fundamentals may weigh on price in the medium term. Watch for further analyst downgrades."
        med_signal = "BEARISH"
    else:
        med_term = "Medium-term outlook is mixed. Earnings guidance and macro data will be key catalysts."
        med_signal = "NEUTRAL"

    # Long term
    if upside and upside > 15 and rg > 0:
        long_term = f"Analyst consensus target of ${target} implies {upside}% upside. Long-term thesis remains intact if growth continues."
        long_signal = "BULLISH"
    elif upside and upside < -10:
        long_term = f"Analyst target of ${target} is below current price, suggesting potential overvaluation on a 3–12 month horizon."
        long_signal = "BEARISH"
    else:
        long_term = "Long-term outlook depends on sector tailwinds and the company's ability to sustain margins amid macro uncertainty."
        long_signal = "NEUTRAL"

    # Contradictions
    contradictions = []
    if sent == "POSITIVE" and tech["pct_30d"] < -10:
        contradictions.append("Positive news sentiment conflicts with a 30-day price decline — this news may already be priced in or insufficient to reverse the trend.")
    if sent == "NEGATIVE" and rsi_val < 35:
        contradictions.append("Negative sentiment but oversold RSI — bad news may already be reflected in price, creating a contrarian opportunity.")
    if verdict == "BULLISH" and tech["pct_from_52w_high"] > -3:
        contradictions.append("Bullish signal but stock is near its 52-week high — limited upside room and increased risk of profit-taking.")
    if fs > 10 and ts < -10:
        contradictions.append("Strong fundamentals but weak technicals — the stock may be in a temporary dip or broader market selloff.")

    # Key drivers
    drivers = []
    if news_result["top_bullish"]: drivers.append(f"Positive news keywords: {', '.join(news_result['top_bullish'][:3])}")
    if news_result["top_bearish"]: drivers.append(f"Negative news keywords: {', '.join(news_result['top_bearish'][:3])}")
    if top_events: drivers.append(f"Primary event types: {', '.join(e[0] for e in top_events)}")
    if rg: drivers.append(f"Revenue growth: {round(rg*100,1)}%")
    pe = fund.get("pe_ratio")
    if pe: drivers.append(f"P/E ratio: {round(pe,1)}")
    beta = fund.get("beta")
    if beta: drivers.append(f"Beta: {round(beta,2)} ({'high' if beta>1.5 else 'moderate' if beta>0.8 else 'low'} volatility)")

    # Risks
    risks = []
    if beta and beta > 1.5: risks.append("High beta — amplified downside in broad market selloffs")
    if rsi_val > 70: risks.append("Overbought RSI — susceptible to technical correction")
    sr = fund.get("short_ratio")
    if sr and sr > 5: risks.append(f"Short ratio of {round(sr,1)} — elevated short interest could pressure price")
    if tech["volume_ratio"] < 0.5: risks.append("Below-average volume — lack of conviction in current move")
    if not risks: risks.append("No major risk flags detected at this time")

    # Impact classification
    impact_type = "Direct Impact"
    if top_events:
        et = top_events[0][0]
        if et == "Macro": impact_type = "Macro/Sector Impact"
        elif et == "Regulatory": impact_type = "Direct Impact (Regulatory)"
        elif et == "M&A": impact_type = "Direct Impact (M&A)"

    return {
        "summary": f"{name} is showing a {verdict.lower()} signal with {conf.lower()} confidence based on a composite of technical, fundamental, and sentiment analysis.",
        "signal":  verdict,
        "confidence": up,
        "short_term": {"signal": short_signal, "text": short_term},
        "med_term":   {"signal": med_signal,   "text": med_term},
        "long_term":  {"signal": long_signal,  "text": long_term},
        "news_sentiment":    sent,
        "fundamental_impact": "Positive" if fs > 10 else "Negative" if fs < -10 else "Neutral",
        "industry_impact":    f"Sector: {sector}. Events: {', '.join(e[0] for e in top_events) if top_events else 'General market news'}.",
        "macro_impact":       "Monitor Fed policy, inflation data, and sector rotation trends.",
        "event_types":        top_events,
        "primary_entity":     name,
        "related_entities":   peers,
        "impact_type":        impact_type,
        "contradictions":     contradictions,
        "key_drivers":        drivers,
        "risks":              risks,
    }


# ── Top Buys / Sells ──────────────────────────────────────────────────────────

def get_top_buys_sells():
    buys, sells = [], []

    def analyze_ticker(ticker):
        try:
            tech = get_technical_indicators(ticker)
            fund = get_fundamentals(ticker)
            if not tech: return None
            articles    = get_ticker_news(ticker, fund.get("name",""), limit=5)
            news_result = score_news_articles(articles)
            ts, _, _    = score_technicals(tech)
            fs, _, _    = score_fundamentals(fund)
            up, dn, verdict, conf = compute_prediction(ts, fs, news_result["sentiment_score"])
            target  = fund.get("analyst_target")
            current = tech["current_price"]
            upside  = round(((target - current) / current) * 100, 1) if target else None
            return {
                "ticker":  ticker,
                "name":    fund.get("name", ticker),
                "price":   current,
                "pct_1d":  tech["pct_1d"],
                "verdict": verdict,
                "up_prob": up,
                "rsi":     tech["rsi"],
                "upside":  upside,
                "target":  target,
                "sentiment": news_result["sentiment_label"],
                "sector":  fund.get("sector","N/A"),
                "score":   (ts * 0.5) + (fs * 0.3) + (news_result["sentiment_score"] * 0.2),
            }
        except: return None

    results = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(analyze_ticker, t): t for t in TOP_MOVERS[:20]}
        for f in as_completed(futures):
            r = f.result()
            if r: results.append(r)

    results.sort(key=lambda x: x["score"], reverse=True)
    buys  = [r for r in results if r["score"] > 15][:5]
    sells = [r for r in results if r["score"] < -15][:5]
    sells.sort(key=lambda x: x["score"])
    return buys, sells


# ── Home screen data ──────────────────────────────────────────────────────────

def get_index_data():
    results = {}
    for name, sym in MARKET_INDICES.items():
        try:
            info  = yf.Ticker(sym).info
            price = info.get("regularMarketPrice") or info.get("currentPrice", 0)
            prev  = info.get("regularMarketPreviousClose") or info.get("previousClose", price)
            pct   = round(((price - prev) / prev) * 100, 2) if prev else 0
            results[name] = {"price": round(price,2), "pct": pct, "symbol": sym}
        except:
            results[name] = {"price": 0, "pct": 0, "symbol": sym}
    return results


def get_movers():
    risers, fallers = [], []
    def fetch_mover(ticker):
        try:
            info  = yf.Ticker(ticker).info
            price = info.get("regularMarketPrice") or info.get("currentPrice")
            prev  = info.get("regularMarketPreviousClose") or info.get("previousClose")
            name  = info.get("shortName", ticker)
            if price and prev and prev > 0:
                pct = round(((price - prev) / prev) * 100, 2)
                return {"ticker": ticker, "name": name, "price": round(price,2), "pct": pct}
        except: pass
        return None

    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(fetch_mover, t): t for t in TOP_MOVERS}
        for f in as_completed(futures):
            r = f.result()
            if r:
                if r["pct"] > 0: risers.append(r)
                elif r["pct"] < 0: fallers.append(r)

    risers.sort(key=lambda x: x["pct"], reverse=True)
    fallers.sort(key=lambda x: x["pct"])
    return risers[:5], fallers[:5]


# ── Chatbot ───────────────────────────────────────────────────────────────────

def extract_ticker(text):
    for word in text.upper().split():
        clean = re.sub(r'[^A-Z]', '', word)
        if 1 < len(clean) <= 5:
            try:
                info = yf.Ticker(clean).info
                if info.get("regularMarketPrice") or info.get("currentPrice"):
                    return clean
            except: pass
    return None


def chatbot_response(message, context_ticker=None):
    msg    = message.lower().strip()
    ticker = extract_ticker(message) or context_ticker

    if any(w in msg for w in ["hello","hi","hey","sup"]):
        return "Hey! Ask me about any stock — try 'How is AAPL doing?' or 'Should I buy TSLA?'"
    if any(w in msg for w in ["help","what can you do"]):
        return ("I can help you analyze stocks. Try:\n"
                "• 'How is AAPL doing?'\n• 'Is NVDA bullish or bearish?'\n"
                "• 'What is the RSI for TSLA?'\n• 'What are the risks for AMZN?'\n"
                "• 'Give me a deep analysis of MSFT'")

    if ticker:
        try:
            tech        = get_technical_indicators(ticker)
            fund        = get_fundamentals(ticker)
            articles    = get_ticker_news(ticker, fund.get("name",""), limit=10)
            news_result = score_news_articles(articles)
            ts, bull_t, bear_t = score_technicals(tech)
            fs, bull_f, bear_f = score_fundamentals(fund)
            up, dn, verdict, conf = compute_prediction(ts, fs, news_result["sentiment_score"])
            name = fund.get("name", ticker)

            if any(w in msg for w in ["deep","analysis","analyze","full","detailed"]):
                da = deep_analysis(ticker, fund, tech, articles, news_result)
                return (f"Deep Analysis — {name} ({ticker})\n\n"
                        f"Signal: {da['signal']} | Confidence: {da['confidence']}%\n\n"
                        f"Summary: {da['summary']}\n\n"
                        f"Short-term ({da['short_term']['signal']}): {da['short_term']['text']}\n\n"
                        f"Medium-term ({da['med_term']['signal']}): {da['med_term']['text']}\n\n"
                        f"Long-term ({da['long_term']['signal']}): {da['long_term']['text']}\n\n"
                        f"Key Risks: {' | '.join(da['risks'][:2])}\n\n"
                        f"Note: This is probabilistic analysis, not financial advice.")

            if any(w in msg for w in ["price","worth","trading","cost"]):
                d = "up" if tech["pct_1d"] > 0 else "down"
                return f"{name} ({ticker}) is at ${tech['current_price']}, {d} {abs(tech['pct_1d'])}% today. {abs(tech['pct_from_52w_high'])}% from 52-week high of ${tech['high_52w']}."
            if "rsi" in msg:
                rsi = tech["rsi"]
                st = "oversold — potential bounce" if rsi < 30 else "overbought — potential pullback" if rsi > 70 else "neutral"
                return f"{ticker} RSI is {rsi}, which is {st}."
            if any(w in msg for w in ["buy","sell","invest","should i"]):
                rec    = fund.get("recommendation","N/A").replace("_"," ").title()
                target = fund.get("analyst_target")
                upside = round(((target - tech["current_price"]) / tech["current_price"]) * 100,1) if target else None
                r = f"{ticker} looks {verdict.lower()} ({conf.lower()} confidence). {up}% up probability. Analyst: '{rec}'"
                if upside: r += f", target ${target} ({'+' if upside>0 else ''}{upside}%)"
                return r + ". Not financial advice."
            if any(w in msg for w in ["bullish","bearish","outlook","trend"]):
                tb = bull_t[0] if bull_t else "no strong bullish signal"
                br = bear_t[0] if bear_t else "no strong bearish signal"
                return f"{ticker} is {verdict} ({conf.lower()}) — {up}% up probability. Bullish: {tb}. Bearish: {br}."
            if any(w in msg for w in ["risk","volatile","danger"]):
                risks = (bear_t + bear_f)[:2]
                beta  = fund.get("beta")
                bs    = f"Beta {round(beta,2)}. " if beta else ""
                return f"{ticker} risk: {bs}{' '.join(risks) or 'No major signals.'} Sentiment: {news_result['sentiment_label'].lower()}."
            if any(w in msg for w in ["news","headline","latest"]):
                if articles: return f"Latest for {ticker}:\n• " + "\n• ".join(a["title"] for a in articles[:3])
                return f"No recent headlines for {ticker}."

            return (f"{name} ({ticker}) — ${tech['current_price']} "
                    f"({'+' if tech['pct_1d']>0 else ''}{tech['pct_1d']}% today). "
                    f"Outlook: {verdict} | Up: {up}% | RSI: {tech['rsi']} | Sentiment: {news_result['sentiment_label']}.")
        except:
            return f"Sorry, couldn't fetch data for {ticker}. Check the ticker symbol."

    return "Try mentioning a stock ticker like AAPL, TSLA, or NVDA. E.g. 'What is the outlook for MSFT?'"


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/home-data")
def home_data():
    try:
        indices         = get_index_data()
        risers, fallers = get_movers()
        market_news     = get_general_market_news(limit=15)
        return jsonify({
            "indices": indices, "risers": risers, "fallers": fallers,
            "market_news": market_news, "updated": datetime.now().strftime("%I:%M %p"),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/top-picks")
def top_picks():
    try:
        buys, sells = get_top_buys_sells()
        return jsonify({"buys": buys, "sells": sells, "updated": datetime.now().strftime("%I:%M %p")})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json()
    ticker    = data.get("ticker","").strip().upper()
    timeframe = data.get("timeframe","1 week")
    if not ticker: return jsonify({"error": "No ticker"}), 400
    try:
        tech = get_technical_indicators(ticker)
        if not tech: return jsonify({"error": f"No data for {ticker}"}), 404
        fund        = get_fundamentals(ticker)
        articles    = get_ticker_news(ticker, fund.get("name",""))
        news_result = score_news_articles(articles)
        ts, bull_t, bear_t = score_technicals(tech)
        fs, bull_f, bear_f = score_fundamentals(fund)
        up, dn, verdict, conf = compute_prediction(ts, fs, news_result["sentiment_score"])
        da      = deep_analysis(ticker, fund, tech, articles, news_result)
        target  = fund.get("analyst_target")
        current = tech["current_price"]
        upside  = round(((target - current) / current) * 100, 1) if target else None
        return jsonify({
            "ticker": ticker, "name": fund.get("name",ticker), "timeframe": timeframe,
            "verdict": verdict, "confidence": conf, "up_prob": up, "dn_prob": dn,
            "tech": tech, "fund": fund, "upside": upside,
            "news": {**news_result, "articles": articles[:8]},
            "factors": {"bullish": (bull_t+bull_f)[:5], "bearish": (bear_t+bear_f)[:5]},
            "scores": {"technical": ts, "fundamental": fs, "news": news_result["sentiment_score"]},
            "deep_analysis": da,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/news", methods=["POST"])
def news():
    data   = request.get_json()
    ticker = data.get("ticker","").strip().upper()
    if not ticker: return jsonify({"error": "No ticker"}), 400
    try:
        fund      = get_fundamentals(ticker)
        articles  = get_ticker_news(ticker, fund.get("name",""), limit=25)
        sentiment = score_news_articles(articles)
        return jsonify({"ticker": ticker, "name": fund.get("name",ticker),
                        "articles": articles, "sentiment": sentiment})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/chart", methods=["POST"])
def chart():
    data   = request.get_json()
    ticker = data.get("ticker","").strip().upper()
    period = data.get("period","3mo")
    if not ticker: return jsonify({"error": "No ticker"}), 400
    try:
        history = get_price_history(ticker, period)
        fund    = get_fundamentals(ticker)
        return jsonify({"ticker": ticker, "name": fund.get("name",ticker), "history": history})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/watchlist", methods=["GET"])
def get_watchlist():
    results = []
    for ticker in watchlist:
        try:
            tech = get_technical_indicators(ticker)
            fund = get_fundamentals(ticker)
            if tech:
                articles = get_ticker_news(ticker, fund.get("name",""), limit=5)
                news_r   = score_news_articles(articles)
                ts,_,_   = score_technicals(tech)
                fs,_,_   = score_fundamentals(fund)
                up,dn,verdict,conf = compute_prediction(ts, fs, news_r["sentiment_score"])
                results.append({"ticker": ticker, "name": fund.get("name",ticker),
                                 "price": tech["current_price"], "pct_1d": tech["pct_1d"],
                                 "pct_7d": tech["pct_7d"], "verdict": verdict, "up_prob": up,
                                 "rsi": tech["rsi"], "sentiment": news_r["sentiment_label"]})
        except: pass
    return jsonify({"watchlist": results})


@app.route("/watchlist/add", methods=["POST"])
def add_watchlist():
    ticker = request.get_json().get("ticker","").strip().upper()
    if ticker and ticker not in watchlist: watchlist.append(ticker)
    return jsonify({"watchlist": watchlist})


@app.route("/watchlist/remove", methods=["POST"])
def remove_watchlist():
    ticker = request.get_json().get("ticker","").strip().upper()
    if ticker in watchlist: watchlist.remove(ticker)
    return jsonify({"watchlist": watchlist})


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    msg  = data.get("message","")
    ctx  = data.get("context_ticker", None)
    if not msg: return jsonify({"error": "No message"}), 400
    return jsonify({"response": chatbot_response(msg, ctx)})


if __name__ == "__main__":
    print("\n  MarketIQ running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
