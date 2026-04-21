"""
Stock Predictor Web App - Flask Backend

Install:
    pip3 install flask yfinance requests beautifulsoup4

Run:
    python3 app.py

Then open your browser to: http://localhost:5000
"""

from flask import Flask, render_template, request, jsonify
import json
from datetime import datetime, timedelta

try:
    import yfinance as yf
except ImportError:
    print("Run: pip3 install yfinance")
    exit(1)

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Run: pip3 install requests beautifulsoup4")
    exit(1)

app = Flask(__name__)


# ── Data Fetchers ─────────────────────────────────────────────────────────────

def get_technical_indicators(ticker):
    stock = yf.Ticker(ticker)
    hist  = stock.history(period="60d")
    if hist.empty:
        return None
    close  = hist["Close"]
    volume = hist["Volume"]
    sma20  = close.rolling(20).mean().iloc[-1]
    sma50  = close.rolling(50).mean().iloc[-1] if len(close) >= 50 else sma20
    delta  = close.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / loss
    rsi    = (100 - 100 / (1 + rs)).iloc[-1]
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
        "current_price":        round(current, 2),
        "prev_close":           round(prev_day, 2),
        "sma20":                round(sma20, 2),
        "sma50":                round(sma50, 2),
        "rsi":                  round(rsi, 1),
        "pct_1d":               round(((current - prev_day)  / prev_day)  * 100, 2),
        "pct_7d":               round(((current - week_ago)  / week_ago)  * 100, 2),
        "pct_30d":              round(((current - month_ago) / month_ago) * 100, 2),
        "volume_ratio":         round(recent_vol / avg_vol if avg_vol else 1.0, 2),
        "high_52w":             round(high_52w, 2),
        "low_52w":              round(low_52w, 2),
        "pct_from_52w_high":    round(((current - high_52w) / high_52w) * 100, 1),
    }


def get_fundamentals(ticker):
    info = yf.Ticker(ticker).info
    return {
        "name":            info.get("longName", ticker),
        "sector":          info.get("sector", "N/A"),
        "pe_ratio":        info.get("trailingPE"),
        "forward_pe":      info.get("forwardPE"),
        "beta":            info.get("beta"),
        "analyst_target":  info.get("targetMeanPrice"),
        "recommendation":  info.get("recommendationKey", "N/A"),
        "revenue_growth":  info.get("revenueGrowth"),
        "earnings_growth": info.get("earningsGrowth"),
        "market_cap":      info.get("marketCap"),
    }


def get_news_headlines(ticker, company_name):
    headlines = []
    url = f"https://finance.yahoo.com/quote/{ticker}/news"
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup.find_all("h3", limit=20):
            text = tag.get_text(strip=True)
            if len(text) > 20:
                headlines.append(text)
    except Exception:
        pass
    if not headlines:
        for n in (yf.Ticker(ticker).news or [])[:15]:
            t = n.get("title", "")
            if t:
                headlines.append(t)
    return headlines[:15]


# ── Scoring ───────────────────────────────────────────────────────────────────

BULLISH_WORDS = ["surge","soar","jump","gain","rally","beat","record","growth",
    "profit","upgrade","buy","strong","positive","rise","boost","exceed",
    "outperform","bullish","expand","partnership","deal","launch","optimistic",
    "recovery","dividend","revenue"]

BEARISH_WORDS = ["fall","drop","decline","loss","miss","downgrade","sell","weak",
    "negative","cut","layoff","lawsuit","investigation","bearish","concern",
    "risk","warning","recall","debt","deficit","fraud","lower","disappoint",
    "reduce","contract","restructure","fine"]


def score_news(headlines):
    bull, bear = 0, 0
    matched = {"bullish": [], "bearish": []}
    for h in headlines:
        lower = h.lower()
        for w in BULLISH_WORDS:
            if w in lower:
                bull += 1
                matched["bullish"].append(w)
        for w in BEARISH_WORDS:
            if w in lower:
                bear += 1
                matched["bearish"].append(w)
    total = bull + bear
    if total == 0:
        score, label = 0, "NEUTRAL"
    else:
        score = round(((bull - bear) / total) * 100, 1)
        label = "POSITIVE" if score > 20 else "NEGATIVE" if score < -20 else "MIXED"
    return {
        "sentiment_label": label,
        "sentiment_score": score,
        "bullish_hits":    bull,
        "bearish_hits":    bear,
        "top_bullish":     list(set(matched["bullish"]))[:5],
        "top_bearish":     list(set(matched["bearish"]))[:5],
    }


def score_technicals(t):
    score, bull, bear = 0, [], []
    if t["current_price"] > t["sma20"]:
        score += 15; bull.append(f"Price above SMA20 (${t['sma20']})")
    else:
        score -= 15; bear.append(f"Price below SMA20 (${t['sma20']})")
    if t["current_price"] > t["sma50"]:
        score += 15; bull.append(f"Price above SMA50 (${t['sma50']})")
    else:
        score -= 15; bear.append(f"Price below SMA50 (${t['sma50']})")
    rsi = t["rsi"]
    if rsi < 30:
        score += 20; bull.append(f"RSI oversold at {rsi} — potential bounce")
    elif rsi > 70:
        score -= 20; bear.append(f"RSI overbought at {rsi} — potential pullback")
    else:
        score += 5;  bull.append(f"RSI neutral at {rsi}")
    if t["pct_7d"] > 3:
        score += 10; bull.append(f"Strong 7-day momentum (+{t['pct_7d']}%)")
    elif t["pct_7d"] < -3:
        score -= 10; bear.append(f"Weak 7-day momentum ({t['pct_7d']}%)")
    if t["pct_30d"] > 5:
        score += 10; bull.append(f"Strong 30-day trend (+{t['pct_30d']}%)")
    elif t["pct_30d"] < -5:
        score -= 10; bear.append(f"Weak 30-day trend ({t['pct_30d']}%)")
    if t["volume_ratio"] > 1.5:
        if t["pct_1d"] > 0:
            score += 10; bull.append(f"High volume on up day ({t['volume_ratio']}x avg)")
        else:
            score -= 10; bear.append(f"High volume on down day ({t['volume_ratio']}x avg)")
    pfh = t["pct_from_52w_high"]
    if pfh > -5:
        score += 5;  bull.append(f"Near 52-week high ({pfh}% away)")
    elif pfh < -30:
        score += 8;  bull.append(f"Far below 52-week high — potential upside ({pfh}%)")
    return score, bull, bear


def score_fundamentals(f):
    score, bull, bear = 0, [], []
    rec = (f.get("recommendation") or "").lower()
    if rec in ("strong_buy", "buy"):
        score += 20; bull.append(f"Analyst rating: {rec.replace('_',' ').title()}")
    elif rec in ("underperform", "sell", "strong_sell"):
        score -= 20; bear.append(f"Analyst rating: {rec.replace('_',' ').title()}")
    pe = f.get("pe_ratio")
    if pe and pe < 15:
        score += 10; bull.append(f"Low P/E ratio ({round(pe,1)}) — potentially undervalued")
    elif pe and pe > 50:
        score -= 5;  bear.append(f"High P/E ratio ({round(pe,1)}) — priced for perfection")
    rg = f.get("revenue_growth")
    if rg and rg > 0.1:
        score += 10; bull.append(f"Strong revenue growth ({round(rg*100,1)}%)")
    elif rg and rg < 0:
        score -= 10; bear.append(f"Negative revenue growth ({round(rg*100,1)}%)")
    eg = f.get("earnings_growth")
    if eg and eg > 0.1:
        score += 10; bull.append(f"Strong earnings growth ({round(eg*100,1)}%)")
    elif eg and eg < 0:
        score -= 10; bear.append(f"Negative earnings growth ({round(eg*100,1)}%)")
    beta = f.get("beta")
    if beta and beta > 2:
        bear.append(f"High beta ({round(beta,2)}) — high volatility")
    elif beta and beta < 0.5:
        bull.append(f"Low beta ({round(beta,2)}) — stable stock")
    return score, bull, bear


def compute_prediction(tech_score, fund_score, news_score_val):
    total    = (tech_score * 0.5) + (fund_score * 0.3) + (news_score_val * 0.2)
    clamped  = max(-80, min(80, total))
    up_prob  = round(50 + (clamped * 0.4))
    dn_prob  = 100 - up_prob
    if up_prob >= 65:
        verdict    = "BULLISH"
        confidence = "HIGH" if up_prob >= 75 else "MEDIUM"
    elif up_prob <= 35:
        verdict    = "BEARISH"
        confidence = "HIGH" if up_prob <= 25 else "MEDIUM"
    else:
        verdict    = "NEUTRAL"
        confidence = "LOW"
    return up_prob, dn_prob, verdict, confidence


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data      = request.get_json()
    ticker    = data.get("ticker", "").strip().upper()
    timeframe = data.get("timeframe", "1 week")

    if not ticker:
        return jsonify({"error": "No ticker provided"}), 400

    try:
        tech = get_technical_indicators(ticker)
        if not tech:
            return jsonify({"error": f"Could not find data for {ticker}. Check the ticker symbol."}), 404

        fund     = get_fundamentals(ticker)
        headlines = get_news_headlines(ticker, fund.get("name", ticker))
        news_result = score_news(headlines)

        tech_score, bull_tech, bear_tech = score_technicals(tech)
        fund_score, bull_fund, bear_fund = score_fundamentals(fund)

        up_prob, dn_prob, verdict, confidence = compute_prediction(
            tech_score, fund_score, news_result["sentiment_score"]
        )

        target  = fund.get("analyst_target")
        current = tech["current_price"]
        upside  = round(((target - current) / current) * 100, 1) if target else None

        return jsonify({
            "ticker":      ticker,
            "name":        fund.get("name", ticker),
            "timeframe":   timeframe,
            "verdict":     verdict,
            "confidence":  confidence,
            "up_prob":     up_prob,
            "dn_prob":     dn_prob,
            "tech":        tech,
            "fund":        fund,
            "upside":      upside,
            "news": {
                **news_result,
                "headlines": headlines[:5],
            },
            "factors": {
                "bullish": (bull_tech + bull_fund)[:5],
                "bearish": (bear_tech + bear_fund)[:5],
            },
            "scores": {
                "technical":   tech_score,
                "fundamental": fund_score,
                "news":        news_result["sentiment_score"],
            }
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Stock Predictor running at http://localhost:5000\n")
    app.run(debug=True, port=5000)
