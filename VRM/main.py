import os, json
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from plotly.offline import plot

# =====================================================

analytics_path = "Files"
os.makedirs(analytics_path, exist_ok=True)
clear = lambda: os.system("cls" if os.name == "nt" else "clear")

title4 = r"""
██████╗ ██████╗  ██████╗      ██╗███████╗ ██████╗████████╗    ██╗   ██╗██████╗ ███╗   ███╗
██╔══██╗██╔══██╗██╔═══██╗     ██║██╔════╝██╔════╝╚══██╔══╝    ██║   ██║██╔══██╗████╗ ████║
██████╔╝██████╔╝██║   ██║     ██║█████╗  ██║        ██║       ██║   ██║██████╔╝██╔████╔██║
██╔═══╝ ██╔══██╗██║   ██║██   ██║██╔══╝  ██║        ██║       ╚██╗ ██╔╝██╔══██╗██║╚██╔╝██║
██║     ██║  ██║╚██████╔╝╚█████╔╝███████╗╚██████╗   ██║        ╚████╔╝ ██║  ██║██║ ╚═╝ ██║
╚═╝     ╚═╝  ╚═╝ ╚═════╝  ╚════╝ ╚══════╝ ╚═════╝   ╚═╝         ╚═══╝  ╚═╝  ╚═╝╚═╝     ╚═╝
"""

output = ""
for ch in title4:
    if ch == "█":output += "\033[97m" + ch + "\033[0m"
    elif ch in "╔╗╚╝═║":output += "\033[92m" + ch + "\033[0m"
    else: output += ch

print(output)

print('\n\n[ \033[92m> \033[0m] Data retrieval in progress.')

# --- List ---
all_symbols = ["K", "EA", "PEP", "T", "TRV", "JNJ", "GIS", "ED", "PG",
    "DUK", "CL", "VZ", "HSY", "COST", "WMT", "MCD", "SO", "MNST", "IBM", "BKR",
    "MSFT", "CDW", "UNH", "AMGN", "ADBE", "LULU", "AMZN", "CRM", "SBUX", "VRTX",
    "META", "BMY", "CDNS", "AMAT", "PYPL", "TTD", "MRK", "BKNG", "GOOG", "NFLX",
    "TSLA", "AMD", "MU", "LINT", "UBER",
    "SHY", "TLT", "AGG", "BND", "LQD"]

# --- Extra Velocity List---
extra_symbols = ["PARK", "BOXX", "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "AMD", "SMH", "SOXX", "AVGO", "IBIT", "BITO", "COIN", "QQQ", "VGT", 
    "ARKK", "COST", "JPM", "XOM", "PFE", "JNJ", "WMT"]

# --- Bond ETFs List---
bond_etfs = ["SHY", "TLT", "AGG", "BND", "LQD"]

all_symbols = [s.strip() for s in all_symbols if " " not in s] 
stock_symbols = [s for s in all_symbols if s not in bond_etfs]

benchmark_ticker = "^IXIC"
investment_amount = 2_000_000_000 
z_99 = 2.33 

sharpe_metrics = {}

# --- Styles ---
def get_style(name: str):
    nm = name.lower()
    if "serenity" in nm:
        return {
            "accent": "#00ff9d", "accent_soft": "#059669", "accent_alt": "#34d399",
            "bg": "#030304", "grid": "rgba(255,255,255,0.1)", "danger": "#ff2d55"
        }
    elif "velocity" in nm:
        return {
            "accent": "#ff2d55", "accent_soft": "#be123c", "accent_alt": "#fb7185",
            "bg": "#030304", "grid": "rgba(255,255,255,0.1)", "danger": "#ff2d55"
        }
    return {"accent": "#60a5fa", "bg": "#030304", "grid": "rgba(255,255,255,0.1)"}


def base_layout(title: str, style: dict):
    return dict(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=14, color=style["accent"], family="Space Mono, monospace")),
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", size=12, color="#d1d5db"),
        margin=dict(l=40, r=120, t=60, b=40),
        xaxis=dict(showgrid=True, gridcolor=style["grid"], zeroline=False),
        yaxis=dict(showgrid=True, gridcolor=style["grid"], zeroline=False),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.01, 
            bgcolor="rgba(0,0,0,0)",
            font=dict(size=10),
            itemsizing='constant'
        )
    )

# --- Step 1 : Data & Scoring ---
print(f"[ \033[92m> \033[0m] Downloading data for {len(all_symbols)} assets.")
try:
    # Force le téléchargement
    df_all = yf.Tickers(all_symbols).history(period="5y", auto_adjust=True)["Close"]
except Exception as e:
    print(f"[ \033[92m> \033[0m] Error : {e}")
    df_all = pd.DataFrame()

df_all = df_all.tz_localize(None).dropna(axis=1, how='all')
valid_symbols = df_all.columns.tolist()

# -- Data Extra Velocity --
print(f"\n[ \033[92m2 \033[0m] Downloading data for {len(extra_symbols)} assets.")
try:df_extra = yf.Tickers(extra_symbols).history(period="5y", auto_adjust=True)["Close"]
except Exception as e:
    print(f"[ \033[92m> \033[0m] Error: {e}")
    df_extra = pd.DataFrame()
df_extra = df_extra.tz_localize(None).dropna(axis=1, how='all')
daily_returns_all = np.log(df_all / df_all.shift(1)).dropna()
std_daily = daily_returns_all.std()
var_10d_assets = z_99 * std_daily * np.sqrt(10) * investment_amount
var_df = pd.DataFrame({"Asset": std_daily.index,"Daily Std": std_daily.values,"VaR_10d_99pct_2bn": var_10d_assets.values})
var_df.to_csv(os.path.join(analytics_path, "asset_10d_var_2bn_99pct.csv"), index=False)

# -- Benchmark & Beta --
try:
    bench_prices = yf.Ticker(benchmark_ticker).history(period="5y", auto_adjust=True)["Close"].tz_localize(None)
    bench_ret = np.log(bench_prices / bench_prices.shift(1)).dropna()
except:bench_ret = pd.Series(dtype=float)

common_idx = daily_returns_all.index.intersection(bench_ret.index)
daily_sync = daily_returns_all.loc[common_idx]
bench_sync = bench_ret.loc[common_idx]

betas = {}
for s in valid_symbols:
    if s in daily_sync.columns and not bench_sync.empty:
        cov = np.cov(daily_sync[s], bench_sync)[0, 1]
        var_b = np.var(bench_sync)
        betas[s] = cov / var_b if var_b > 0 else 0
    else:betas[s] = 1.0
beta_series = pd.Series(betas)

# -- Scoring --
std_norm = (std_daily - std_daily.min()) / (std_daily.max() - std_daily.min())
beta_norm = (beta_series - beta_series.min()) / (beta_series.max() - beta_series.min())
std_norm = std_norm.reindex(valid_symbols).fillna(0)
beta_norm = beta_norm.reindex(valid_symbols).fillna(0)

risk_scores = []
for s in valid_symbols:
    if s in bond_etfs:
        score = std_norm[s]
    else:
        score = 0.7 * std_norm[s] + 0.3 * beta_norm[s]
    risk_scores.append(score)

risk_table = pd.DataFrame({"Asset": valid_symbols, "RiskScore": risk_scores}).sort_values("RiskScore")

half = len(valid_symbols) // 2
serenity_symbols = risk_table.head(half)["Asset"].tolist()
velocity_symbols = risk_table.tail(len(valid_symbols) - half)["Asset"].tolist()

print(f"[ \033[92m> \033[0m] Serenity: {len(serenity_symbols)} assets | Velocity: {len(velocity_symbols)} assets.")

# -- Graphics --
def plot_base_100(prices, name):
    style = get_style(name)
    df100 = prices / prices.iloc[0] * 100
    fig = go.Figure()
    for col in df100.columns:fig.add_trace(go.Scatter(x=df100.index, y=df100[col], mode="lines", line=dict(width=1.5), opacity=0.8, name=col))
    lay = base_layout(f"BASE 100 INDEX (5Y)", style)
    lay['margin'] = dict(r=250, l=50, t=50, b=40) 
    fig.update_layout(**lay)
    plot(fig, filename=os.path.join(analytics_path, f"{name.lower()}_base100.html"), auto_open=False)
def plot_log_returns(returns, name):
    style = get_style(name)
    fig = go.Figure()
    for col in returns.columns:fig.add_trace(go.Scatter(x=returns.index, y=returns[col], mode="lines", line=dict(width=0.8), opacity=0.6, name=col))
    lay = base_layout(f"LOGARITHMIC RETURNS (5Y)", style)
    lay['margin'] = dict(r=250, l=50, t=50, b=40)
    fig.update_layout(**lay)
    plot(fig, filename=os.path.join(analytics_path, f"{name.lower()}_log_returns.html"), auto_open=False)
def plot_vol_stats(name, daily_ret):
    style = get_style(name)
    ann_vol = daily_ret.std() * np.sqrt(252)
    fig_vol = go.Figure(go.Bar(x=ann_vol.index, y=ann_vol.values, marker=dict(color=style["accent_soft"])))
    lay_v = base_layout(f"IMPLIED VOLATILITY", style)
    lay_v['xaxis']['tickangle'] = -45
    lay_v['xaxis']['tickfont'] = dict(size=10)
    lay_v['margin'] = dict(b=100, r=100, t=50, l=50) 
    fig_vol.update_layout(**lay_v)
    plot(fig_vol, filename=os.path.join(analytics_path, f"{name.lower()}_volatility_chart.html"), auto_open=False)
    ann_ret = daily_ret.mean() * 252
    sorted_ret = ann_ret.sort_values()
    fig_pyr = go.Figure(go.Bar(x=sorted_ret.values, y=sorted_ret.index, orientation="h", marker=dict(color=style["accent"])))
    lay_p = base_layout(f"RETURN SKEW PYRAMID", style)
    lay_p['margin'] = dict(l=100, r=50, t=50, b=40)
    fig_pyr.update_layout(**lay_p)
    plot(fig_pyr, filename=os.path.join(analytics_path, f"{name.lower()}_returns_pyramid.html"), auto_open=False)
def plot_var_curve(name, optimal_vol):
    style = get_style(name)
    sigma_daily = optimal_vol / np.sqrt(252)
    days = np.arange(1, 11)
    var_vals = z_99 * sigma_daily * np.sqrt(days) * investment_amount
    fig = go.Figure(go.Scatter(x=days, y=var_vals, mode="lines+markers", line=dict(color=style["accent"], width=3)))
    lay = base_layout(f"VaR 99% CURVE (USD)", style)
    fig.update_layout(**lay)
    plot(fig, filename=os.path.join(analytics_path, f"{name.lower()}_var_10d_curve.html"), auto_open=False)

# -- Optimization --
def run_optimization(fund_name, symbols, custom_df=None):
    print(f"\n\n{'='*20} PROCESSING : {fund_name.upper()} {'='*20}")
    if custom_df is not None:raw_df = custom_df[symbols].copy()
    else:
        valid_s = [s for s in symbols if s in df_all.columns]
        raw_df = df_all[valid_s].copy()
    df_display = raw_df.ffill().bfill()
    daily_returns_display = np.log(df_display / df_display.shift(1)).fillna(0)
    df_optim = raw_df.dropna()
    if df_optim.empty:
        print(f"[ \033[92m> \033[0m] Error: empty data for {fund_name}.")
        df_optim = raw_df.fillna(0)
    daily_returns_optim = np.log(df_optim / df_optim.shift(1)).dropna()
    if daily_returns_optim.empty:
        print("[ \033[92m> \033[0m] Error")
        return
    mean_returns = daily_returns_optim.mean() * 252
    cov_matrix = daily_returns_optim.cov() * 252
    plot_base_100(df_display, fund_name)
    plot_log_returns(daily_returns_display, fund_name)
    plot_vol_stats(fund_name, daily_returns_optim) 
    print(f"[ \033[92m> \033[0m] Running Monte Carlo for {fund_name}.")
    np.random.seed(42)
    num_portfolios = 75000
    results = np.zeros((num_portfolios, 3))
    weights_list = []
    if cov_matrix.empty:
         print("[ \033[92m> \033[0m] Matrix Covariance empty. Skip.")
         return
    # -- Risk Free Rate --
    rf_series = yf.Ticker('^IRX').history(period='5y')['Close']
    rf = rf_series.dropna().mean() / 100.0

    # -- Sharpe Ratio Calculation --
    for i in range(num_portfolios):
        weights = np.random.random(len(daily_returns_optim.columns))
        weights /= np.sum(weights)
        w = pd.Series(weights, index=daily_returns_optim.columns)
        port_return = float(w @ mean_returns)
        port_volatility = float(np.sqrt(w @ cov_matrix @ w))
        sharpe_ratio = (port_return - rf) / port_volatility if port_volatility > 0 else 0
        results[i, 0] = port_return
        results[i, 1] = port_volatility
        results[i, 2] = sharpe_ratio
        weights_list.append(weights)
    results_df = pd.DataFrame(results, columns=['Return', 'Volatility', 'Sharpe Ratio'])
    max_idx = results_df['Sharpe Ratio'].idxmax()
    optimal_ret = results_df.loc[max_idx, 'Return']
    optimal_vol = results_df.loc[max_idx, 'Volatility']
    max_sharpe_val = results_df.loc[max_idx, 'Sharpe Ratio']
    best_weights = weights_list[max_idx]
    print(f"[ \033[92m> \033[0m] {fund_name} Max Sharpe : {max_sharpe_val:.2f}")
    sharpe_metrics[fund_name.lower() + "_sharpe"] = float(max_sharpe_val)
    best_weights_df = pd.DataFrame({'Asset': daily_returns_optim.columns, 'Weight': best_weights})
    best_weights_df = best_weights_df.sort_values(by="Weight", ascending=False)
    best_weights_df_visible = best_weights_df[best_weights_df['Weight'] > 0.01] 
    style = get_style(fund_name)
    cml_x = np.linspace(0, max(results_df['Volatility']), 120)
    cml_y = rf + (optimal_ret - rf) / optimal_vol * cml_x
    fig_front = go.Figure()
    fig_front.add_trace(go.Scatter(x=results_df["Volatility"], 
    y=results_df["Return"],mode="markers",
    marker=dict(size=4, opacity=0.3, color=style["accent_soft"]), 
    name="Portfolios"))
    label_sharpe = f"Max Sharpe Ratio: {max_sharpe_val:.2f}" 
    fig_front.add_trace(go.Scatter(
        x=[optimal_vol], y=[optimal_ret], mode="markers+text", 
        marker=dict(size=14, color=style["accent"], line=dict(color="white", width=2)), 
        text=[label_sharpe], textposition="top center", name="Max Sharpe"))
    fig_front.add_trace(go.Scatter(
        x=cml_x, y=cml_y, mode="lines", 
        line=dict(dash="dash", color=style["accent"]), name="CML"))
    if fund_name == "Extra_Velocity":chart_title = f"Sharpe: {max_sharpe_val:.2f}"
    else:chart_title = f"Sharpe: {max_sharpe_val:.2f}"
    lay_f = base_layout(chart_title, style)
    lay_f['legend'] = dict(x=0.02, y=0.98, bgcolor='rgba(0,0,0,0.5)')
    fig_front.update_layout(**lay_f)
    plot(fig_front, filename=os.path.join(analytics_path, f"{fund_name.lower()}_markowitz_sharpe.html"), auto_open=False)
    
    fig_alloc = go.Figure(go.Pie(
        labels=best_weights_df_visible['Asset'], 
        values=best_weights_df_visible['Weight'], hole=0.5, 
        textinfo="label+percent",textposition="outside",
        marker=dict(colors=[style['accent'], style['accent_soft'], style['accent_alt'], '#444', '#333'])))
    lay_a = base_layout(f"OPTIMAL ALLOCATION", style)
    lay_a['showlegend'] = False
    fig_alloc.update_layout(**lay_a)
    plot(fig_alloc, filename=os.path.join(analytics_path, f"{fund_name.lower()}_allocation_chart.html"), auto_open=False)
    port_idx = (df_optim * best_weights).sum(axis=1)
    port_idx = port_idx / port_idx.iloc[0] * 100
    dd = (port_idx - port_idx.cummax()) / port_idx.cummax()
    fig_dd = go.Figure(go.Scatter(x=dd.index, y=dd, mode="lines", line=dict(color=style["danger"], width=2), fill="tozeroy", fillcolor="rgba(255, 45, 85, 0.1)"))
    lay_d = base_layout(f"MAX DRAWDOWN", style)
    fig_dd.update_layout(**lay_d)
    plot(fig_dd, filename=os.path.join(analytics_path, f"{fund_name.lower()}_drawdown.html"), auto_open=False)
    plot_var_curve(fund_name, optimal_vol)

# -- Debt Ratio --
def compute_top3_debt_ratio():
    print(f"\n{'='*20} Debt Ratio top 3 {'='*20}")
    print(f"[ \033[92m> \033[0m] Calculation Debt Ratio.")
    potential_giants = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'META', 'TSLA']
    top_candidates = [t for t in potential_giants]
    selection = top_candidates[:3]
    ratios = []
    for t in selection:
        try:
            info = yf.Ticker(t).info
            total_debt = info.get('totalDebt', 0)
            market_cap = info.get('marketCap', 0)
            
            if total_debt and market_cap:
                ratio = (total_debt / (total_debt + market_cap)) * 100
                ratios.append(ratio)
                
                print(f"[ \033[92m> \033[0m] {t:<5}| Debt: ${total_debt/1e9:>7.1f}B | Cap: ${market_cap/1e9:>7.1f}B | Ratio: \033[96m{ratio:.2f}%\033[0m")
            else:
                print(f"[ \033[92m> \033[0m] {t:<5} | Data incomplete.")
        except Exception as e:
            print(f"[ \033[92m> \033[0m] {t:<5} | Error: {e}")
    if ratios:
        avg_ratio = sum(ratios) / len(ratios)
        print(f"\n[ \033[92m> \033[0m] Average Debt Ratio (Top 3): \033[92m{avg_ratio:.2f}%\033[0m")

# -- Execution --

# -- Serenity --
if len(serenity_symbols) > 0:run_optimization("Serenity", serenity_symbols)

# -- Velocity --
if len(velocity_symbols) > 0:run_optimization("Velocity", velocity_symbols)

# -- Extra Serenity --
if len(extra_symbols) > 0:run_optimization("Extra_Velocity", extra_symbols, custom_df=df_extra)

# -- Metric.js File --
print("\n[ \033[92m> \033[0m] Files generation metrics.js")
with open(os.path.join(analytics_path, "metrics.js"), "w") as f:f.write(f"const generated_metrics = {str(sharpe_metrics)};")

# -- Debt Ratio Execution --
compute_top3_debt_ratio()

# -- News Scraper --
print("[ \033[92m> \033[0m] Scraping news with Yfinance")

try:
    news_raw = yf.Ticker(benchmark_ticker).news
    news_list = []
    for item in news_raw:
        content = item.get("content", item)
        title = content.get("title")
        summary = content.get("summary", "")
        publisher = content.get("provider", {}).get("displayName")
        pubDate = content.get("pubDate") 
        link = content.get("canonicalUrl", {}).get("url")
        thumbnail_data = content.get("thumbnail") or {}
        thumbnail_url = thumbnail_data.get("originalUrl")
        pubDate_iso = None
        if pubDate:
            if 'T' in pubDate and (pubDate.endswith('Z') or '+' in pubDate or '-' in pubDate[-6:]):pubDate_iso = pubDate 
            else:pubDate_iso = pubDate + "Z"
        entry = {
            "title": title,"summary": summary,
            "publisher": publisher,"pubDate": pubDate_iso,
            "link": link,"thumbnail": thumbnail_url} 
        if entry["title"] and entry["link"]:news_list.append(entry)
    news_js_path = os.path.join(analytics_path, "news_data.js")
    with open(news_js_path, "w", encoding="utf-8") as f:
        js_content = f"const EXTERNAL_NEWS_DATA = {json.dumps(news_list, indent=4, ensure_ascii=False)};"
        f.write(js_content)     
    print(f"[ \033[92m> \033[0m] {len(news_list)} news scraped in : {news_js_path}")
except Exception as e:
    print(f"[ \033[92m> \033[0m] Error : {e}")
print("\n[ \033[92m> \033[0m] Website ready.")