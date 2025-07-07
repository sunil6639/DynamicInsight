import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
from datetime import datetime, timedelta

# --- Configuration ---
magnificent_7_tickers = [
    "AAPL",  # Apple Inc.
    "MSFT",  # Microsoft Corp.
    "GOOGL", # Alphabet Inc. (Class A)
    "AMZN",  # Amazon.com, Inc.
    "NVDA",  # NVIDIA Corp.
    "META",  # Meta Platforms, Inc.
    "TSLA"   # Tesla, Inc.
]

# --- Custom CSS for Background ---
# You can choose one of these options, or combine them carefully.
# Option 1: Solid Color Background (Simple and clean)
# background_css = """
# <style>
# body {
#     background-color: #f0f2f6; /* Light gray */
# }
# </style>
# """

# Option 2: Gradient Background (Modern and visually appealing)
background_css = """
<style>
body {
    background: linear-gradient(to right, #ece9e6, #ffffff); /* A subtle gradient from light warm gray to white */
}
.stApp {
    background-color: rgba(0,0,0,0); /* Ensure the main app content is transparent */
}
</style>
"""

# Option 3: Image Background (Needs a publicly accessible URL)
# Uncomment and replace YOUR_IMAGE_URL with an actual URL
# background_css = """
# <style>
# body {
#     background-image: url("https://images.unsplash.com/photo-1519681393784-a4b162fa0150?q=80&w=2940&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"); /* Example: abstract subtle pattern */
#     background-size: cover;
#     background-position: center;
#     background-repeat: no-repeat;
#     background-attachment: fixed; /* Keeps background fixed when scrolling */
# }
# .stApp {
#     background-color: rgba(255,255,255,0.8); /* Semi-transparent white to make text readable */
# }
# </style>
# """

st.markdown(background_css, unsafe_allow_html=True)


# --- Data Fetching (with caching and error handling) ---

# Global cache for Ticker objects to avoid re-initializing them
@st.cache_resource(ttl=3600) # Cache the yfinance.Ticker objects for 1 hour
def get_yf_ticker_object(ticker_symbol):
    """Returns a cached yfinance Ticker object."""
    return yf.Ticker(ticker_symbol)

@st.cache_data(ttl=3600) # Cache fundamental data for 1 hour
def get_magnificent_7_fundamentals(tickers):
    """
    Fetches fundamental data for the Magnificent 7 stocks from Yahoo Finance.
    """
    fundamental_data = {}
    for ticker_symbol in tickers:
        try:
            ticker = get_yf_ticker_object(ticker_symbol)
            info = ticker.info
            
            def get_safe_info(key):
                return info.get(key)

            fundamental_data[ticker_symbol] = {
                "Short Name": get_safe_info("shortName"),
                "Sector": get_safe_info("sector"),
                "Industry": get_safe_info("industry"),
                "Market Cap": get_safe_info("marketCap"),
                "Enterprise Value": get_safe_info("enterpriseValue"),
                "Trailing P/E": get_safe_info("trailingPE"),
                "Forward P/E": get_safe_info("forwardPE"),
                "Dividend Yield": get_safe_info("dividendYield"),
                "Beta": get_safe_info("beta"),
                "Revenue (TTM)": get_safe_info("trailingAnnualRevenue"),
                "Gross Margins": get_safe_info("grossMargins"),
                "Profit Margins": get_safe_info("profitMargins"),
                "Return on Equity": get_safe_info("returnOnEquity"),
                "Debt to Equity": get_safe_info("debtToEquity"),
                "Current Ratio": get_safe_info("currentRatio"),
                "Total Cash": get_safe_info("totalCash"),
                "Total Debt": get_safe_info("totalDebt"),
                "Free Cashflow (TTM)": get_safe_info("freeCashflow"),
                "Operating Cashflow (TTM)": get_safe_info("operatingCashflow"),
                "Earnings Growth (YoY)": get_safe_info("earningsGrowth"),
                "Recommendation Key": get_safe_info("recommendationKey"),
                "Target High Price": get_safe_info("targetHighPrice"),
                "Target Low Price": get_safe_info("targetLowPrice"),
                "Target Mean Price": get_safe_info("targetMeanPrice"),
                "Target Median Price": get_safe_info("targetMedianPrice"),
            }
        except Exception as e:
            st.error(f"Could not retrieve fundamental data for {ticker_symbol}: {e}")
            fundamental_data[ticker_symbol] = {"Error": str(e)}
    return fundamental_data

@st.cache_data(ttl=3600) # Cache historical data for 1 hour
def get_historical_data(ticker_symbol, start_date=None, end_date=None, period="1y", interval="1d"):
    """
    Fetches historical stock price data for a given ticker, period, or date range.
    Prioritizes start_date/end_date if provided.
    """
    try:
        # Use get_yf_ticker_object to get the cached Ticker object
        ticker = get_yf_ticker_object(ticker_symbol)
        
        if start_date and end_date:
            data = ticker.history(start=start_date, end=end_date, interval=interval)
        else:
            data = ticker.history(period=period, interval=interval)

        if data.empty:
            st.warning(f"No historical data returned for {ticker_symbol} for period {period} (or {start_date}-{end_date}), interval {interval}. This might be due to rate limiting, no available data, or too short a period/interval.")
            return pd.DataFrame({"Error": ["No data or rate limited"]})
        
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in data.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in data.columns]
            st.warning(f"Missing expected columns {missing_cols} in historical data for {ticker_symbol}. Data might be incomplete or malformed.")
            return pd.DataFrame({"Error": [f"Missing columns: {missing_cols}"]})

        data.index = pd.to_datetime(data.index) # Ensure datetime index
        return data
    except Exception as e:
        st.error(f"Error fetching historical data for {ticker_symbol} (Period: {period}, Interval: {interval}, Dates: {start_date} to {end_date}): {e}")
        return pd.DataFrame({"Error": [str(e)]})

@st.cache_data(ttl=3600)
def get_dividends(ticker_symbol):
    try:
        ticker = get_yf_ticker_object(ticker_symbol)
        dividends = ticker.dividends
        if dividends.empty:
            return pd.DataFrame({"Error": ["No dividend data"]})
        dividends = dividends.reset_index()
        dividends.columns = ['Date', 'Dividend']
        dividends['Date'] = pd.to_datetime(dividends['Date'])
        return dividends
    except Exception as e:
        # st.error(f"Error fetching dividend data for {ticker_symbol}: {e}")
        return pd.DataFrame({"Error": [str(e)]})

@st.cache_data(ttl=3600)
def get_earnings(ticker_symbol):
    try:
        ticker = get_yf_ticker_object(ticker_symbol)
        earnings = ticker.earnings # Annual earnings
        quarterly_earnings = ticker.quarterly_earnings # Quarterly earnings

        earnings_data = {}
        if not earnings.empty:
            earnings_data['annual'] = earnings.reset_index()
            earnings_data['annual']['Date'] = pd.to_datetime(earnings_data['annual']['Date'])
        if not quarterly_earnings.empty:
            earnings_data['quarterly'] = quarterly_earnings.reset_index()
            earnings_data['quarterly']['Date'] = pd.to_datetime(earnings_data['quarterly']['Date'])
        
        if not earnings_data:
             return {"Error": "No earnings data"}
        return earnings_data
    except Exception as e:
        # st.error(f"Error fetching earnings data for {ticker_symbol}: {e}")
        return {"Error": str(e)}

@st.cache_data(ttl=3600)
def get_institutional_holders(ticker_symbol):
    try:
        ticker = get_yf_ticker_object(ticker_symbol)
        holders = ticker.institutional_holders
        if holders.empty:
            return pd.DataFrame({"Error": ["No institutional holder data"]})
        holders.columns = ['Holder', 'Shares', 'Date Reported', '% Out', 'Value'] # Clean up column names
        return holders
    except Exception as e:
        # st.error(f"Error fetching institutional holder data for {ticker_symbol}: {e}")
        return pd.DataFrame({"Error": [str(e)]})


# --- Streamlit App Layout ---
st.set_page_config(layout="wide", page_title="Magnificent 7 Analysis - Advanced", page_icon="📈")

# --- GLOBAL HEADER WITH NAME ---
st.title("Magnificent 7 Stock Analysis: Advanced Dashboard")
st.markdown("Developed by **Sunil Satyanarayan**")
st.markdown("""
    Explore key fundamental metrics, historical price performance, and deeper insights
    for the "Magnificent 7" stocks with an enhanced interactive interface.
""")
st.markdown("---")


# --- Global Sidebar Controls ---
st.sidebar.header("Global Settings")
selected_m7_tickers_for_display = st.sidebar.multiselect(
    "Select Magnificent 7 Stocks to Display:",
    options=magnificent_7_tickers,
    default=magnificent_7_tickers, # All selected by default
    help="Choose which of the Magnificent 7 stocks to include in the analysis."
)

st.sidebar.markdown("---")
st.sidebar.info("Data refreshed every 1 hour (cached).")
st.sidebar.caption("Built with Streamlit, yfinance, Plotly.")

# --- Fetch Data (only for selected tickers from sidebar) ---
with st.spinner("Fetching all data for selected stocks... This might take a moment."):
    all_fundamentals = get_magnificent_7_fundamentals(selected_m7_tickers_for_display)

# Filter out tickers that had errors in fundamental data for main display
valid_tickers_with_data = [t for t in selected_m7_tickers_for_display if "Error" not in all_fundamentals.get(t, {})]
if not valid_tickers_with_data:
    st.error("No data could be retrieved for the selected tickers. Please check your internet connection or try again later.")
    st.stop() # Stop the app execution if no valid data


# --- Section 1: Overview Table (with Expander for details) ---
st.header("1. Comprehensive Overview")
st.markdown("A deep dive into the financial health, valuation, and analyst sentiment.")

overview_tab, analyst_tab = st.tabs(["Key Fundamentals", "Analyst Ratings"])

with overview_tab:
    display_data_overview = []
    for ticker in valid_tickers_with_data:
        data = all_fundamentals[ticker]
        row = {"Ticker": ticker}
        # Basic overview columns for main table
        row["Short Name"] = data.get("Short Name", "N/A")
        row["Sector"] = data.get("Sector", "N/A")
        row["Market Cap"] = f"${data.get('Market Cap', 0):,.0f}" if data.get('Market Cap') is not None else "N/A"
        row["Trailing P/E"] = f"{data.get('Trailing P/E', 0):.2f}" if data.get('Trailing P/E') is not None else "N/A"
        row["Profit Margins"] = f"{data.get('Profit Margins', 0):.2%}" if data.get('Profit Margins') is not None else "N/A"
        row["Revenue (TTM)"] = f"${data.get('Revenue (TTM)', 0):,.0f}" if data.get('Revenue (TTM)') is not None else "N/A"
        row["Return on Equity"] = f"{data.get('Return on Equity', 0):.2%}" if data.get('Return on Equity') is not None else "N/A"
        row["Dividend Yield"] = f"{data.get('Dividend Yield', 0):.2%}" if data.get('Dividend Yield') is not None else "N/A"
        display_data_overview.append(row)

    df_overview = pd.DataFrame(display_data_overview)

    if not df_overview.empty:
        st.dataframe(df_overview.set_index("Ticker"), use_container_width=True)

    with st.expander("More Detailed Fundamental Metrics (Click to Expand)", expanded=False):
        detailed_cols = [
            "Ticker", "Short Name", "Industry", "Enterprise Value", "Forward P/E",
            "Beta", "Debt to Equity", "Current Ratio", "Total Cash", "Total Debt",
            "Free Cashflow (TTM)", "Operating Cashflow (TTM)", "Earnings Growth (YoY)"
        ]
        
        display_data_detailed = []
        for ticker in valid_tickers_with_data:
            data = all_fundamentals[ticker]
            row = {"Ticker": ticker}
            for col in detailed_cols:
                original_key = col.replace(" (TTM)", "").replace(" (YoY)", "").replace(" ", "") # Simple key mapping
                # Handle specific formatting for detailed view
                val = data.get(original_key) # Use original key from yfinance 'info'
                if val is not None:
                    if "Value" in col or "Cash" in col or "Debt" in col or "Revenue" in col or "Cashflow" in col:
                        row[col] = f"${val:,.0f}"
                    elif "Ratio" in col or "Beta" in col or "P/E" in col:
                        row[col] = f"{val:.2f}"
                    elif "Growth" in col:
                        row[col] = f"{val:.2%}"
                    else:
                        row[col] = val
                else:
                    row[col] = "N/A"
            display_data_detailed.append(row)

        df_detailed = pd.DataFrame(display_data_detailed)
        if not df_detailed.empty:
            st.dataframe(df_detailed.set_index("Ticker"), use_container_width=True)
        else:
            st.info("No detailed fundamental data available for the selected companies.")

with analyst_tab:
    st.markdown("### Analyst Ratings and Price Targets")
    analyst_data = []
    for ticker in valid_tickers_with_data:
        data = all_fundamentals[ticker]
        row = {
            "Ticker": ticker,
            "Short Name": data.get("Short Name", "N/A"),
            "Recommendation": data.get("Recommendation Key", "N/A"),
            "Target High": f"${data.get('Target High Price', 0):,.2f}" if data.get('Target High Price') is not None else "N/A",
            "Target Low": f"${data.get('Target Low Price', 0):,.2f}" if data.get('Target Low Price') is not None else "N/A",
            "Target Mean": f"${data.get('Target Mean Price', 0):,.2f}" if data.get('Target Mean Price') is not None else "N/A",
            "Target Median": f"${data.get('Target Median Price', 0):,.2f}" if data.get('Target Median Price') is not None else "N/A",
        }
        analyst_data.append(row)
    
    df_analyst = pd.DataFrame(analyst_data)
    if not df_analyst.empty:
        st.dataframe(df_analyst.set_index("Ticker"), use_container_width=True)
    else:
        st.info("No analyst rating data available for the selected companies.")

st.markdown("---")

# --- Section 2: Individual Company Deep Dive (Tabs for different views) ---
st.header("2. Individual Company Deep Dive")
st.markdown("Explore detailed historical data, dividends, and earnings for a selected company.")

selected_company_for_deep_dive = st.selectbox(
    "Select a Company:",
    options=[all_fundamentals[t].get("Short Name", t) for t in valid_tickers_with_data],
    format_func=lambda x: x,
    key="deep_dive_company_select"
)

# Extract the ticker from the selected company name
deep_dive_ticker = None
for ticker, data in all_fundamentals.items():
    if data.get("Short Name") == selected_company_for_deep_dive or ticker == selected_company_for_deep_dive:
        deep_dive_ticker = ticker
        break

if deep_dive_ticker:
    st.subheader(f"Deep Dive: {all_fundamentals[deep_dive_ticker].get('Short Name', deep_dive_ticker)}")

    price_chart_tab, dividends_tab, earnings_tab, institutional_tab = st.tabs(
        ["Price Chart", "Dividend History", "Earnings History", "Institutional Holders"]
    )

    with price_chart_tab:
        st.markdown("#### Historical Price & Volume")
        col_price_chart_controls, col_price_chart = st.columns([1, 3])

        with col_price_chart_controls:
            period_options = {
                "1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo",
                "1 Year": "1y", "3 Years": "3y", "5 Years": "5y", "Max": "max"
            }
            selected_period_label = st.radio(
                "Select Period:",
                list(period_options.keys()),
                index=3, # Default to 1 Year
                key="price_period_radio"
            )
            selected_period = period_options[selected_period_label]
            
            st.markdown("---")
            st.markdown("**Or Custom Date Range:**")
            
            today = datetime.now().date()
            default_start = today - timedelta(days=365)
            
            custom_start_date = st.date_input("Start Date:", value=default_start, key="custom_start_date")
            custom_end_date = st.date_input("End Date:", value=today, key="custom_end_date")
            
            fetch_by_custom_range = st.checkbox("Use Custom Date Range", value=False, key="use_custom_range_checkbox")

        with col_price_chart:
            historical_data = None
            if fetch_by_custom_range:
                historical_data = get_historical_data(deep_dive_ticker, start_date=custom_start_date, end_date=custom_end_date)
            else:
                historical_data = get_historical_data(deep_dive_ticker, period=selected_period)
            
            if not historical_data.empty and "Error" not in historical_data.columns:
                fig = go.Figure()

                # Candlestick chart
                fig.add_trace(go.Candlestick(
                    x=historical_data.index,
                    open=historical_data['Open'],
                    high=historical_data['High'],
                    low=historical_data['Low'],
                    close=historical_data['Close'],
                    name='Candlestick',
                    increasing_line_color='green',
                    decreasing_line_color='red'
                ))

                # Volume bar chart (on a secondary y-axis)
                fig.add_trace(go.Bar(
                    x=historical_data.index,
                    y=historical_data['Volume'],
                    name='Volume',
                    yaxis='y2',
                    marker_color='rgba(0,128,0,0.5)' # Semi-transparent green
                ))

                fig.update_layout(
                    title=f'{deep_dive_ticker} Price and Volume',
                    xaxis_rangeslider_visible=False,
                    xaxis_title="Date",
                    yaxis_title="Price",
                    yaxis2=dict(
                        title="Volume",
                        overlaying="y",
                        side="right",
                        showgrid=False,
                        range=[0, historical_data['Volume'].max() * 1.5]
                    ),
                    legend_orientation="h",
                    legend_yanchor="bottom",
                    legend_y=1.02,
                    legend_xanchor="right",
                    legend_x=1,
                    hovermode="x unified",
                    height=550
                )
                st.plotly_chart(fig, use_container_width=True)
            elif "Error" in historical_data.columns:
                st.error(f"Error loading historical data: {historical_data['Error'].iloc[0]}")
            else:
                st.warning(f"No historical data available for {deep_dive_ticker} for the selected criteria.")

    with dividends_tab:
        st.markdown("#### Dividend History")
        dividends_df = get_dividends(deep_dive_ticker)
        if not dividends_df.empty and "Error" not in dividends_df.columns:
            st.dataframe(dividends_df.sort_values(by="Date", ascending=False), use_container_width=True)

            fig_div = px.bar(
                dividends_df,
                x="Date",
                y="Dividend",
                title=f'{deep_dive_ticker} Dividend Payout History',
                labels={"Dividend": "Dividend Per Share"},
                hover_data={"Dividend": ':.2f'}
            )
            fig_div.update_layout(xaxis_title="Date", yaxis_title="Dividend Per Share")
            st.plotly_chart(fig_div, use_container_width=True)
        elif "Error" in dividends_df.columns:
            st.error(f"Error loading dividend data: {dividends_df['Error'].iloc[0]}")
        else:
            st.info(f"No dividend history available for {deep_dive_ticker}.")

    with earnings_tab:
        st.markdown("#### Earnings History")
        earnings_data = get_earnings(deep_dive_ticker)
        
        if "Error" not in earnings_data:
            annual_earnings_df = earnings_data.get('annual')
            quarterly_earnings_df = earnings_data.get('quarterly')

            if annual_earnings_df is not None and not annual_earnings_df.empty:
                st.subheader("Annual Earnings")
                st.dataframe(annual_earnings_df.sort_values(by="Date", ascending=False), use_container_width=True)
                fig_annual_earnings = px.bar(
                    annual_earnings_df,
                    x="Date",
                    y=["Revenue", "Earnings"],
                    barmode="group",
                    title=f'{deep_dive_ticker} Annual Revenue & Earnings',
                    labels={"value": "Amount", "variable": "Metric"},
                    hover_data={"value": ':,.0f'}
                )
                st.plotly_chart(fig_annual_earnings, use_container_width=True)
            else:
                st.info(f"No annual earnings data available for {deep_dive_ticker}.")

            if quarterly_earnings_df is not None and not quarterly_earnings_df.empty:
                st.subheader("Quarterly Earnings")
                st.dataframe(quarterly_earnings_df.sort_values(by="Date", ascending=False), use_container_width=True)
                fig_quarterly_earnings = px.bar(
                    quarterly_earnings_df,
                    x="Date",
                    y=["Revenue", "Earnings"],
                    barmode="group",
                    title=f'{deep_dive_ticker} Quarterly Revenue & Earnings',
                    labels={"value": "Amount", "variable": "Metric"},
                    hover_data={"value": ':,.0f'}
                )
                st.plotly_chart(fig_quarterly_earnings, use_container_width=True)
            else:
                st.info(f"No quarterly earnings data available for {deep_dive_ticker}.")
        else:
            st.error(f"Error loading earnings data: {earnings_data['Error']}")


    with institutional_tab:
        st.markdown("#### Top Institutional Holders")
        institutional_holders_df = get_institutional_holders(deep_dive_ticker)
        if not institutional_holders_df.empty and "Error" not in institutional_holders_df.columns:
            st.dataframe(institutional_holders_df, use_container_width=True)
            
            top_n = st.slider("Show Top Institutional Holders (by Shares):", 5, 20, 10, key="top_holders_slider")
            if not institutional_holders_df.empty and 'Shares' in institutional_holders_df.columns:
                df_top_holders = institutional_holders_df.head(top_n).copy()
                df_top_holders['Shares'] = pd.to_numeric(df_top_holders['Shares'], errors='coerce')
                df_top_holders = df_top_holders.dropna(subset=['Shares'])

                if not df_top_holders.empty:
                    fig_holders = px.bar(
                        df_top_holders.sort_values(by='Shares', ascending=True),
                        x='Shares',
                        y='Holder',
                        orientation='h',
                        title=f'Top {top_n} Institutional Holders of {deep_dive_ticker}',
                        labels={'Shares': 'Shares Held', 'Holder': 'Institution'},
                        hover_data={'Shares': ':,.0f', '% Out': ':.2%'}
                    )
                    st.plotly_chart(fig_holders, use_container_width=True)
                else:
                    st.info(f"Not enough valid data to plot top institutional holders for {deep_dive_ticker}.")
            else:
                st.info(f"No valid shares data to plot institutional holders for {deep_dive_ticker}.")
        elif "Error" in institutional_holders_df.columns:
            st.error(f"Error loading institutional holders data: {institutional_holders_df['Error'].iloc[0]}")
        else:
            st.info(f"No institutional holder data available for {deep_dive_ticker}.")

st.markdown("---")

# --- Section 3: Comparative Analysis Across Magnificent 7 (Multi-select Metrics) ---
st.header("3. Comparative Analysis Across Magnificent 7")
st.markdown("Compare various fundamental metrics among the selected Magnificent 7 stocks.")

valid_companies_for_comparison = {t: data for t, data in all_fundamentals.items() if "Error" not in data and t in selected_m7_tickers_for_display}

if valid_companies_for_comparison:
    chart_comparative_data = []
    for t, data in valid_companies_for_comparison.items():
        row = {}
        row["Ticker"] = f"{data['Short Name']} ({t})"
        
        market_cap = data.get("Market Cap")
        row["Market Cap (Billions)"] = market_cap / 1e9 if market_cap is not None else np.nan

        pe = data.get("Trailing P/E")
        row["Trailing P/E"] = pe if pe is not None else np.nan

        fwd_pe = data.get("Forward P/E")
        row["Forward P/E"] = fwd_pe if fwd_pe is not None else np.nan

        profit_margins = data.get("Profit Margins")
        row["Profit Margins (%)"] = profit_margins * 100 if profit_margins is not None else np.nan

        roe = data.get("Return on Equity")
        row["Return on Equity (%)"] = roe * 100 if roe is not None else np.nan

        revenue_ttm = data.get("Revenue (TTM)")
        row["Revenue (TTM) (Billions)"] = revenue_ttm / 1e9 if revenue_ttm is not None else np.nan
        
        debt_to_equity = data.get("Debt to Equity")
        row["Debt to Equity"] = debt_to_equity if debt_to_equity is not None else np.nan

        current_ratio = data.get("Current Ratio")
        row["Current Ratio"] = current_ratio if current_ratio is not None else np.nan

        total_cash = data.get("Total Cash")
        row["Total Cash (Billions)"] = total_cash / 1e9 if total_cash is not None else np.nan
        
        total_debt = data.get("Total Debt")
        row["Total Debt (Billions)"] = total_debt / 1e9 if total_debt is not None else np.nan

        earnings_growth = data.get("Earnings Growth (YoY)")
        row["Earnings Growth (YoY, %)"] = earnings_growth * 100 if earnings_growth is not None else np.nan

        chart_comparative_data.append(row)

    chart_comparative_df = pd.DataFrame(chart_comparative_data)
    
    if not chart_comparative_df.empty:
        comparative_metric_options = {
            "Market Cap (Billions)": "Market Cap (Billions)",
            "Trailing P/E": "Trailing P/E",
            "Forward P/E": "Forward P/E",
            "Profit Margins (%)": "Profit Margins (%)",
            "Return on Equity (%)": "Return on Equity (%)",
            "Revenue (TTM) (Billions)": "Revenue (TTM) (Billions)",
            "Debt to Equity": "Debt to Equity",
            "Current Ratio": "Current Ratio",
            "Total Cash (Billions)": "Total Cash (Billions)",
            "Total Debt (Billions)": "Total Debt (Billions)",
            "Earnings Growth (YoY, %)": "Earnings Growth (YoY, %)",
        }

        selected_comparative_metrics = st.multiselect(
            "Select Metrics for Comparison Chart:",
            options=list(comparative_metric_options.keys()),
            default=["Market Cap (Billions)", "Trailing P/E"],
            help="Choose one or more metrics to compare across companies. If multiple selected, a grouped bar chart will be shown."
        )

        if selected_comparative_metrics:
            if len(selected_comparative_metrics) > 1:
                df_to_melt = chart_comparative_df[['Ticker'] + selected_comparative_metrics]
                melted_df = df_to_melt.melt(id_vars=['Ticker'], var_name='Metric', value_name='Value')
                melted_df = melted_df.dropna(subset=['Value'])
                
                if not melted_df.empty:
                    fig_multi_bar = px.bar(
                        melted_df,
                        x="Ticker",
                        y="Value",
                        color="Metric",
                        barmode="group",
                        title="Selected Metrics Comparison Across Magnificent 7",
                        labels={"Ticker": "Company", "Value": "Value"},
                        hover_data={"Value": True}
                    )
                    fig_multi_bar.update_layout(xaxis_title="", yaxis_title="Value")
                    st.plotly_chart(fig_multi_bar, use_container_width=True)
                else:
                    st.warning("No valid data to generate the multi-metric comparison chart for the selected metrics.")
            else:
                selected_metric = selected_comparative_metrics[0]
                chart_data_for_plot = chart_comparative_df.dropna(subset=[selected_metric])

                if not chart_data_for_plot.empty:
                    fig_bar = px.bar(
                        chart_data_for_plot.sort_values(by=selected_metric, ascending=False),
                        x="Ticker",
                        y=selected_metric,
                        title=f'{selected_metric} Comparison Across Magnificent 7',
                        labels={"Ticker": "Company", selected_metric: selected_metric},
                        hover_data={selected_metric: ':.2f'}
                    )
                    fig_bar.update_layout(xaxis_title="", yaxis_title=selected_metric)
                    st.plotly_chart(fig_bar, use_container_width=True)
                else:
                    st.warning(f"No valid data to generate '{selected_metric}' comparison chart. Try selecting a different metric.")
        else:
            st.info("Please select at least one metric for comparison.")
    else:
        st.warning("Not enough valid fundamental data to generate comparative charts.")
else:
    st.warning("No valid company data available for comparative charts based on your selections.")

st.markdown("---")
st.caption("Data provided by Yahoo Finance. Charts are for informational purposes only and not financial advice.")