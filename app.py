from flask import Flask, render_template, request, jsonify
import yfinance as yf
import pandas as pd
import datetime
from io import StringIO
import os

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit_tickers', methods=['POST'])
def submit_tickers():
    data = request.json
    tickers = data['tickers'].split(',')
    start_date = '2020-01-01'
    end_date = datetime.datetime.today().strftime('%Y-%m-%d')

    combined_data = process_tickers(tickers, start_date, end_date)
    return jsonify(combined_data)

def process_tickers(tickers, start_date, end_date):
    window = 14
    short_window = 50
    long_window = 200
    combined_data = {'Ticker': [], 'Latest Close': [], 'RSI': [], 'P/E Ratio': [], 'P/S Ratio': [],
                     f'SMA_{short_window}': [], f'SMA_{long_window}': [],
                     'Percent Change from 50 to 200': [], 'Volume Surge 100 Days': [],
                     'Gross Margin': [], 'Operating Margin': [], 'LTM Revenue': [],
                     'LTM FCF': []}

    for ticker in tickers:
        try:
            data_with_rsi = fetch_and_calculate_rsi(ticker, start_date, end_date, window)
            if data_with_rsi is None:
                print(f"No data available for ticker {ticker}. Skipping.")
                continue

            prma_data = fetch_and_calculate_prma(ticker, start_date, end_date, short_window, long_window)
            if prma_data is None:
                print(f"No data available for ticker {ticker}. Skipping.")
                continue

            pe_ratio = get_pe_ratio([ticker])[ticker]
            ps_ratios = get_ps_ratio([ticker])

            if ticker not in ps_ratios or isinstance(ps_ratios[ticker], str) and ps_ratios[ticker].startswith("Error"):
                print(f"Skipping ticker {ticker} due to error in P/S ratio calculation.")
                continue

            ps_ratio = ps_ratios[ticker]
            vs = get_vs(ticker, (datetime.datetime.today() - datetime.timedelta(100)).strftime('%Y-%m-%d'), end_date)

            if not data_with_rsi.empty:
                rsi = data_with_rsi['RSI'].iloc[-1]

                combined_data['Ticker'].append(ticker)
                combined_data['Latest Close'].append(prma_data['Latest Close'])
                combined_data['RSI'].append(rsi)
                combined_data['P/E Ratio'].append(pe_ratio)
                combined_data['P/S Ratio'].append(ps_ratio)
                combined_data[f'SMA_{short_window}'].append(prma_data[f'SMA_{short_window}'])
                combined_data[f'SMA_{long_window}'].append(prma_data[f'SMA_{long_window}'])
                combined_data['Percent Change from 50 to 200'].append(prma_data['Percent Change from 50 to 200'])
                combined_data['Volume Surge 100 Days'].append(vs)
                combined_data['Gross Margin'].append(get_gross_margin([ticker]))
                combined_data['Operating Margin'].append(get_operating_margin([ticker]))
                combined_data['LTM Revenue'].append(get_ltm_revenue([ticker]))

        except Exception as e:
            print(f"Error processing ticker {ticker}: {e}")
            continue

    return combined_data

def fetch_and_calculate_rsi(ticker, start_date, end_date, window=14):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    data['RSI'] = calculate_rsi(data, window)
    return data

def calculate_rsi(data, window=14):
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def fetch_and_calculate_prma(ticker, start_date, end_date, short_window=50, long_window=200):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    data = calculate_moving_averages(data, short_window, long_window)
    latest_close = data['Close'].iloc[-1]
    latest_short_ma = data[f'SMA_{short_window}'].iloc[-1]
    latest_long_ma = data[f'SMA_{long_window}'].iloc[-1]
    percent_change = ((latest_long_ma - latest_short_ma) / latest_short_ma) * 100

    return {
        'Ticker': ticker,
        'Latest Close': latest_close,
        f'SMA_{short_window}': latest_short_ma,
        f'SMA_{long_window}': latest_long_ma,
        'Percent Change from 50 to 200': f"{percent_change:.2f}%",
        'Close > SMA_50': latest_close > latest_short_ma,
        'Close > SMA_200': latest_close > latest_long_ma
    }

def calculate_moving_averages(data, short_window=50, long_window=200):
    data[f'SMA_{short_window}'] = data['Close'].rolling(window=short_window).mean()
    data[f'SMA_{long_window}'] = data['Close'].rolling(window=long_window).mean()
    return data

def get_pe_ratio(tickers):
    pe_ratios = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_price = stock.history(period="1d")['Close'].iloc[0]
            eps = stock.info['trailingEps']
            pe_ratio = stock_price / eps if eps != 0 else 'N/A'
            pe_ratios[ticker] = pe_ratio
        except Exception as e:
            pe_ratios[ticker] = f"Error: {e}"
    return pe_ratios

def get_ps_ratio(tickers):
    ps_ratios = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            stock_price = stock.history(period="1d")['Close'].iloc[0]

            if 'sharesOutstanding' not in stock.info or 'totalRevenue' not in stock.info:
                raise ValueError(f"Required data not found for ticker {ticker}")

            shares_outstanding = stock.info['sharesOutstanding']
            market_cap = stock_price * shares_outstanding
            total_revenue = stock.info['totalRevenue']

            ps_ratio = market_cap / total_revenue
            ps_ratios[ticker] = ps_ratio
        except Exception as e:
            ps_ratios[ticker] = f"Error: {e}"
            print(f"Error fetching data for {ticker}: {e}")

    return ps_ratios

def calculate_vs(data):
    delta = data['Volume']
    avg_volume = delta.mean()
    curr_volume = delta.iloc[-1]
    return 100 * (curr_volume - avg_volume) / avg_volume

def get_vs(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        return None
    return calculate_vs(data)

def get_gross_margin(tickers):
    gross_margins = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            profit_margin = stock.financials.loc['Gross Profit'].iloc[0]
            total_revenue = stock.financials.loc['Total Revenue'].iloc[0]
            if total_revenue != 0:
                gross_margin = 100 * profit_margin / total_revenue
            else:
                gross_margin = 'N/A'
            gross_margins[ticker] = gross_margin
        except Exception as e:
            gross_margins[ticker] = f"Error: {e}"
    return gross_margins

def get_operating_margin(tickers):
    operating_margins = {}

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.quarterly_financials
            if len(income_statement.columns) < 1:
                raise ValueError("Not enough data to calculate operating margin")
            operating_income = income_statement.loc['Operating Income'].iloc[0]
            revenue = income_statement.loc['Total Revenue'].iloc[0]
            if revenue == 0:
                raise ValueError("Revenue cannot be zero")
            operating_margin = (operating_income / revenue) * 100
            operating_margins[ticker] = operating_margin
        except Exception as e:
            operating_margins[ticker] = f"Error: {e}"

    return operating_margins

def get_ltm_revenue(tickers):
    ltm_revenues = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            income_statement = stock.financials
            if len(income_statement.columns) < 4:
                raise ValueError("Not enough data to calculate LTM revenue")
            ltm_revenue = income_statement.loc['Total Revenue'].iloc[:4].sum()
            ltm_revenues[ticker] = ltm_revenue
        except Exception as e:
            ltm_revenues[ticker] = f"Error: {e}"
    return ltm_revenues

def get_ltm_fcf(tickers):
    ltm_fcfs = {}
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            cash_flow_statement = stock.cashflow
            if len(cash_flow_statement.columns) < 4:
                raise ValueError("Not enough data to calculate LTM FCF")
            fcf_ltm = (
                cash_flow_statement.loc['Total Cash From Operating Activities'].iloc[:4].sum() -
                cash_flow_statement.loc['Capital Expenditures'].iloc[:4].sum()
            )
            ltm_fcfs[ticker] = fcf_ltm
        except Exception as e:
            ltm_fcfs[ticker] = f"Error: {e}"

    return ltm_fcfs

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
