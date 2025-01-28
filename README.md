![image](https://github.com/user-attachments/assets/879c8978-faaa-46a8-bef7-001d6a0f3415)



# YAI
Leveraging AI and the yfinance python module to investigate the stock market

This Python script combines real-time market data with AI-powered analysis to provide comprehensive stock assessments and trading recommendations. It leverages Yahoo Finance for market data and Claude (Anthropic's AI) for intelligent analysis.

## Features

- Real-time stock data fetching using yfinance
- Technical analysis including:
  - Moving averages (50-day, 200-day SMA)
  - RSI (Relative Strength Index)
  - Volume analysis
  - Volatility metrics
- Fundamental analysis including:
  - Market cap
  - P/E ratio
  - Profit margins
  - Revenue growth
  - Debt/Equity ratios
- Market context analysis:
  - Sector performance
  - Industry positioning
  - Beta and relative strength vs S&P500
- AI-powered analysis providing:
  - Technical trend interpretation
  - Fundamental value assessment
  - Market condition evaluation
  - Clear BUY/HOLD/SELL recommendations

## Prerequisites

```bash
pip install yfinance anthropic numpy pandas
```

## Setup

1. Get an API key from Anthropic (https://www.anthropic.com/)
2. Create a file named `key.txt` in the project directory
3. Paste your Anthropic API key in `key.txt`

## Usage

```bash
python yai.py
```

When prompted, enter a stock symbol (e.g., AAPL, MSFT, GOOGL).

## Output

The script provides:
- Detailed technical analysis
- Fundamental analysis
- Market context analysis
- Color-coded final recommendation (🟢 BUY, 🟡 HOLD, 🔴 SELL)
- Analysis cost based on API usage

Example output:
```
📊 Analysis Report: AAPL
==================================================

📍 Technical Analysis
------------------------------
[Technical analysis details...]

📍 Fundamental Analysis
------------------------------
[Fundamental analysis details...]

📍 Market Analysis
------------------------------
[Market analysis details...]

🎯 FINAL RECOMMENDATION
==================================================
Signal: BUY
Rationale: [AI-generated explanation]

💰 Analysis Cost: ~$0.30
```

## Security Note

Never commit your `key.txt` file to version control. Make sure it's listed in your `.gitignore`:
```
key.txt
```

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License - See LICENSE file for details

## Disclaimer

This tool is for educational purposes only. Always do your own research and consult with financial advisors before making investment decisions. The recommendations provided are based on automated analysis and should not be the sole basis for trading decisions.
