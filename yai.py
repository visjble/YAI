import asyncio
import yfinance as yf
from typing import Dict, Any, List
from anthropic import AsyncAnthropic
import numpy as np
import pandas as pd

def load_api_key():
    try:
        with open("key.txt", "r") as file:  #THIS FILE IN THE SAME DIRECTORY CONTAINS YOUR ANTHROPIC API KEY
            api_key = file.read().strip()
            if not api_key:
                raise ValueError("API key is empty")
            return api_key
    except FileNotFoundError:
        print("Error: key.txt not found")
        exit(1)
    except Exception as e:
        print(f"Error reading API key: {e}")
        exit(1)

# Get API key at module level
API_KEY = load_api_key()


class StockData:
    @staticmethod
    def get_data(symbol: str) -> Dict[str, Any]:
        """Fetch and process stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(symbol)
            hist = stock.history(period="1y")
            info = stock.info

            # Calculate technical indicators
            hist['SMA50'] = hist['Close'].rolling(window=50).mean()
            hist['SMA200'] = hist['Close'].rolling(window=200).mean()
            
            # Calculate RSI
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            hist['RSI'] = 100 - (100 / (1 + rs))

            # Get S&P 500 comparison
            spy = yf.Ticker("SPY")
            spy_hist = spy.history(period="1y")
            stock_return = hist['Close'].iloc[-1] / hist['Close'].iloc[0]
            market_return = spy_hist['Close'].iloc[-1] / spy_hist['Close'].iloc[0]
            
            return {
                "symbol": symbol,
                "technical": {
                    "price": hist['Close'].iloc[-1],
                    "sma50": hist['SMA50'].iloc[-1],
                    "sma200": hist['SMA200'].iloc[-1],
                    "rsi": hist['RSI'].iloc[-1],
                    "volume": hist['Volume'].iloc[-1],
                    "avg_volume": hist['Volume'].mean(),
                    "volatility": hist['Close'].pct_change().std() * np.sqrt(252),
                    "daily_change": hist['Close'].pct_change().iloc[-1],
                    "monthly_change": hist['Close'].pct_change(20).iloc[-1]
                },
                "fundamental": {
                    "market_cap": info.get('marketCap'),
                    "pe_ratio": info.get('forwardPE'),
                    "pb_ratio": info.get('priceToBook'),
                    "profit_margin": info.get('profitMargins'),
                    "revenue_growth": info.get('revenueGrowth'),
                    "debt_to_equity": info.get('debtToEquity')
                },
                "market": {
                    "beta": info.get('beta'),
                    "relative_strength": stock_return / market_return,
                    "sector": info.get('sector'),
                    "industry": info.get('industry')
                }
            }
        except Exception as e:
            raise Exception(f"Error fetching data for {symbol}: {str(e)}")

class Agent:
    def __init__(self, name: str, role: str, system_prompt: str):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.client = AsyncAnthropic(api_key=API_KEY)

    async def analyze(self, data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            prompt = self._create_prompt(data)
            response = await self.client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1024,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Handle the response content properly
            content = response.content
            if isinstance(content, list):
                content = content[0].text if hasattr(content[0], 'text') else str(content[0])
            
            return {
                "agent": self.name,
                "analysis": content,  # Use processed content
                "tokens": response.usage.input_tokens + response.usage.output_tokens
            }
        except Exception as e:
            return {
                "agent": self.name,
                "analysis": f"Analysis failed: {str(e)}",
                "tokens": 0
            }
    def _create_prompt(self, data: Dict[str, Any]) -> str:
        raise NotImplementedError

class TechnicalAnalyst(Agent):
    def __init__(self):
        super().__init__(
            name="Technical Analysis",
            role="Technical Analyst",
            system_prompt="You are a technical analysis expert. Provide concise insights about price trends, momentum, and technical indicators. Be specific about support/resistance levels and trend directions."
        )
    
    def _create_prompt(self, data: Dict[str, Any]) -> str:
        t = data["technical"]
        return f"""
        Analyze technical indicators for {data['symbol']}:
        
        Price Action:
        - Current Price: ${t['price']:.2f}
        - Daily Change: {t['daily_change']*100:.1f}%
        - Monthly Change: {t['monthly_change']*100:.1f}%
        
        Technical Indicators:
        - 50-day SMA: ${t['sma50']:.2f}
        - 200-day SMA: ${t['sma200']:.2f}
        - RSI (14): {t['rsi']:.1f}
        - Volatility: {t['volatility']*100:.1f}%
        
        Volume Analysis:
        - Current Volume: {t['volume']:,.0f}
        - Average Volume: {t['avg_volume']:,.0f}
        
        Provide a concise technical analysis focusing on trend direction and key levels.
        """

class FundamentalAnalyst(Agent):
    def __init__(self):
        super().__init__(
            name="Fundamental Analysis",
            role="Financial Analyst",
            system_prompt="You are a fundamental analysis expert. Provide concise insights about company financials, valuation, and growth metrics. Compare metrics to industry standards where relevant."
        )
    
    def _create_prompt(self, data: Dict[str, Any]) -> str:
        f = data["fundamental"]
        m = data["market"]
        return f"""
        Analyze fundamentals for {data['symbol']}:
        
        Valuation Metrics:
        - Market Cap: ${f['market_cap']:,.0f}
        - P/E Ratio: {f['pe_ratio']:.2f}
        - P/B Ratio: {f['pb_ratio']:.2f}
        
        Financial Health:
        - Profit Margin: {f['profit_margin']*100:.1f}%
        - Revenue Growth: {f['revenue_growth']*100:.1f}%
        - Debt/Equity: {f['debt_to_equity']:.2f}
        
        Market Position:
        - Sector: {m['sector']}
        - Industry: {m['industry']}
        - Beta: {m['beta']:.2f}
        
        Provide a concise fundamental analysis focusing on valuation and growth prospects.
        """

class MarketAnalyst(Agent):
    def __init__(self):
        super().__init__(
            name="Market Analysis",
            role="Market Strategist",
            system_prompt="You are a market analysis expert. Provide concise insights about market conditions, sector trends, and relative performance. Focus on key market dynamics affecting the stock."
        )
    
    def _create_prompt(self, data: Dict[str, Any]) -> str:
        m = data["market"]
        return f"""
        Analyze market context for {data['symbol']}:
        
        Market Position:
        - Sector: {m['sector']}
        - Industry: {m['industry']}
        - Beta: {m['beta']:.2f}
        - Relative Strength vs S&P500: {m['relative_strength']:.2f}
        
        Provide a concise market analysis focusing on sector trends and market positioning.
        """

async def generate_recommendation(analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
    try:
        client = AsyncAnthropic(api_key=API_KEY)
        # rest of the function remains the same
    except Exception as e:
        print(f"Error initializing Anthropic client in recommendation: {e}")
        return {
            "signal": "ERROR",
            "explanation": f"Failed to initialize client: {str(e)}",
            "tokens": 0
        }  
    # Combine all analyses
    combined = "\n\n".join([f"{a['agent']}:\n{a['analysis']}" for a in analyses])
    
    prompt = f"""
    Based on the following analyses, provide a clear BUY, HOLD, or SELL recommendation.
    Start with exactly one of these words: BUY, HOLD, or SELL.
    Then provide a concised explanation of your recommendation.

    {combined}
    """
    
    try:
        response = await client.messages.create(
            model="claude-3-sonnet-20240229",
            max_tokens=150,
            system="You are a decisive financial advisor. Always start with BUY, HOLD, or SELL.",
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Handle the response content properly
        content = response.content
        if isinstance(content, list):
            content = content[0].text if hasattr(content[0], 'text') else str(content[0])
        
        # Extract signal from processed content
        if "BUY" in content.upper():
            signal = "BUY"
        elif "SELL" in content.upper():
            signal = "SELL"
        else:
            signal = "HOLD"
            
        explanation = content.replace(signal, "", 1).strip()
        
        return {
            "signal": signal,
            "explanation": explanation,
            "tokens": response.usage.input_tokens + response.usage.output_tokens
        }
    except Exception as e:
        return {
            "signal": "ERROR",
            "explanation": str(e),
            "tokens": 0
        }

async def analyze_stock(symbol: str):
    try:
        # Get stock data
        print(f"\nFetching data for {symbol}...")
        data = StockData.get_data(symbol)
        
        # Create analysts
        analysts = [
            TechnicalAnalyst(),
            FundamentalAnalyst(),
            MarketAnalyst()
        ]
        
        # Run analyses
        analyses = await asyncio.gather(*[
            analyst.analyze(data) for analyst in analysts
        ])
        
        # Get recommendation
        recommendation = await generate_recommendation(analyses)
        
        # Calculate total tokens
        total_tokens = sum(a['tokens'] for a in analyses) + recommendation['tokens']
        
        # Print report
        print(f"\n📊 ANALYSIS REPORT: {symbol}")
        print("=" * 50)
        
        for analysis in analyses:
            print(f"\n📍 {analysis['agent']}")
            print("-" * 30)
            print(analysis['analysis'].strip())
        
        # Print recommendation with color
        colors = {
            "BUY": "\033[92m",    # Green
            "SELL": "\033[91m",   # Red
            "HOLD": "\033[93m",   # Yellow
            "ERROR": "\033[91m"    # Red
        }
        
        print("\n🎯 FINAL RECOMMENDATION")
        print("=" * 50)
        signal = recommendation['signal']
        print(f"Signal: {colors.get(signal, '')}{signal}\033[0m")
        print(f"Rationale: {recommendation['explanation']}")
        
        # Print cost
        print(f"\n💰 Analysis Cost: ${(total_tokens * 0.00015):.2f}")
        
    except Exception as e:
        print(f"Error: {str(e)}")

async def main():
 
    # Verify API key at startup
    try:
        test_client = AsyncAnthropic(api_key=API_KEY)
    except Exception as e:
        print(f"Error: Invalid API key configuration: {e}")
        return

    while True:
        symbol = input("\nEnter stock symbol (or 'quit' to exit): ").upper()
        if symbol in ['QUIT', 'Q']:
            break
        
        if symbol:
            await analyze_stock(symbol)
        else:
            print("Please enter a valid symbol")

if __name__ == "__main__":
    asyncio.run(main())
