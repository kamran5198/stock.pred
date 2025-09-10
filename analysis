from sklearn.metrics import accuracy_score, classification_report
# ============================= #### final model
# Markov Chain Predictor
# =============================
class MarkovChainPredictor:
    def __init__(self, n_states=3):
        self.n_states = n_states
        self.transition_matrix = None
        self.state_mapping = None
        self.state_bins = None
    def create_states(self, prices):
        quantiles = np.linspace(0, 1, self.n_states + 1)
        self.state_bins = np.quantile(prices, quantiles)
        states = np.digitize(prices, self.state_bins) - 1
        return states
    def build_transition_matrix(self, states):
        n = self.n_states
        transition_matrix = np.zeros((n, n))
        for i in range(len(states) - 1):
            current_state = int(np.clip(states[i], 0, n - 1))
            next_state = int(np.clip(states[i + 1], 0, n - 1))
            transition_matrix[current_state, next_state] += 1
        row_sums = transition_matrix.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        transition_matrix = transition_matrix / row_sums
        self.transition_matrix = transition_matrix
        return transition_matrix
    def train(self, prices):
        states = self.create_states(prices)
        return self.build_transition_matrix(states)
        return states
    def predict_next_state(self, current_state):
        if self.transition_matrix is None:
            return None
        probabilities = self.transition_matrix[current_state, :]
        next_state = np.argmax(probabilities)
        confidence = probabilities[next_state]
        return next_state, confidence
    def predict_sequence(self, initial_state, n_steps=10):
        current_state = initial_state
        predictions = [current_state]
        confidences = [1.0]
        for _ in range(n_steps):
            next_state, confidence = self.predict_next_state(current_state)
            predictions.append(next_state)
            confidences.append(confidence)
            current_state = next_state
        return predictions, confidences
    def state_to_price(self, state):
        if self.state_bins is None:
            return None
        if state == 0:
            price_estimate = self.state_bins[0] + (self.state_bins[1] - self.state_bins[0]) / 2
        elif state == self.n_states - 1:
            price_estimate = self.state_bins[-2] + (self.state_bins[-1] - self.state_bins[-2]) / 2
        else:
            price_estimate = self.state_bins[state] + (self.state_bins[state+1] - self.state_bins[state]) / 2
        return price_estimate
from tensorflow.keras.layers import GRU, Conv1D, MaxPooling1D, Flatten
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
import streamlit as st ### final model
import requests
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import ta
import warnings
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import json
import io
import base64
import sqlite3
import os
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
import matplotlib.pyplot as plt

# Add Prophet for time series analysis
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophet not available. Install with: pip install prophet")

import tempfile
from io import BytesIO

# LangChain imports
try:
    from langchain.llms import OpenAI
    from langchain.chat_models import ChatOpenAI
    from langchain.schema import HumanMessage, SystemMessage
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain.agents import create_pandas_dataframe_agent
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False

warnings.filterwarnings('ignore')

# =============================
# Database Setup
# =============================
def init_db():
    """Initialize SQLite database for storing predictions"""
    conn = sqlite3.connect('stock_predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  symbol TEXT,
                  date TEXT,
                  current_price REAL,
                  predicted_price REAL,
                  sentiment_score REAL,
                  fundamental_score REAL,
                  model_type TEXT,
                  prediction_days INTEGER,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def save_prediction(symbol, current_price, predicted_price, sentiment_score, fundamental_score, model_type, prediction_days):
    """Save prediction to database"""
    conn = sqlite3.connect('stock_predictions.db')
    c = conn.cursor()
    c.execute('''INSERT INTO predictions 
                 (symbol, date, current_price, predicted_price, sentiment_score, fundamental_score, model_type, prediction_days)
                 VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
              (symbol, datetime.now().strftime('%Y-%m-%d'), current_price, predicted_price, 
               sentiment_score, fundamental_score, model_type, prediction_days))
    conn.commit()
    conn.close()

def get_historical_predictions(symbol, days=30):
    """Retrieve historical predictions for a symbol"""
    conn = sqlite3.connect('stock_predictions.db')
    c = conn.cursor()
    c.execute('''SELECT date, current_price, predicted_price, sentiment_score, fundamental_score, model_type
                 FROM predictions 
                 WHERE symbol = ? 
                 ORDER BY date DESC LIMIT ?''', (symbol, days))
    results = c.fetchall()
    conn.close()
    
    if not results:
        return pd.DataFrame()
    
    df = pd.DataFrame(results, columns=['date', 'current_price', 'predicted_price', 'sentiment_score', 'fundamental_score', 'model_type'])
    df['date'] = pd.to_datetime(df['date'])
    return df

# =============================
# PDF Report Generation
# =============================
def generate_pdf_report(symbol, analysis, historical_data, predictions, sentiment_score, 
                       news_data, fundamental_score, strategy_recommendations, 
                       technical_analysis, model_performance):
    """Generate a PDF report of the analysis"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=20,
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph(f"Stock Analysis Report: {symbol}", title_style))
    
    # Executive Summary
    story.append(Paragraph("Executive Summary", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    current_price = historical_data['Close'].iloc[-1]
    predicted_price = predictions[-1] if predictions else current_price
    price_change_pct = ((predicted_price - current_price) / current_price) * 100
    
    summary_data = [
    ["Metric", "Value"],
    ["Symbol", symbol],
    ["Current Price", f"${current_price:.2f}"],
    ["Predicted Price", f"${predicted_price:.2f}"],
    ["Predicted Change", f"{price_change_pct:.2f}%"],
    ["Sentiment Score", f"{sentiment_score:.3f}"],
    ["Fundamental Score", f"{fundamental_score:.3f}"],
    ["Analysis Date", datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    ]
    
    summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Price Chart
    story.append(Paragraph("Price Chart", styles['Heading2']))
    story.append(Spacer(1, 12))
    
    # Create and save plot to buffer
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=historical_data.index,
        open=historical_data['Open'],
        high=historical_data['High'],
        low=historical_data['Low'],
        close=historical_data['Close'],
        name="Price"
    ))
    fig.update_layout(title=f"{symbol} Stock Price", yaxis_title="Price ($)", xaxis_title="Date", height=500)
    
    img_buffer = BytesIO()
    fig.write_image(img_buffer, format='png')
    img_buffer.seek(0)
    
    # Add image to PDF
    story.append(Image(img_buffer, width=6*inch, height=4*inch))
    story.append(Spacer(1, 20))
    
    # AI Analysis
    story.append(Paragraph("AI Analysis", styles['Heading2']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(analysis, styles['BodyText']))
    story.append(Spacer(1, 20))
    
    # Strategy Recommendations
    if strategy_recommendations:
        story.append(Paragraph("Strategy Recommendations", styles['Heading2']))
        story.append(Spacer(1, 12))
        
        for i, rec in enumerate(strategy_recommendations):
            story.append(Paragraph(f"Recommendation {i+1}: {rec['strategy']}", styles['Heading3']))
            story.append(Paragraph(f"Confidence: {rec['confidence']}", styles['BodyText']))
            story.append(Paragraph(f"Rationale: {rec['rationale']}", styles['BodyText']))
            story.append(Paragraph(f"Risk Level: {rec['risk_level']}", styles['BodyText']))
            story.append(Paragraph(f"Timeframe: {rec['timeframe']}", styles['BodyText']))
            story.append(Paragraph("Recommended Actions:", styles['Heading4']))
            
            for action in rec['actions']:
                story.append(Paragraph(f"• {action}", styles['BodyText']))
            
            story.append(Spacer(1, 12))
    
    # Build PDF
    doc.build(story)
    buffer.seek(0)
    return buffer

def create_comprehensive_predictions_csv(symbol, stock_data, future_prices, sentiment_score, 
                                       fundamental_score, news_df, strategy_recommendations):
    """Create comprehensive CSV with all prediction data"""
    current_price = float(stock_data['Close'].iloc[-1])
    last_date = stock_data.index[-1]
    future_dates = [last_date + timedelta(days=i+1) for i in range(len(future_prices))]
    
    # Create main predictions dataframe
    predictions_df = pd.DataFrame({
        'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
        'Predicted_Price': [f"{p:.2f}" for p in future_prices],
        'Current_Price': f"{current_price:.2f}",
        'Price_Change_Pct': [f"{((p - current_price) / current_price * 100):.2f}%" for p in future_prices],
        'Sentiment_Score': f"{sentiment_score:.3f}",
        'Fundamental_Score': f"{fundamental_score:.3f}",
        'Analysis_Date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    })
    
    # Add strategy recommendations as additional columns
    if strategy_recommendations:
        strategies = "; ".join([rec['strategy'] for rec in strategy_recommendations])
        predictions_df['Recommended_Strategies'] = strategies
    
    # Add recent news summary
    if news_df is not None and not news_df.empty:
        news_summary = "; ".join(news_df.head(3)['title'].fillna('').astype(str))
        predictions_df['Recent_News'] = news_summary
    
    return predictions_df

# =============================
# Prophet Time Series Analysis
# =============================
class ProphetPredictor:
    def __init__(self, growth='linear', yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False):
        self.model = None
        self.growth = growth
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        
    def prepare_data(self, df):
        """Prepare data for Prophet model"""
        prophet_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        return prophet_df
        
    def train(self, df):
        """Train the Prophet model"""
        if not PROPHET_AVAILABLE:
            st.error("Prophet not available. Install with: pip install prophet")
            return None
            
        try:
            self.model = Prophet(
                growth=self.growth,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )
            prophet_df = self.prepare_data(df)
            # Remove timezone info from 'ds' column for Prophet
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds']).dt.tz_localize(None)
            self.model.fit(prophet_df)
            return True
        except Exception as e:
            st.error(f"Error training Prophet model: {e}")
            return False
            
    def predict(self, periods=30):
        """Make predictions with Prophet"""
        if self.model is None:
            return pd.DataFrame()
            
        try:
            future = self.model.make_future_dataframe(periods=periods)
            forecast = self.model.predict(future)
            return forecast
        except Exception as e:
            st.error(f"Error predicting with Prophet: {e}")
            return pd.DataFrame()
            
    def plot_components(self, forecast):
        """Plot Prophet components"""
        if self.model is None:
            return None
            
        try:
            fig = self.model.plot_components(forecast)
            return fig
        except Exception as e:
            st.error(f"Error plotting Prophet components: {e}")
            return None

# =============================
# Sector and Industry Lists
# =============================
SECTORS = [
    "Technology", "Healthcare", "Financial Services", "Consumer Cyclical",
    "Communication Services", "Industrials", "Energy", "Utilities",
    "Real Estate", "Materials", "Consumer Defensive", "Other"
]

INDUSTRIES = {
    "Technology": [
        "Software", "Hardware", "Semiconductors", "IT Services",
        "Internet", "Telecom Equipment", "Electronics", "Other"
    ],
    "Healthcare": [
        "Pharmaceuticals", "Biotechnology", "Medical Devices",
        "Healthcare Providers", "Medical Diagnostics", "Other"
    ],
    "Financial Services": [
        "Banks", "Insurance", "Investment Services", "Asset Management",
        "Credit Services", "Other"
    ],
    "Consumer Cyclical": [
        "Automotive", "Retail", "Apparel", "Travel & Leisure",
        "Homebuilding", "Other"
    ],
    "Communication Services": [
        "Media", "Entertainment", "Telecom", "Advertising",
        "Publishing", "Other"
    ],
    "Industrials": [
        "Aerospace & Defense", "Machinery", "Construction",
        "Electrical Equipment", "Transportation", "Other"
    ],
    "Energy": [
        "Oil & Gas", "Renewable Energy", "Energy Equipment",
        "Oil & Gas Services", "Other"
    ],
    "Utilities": [
        "Electric Utilities", "Gas Utilities", "Water Utilities",
        "Multi-Utilities", "Other"
    ],
    "Real Estate": [
        "Real Estate Services", "REITs", "Real Estate Development",
        "Real Estate Operations", "Other"
    ],
    "Materials": [
        "Chemicals", "Metals & Mining", "Construction Materials",
        "Paper & Forest Products", "Other"
    ],
    "Consumer Defensive": [
        "Food & Beverage", "Household Products", "Tobacco",
        "Personal Products", "Other"
    ],
    "Other": ["Other"]
}

# =============================
# Utility helpers
# =============================

def safe_pct_change(series: pd.Series) -> pd.Series:
    try:
        return series.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception:
        return pd.Series([0]*len(series), index=series.index)


def nan_safe(values) -> bool:
    """Return True if array-like has no NaNs/inf."""
    arr = np.asarray(values, dtype=float)
    return np.isfinite(arr).all()

# =============================
# AI Analysis with LangChain
# =============================
class AIAnalysisAgent:
    def __init__(self, openai_api_key=None):
        self.openai_api_key = openai_api_key
        self.llm = None
        if openai_api_key and LANGCHAIN_AVAILABLE:
            try:
                self.llm = ChatOpenAI(
                    temperature=0.3,
                    model_name="gpt-3.5-turbo",
                    openai_api_key=openai_api_key
                )
            except Exception as e:
                st.error(f"Error initializing AI agent: {e}")
                self.llm = None

    def analyze_predictions(self, symbol, historical_data, predictions, sentiment_score, news_data, fundamental_score):
        """Use AI to analyze predictions and provide insights"""
        if not self.llm:
            return "AI analysis not available. Please provide an OpenAI API key."
        
        try:
            # Prepare data for analysis
            current_price = historical_data['Close'].iloc[-1]
            predicted_price = predictions[-1] if predictions else current_price
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            # Create a summary of the data
            data_summary = f"""
            Stock: {symbol}
            Current Price: ${current_price:.2f}
            Predicted Price: ${predicted_price:.2f}
            Predicted Change: {price_change_pct:.2f}%
            Sentiment Score: {sentiment_score:.3f}
            Fundamental Score: {fundamental_score:.3f}
            """
            
            # News summary
            news_summary = "Recent News:\n"
            if news_data is not None and not news_data.empty:
                for i, row in news_data.head(3).iterrows():
                    news_summary += f"- {row.get('title', 'N/A')} (Sentiment: {row.get('sentiment', 0):.3f})\n"
            
            # Create prompt for analysis
            prompt = f"""
            You are a financial analyst with expertise in stock market predictions. 
            Analyze the following stock data and provide insights and recommendations.
            
            {data_summary}
            
            {news_summary}
            
            Based on this information, please provide:
            1. A brief analysis of the current situation
            2. Key factors influencing the prediction
            3. Investment recommendations (Buy/Hold/Sell) with reasoning
            4. Potential risks to consider
            5. Suggested timeframe for the investment
            
            Provide your response in a structured format with clear sections.
            """
            
            # Get AI analysis
            messages = [
                SystemMessage(content="You are a financial analyst providing stock market insights."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            return response.content
        except Exception as e:
            return f"Error in AI analysis: {str(e)}"

    def generate_report(self, symbol, analysis, historical_data, predictions, sentiment_score, news_data, fundamental_score, strategy_recommendations):
        """Generate a comprehensive report using AI"""
        if not self.llm:
            return "Report generation not available. Please provide an OpenAI API key."
        
        try:
            # Prepare data for report
            current_price = historical_data['Close'].iloc[-1]
            predicted_price = predictions[-1] if predictions else current_price
            
            # Create prompt for report generation
            prompt = f"""
            Create a comprehensive investment report for {symbol} based on the following data:
            
            Current Price: ${current_price:.2f}
            Predicted Price: ${predicted_price:.2f}
            Sentiment Score: {sentiment_score:.3f}
            Fundamental Score: {fundamental_score:.3f}
            
            Analysis Insights:
            {analysis}
            
            Strategy Recommendations:
            {json.dumps(strategy_recommendations, indent=2) if strategy_recommendations else "No specific strategies"}
            
            Please structure the report with:
            1. Executive Summary
            2. Current Market Position
            3. Technical Analysis
            4. Fundamental Analysis
            5. Sentiment Analysis
            6. Price Predictions
            7. Investment Recommendations
            8. Risk Assessment
            9. Strategic Actions
            10. Conclusion
            
            Make the report professional and suitable for investors.
            """
            
            # Get AI report
            messages = [
                SystemMessage(content="You are a financial analyst creating investment reports."),
                HumanMessage(content=prompt)
            ]
            
            response = self.llm(messages)
            return response.content
        except Exception as e:
            return f"Error generating report: {str(e)}"

# =============================
# Strategic Choices (unchanged with minor robustness)
# =============================
class StrategyRecommender:
    def __init__(self):
        self.strategy_keywords = {
            'growth': ['expansion', 'growth', 'invest', 'funding', 'acquisition', 'merger', 'scaling', 'development'],
            'market_penetration': ['market share', 'competitive', 'pricing', 'promotion', 'sales', 'customer acquisition'],
            'product_development': ['innovation', 'new product', 'r&d', 'research', 'technology', 'patent', 'launch'],
            'market_development': ['international', 'global', 'new market', 'geographic', 'region', 'overseas'],
            'diversification': ['diversify', 'new business', 'vertical', 'horizontal', 'portfolio'],
            'cost_leadership': ['efficiency', 'cost reduction', 'optimization', 'automation', 'streamline'],
            'consolidation': ['restructuring', 'layoffs', 'consolidation', 'downsizing', 'cost cutting', 'bankruptcy']
        }

    def analyze_news_for_strategies(self, news_df):
        if news_df is None or news_df.empty:
            return {}
        strategy_scores = {strategy: 0 for strategy in self.strategy_keywords.keys()}
        for _, row in news_df.iterrows():
            title_text = str(row.get('title', '')).lower()
            desc_text = str(row.get('description', '')).lower()
            combined_text = f"{title_text} {desc_text}"
            for strategy, keywords in self.strategy_keywords.items():
                for keyword in keywords:
                    if keyword in combined_text:
                        sentiment_weight = max(0.1, float(row.get('sentiment', 0)) + 1)
                        strategy_scores[strategy] += sentiment_weight
        total_articles = len(news_df)
        if total_articles > 0:
            strategy_scores = {k: v/total_articles for k, v in strategy_scores.items()}
        return strategy_scores

    def get_strategy_recommendation(self, prediction_trend, sentiment_score, strategy_scores, current_price, predicted_prices):
        recommendations = []
        if not predicted_prices:
            return recommendations
        price_change_pct = ((predicted_prices[-1] - current_price) / current_price) * 100
        volatility = np.std(predicted_prices) / max(1e-9, np.mean(predicted_prices)) * 100
        if price_change_pct > 5:
            trend = "bullish"
        elif price_change_pct < -5:
            trend = "bearish"
        else:
            trend = "neutral"

        dominant_strategy = max(strategy_scores, key=strategy_scores.get) if strategy_scores else None

        if trend == "bullish" and sentiment_score > 0.1:
            recommendations.extend([
                {
                    "strategy": "Growth Strategy",
                    "confidence": "High",
                    "rationale": f"Strong upward trend (+{price_change_pct:.1f}%) with positive sentiment ({sentiment_score:.3f})",
                    "actions": [
                        "Increase R&D investments for product development",
                        "Expand market presence through strategic partnerships",
                        "Consider stock buyback programs to boost shareholder value",
                        "Accelerate growth initiatives and capital expenditure"
                    ],
                    "risk_level": "Medium",
                    "timeframe": "6-12 months"
                }
            ])
            if dominant_strategy == 'product_development' and strategy_scores.get(dominant_strategy, 0) > 0.3:
                recommendations.append({
                    "strategy": "Innovation-Led Growth",
                    "confidence": "High",
                    "rationale": "Strong product development signals in news with positive market outlook",
                    "actions": [
                        "Fast-track new product launches",
                        "Increase innovation budget allocation",
                        "Strengthen IP portfolio through patents",
                        "Form technology partnerships"
                    ],
                    "risk_level": "Medium-High",
                    "timeframe": "3-9 months"
                })
        elif trend == "bearish" and sentiment_score < -0.1:
            recommendations.extend([
                {
                    "strategy": "Defensive Strategy",
                    "confidence": "High",
                    "rationale": f"Declining trend ({price_change_pct:.1f}%) with negative sentiment ({sentiment_score:.3f})",
                    "actions": [
                        "Implement cost optimization measures",
                        "Focus on cash flow preservation",
                        "Strengthen core business operations",
                        "Consider strategic asset divestiture"
                    ],
                    "risk_level": "Low-Medium",
                    "timeframe": "3-6 months"
                }
            ])
            if dominant_strategy == 'consolidation' and strategy_scores.get(dominant_strategy, 0) > 0.2:
                recommendations.append({
                    "strategy": "Restructuring & Consolidation",
                    "confidence": "Medium-High",
                    "rationale": "Market pressures indicate need for operational efficiency",
                    "actions": [
                        "Streamline operations and reduce redundancies",
                        "Consolidate underperforming business units",
                        "Renegotiate supplier contracts for better terms",
                        "Focus resources on profitable segments"
                    ],
                    "risk_level": "High",
                    "timeframe": "6-18 months"
                })
        else:
            recommendations.extend([
                {
                    "strategy": "Balanced Approach",
                    "confidence": "Medium",
                    "rationale": f"Stable/uncertain conditions with neutral to mixed sentiment",
                    "actions": [
                        "Maintain steady operational performance",
                        "Explore selective growth opportunities",
                        "Strengthen competitive positioning",
                        "Build strategic reserves for future investments"
                    ],
                    "risk_level": "Low",
                    "timeframe": "6-12 months"
                }
            ])

        if dominant_strategy and strategy_scores.get(dominant_strategy, 0) > 0.2:
            if dominant_strategy == 'market_development':
                recommendations.append({
                    "strategy": "Market Expansion",
                    "confidence": "Medium-High",
                    "rationale": "Strong geographic expansion signals in recent news",
                    "actions": [
                        "Evaluate international market opportunities",
                        "Develop localization strategies for new regions",
                        "Form strategic alliances in target markets",
                        "Adapt products for regional preferences"
                    ],
                    "risk_level": "Medium-High",
                    "timeframe": "9-24 months"
                })
            elif dominant_strategy == 'market_penetration':
                recommendations.append({
                    "strategy": "Market Share Capture",
                    "confidence": "High",
                    "rationale": "Competitive positioning opportunities identified in market analysis",
                    "actions": [
                        "Aggressive pricing strategies to gain market share",
                        "Increase marketing and promotional activities",
                        "Enhance customer acquisition programs",
                        "Improve product/service differentiation"
                    ],
                    "risk_level": "Medium",
                    "timeframe": "3-12 months"
                })

        if volatility > 15:
            recommendations.append({
                "strategy": "Risk Management Focus",
                "confidence": "High",
                "rationale": f"High price volatility ({volatility:.1f}%) requires active risk management",
                "actions": [
                    "Implement dynamic hedging strategies",
                    "Diversify revenue streams to reduce dependency",
                    "Maintain higher cash reserves for flexibility",
                    "Consider scenario-based strategic planning"
                ],
                "risk_level": "Medium",
                "timeframe": "Immediate-6 months"
            })
        return recommendations


def display_strategy_recommendations(symbol, sentiment_score, news_df, current_price, future_prices):
    st.subheader("🎯 Strategic Recommendations")
    if not future_prices:
        st.warning("Unable to generate strategic recommendations without price predictions.")
        return
    strategy_recommender = StrategyRecommender()
    strategy_scores = strategy_recommender.analyze_news_for_strategies(news_df)
    trend = "bullish" if future_prices[-1] > current_price else "bearish"
    recommendations = strategy_recommender.get_strategy_recommendation(
        trend, sentiment_score, strategy_scores, current_price, future_prices
    )
    if strategy_scores:
        st.write("**Strategic Themes Identified in News:**")
        sorted_strategies = sorted(strategy_scores.items(), key=lambda x: x[1], reverse=True)
        cols = st.columns(min(4, len(sorted_strategies)))
        for i, (strategy, score) in enumerate(sorted_strategies[:4]):
            if score > 0.05:
                with cols[i]:
                    st.metric(
                        strategy.replace('_', ' ').title(),
                        f"{score:.2f}",
                        help=f"Frequency of {strategy.replace('_', ' ')} related keywords in news"
                    )
    if recommendations:
        for rec in recommendations:
            with st.expander(f"📋 {rec['strategy']} (Confidence: {rec['confidence']})"):
                st.write(f"**Rationale:** {rec['rationale']}")
                st.write(f"**Risk Level:** {rec['risk_level']}")
                st.write(f"**Timeframe:** {rec['timeframe']}")
                st.write("**Recommended Actions:**")
                for action in rec['actions']:
                    st.write(f"• {action}")
    else:
        st.info("No specific strategic recommendations available. Consider maintaining current strategy.")

    if len(recommendations) > 1:
        st.subheader("📊 Strategy Priority Matrix")
        priority_data = []
        for rec in recommendations:
            confidence_score = {'Low': 1, 'Medium': 2, 'Medium-High': 2.5, 'High': 3}[rec['confidence']]
            risk_score = {'Low': 1, 'Low-Medium': 1.5, 'Medium': 2, 'Medium-High': 2.5, 'High': 3}[rec['risk_level']]
            priority_data.append({
                'Strategy': rec['strategy'],
                'Confidence': confidence_score,
                'Risk': risk_score,
                'Priority Score': confidence_score / max(1e-9, risk_score)
            })
        priority_df = pd.DataFrame(priority_data)
        priority_df = priority_df.sort_values('Priority Score', ascending=False)
        st.dataframe(priority_df.round(2), use_container_width=True)
    
    return recommendations

# =============================
# Fundamentals aggregator (NEW)
# =============================
class FundamentalFactors:
    """
    Aggregates economy/sector/industry metrics and user provisional inputs into
    a single scalar 'fundamental_score' in range ~[-1, 1].
    """
    def __init__(self):
        self.factors = {
            'interest_rate': 0.0,
            'inflation': 0.0,
            'gdp_growth': 0.0,
            'sector_perf': 0.0,
            'industry_perf': 0.0,
            'personal_factor': 0.0
        }
        # Default weights (can be tuned). Sign encodes expected relation to equities.
        self.weights = {
            'interest_rate': -0.3,
            'inflation': -0.2,
            'gdp_growth': 0.4,
            'sector_perf': 0.5,
            'industry_perf': 0.6,
            'personal_factor': 1.0
        }

    def set_factor(self, name: str, value: float):
        if name in self.factors:
            try:
                self.factors[name] = float(value)
            except Exception:
                pass

    def set_weight(self, name: str, weight: float):
        if name in self.weights:
            try:
                self.weights[name] = float(weight)
            except Exception:
                pass

    def compute_score(self) -> float:
        score = 0.0
        for key, value in self.factors.items():
            score += self.weights.get(key, 0.0) * float(value)
        # Clamp to a reasonable range
        return float(np.clip(score, -1.0, 1.0))

# =============================
# News Analyzer
# =============================
class NewsAnalyzer:
    def __init__(self, api_key: str | None):
        self.api_key = api_key
        self.newsapi = None
        # Store for manually added news
        self.manual_news = pd.DataFrame(columns=['title', 'description', 'publishedAt', 'sentiment'])

    def add_manual_news(self, title, description, sentiment_score):
        """Add manually entered news with sentiment score"""
        new_news = pd.DataFrame({
            'title': [title],
            'description': [description],
            'publishedAt': [datetime.now()],
            'sentiment': [sentiment_score],
            'source': 'manual'
        })
        self.manual_news = pd.concat([self.manual_news, new_news], ignore_index=True)

    def preprocess_news_data(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        try:
            df['publishedAt'] = pd.to_datetime(df['publishedAt'], errors='coerce')
            df = df.dropna(subset=['publishedAt'])
            keep_cols = [c for c in ['author', 'title', 'description', 'publishedAt', 'url', 'source'] if c in df.columns]
            df = df[keep_cols]
            return df
        except Exception as e:
            st.error(f"Error preprocessing news: {e}")
            return df

    def analyze_sentiment(self, text):
        if pd.isna(text) or text == "":
            return 0.0
        try:
            blob = TextBlob(str(text))
            return float(blob.sentiment.polarity)
        except Exception:
            return 0.0

    def get_news_sentiment(self, symbol: str, days: int = 10):
        # Combine with manual news
        df = self.manual_news.copy() if not self.manual_news.empty else pd.DataFrame()
        
        if df.empty:
            return 0.0, df
        
        # Calculate average sentiment
        avg_sentiment = float(df['sentiment'].mean()) if 'sentiment' in df.columns else 0.0
        return avg_sentiment, df

# =============================
# LSTM Model
# =============================
class LSTMPredictor:
    def __init__(self, lookback=60, units=50, dropout=0.2):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        class LSTMPredictor:
            def __init__(self, lookback=60, units=50, dropout=0.2, predict_ohlc=False):
                self.lookback = lookback
                self.units = units
                self.dropout = dropout
                self.model = None
                self.scaler = MinMaxScaler()
                self.predict_ohlc = predict_ohlc

            def prepare_data(self, data, feature_columns=['Close']):
                """Prepare data for LSTM model - now supports OHLC prediction"""
                scaled_data = self.scaler.fit_transform(data[feature_columns])
                X, y = [], []
                for i in range(self.lookback, len(scaled_data)):
                    X.append(scaled_data[i-self.lookback:i])
                    if self.predict_ohlc and 'Open' in feature_columns and 'High' in feature_columns and 'Low' in feature_columns:
                        y.append(scaled_data[i, :4])
                    else:
                        y.append(scaled_data[i, 0])
                return np.array(X), np.array(y)

            def build_model(self, input_shape):
                model = Sequential()
                model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
                model.add(Dropout(self.dropout))
                model.add(LSTM(units=self.units, return_sequences=True))
                model.add(Dropout(self.dropout))
                model.add(LSTM(units=self.units))
                model.add(Dropout(self.dropout))
                if self.predict_ohlc:
                    model.add(Dense(units=4))
                else:
                    model.add(Dense(units=1))
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                return model

            def predict(self, X):
                """Make predictions - handle both OHLC and Close-only predictions"""
                if self.model is None:
                    return np.array([])
                predictions = self.model.predict(X)
                if self.predict_ohlc:
                    return self.scaler.inverse_transform(predictions)
                else:
                    return self.scaler.inverse_transform(
                        np.concatenate((predictions, np.zeros((len(predictions), X.shape[2]-1))), axis=1)
                    )[:, 0]
    # ...existing code...
    # The following block is misplaced and should be removed or refactored
    # If GRU support is needed, define a GRUPredictor class similar to LSTMPredictor
    pass

class CNNLSTMPredictor(LSTMPredictor):
    def build_model(self, input_shape):
        model = Sequential()
        model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model
    def __init__(self, lookback=60, units=50, dropout=0.2):
        self.lookback = lookback
        self.units = units
        self.dropout = dropout
        self.model = None
        self.scaler = MinMaxScaler()

    def prepare_data(self, data, feature_columns=['Close']):
        """Prepare data for LSTM model"""
        # Scale the data
        scaled_data = self.scaler.fit_transform(data[feature_columns])
        
        # Create sequences for LSTM
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i])
            y.append(scaled_data[i, 0])  # Predict Close price
        
        return np.array(X), np.array(y)

    def build_model(self, input_shape):
        """Build LSTM model"""
        model = Sequential()
        model.add(LSTM(units=self.units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.dropout))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.dropout))
        model.add(Dense(units=1))
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        return model

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """Train the LSTM model"""
        if X_train.shape[0] == 0:
            return None
        
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(
            X_train, y_train, 
            epochs=epochs, 
            batch_size=batch_size, 
            validation_split=0.2,
            verbose=0
        )
        return history

    def predict(self, X):
        """Make predictions"""
        if self.model is None:
            return np.array([])
        predictions = self.model.predict(X)
        return self.scaler.inverse_transform(np.concatenate((predictions, np.zeros((len(predictions), X.shape[2]-1))), axis=1))[:, 0]

    def forecast_future(self, last_sequence, days=30):
        """Forecast future prices"""
        if self.model is None:
            return []
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days):
            # Predict next value
            next_pred = self.model.predict(current_sequence.reshape(1, self.lookback, current_sequence.shape[1]), verbose=0)
            
            # Create new row for the sequence
            new_row = np.zeros((1, current_sequence.shape[1]))
            new_row[0, 0] = next_pred[0, 0]  # Only update the Close price
            
            # Update sequence
            current_sequence = np.vstack([current_sequence[1:], new_row])
            
            # Inverse transform to get actual price
            actual_pred = self.scaler.inverse_transform(
                np.concatenate((next_pred, np.zeros((1, current_sequence.shape[1]-1))), axis=1)
            )[0, 0]
            
            predictions.append(actual_pred)
        
        return predictions

# =============================
# Stock Predictor
# =============================
class StockPredictor:
    def train_xgboost(self, X, y):
        if not XGBOOST_AVAILABLE:
            st.error("XGBoost not available. Install with: pip install xgboost")
            return None
        model = xgb.XGBRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        return model

    def train_lightgbm(self, X, y):
        if not LIGHTGBM_AVAILABLE:
            st.error("LightGBM not available. Install with: pip install lightgbm")
            return None
        model = lgb.LGBMRegressor(n_estimators=200, random_state=42)
        model.fit(X, y)
        return model
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.model = None
        self.lstm_model = None
        self.prophet_model = None
        self.markov_model = None
        self.lookback = 30

    def train_markov(self, prices, n_states=5):
        self.markov_model = MarkovChainPredictor(n_states=n_states)
        states = self.markov_model.train(prices)
        return states

    def predict_future_markov(self, current_price, days=30):
        if self.markov_model is None:
            return []
        current_state = np.digitize([current_price], self.markov_model.state_bins)[0] - 1
        current_state = max(0, min(current_state, self.markov_model.n_states - 1))
        state_predictions, confidences = self.markov_model.predict_sequence(current_state, n_steps=days)
        price_predictions = [self.markov_model.state_to_price(state) for state in state_predictions]
        return price_predictions[1:], confidences[1:]

    def get_stock_data(self, symbol, period="1y"):
        try:
            stock = yf.Ticker(symbol)
            df = stock.history(period=period)
            if df is None or df.empty:
                return pd.DataFrame()
            df = df.dropna()
            return df
        except Exception as e:
            st.error(f"Error fetching stock data for {symbol}: {e}")
            return pd.DataFrame()

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        try:
            df = df.copy()
            # Moving averages
            df['MA_20'] = ta.trend.sma_indicator(df['Close'], window=20)
            df['MA_50'] = ta.trend.sma_indicator(df['Close'], window=50)
            # RSI
            df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
            # MACD
            macd = ta.trend.MACD(df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            # Bollinger
            boll = ta.volatility.BollingerBands(df['Close'])
            df['BB_Upper'] = boll.bollinger_hband()
            df['BB_Lower'] = boll.bollinger_lband()
            # Volume MA
            if 'Volume' in df.columns:
                df['Volume_MA'] = ta.trend.sma_indicator(df['Volume'], window=20)
            # Add daily sentiment as a feature
            if 'daily_sentiment' not in df.columns:
                df['daily_sentiment'] = 0.0
            df = df.replace([np.inf, -np.inf], np.nan).dropna()
            return df
        except Exception as e:
            st.error(f"Error adding technical indicators: {e}")
            return df

    def _build_flat_features(self, window_df: pd.DataFrame, sentiment_score: float, fundamental_score: float) -> list:
        price_features = [
            window_df['Close'].values,
            window_df['Volume'].values if 'Volume' in window_df.columns else np.zeros(len(window_df)),
            window_df['MA_20'].values,
            window_df['MA_50'].values,
            window_df['RSI'].values,
        ]
        flat_features = []
        for arr in price_features:
            flat_features.extend(list(arr))
        flat_features.append(float(sentiment_score))
        flat_features.append(float(fundamental_score))
        return flat_features

    def prepare_features(self, df: pd.DataFrame, sentiment_score=0.0, fundamental_score=0.0):
        df = self.add_technical_indicators(df)
        features, target = [], []
        L = self.lookback
        for i in range(L, len(df)):
            window = df.iloc[i-L:i]
            if not nan_safe(window[['Close', 'MA_20', 'MA_50', 'RSI']].values):
                continue
            flat = self._build_flat_features(window, sentiment_score, fundamental_score)
            # Verify feature length
            expected = L * 5 + 2  # 5 arrays in price_features + 2 meta features
            if len(flat) == expected:
                features.append(flat)
                target.append(df['Close'].iloc[i])
        return np.array(features, dtype=float), np.array(target, dtype=float)

    def train_model(self, X, y, model_type='rf'):
        try:
            if X is None or len(X) == 0:
                return None
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test) if len(X_test) else X_test
            if model_type == 'rf':
                self.model = RandomForestRegressor(n_estimators=200, random_state=42)
            else:
                self.model = LinearRegression()
            self.model.fit(X_train_scaled, y_train)
            train_pred = self.model.predict(X_train_scaled)
            test_pred = self.model.predict(X_test_scaled) if len(X_test) else np.array([])
            train_mae = mean_absolute_error(y_train, train_pred)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            if len(X_test):
                test_mae = mean_absolute_error(y_test, test_pred)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            else:
                test_mae = np.nan
                test_rmse = np.nan
            return {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'y_train': y_train,
                'y_test': y_test
            }
        except Exception as e:
            st.error(f"Error training model: {e}")
            return None

    def predict_future(self, df, sentiment_score=0.0, fundamental_score=0.0, days=5):
        try:
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add sentiment and fundamental scores as features
            df['sentiment_score'] = sentiment_score
            df['fundamental_score'] = fundamental_score

            # Prepare data for LSTM
            feature_columns = ['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'sentiment_score', 'fundamental_score']
            available_features = [col for col in feature_columns if col in df.columns]

            if len(available_features) < 2:  # Need at least Close and one other feature
                available_features = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']

            self.lstm_model = LSTMPredictor(lookback=60)
            X, y = self.lstm_model.prepare_data(df, available_features)

            if len(X) == 0:
                return None

            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]

            # Train model
            history = self.lstm_model.train(X_train, y_train, epochs=50, batch_size=32)

            # Evaluate model
            train_pred = self.lstm_model.predict(X_train)
            test_pred = self.lstm_model.predict(X_test)

            train_mae = mean_absolute_error(y_train, train_pred) if len(train_pred) > 0 else np.nan
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred)) if len(train_pred) > 0 else np.nan
            test_mae = mean_absolute_error(y_test, test_pred) if len(test_pred) > 0 else np.nan
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred)) if len(test_pred) > 0 else np.nan

            return {
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_pred': train_pred,
                'test_pred': test_pred,
                'y_train': y_train,
                'y_test': y_test,
                'history': history
            }
        except Exception as e:
            st.error(f"Error training LSTM model: {e}")
            return None

    def predict_future_lstm(self, df, sentiment_score=0.0, fundamental_score=0.0, days=30):
        """Predict future prices using LSTM with sentiment and fundamental scores"""
        try:
            if self.lstm_model is None or self.lstm_model.model is None:
                return []
            
            # Add technical indicators
            df = self.add_technical_indicators(df)
            
            # Add sentiment and fundamental scores as features
            df['sentiment_score'] = sentiment_score
            df['fundamental_score'] = fundamental_score
            
            # Prepare features for LSTM
            feature_columns = ['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'sentiment_score', 'fundamental_score']
            available_features = [col for col in feature_columns if col in df.columns]
            
            if len(available_features) < 2:
                available_features = ['Close', 'Volume'] if 'Volume' in df.columns else ['Close']
            
            # Get the last sequence
            scaled_data = self.lstm_model.scaler.transform(df[available_features].tail(self.lstm_model.lookback))
            
            # Forecast future prices
            predictions = self.lstm_model.forecast_future(scaled_data, days=days)
            
            return predictions
        except Exception as e:
            st.error(f"Error predicting with LSTM: {e}")
            return []

    def train_prophet(self, df):
        """Train Prophet model"""
        try:
            if not PROPHET_AVAILABLE:
                st.error("Prophet not available. Install with: pip install prophet")
                return None
                
            self.prophet_model = ProphetPredictor()
            success = self.prophet_model.train(df)
            return success
        except Exception as e:
            st.error(f"Error training Prophet model: {e}")
            return False

    def predict_future_prophet(self, days=30):
        """Predict future prices using Prophet"""
        try:
            if self.prophet_model is None or self.prophet_model.model is None:
                return []
                
            forecast = self.prophet_model.predict(periods=days)
            if forecast.empty:
                return []
                
            # Get the future predictions
            future_predictions = forecast.tail(days)['yhat'].values
            return future_predictions.tolist()
        except Exception as e:
            st.error(f"Error predicting with Prophet: {e}")
            return []

# =============================
# Download Report Function
# =============================
def create_download_link(content, filename="stock_analysis_report.txt"):
    """Generate a download link for the report"""
    b64 = base64.b64encode(content.encode()).decode()
    href = f'<a href="data:file/txt;base64,{b64}" download="{filename}">Download Report</a>'
    return href

def create_pdf_download_link(pdf_buffer, filename="stock_analysis_report.pdf"):
    """Generate a download link for PDF report"""
    b64 = base64.b64encode(pdf_buffer.getvalue()).decode()
    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Download PDF Report</a>'
    return href

# =============================
# Streamlit App
# =============================
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
<style>
    .main-header { font-size: 3rem; color: #1E88E5; text-align: center; margin-bottom: 2rem; }
    .metric-card { background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; color: white; text-align: center; }
    .news-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; border-left: 4px solid #1E88E5; }
</style>
""",
    unsafe_allow_html=True,
)


def main():
    # Initialize database
    init_db()
    
    st.markdown('<h1 class="main-header">📈 Stock Prediction Dashboard</h1>', unsafe_allow_html=True)

    # Sidebar configuration
    st.sidebar.header("Configuration")
    openai_api_key = st.sidebar.text_input("OpenAI API Key (Optional)", type="password", help="Required for AI analysis and report generation")
    if openai_api_key:
        import openai
        openai.api_key = openai_api_key
    symbol = st.sidebar.text_input("Stock Symbol", value="AAPL", help="Enter stock ticker symbol (e.g., AAPL, GOOGL)")
    
    # Sector and Industry dropdowns
    selected_sector = st.sidebar.selectbox("Sector", SECTORS)
    selected_industry = st.sidebar.selectbox("Industry", INDUSTRIES.get(selected_sector, ["Other"]))
    
    period = st.sidebar.selectbox("Data Period", ["1y", "2y", "5y", "max"], index=0)
    model_type = st.sidebar.selectbox(
        "Prediction Model",
        ["Random Forest", "Linear Regression", "LSTM", "GRU", "CNN-LSTM", "XGBoost", "LightGBM", "Prophet", "Markov Chain"]
    )
    model_key = {
        "Random Forest": "rf",
        "Linear Regression": "lr",
        "LSTM": "lstm",
        "GRU": "gru",
        "CNN-LSTM": "cnn_lstm",
        "XGBoost": "xgb",
        "LightGBM": "lgb",
        "Prophet": "prophet",
        "Markov Chain": "markov"
    }[model_type]
    pred_days = st.sidebar.slider("Prediction Days", 1, 30, 10)

    # === Fundamental inputs (NEW) ===
    st.sidebar.subheader("Fundamentals & Provisional Inputs")
    # Sector/industry proxies using ETFs
    sector_etf = st.sidebar.text_input("Sector ETF (proxy)", value="XLK")
    industry_etf = st.sidebar.text_input("Industry ETF (proxy)", value="SOXX")

    with st.sidebar.expander("Manual Macro Inputs (-1 bearish to +1 bullish)"):
        interest_rate = st.slider("Interest Rate Impact", -1.0, 1.0, 0.0)
        inflation = st.slider("Inflation Impact", -1.0, 1.0, 0.0)
        gdp_growth = st.slider("GDP Growth Impact", -1.0, 1.0, 0.0)
    personal_factor = st.sidebar.slider("Personal/Provisional Adjustment", -1.0, 1.0, 0.0)

    # Optional: adjust weights
    with st.sidebar.expander("Weights (advanced)"):
        w_ir = st.number_input("Weight: Interest Rate", value=-0.3, step=0.1, format="%.2f")
        w_inf = st.number_input("Weight: Inflation", value=-0.2, step=0.1, format="%.2f")
        w_gdp = st.number_input("Weight: GDP Growth", value=0.4, step=0.1, format="%.2f")
        w_sector = st.number_input("Weight: Sector Perf", value=0.5, step=0.1, format="%.2f")
        w_industry = st.number_input("Weight: Industry Perf", value=0.6, step=0.1, format="%.2f")
        w_personal = st.number_input("Weight: Personal", value=1.0, step=0.1, format="%.2f")

    # Manual news input
    st.sidebar.subheader("Add Manual News")
    with st.sidebar.form("manual_news_form"):
        news_title = st.text_input("News Title")
        news_description = st.text_area("News Description")
        news_sentiment = st.slider("Sentiment Score", -1.0, 1.0, 0.0, 0.1)
        submitted = st.form_submit_button("Add News")
        if submitted and news_title:
            st.sidebar.success("News added successfully!")

    # Historical predictions comparison
    st.sidebar.subheader("Historical Predictions")
    if st.sidebar.button("View Historical Predictions"):
        historical_predictions = get_historical_predictions(symbol)
        if not historical_predictions.empty:
            st.sidebar.write(f"Historical Predictions for {symbol}")
            st.sidebar.dataframe(historical_predictions)
        else:
            st.sidebar.info("No historical predictions found for this symbol.")

    if st.sidebar.button("Analyze Stock", type="primary"):
        with st.spinner("Analyzing stock data, fundamentals, and news sentiment..."):
            news_analyzer = NewsAnalyzer(None)
            
            # Add manual news if provided
            if submitted and news_title:
                news_analyzer.add_manual_news(news_title, news_description, news_sentiment)
            
            stock_predictor = StockPredictor()
            ai_agent = AIAnalysisAgent(openai_api_key)

            # Fetch stock data
            stock_data = stock_predictor.get_stock_data(symbol, period)
            if stock_data.empty:
                st.error("Failed to fetch stock data. Please check the symbol.")
                return

            # Compute sector/industry performance (simple mean return over period)
            sector_perf = 0.0
            industry_perf = 0.0
            try:
                sector_df = yf.Ticker(sector_etf).history(period=period)
                industry_df = yf.Ticker(industry_etf).history(period=period)
                if not sector_df.empty:
                    sector_perf = float(safe_pct_change(sector_df['Close']).mean())
                if not industry_df.empty:
                    industry_perf = float(safe_pct_change(industry_df['Close']).mean())
            except Exception:
                pass

            # News sentiment
            sentiment_score, news_df = news_analyzer.get_news_sentiment(symbol, 10)

            # Build fundamental score
            fundamentals = FundamentalFactors()
            fundamentals.set_weight('interest_rate', w_ir)
            fundamentals.set_weight('inflation', w_inf)
            fundamentals.set_weight('gdp_growth', w_gdp)
            fundamentals.set_weight('sector_perf', w_sector)
            fundamentals.set_weight('industry_perf', w_industry)
            fundamentals.set_weight('personal_factor', w_personal)

            fundamentals.set_factor('interest_rate', interest_rate)
            fundamentals.set_factor('inflation', inflation)
            fundamentals.set_factor('gdp_growth', gdp_growth)
            fundamentals.set_factor('sector_perf', sector_perf)
            fundamentals.set_factor('industry_perf', industry_perf)
            fundamentals.set_factor('personal_factor', personal_factor)

            fundamental_score = fundamentals.compute_score()

            # Create tabs
            tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📊 Stock Analysis", "🔮 Predictions", "📈 Technical Analysis", "🤖 AI Analysis", "📋 Historical Comparison", "📊 Prophet Analysis"])

            with tab1:
                current_price = float(stock_data['Close'].iloc[-1])
                prev_price = float(stock_data['Close'].iloc[-2]) if len(stock_data) > 1 else current_price
                price_change = current_price - prev_price
                price_change_pct = (price_change / prev_price) * 100 if prev_price else 0

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>{symbol}</h3>
                        <h2>${current_price:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    color = "#00e676" if price_change >= 0 else "#ff5252"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Change</h3>
                        <h2 style="color: {color}">${price_change:.2f}</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    color = "#00e676" if price_change_pct >= 0 else "#ff5252"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Change %</h3>
                        <h2 style="color: {color}">{price_change_pct:.2f}%</h2>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    sentiment_color = "#00e676" if sentiment_score > 0.05 else "#ff5252" if sentiment_score < -0.05 else "#bdbdbd"
                    st.markdown(f"""
                    <div class="metric-card">
                        <h3>Sentiment</h3>
                        <h2 style="color: {sentiment_color}">{sentiment_score:.3f}</h2>
                    </div>
                    """, unsafe_allow_html=True)

                # Price chart
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=stock_data.index,
                    open=stock_data['Open'],
                    high=stock_data['High'],
                    low=stock_data['Low'],
                    close=stock_data['Close'],
                    name="Price"
                ))
                fig.update_layout(title=f"{symbol} Stock Price", yaxis_title="Price ($)", xaxis_title="Date", height=500)
                st.plotly_chart(fig, use_container_width=True)

            with tab2:
                if model_type == "Markov Chain":
                    with st.spinner("Training Markov model..."):
                        prices = stock_data['Close'].values
                        states = stock_predictor.train_markov(prices, n_states=5)
                        st.info(f"Markov Model: {len(np.unique(states))} states created")
                        if stock_predictor.markov_model.transition_matrix is not None:
                            st.subheader("Markov Transition Matrix")
                            transition_df = pd.DataFrame(
                                stock_predictor.markov_model.transition_matrix,
                                index=[f"State {i}" for i in range(stock_predictor.markov_model.n_states)],
                                columns=[f"State {i}" for i in range(stock_predictor.markov_model.n_states)]
                            )
                            st.dataframe(transition_df.style.format("{:.3f}").background_gradient(cmap='Blues'))
                        current_price = float(stock_data['Close'].iloc[-1])
                        future_prices, confidences = stock_predictor.predict_future_markov(current_price, pred_days)
                        if future_prices:
                            last_date = stock_data.index[-1]
                            future_dates = [last_date + timedelta(days=i+1) for i in range(pred_days)]
                            fig_pred = go.Figure()
                            fig_pred.add_trace(go.Scatter(
                                x=stock_data.index[-30:], 
                                y=stock_data['Close'].iloc[-30:], 
                                mode='lines', 
                                name='Historical', 
                                line=dict(color='blue')
                            ))
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates, 
                                y=future_prices, 
                                mode='lines+markers', 
                                name='Predicted', 
                                line=dict(color='red', dash='dash')
                            ))
                            upper_bound = [p * (1 + (1 - c) * 0.1) for p, c in zip(future_prices, confidences)]
                            lower_bound = [p * (1 - (1 - c) * 0.1) for p, c in zip(future_prices, confidences)]
                            fig_pred.add_trace(go.Scatter(
                                x=future_dates + future_dates[::-1],
                                y=upper_bound + lower_bound[::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='Confidence Interval'
                            ))
                            fig_pred.update_layout(
                                title=f"{symbol} Markov Chain Prediction ({pred_days} days)",
                                yaxis_title="Price ($)",
                                xaxis_title="Date",
                                height=500
                            )
                            st.plotly_chart(fig_pred, use_container_width=True)
                            pred_df = pd.DataFrame({
                                'Date': [d.strftime('%Y-%m-%d') for d in future_dates],
                                'Predicted Price': [f"${p:.2f}" for p in future_prices],
                                'Confidence': [f"{c*100:.1f}%" for c in confidences]
                            })
                            st.subheader("Markov Chain Predictions")
                            st.dataframe(pred_df, use_container_width=True)
                            predicted_price = future_prices[-1] if future_prices else current_price
                            save_prediction(symbol, current_price, predicted_price, 
                                          sentiment_score, fundamental_score, model_type, pred_days)
                    st.subheader("Stock Price Predictions")
                
                available_features = ['Close', 'Volume', 'MA_20', 'MA_50', 'RSI', 'MACD', 'sentiment_score', 'fundamental_score']
                available_features = [col for col in available_features if col in stock_data.columns]
                if model_type == "LSTM":
                    with st.spinner("Training LSTM model..."):
                        results = stock_predictor.train_lstm(stock_data, sentiment_score, fundamental_score)
                    if not results:
                        st.error("Failed to train LSTM model")
                    else:
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Training MAE", f"${results['train_mae']:.2f}")
                            st.metric("Training RMSE", f"${results['train_rmse']:.2f}")
                        with col2:
                            test_mae = results['test_mae']
                            test_rmse = results['test_rmse']
                            st.metric("Testing MAE", "N/A" if pd.isna(test_mae) else f"${test_mae:.2f}")
                            st.metric("Testing RMSE", "N/A" if pd.isna(test_rmse) else f"${test_rmse:.2f}")
                        future_prices = stock_predictor.predict_future_lstm(stock_data, sentiment_score, fundamental_score, pred_days)
                elif model_type == "GRU":
                    with st.spinner("Training GRU model..."):
                        # stock_predictor.lstm_model = GRUPredictor(lookback=60) # Fix: GRUPredictor not defined. Use LSTMPredictor or define GRUPredictor if needed.
                        X, y = stock_predictor.lstm_model.prepare_data(stock_data, available_features)
                        if len(X) > 0:
                            history = stock_predictor.lstm_model.train(X, y, epochs=50, batch_size=32)
                            future_prices = stock_predictor.lstm_model.forecast_future(X[-1], days=pred_days)
                        else:
                            st.error("Insufficient data for GRU model.")
                elif model_type == "CNN-LSTM":
                    with st.spinner("Training CNN-LSTM model..."):
                        stock_predictor.lstm_model = CNNLSTMPredictor(lookback=60)
                        X, y = stock_predictor.lstm_model.prepare_data(stock_data, available_features)
                        if len(X) > 0:
                            history = stock_predictor.lstm_model.train(X, y, epochs=50, batch_size=32)
                            future_prices = stock_predictor.lstm_model.forecast_future(X[-1], days=pred_days)
                        else:
                            st.error("Insufficient data for CNN-LSTM model.")
                elif model_type == "XGBoost":
                    with st.spinner("Training XGBoost model..."):
                        X, y = stock_predictor.prepare_features(stock_data, sentiment_score, fundamental_score)
                        if len(X) > 0:
                            model = stock_predictor.train_xgboost(X, y)
                            stock_predictor.model = model
                            future_prices = stock_predictor.predict_future(stock_data, sentiment_score, fundamental_score, pred_days)
                        else:
                            st.error("Insufficient data for XGBoost model.")
                elif model_type == "LightGBM":
                    with st.spinner("Training LightGBM model..."):
                        X, y = stock_predictor.prepare_features(stock_data, sentiment_score, fundamental_score)
                        if len(X) > 0:
                            model = stock_predictor.train_lightgbm(X, y)
                            stock_predictor.model = model
                            future_prices = stock_predictor.predict_future(stock_data, sentiment_score, fundamental_score, pred_days)
                        else:
                            st.error("Insufficient data for LightGBM model.")
                elif model_type == "Prophet":
                    with st.spinner("Training Prophet model..."):
                        success = stock_predictor.train_prophet(stock_data)
                    if not success:
                        st.error("Failed to train Prophet model")
                    else:
                        future_prices = stock_predictor.predict_future_prophet(pred_days)
                else:
                    X, y = stock_predictor.prepare_features(stock_data, sentiment_score, fundamental_score)
                    if len(X) == 0:
                        st.error("Insufficient data for prediction model after feature preparation.")
                    else:
                        results = stock_predictor.train_model(X, y, model_key)
                        if not results:
                            st.error("Failed to train prediction model")
                        else:
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Training MAE", f"${results['train_mae']:.2f}")
                                st.metric("Training RMSE", f"${results['train_rmse']:.2f}")
                            with col2:
                                test_mae = results['test_mae']
                                test_rmse = results['test_rmse']
                                st.metric("Testing MAE", "N/A" if pd.isna(test_mae) else f"${test_mae:.2f}")
                                st.metric("Testing RMSE", "N/A" if pd.isna(test_rmse) else f"${test_rmse:.2f}")
                            future_prices = stock_predictor.predict_future(stock_data, sentiment_score, fundamental_score, pred_days)
                
                if not future_prices:
                    st.error("Failed to generate predictions")
                else:
                    # Save prediction to database
                    current_price = float(stock_data['Close'].iloc[-1])
                    predicted_price = future_prices[-1] if future_prices else current_price
                    save_prediction(symbol, current_price, predicted_price, sentiment_score, fundamental_score, model_type, pred_days)
                    
                    last_date = stock_data.index[-1]
                    future_dates = [last_date + timedelta(days=i+1) for i in range(pred_days)]
                    fig_pred = go.Figure()
                    fig_pred.add_trace(go.Scatter(x=stock_data.index[-30:], y=stock_data['Close'].iloc[-30:], mode='lines', name='Historical', line=dict(color='blue')))
                    fig_pred.add_trace(go.Scatter(x=future_dates, y=future_prices, mode='lines+markers', name='Predicted', line=dict(color='red', dash='dash')))
                    fig_pred.update_layout(title=f"{symbol} Price Prediction ({pred_days} days)", yaxis_title="Price ($)", xaxis_title="Date", height=500)
                    st.plotly_chart(fig_pred, use_container_width=True)
                    pred_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': [f"${p:.2f}" for p in future_prices]})
                    st.subheader("Price Predictions")
                    st.dataframe(pred_df, use_container_width=True)

                    strategy_recommendations = display_strategy_recommendations(symbol, sentiment_score, news_df, current_price, future_prices)

            with tab3:
                st.subheader("Technical Analysis")
                tech_data = stock_predictor.add_technical_indicators(stock_data)
                # RSI
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(x=tech_data.index, y=tech_data['RSI'], mode='lines', name='RSI', line=dict(color='purple')))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI Indicator", yaxis_title="RSI", height=400)
                st.plotly_chart(fig_rsi, use_container_width=True)
                # MAs
                fig_ma = go.Figure()
                fig_ma.add_trace(go.Scatter(x=tech_data.index, y=tech_data['Close'], name='Close Price'))
                fig_ma.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA_20'], name='MA 20'))
                fig_ma.add_trace(go.Scatter(x=tech_data.index, y=tech_data['MA_50'], name='MA 50'))
                fig_ma.update_layout(title="Moving Averages", yaxis_title="Price ($)", height=400)
                st.plotly_chart(fig_ma, use_container_width=True)
                # Metrics
                current_rsi = float(tech_data['RSI'].iloc[-1])
                current_ma20 = float(tech_data['MA_20'].iloc[-1])
                current_ma50 = float(tech_data['MA_50'].iloc[-1])
                current_price = float(stock_data['Close'].iloc[-1])
                col1, col2, col3 = st.columns(3)
                with col1:
                    rsi_signal = "Overbought" if current_rsi > 70 else "Oversold" if current_rsi < 30 else "Neutral"
                    st.metric("RSI", f"{current_rsi:.2f}", rsi_signal)
                with col2:
                    ma_signal = "Bullish" if current_price > current_ma20 else "Bearish"
                    st.metric("MA 20", f"${current_ma20:.2f}", ma_signal)
                with col3:
                    ma_trend = "Uptrend" if current_ma20 > current_ma50 else "Downtrend"
                    st.metric("MA 50", f"${current_ma50:.2f}", ma_trend)

            with tab4:
                st.subheader("AI Analysis")
                
                if not openai_api_key:
                    st.warning("Please provide an OpenAI API key to enable AI analysis")
                else:
                    with st.spinner("Generating AI analysis..."):
                        ai_analysis = ai_agent.analyze_predictions(
                            symbol, stock_data, future_prices, sentiment_score, 
                            news_df, fundamental_score
                        )
                    
                    st.write(ai_analysis)
                    
                    # Generate and download report
                    with st.spinner("Generating comprehensive report..."):
                        report_content = ai_agent.generate_report(
                            symbol, ai_analysis, stock_data, future_prices, 
                            sentiment_score, news_df, fundamental_score, 
                            strategy_recommendations if 'strategy_recommendations' in locals() else []
                        )
                    
                    st.markdown("### 📄 Download Report")
                    st.markdown(create_download_link(report_content, f"{symbol}_analysis_report.txt"), unsafe_allow_html=True)
                    
                    # Generate PDF report
                    pdf_buffer = generate_pdf_report(
                        symbol, ai_analysis, stock_data, future_prices,
                        sentiment_score, news_df, fundamental_score,
                        strategy_recommendations if 'strategy_recommendations' in locals() else [],
                        {}, {}  # Placeholder for technical_analysis and model_performance
                    )
                    st.markdown(create_pdf_download_link(pdf_buffer, f"{symbol}_analysis_report.pdf"), unsafe_allow_html=True)

            with tab5:
                st.subheader("Historical Predictions Comparison")
                historical_predictions = get_historical_predictions(symbol)
                
                if not historical_predictions.empty:
                    # Calculate accuracy of past predictions
                    historical_predictions['accuracy'] = historical_predictions.apply(
                        lambda row: (row['predicted_price'] - row['current_price']) / row['current_price'] * 100, 
                        axis=1
                    )
                    
                    # Display historical predictions
                    st.dataframe(historical_predictions, use_container_width=True)
                    
                    # Create comparison chart
                    fig_comp = go.Figure()
                    fig_comp.add_trace(go.Scatter(
                        x=historical_predictions['date'], 
                        y=historical_predictions['current_price'], 
                        mode='lines+markers', 
                        name='Actual Price',
                        line=dict(color='blue')
                    ))
                    fig_comp.add_trace(go.Scatter(
                        x=historical_predictions['date'], 
                        y=historical_predictions['predicted_price'], 
                        mode='lines+markers', 
                        name='Predicted Price',
                        line=dict(color='red', dash='dash')
                    ))
                    fig_comp.update_layout(
                        title=f"{symbol} Historical Predictions vs Actual Prices",
                        yaxis_title="Price ($)",
                        xaxis_title="Date",
                        height=500
                    )
                    st.plotly_chart(fig_comp, use_container_width=True)
                    
                    # Display accuracy metrics
                    avg_accuracy = historical_predictions['accuracy'].mean()
                    st.metric("Average Prediction Accuracy", f"{avg_accuracy:.2f}%")
                    
                    # Model performance comparison
                    if len(historical_predictions['model_type'].unique()) > 1:
                        st.subheader("Model Performance Comparison")
                        model_performance = historical_predictions.groupby('model_type')['accuracy'].agg(['mean', 'std', 'count']).round(2)
                        st.dataframe(model_performance, use_container_width=True)
                else:
                    st.info("No historical predictions found for this symbol.")
                    
            with tab6:
                st.subheader("Prophet Time Series Analysis")
                
                if not PROPHET_AVAILABLE:
                    st.error("Prophet not available. Install with: pip install prophet")
                else:
                    # Train Prophet model if not already trained
                    if stock_predictor.prophet_model is None:
                        with st.spinner("Training Prophet model..."):
                            success = stock_predictor.train_prophet(stock_data)
                        if not success:
                            st.error("Failed to train Prophet model")
                    
                    if stock_predictor.prophet_model and stock_predictor.prophet_model.model is not None:
                        # Make predictions
                        forecast = stock_predictor.prophet_model.predict(pred_days)
                        
                        if not forecast.empty:
                            # Plot the forecast
                            fig_prophet = go.Figure()
                            
                            # Historical data
                            fig_prophet.add_trace(go.Scatter(
                                x=stock_data.index, 
                                y=stock_data['Close'], 
                                mode='lines', 
                                name='Historical',
                                line=dict(color='blue')
                            ))
                            
                            # Forecast
                            fig_prophet.add_trace(go.Scatter(
                                x=forecast['ds'], 
                                y=forecast['yhat'], 
                                mode='lines', 
                                name='Forecast',
                                line=dict(color='red', dash='dash')
                            ))
                            
                            # Confidence interval
                            fig_prophet.add_trace(go.Scatter(
                                x=forecast['ds'].tolist() + forecast['ds'].tolist()[::-1],
                                y=forecast['yhat_upper'].tolist() + forecast['yhat_lower'].tolist()[::-1],
                                fill='toself',
                                fillcolor='rgba(255,0,0,0.2)',
                                line=dict(color='rgba(255,255,255,0)'),
                                name='Confidence Interval'
                            ))
                            
                            fig_prophet.update_layout(
                                title=f"{symbol} Prophet Forecast",
                                yaxis_title="Price ($)",
                                xaxis_title="Date",
                                height=500
                            )
                            st.plotly_chart(fig_prophet, use_container_width=True)
                            
                            # Show components
                            st.subheader("Prophet Components")
                            components_fig = stock_predictor.prophet_model.plot_components(forecast)
                            if components_fig:
                                st.pyplot(components_fig)
                            
                            # Show forecast table
                            st.subheader("Prophet Forecast Details")
                            forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(pred_days)
                            forecast_display.columns = ['Date', 'Predicted', 'Lower Bound', 'Upper Bound']
                            forecast_display['Date'] = forecast_display['Date'].dt.strftime('%Y-%m-%d')
                            forecast_display['Predicted'] = forecast_display['Predicted'].apply(lambda x: f"${x:.2f}")
                            forecast_display['Lower Bound'] = forecast_display['Lower Bound'].apply(lambda x: f"${x:.2f}")
                            forecast_display['Upper Bound'] = forecast_display['Upper Bound'].apply(lambda x: f"${x:.2f}")
                            st.dataframe(forecast_display, use_container_width=True)
                        else:
                            st.error("Failed to generate Prophet forecast")
                    else:
                        st.error("Prophet model not available")


if __name__ == "__main__":
    main()
