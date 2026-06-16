import json
from typing import TypedDict, List, Dict, Any
import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


# ====================== STATE ======================
class FinanceState(TypedDict):
    user_profile: Dict[str, Any]
    blackboard: List[str]
    simulations: List[Dict]
    final_recommendation: str


# ====================== MONTE CARLO SIMULATOR (Improved) ======================
class MonteCarloSimulator:
    def __init__(self):
        self.default_tickers = ['SPY', 'QQQ', 'VTI', 'BND', 'GLD']
    
    def get_historical_data(self, tickers: List[str], period="5y"):
        """Robust data fetching"""
        data = yf.download(
            tickers, 
            period=period, 
            progress=False, 
            auto_adjust=True,
            threads=True
        )
        
        if isinstance(data.columns, pd.MultiIndex):
            prices = data['Adj Close'] if 'Adj Close' in data.columns.get_level_values(0) else data['Close']
        else:
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0] if isinstance(tickers, list) else str(tickers))
        
        prices = prices.dropna(how='all')
        return prices
    
    def run_monte_carlo(self, 
                       initial_investment: float,
                       tickers: List[str],
                       weights: List[float] = None,
                       time_horizon_years: int = 3,
                       num_simulations: int = 5000):
        
        if weights is None:
            weights = [1.0 / len(tickers)] * len(tickers)
        
        prices = self.get_historical_data(tickers)
        returns = prices.pct_change().dropna()
        
        if returns.empty:
            raise ValueError(f"No data returned for tickers: {tickers}")
        
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        annual_vol = returns.std() * np.sqrt(252)
        
        print(f"📊 Historical Annual Volatility: {annual_vol.mean():.1%}")
        
        days = time_horizon_years * 252
        portfolio_sims = np.zeros((days, num_simulations))
        portfolio_sims[0] = initial_investment
        
        # More realistic simulation with slight pessimism
        adjusted_drift = mean_returns - 0.0003  # small risk adjustment
        
        for t in range(1, days):
            Z = np.random.normal(0, 1, size=(len(tickers), num_simulations))
            L = np.linalg.cholesky(cov_matrix.values)
            daily_returns = adjusted_drift.values.reshape(-1, 1) + L @ Z * (1/252)**0.5
            portfolio_sims[t] = portfolio_sims[t-1] * (1 + weights @ daily_returns)
        
        final_values = portfolio_sims[-1]
        
        return {
            "mean_final_value": float(np.mean(final_values)),
            "median_final_value": float(np.median(final_values)),
            "worst_5_percent": float(np.percentile(final_values, 5)),
            "best_5_percent": float(np.percentile(final_values, 95)),
            "success_probability_15pct": float(np.mean(final_values > initial_investment * 1.15)),
            "success_probability_0pct": float(np.mean(final_values > initial_investment)),
            "annual_volatility": float(annual_vol.mean()),
            "num_simulations": num_simulations,
            "time_horizon_years": time_horizon_years,
            "tickers": tickers,
            "initial_investment": initial_investment
        }


# ====================== MENTAL LOOP DECISION ======================
class MentalLoopDecision(BaseModel):
    chosen_scenario: str
    reasoning: str
    risk_assessment: str
    recommendation: str


def create_mental_loop_agent(llm, simulator: MonteCarloSimulator):
    
    def mental_loop_node(state: FinanceState):
        print("🤖 Running Mental Loop Simulation...")
        
        user = state["user_profile"]
        initial = float(user.get("initial_investment", 50000))
        horizon = int(user.get("time_horizon", 3))
        tickers = user.get("preferred_tickers", ["SPY", "QQQ", "BND"])
        
        # Run Simulation
        sim_result = simulator.run_monte_carlo(
            initial_investment=initial,
            tickers=tickers,
            time_horizon_years=horizon
        )
        
        state["simulations"].append(sim_result)
        
        # Stronger Mental Loop Prompt
        prompt = ChatPromptTemplate.from_template("""
You are a world-class Mental Loop Personal Finance Advisor.

User Profile:
{user_profile}

Monte Carlo Simulation Results (based on real market data):
{simulation_results}

Perform deep Mental Loop analysis:
- Simulate multiple future market regimes
- Consider user's goals and risk tolerance
- Choose the optimal allocation strategy

Be honest about risks and provide actionable advice.
""")
        
        chain = prompt | llm.with_structured_output(MentalLoopDecision)
        
        result = chain.invoke({
            "user_profile": json.dumps(user, indent=2),
            "simulation_results": json.dumps(sim_result, indent=2)
        })
        
        report = f"""**Mental Loop Finance Report**

**Chosen Strategy:** {result.chosen_scenario}

**Reasoning:**  
{result.reasoning}

**Risk Assessment:**  
{result.risk_assessment}

**Final Recommendation:**  
{result.recommendation}

**Monte Carlo Simulation Summary:**
- Initial Investment: ${sim_result['initial_investment']:,.0f}
- Expected Final Value: ${sim_result['mean_final_value']:,.0f}
- Median Final Value: ${sim_result['median_final_value']:,.0f}
- Worst 5% Scenario: ${sim_result['worst_5_percent']:,.0f}
- Success Probability (>15% growth): {sim_result['success_probability_15pct']*100:.1f}%
- Overall Positive Return Probability: {sim_result['success_probability_0pct']*100:.1f}%
- Historical Annual Volatility: {sim_result['annual_volatility']:.1%}
"""
        
        state["blackboard"].append(report)
        state["final_recommendation"] = result.recommendation
        
        return state
    
    return mental_loop_node


# ====================== BUILD ADVISOR ======================
def build_finance_advisor(llm):
    simulator = MonteCarloSimulator()
    mental_loop = create_mental_loop_agent(llm, simulator)
    
    def run_advisor(user_input: Dict) -> Dict:
        state: FinanceState = {
            "user_profile": user_input,
            "blackboard": [],
            "simulations": [],
            "final_recommendation": ""
        }
        return mental_loop(state)
    
    return run_advisor, simulator