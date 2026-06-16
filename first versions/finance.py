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


# ====================== MONTE CARLO SIMULATOR (Robust Version) ======================
class MonteCarloSimulator:
    def __init__(self):
        self.default_tickers = ['SPY', 'QQQ', 'VTI', 'BND', 'GLD']
    
    def get_historical_data(self, tickers: List[str], period="5y"):
        """Robust historical data fetch"""
        data = yf.download(
            tickers, 
            period=period, 
            progress=False, 
            auto_adjust=True,
            threads=True
        )
        
        # Handle different return structures from yfinance
        if isinstance(data.columns, pd.MultiIndex):
            # Multi-ticker case
            if 'Adj Close' in data.columns.get_level_values(0):
                prices = data['Adj Close']
            else:
                prices = data['Close']
        else:
            # Single ticker case
            prices = data['Adj Close'] if 'Adj Close' in data.columns else data['Close']
            if isinstance(prices, pd.Series):
                prices = prices.to_frame(name=tickers[0] if isinstance(tickers, list) else tickers)
        
        prices = prices.dropna(how='all')
        return prices
    
    def run_monte_carlo(self, 
                       initial_investment: float,
                       tickers: List[str],
                       weights: List[float] = None,
                       time_horizon_years: int = 3,
                       num_simulations: int = 3000):
        
        if weights is None:
            weights = [1.0 / len(tickers)] * len(tickers)
        
        prices = self.get_historical_data(tickers)
        returns = prices.pct_change().dropna()
        
        if returns.empty:
            raise ValueError(f"No historical data for tickers: {tickers}")
        
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Monte Carlo
        days = time_horizon_years * 252
        portfolio_sims = np.zeros((days, num_simulations))
        portfolio_sims[0] = initial_investment
        
        for t in range(1, days):
            Z = np.random.normal(0, 1, size=(len(tickers), num_simulations))
            L = np.linalg.cholesky(cov_matrix.values)
            daily_returns = mean_returns.values.reshape(-1, 1) + L @ Z * (1/252)**0.5
            portfolio_sims[t] = portfolio_sims[t-1] * (1 + weights @ daily_returns)
        
        final_values = portfolio_sims[-1]
        
        return {
            "mean_final_value": float(np.mean(final_values)),
            "median_final_value": float(np.median(final_values)),
            "worst_5_percent": float(np.percentile(final_values, 5)),
            "best_5_percent": float(np.percentile(final_values, 95)),
            "success_probability_15pct": float(np.mean(final_values > initial_investment * 1.15)),
            "num_simulations": num_simulations,
            "time_horizon_years": time_horizon_years,
            "tickers": tickers
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
        try:
            sim_result = simulator.run_monte_carlo(
                initial_investment=initial,
                tickers=tickers,
                time_horizon_years=horizon
            )
            state["simulations"].append(sim_result)
        except Exception as e:
            print(f"Simulation Error: {e}")
            sim_result = {"error": str(e)}
            state["simulations"].append(sim_result)
        
        # Mental Loop with LLM
        prompt = ChatPromptTemplate.from_template("""
You are a top-tier Mental Loop Personal Finance Advisor.

User Profile:
{user_profile}

Monte Carlo Simulation (based on real 5y market data):
{simulation_results}

Analyze deeply and choose the best strategy.
""")
        
        chain = prompt | llm.with_structured_output(MentalLoopDecision)
        
        result = chain.invoke({
            "user_profile": json.dumps(user, indent=2),
            "simulation_results": json.dumps(sim_result, indent=2)
        })
        
        report = f"""**Mental Loop Finance Report**

**Chosen Strategy:** {result.chosen_scenario}

**Reasoning:** {result.reasoning}

**Risk Assessment:** {result.risk_assessment}

**Recommendation:** {result.recommendation}

**Monte Carlo Summary:**
- Expected Final Value: ${sim_result.get('mean_final_value', 'N/A'):,.0f}
- Worst 5% Scenario: ${sim_result.get('worst_5_percent', 'N/A'):,.0f}
- Success Probability (>15% growth): {sim_result.get('success_probability_15pct', 0)*100:.1f}%
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