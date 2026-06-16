import json
from typing import TypedDict, List, Dict, Any, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END


# ====================== STATE ======================
class AgentState(TypedDict):
    user_profile: Dict[str, Any]
    market_analyst_proposal: Optional[Dict]
    simulation_results: Optional[List[Dict]]
    risk_evaluation: Optional[Dict]
    final_recommendation: Optional[Dict]
    blackboard: List[str]
    loop_count: int


# ====================== PYDANTIC MODELS ======================
class MarketAnalystProposal(BaseModel):
    strategy: str = Field(description="High-level investment strategy")
    reasoning: str = Field(description="Reasoning behind the proposed strategy")
    allocation: Dict[str, float] = Field(description="Suggested asset allocation (e.g. {'SPY': 0.4, 'BND': 0.4})")


class SimulationResult(BaseModel):
    scenario: str
    mean_final_value: float
    worst_5pct: float
    success_probability: float
    summary: str


class RiskEvaluation(BaseModel):
    assessment: str
    concerns: str
    suggested_adjustments: str
    confidence: float
    should_revise: bool = Field(description="Should we go back to analyst and revise strategy?")


class FinalRecommendation(BaseModel):
    final_strategy: str
    allocation: Dict[str, float]
    reasoning: str
    expected_outcome: str
    risk_summary: str


# ====================== MONTE CARLO SIMULATOR ======================
class MonteCarloSimulator:
    # def get_historical_data(self, tickers: List[str], period="5y"):
    #     data = yf.download(tickers, period=period, progress=False, auto_adjust=True)
    #     if isinstance(data.columns, pd.MultiIndex):
    #         prices = data['Adj Close']
    #     else:
    #         prices = data['Adj Close']
    #     return prices.dropna(how='all')
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
    def run_simulation(self, initial_investment: float, tickers: List[str], weights: List[float], 
                      time_horizon_years: int = 5, num_simulations: int = 3000):
        prices = self.get_historical_data(tickers)
        returns = prices.pct_change().dropna()
        
        mean_ret = returns.mean()
        cov = returns.cov()
        
        days = time_horizon_years * 252
        portfolio_sims = np.zeros((days, num_simulations))
        portfolio_sims[0] = initial_investment
        
        for t in range(1, days):
            Z = np.random.normal(0, 1, (len(tickers), num_simulations))
            L = np.linalg.cholesky(cov.values)
            daily_ret = mean_ret.values.reshape(-1,1) + L @ Z * (1/252)**0.5
            portfolio_sims[t] = portfolio_sims[t-1] * (1 + weights @ daily_ret)
        
        final_values = portfolio_sims[-1]
        
        return {
            "scenario": "Base Case",
            "mean_final_value": float(np.mean(final_values)),
            "worst_5pct": float(np.percentile(final_values, 5)),
            "success_probability": float(np.mean(final_values > initial_investment * 1.15)),
            "summary": f"Expected growth: +{((np.mean(final_values)/initial_investment)-1)*100:.1f}%"
        }


# ====================== NODES ======================
def market_analyst_node(state: AgentState, llm) -> Dict:
    print("📊 Market Analyst is proposing strategy...")
    print(f"📊 Market Analyst (Loop {state.get('loop_count', 1)})")

    prompt = ChatPromptTemplate.from_template(
        """You are a professional Market Analyst.
        Based on the user's profile, propose a clear investment strategy and asset allocation.

        User Profile:
        {user_profile}
        """
    )
    
    chain = prompt | llm.with_structured_output(MarketAnalystProposal)
    proposal = chain.invoke({"user_profile": json.dumps(state["user_profile"], indent=2)})
    
    report = f"**Market Analyst Proposal**\nStrategy: {proposal.strategy}\nAllocation: {proposal.allocation}"
    print(f"market analysis report:\t{report}")
    state["blackboard"].append(report)
    state["market_analyst_proposal"] = proposal.dict()
    state["loop_count"] = state.get("loop_count", 0) + 1

    return state


def simulation_node(state: AgentState, simulator: MonteCarloSimulator) -> Dict:
    print("🔬 Running Monte Carlo Simulations...")
    
    proposal = state["market_analyst_proposal"]
    allocation = proposal["allocation"]
    tickers = list(allocation.keys())
    weights = list(allocation.values())
    
    sim = simulator.run_simulation(
        initial_investment=state["user_profile"].get("initial_investment", 50000),
        tickers=tickers,
        weights=weights,
        time_horizon_years=state["user_profile"].get("time_horizon", 5)
    )
    
    state["simulation_results"] = [sim]
    report = f"**Simulation Results**\n{sim['summary']}\nMean Final Value: ${sim['mean_final_value']:,.0f}"
    print(f"simulatoin node report:\t{report}")

    state["blackboard"].append(report)
    
    return state


def risk_manager_node(state: AgentState, llm) -> Dict:
    print("🛡️ Risk Manager is evaluating...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a cautious Risk Manager.
        Review the analyst proposal and simulation results. Provide honest risk assessment.

        Analyst Proposal: {proposal}
        Simulation Results: {simulation}
        User Profile: {user_profile}
        """
    )
    
    chain = prompt | llm.with_structured_output(RiskEvaluation)
    evaluation = chain.invoke({
        "proposal": json.dumps(state["market_analyst_proposal"]),
        "simulation": json.dumps(state["simulation_results"]),
        "user_profile": json.dumps(state["user_profile"])
    })
    
    report = f"**Risk Manager Evaluation**\nAssessment: {evaluation.assessment}\nSuggested Adjustments: {evaluation.suggested_adjustments}"
    print(f"risk manager report:\t{report}")

    state["blackboard"].append(report)
    state["risk_evaluation"] = evaluation.dict()
    
    return state


def final_advisor_node(state: AgentState, llm) -> Dict:
    print("✅ Final Advisor is making recommendation...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are the Final Personal Finance Advisor.
        Synthesize all previous steps and give a clear, actionable recommendation to the user.

        Analyst Proposal: {proposal}
        Simulation: {simulation}
        Risk Evaluation: {risk}
        """
    )
    
    chain = prompt | llm.with_structured_output(FinalRecommendation)
    final_rec = chain.invoke({
        "proposal": json.dumps(state["market_analyst_proposal"]),
        "simulation": json.dumps(state["simulation_results"]),
        "risk": json.dumps(state["risk_evaluation"])
    })
    
    report = f"""**Final Recommendation**
Strategy: {final_rec.final_strategy}
Allocation: {final_rec.allocation}
Reasoning: {final_rec.reasoning}
Expected Outcome: {final_rec.expected_outcome}"""
    
    state["blackboard"].append(report)
    state["final_recommendation"] = final_rec.dict()
    
    return state


# ====================== CONDITIONAL ROUTING ======================
def should_revise_strategy(state: AgentState) -> str:
    """تصمیم‌گیری برای بازگشت یا ادامه"""
    risk = state.get("risk_evaluation", {})
    if risk.get("should_revise", False) and state.get("loop_count", 1) < 4:
        return "revise"          # برگشت به Analyst
    return "finalize"            # برو به Final Advisor



# ====================== BUILD GRAPH ======================
def build_finance_advisor(llm):
    simulator = MonteCarloSimulator()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("market_analyst", lambda s: market_analyst_node(s, llm))
    workflow.add_node("simulator", lambda s: simulation_node(s, simulator))
    workflow.add_node("risk_manager", lambda s: risk_manager_node(s, llm))
    workflow.add_node("final_advisor", lambda s: final_advisor_node(s, llm))
    
    workflow.set_entry_point("market_analyst")
    workflow.add_edge("market_analyst", "simulator")
    workflow.add_edge("simulator", "risk_manager")
    # workflow.add_edge("risk_manager", "final_advisor")

    workflow.add_conditional_edges(
        "risk_manager",
        should_revise_strategy,
        {
            "revise": "market_analyst",   # بازگشت برای اصلاح استراتژی
            "finalize": "final_advisor"
        }
    )
    
    workflow.add_edge("final_advisor", END)
    
    return workflow.compile(), simulator


# ====================== TEST FUNCTION ======================
def test_advisor():
    from langchain_google_genai import ChatGoogleGenerativeAI
    import os
    from dotenv import load_dotenv, dotenv_values 
    load_dotenv() 

    llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.5, google_api_key=os.getenv("GOOGLE_API_KEY"))
    advisor, simulator = build_finance_advisor(llm)
    
    user_profile = {
        "initial_investment": 75000,
        "time_horizon": 5,
        "risk_tolerance": "Moderate",
        "goals": "Save for house down payment and long-term retirement",
        "monthly_contribution": 1500
    }
    
    initial_state = {
        "user_profile": user_profile,
        "blackboard": [],
        "market_analyst_proposal": None,
        "simulation_results": None,
        "risk_evaluation": None,
        "final_recommendation": None
    }
    
    result = advisor.invoke(initial_state)
    
    print("\n" + "="*60)
    print("FINAL RECOMMENDATION")
    print("="*60)
    print("final strategy")
    print(result["final_recommendation"]["final_strategy"])
    print("reasoning")
    print(result["final_recommendation"]["reasoning"])
    print(result["final_recommendation"]["expected_outcome"])

if __name__ == "__main__":
    test_advisor()