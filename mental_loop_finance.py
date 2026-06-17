import json
import numpy as np
import pandas as pd
import yfinance as yf
from pydantic import BaseModel, Field
from typing import TypedDict, List, Dict, Any, Optional
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults
import warnings
warnings.filterwarnings('ignore')

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
        Based on the user's profile and risk manger agent suggestion if it is provided, propose a clear investment strategy and asset allocation.

        User Profile:
        {user_profile}

        Risk  Manager Suggestion:
        {risk_manager_suggestion}
        """
    )
    
    chain = prompt | llm.with_structured_output(MarketAnalystProposal)
    risk_evaluation = ""
    try:
        risk_evaluation = state["risk_evaluation"].get("suggested_adjustments", "")
    except :
        risk_evaluation = ""

    proposal = chain.invoke({"user_profile": json.dumps(state["user_profile"], indent=2), 
                             "risk_manager_suggestion": risk_evaluation})
    
    report = f"**Market Analyst Proposal**\nStrategy: {proposal.strategy}\nAllocation: {proposal.allocation}"
    print(f"market analysis report:\t{report}")
    state["blackboard"].append(report)
    state["market_analyst_proposal"] = proposal.model_dump()
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



# ====================== NEWS / FUNDAMENTAL ANALYST NODE ======================
def fundamental_news_analyst_node(state: AgentState, llm, tavily_tool: TavilySearchResults):
    print("📰 Fundamental News Analyst is researching...")
    
    user = state["user_profile"]
    tickers = list(state["market_analyst_proposal"]["allocation"].keys()) if state.get("market_analyst_proposal") else ["SPY", "QQQ"]
    
    # Search for recent news
    query = f"Recent news and fundamental analysis for {', '.join(tickers)} stocks and global market outlook"
    
    search_results = tavily_tool.invoke({"query": query})
    
    # Summarize with LLM
    prompt = ChatPromptTemplate.from_template(
        """You are a Fundamental News Analyst.
        Summarize the most important recent news and fundamental developments for the mentioned assets.
        Focus on factors that can significantly impact the investment decision in the next 6-12 months.

        Search Results:
        {search_results}

        User Profile Context:
        {user_profile}
        """
    )
    
    chain = prompt | llm
    summary = chain.invoke({
        "search_results": search_results,
        "user_profile": json.dumps(user, indent=2)
    })
    
    report = f"""**📰 Fundamental News Analyst Report**

**Key Insights:**
{summary.content[0]["text"] if hasattr(summary, 'content') else summary}
"""
    print(f"fundamental analysis report:\t{report}")
    
    state["blackboard"].append(report)
    state["news_summary"] = summary.content[0]["text"] if hasattr(summary, 'content') else str(summary)
    
    return state



def risk_manager_node(state: AgentState, llm) -> Dict:
    print("🛡️ Risk Manager is evaluating...")
    
    prompt = ChatPromptTemplate.from_template(
        """You are a cautious Risk Manager.
        Review the analyst proposal and simulation results. Provide honest risk assessment.

        Analyst Proposal: {proposal}
        Simulation Results: {simulation}
        User Profile: {user_profile}
        Fundamental News Summary: {news_summary}
        """
    )
    
    chain = prompt | llm.with_structured_output(RiskEvaluation)
    evaluation = chain.invoke({
        "proposal": json.dumps(state["market_analyst_proposal"]),
        "simulation": json.dumps(state["simulation_results"]),
        "user_profile": json.dumps(state["user_profile"]),
        "news_summary": state.get("news_summary", "No news summary available")
    })
    
    report = f"**Risk Manager Evaluation**\nAssessment: {evaluation.assessment}\nSuggested Adjustments: {evaluation.suggested_adjustments}"
    print(f"risk manager report:\t{report}")

    state["blackboard"].append(report)
    state["risk_evaluation"] = evaluation.model_dump()
    
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
    
    print(f"final advisor report:\t\t\t{report}")
    
    state["blackboard"].append(report)
    state["final_recommendation"] = final_rec.model_dump()
    
    return state


# ====================== CONDITIONAL ROUTING ======================
def should_revise_strategy(state: AgentState) -> str:
    """تصمیم‌گیری برای بازگشت یا ادامه"""
    risk = state.get("risk_evaluation", {})
    if risk.get("should_revise", False) and state.get("loop_count", 1) < 4:
        return "revise"          
    return "finalize"            



# ====================== BUILD GRAPH ======================
def build_finance_advisor(llm, tavily_tool):
    simulator = MonteCarloSimulator()
    
    workflow = StateGraph(AgentState)
    
    workflow.add_node("market_analyst", lambda s: market_analyst_node(s, llm))
    workflow.add_node("simulator", lambda s: simulation_node(s, simulator))
    workflow.add_node("fundamental_news", lambda s: fundamental_news_analyst_node(s, llm, tavily_tool))
    workflow.add_node("risk_manager", lambda s: risk_manager_node(s, llm))
    workflow.add_node("final_advisor", lambda s: final_advisor_node(s, llm))

    workflow.set_entry_point("market_analyst")
    workflow.add_edge("market_analyst", "simulator")
    workflow.add_edge("simulator", "fundamental_news")
    workflow.add_edge("fundamental_news", "risk_manager")

    workflow.add_conditional_edges(
        "risk_manager",
        should_revise_strategy,
        {
            "revise": "market_analyst",  
            "finalize": "final_advisor"
        }
    )

    workflow.add_edge("final_advisor", END)
    
    return workflow.compile(), simulator
