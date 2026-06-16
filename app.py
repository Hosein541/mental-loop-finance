import streamlit as st
import json
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from mental_loop_finance import build_finance_advisor   # your main graph file
from chat import handle_chat_message

st.set_page_config(page_title="MentalLoop Finance Advisor", layout="wide")
st.title("💰 MentalLoop Personal Finance Advisor")
st.markdown("**Mental Loop Architecture + Real Market Data + News**")

# ====================== SIDEBAR ======================
with st.sidebar:
    st.header("API Keys")
    gemini_key = st.text_input("Gemini API Key", type="password")
    tavily_key = st.text_input("Tavily API Key", type="password")

    st.header("Your Profile")
    initial_investment = st.number_input("Initial Investment ($)", value=75000, step=500, min_value=1000, max_value=500000)
    time_horizon = st.number_input("Time Horizon (years)", value=5, min_value=1, max_value=10)
    risk_tolerance = st.selectbox("Risk Tolerance", ["Low", "Moderate", "High"])
    monthly_contribution = st.number_input("Monthly Contribution ($)", value=1500, step=100, min_value=100, max_value=20000)
    goals = st.text_area("Explain about your golas", placeholder="Build wealth and prepare for major life goals", height=90)

    start_button = st.button("🚀 Generate Initial Analysis", type="primary")

# ====================== MAIN AREA ======================
if "advisor" not in st.session_state and gemini_key and tavily_key and start_button:
    st.session_state.llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.6, google_api_key=gemini_key)
    st.session_state.tavily_tool = TavilySearchResults(max_results=5, tavily_api_key=tavily_key)
    
    st.session_state.advisor, st.session_state.simulator = build_finance_advisor(st.session_state.llm, st.session_state.tavily_tool)
    
    st.session_state.current_state = {
        "user_profile": {
            "initial_investment": initial_investment,
            "time_horizon": time_horizon,
            "risk_tolerance": risk_tolerance,
            "monthly_contribution": monthly_contribution,
            "goals": [goals if len(goals) > 0 else "Build wealth and prepare for major life goals"]
        },
        "blackboard": [],
        "market_analyst_proposal": None,
        "simulation_results": None,
        "risk_evaluation": None,
        "final_recommendation": None,
        "news_summary": None
    }

if "current_state" in st.session_state:
    state = st.session_state.current_state
    
    # Initial Analysis
    if start_button and not state.get("final_recommendation"):
        with st.spinner("Running full Mental Loop Analysis..."):
            user_input = st.session_state.current_state["user_profile"]
            st.info(f"user profile :\t\t\t{user_input}")

            result = st.session_state.advisor.invoke(state)
            st.session_state.current_state = result
    dispaly_state = st.session_state.current_state

    # Display Analysis
    if dispaly_state.get("final_recommendation"):
        st.subheader("📌 Final Recommendation")
        st.markdown(dispaly_state["final_recommendation"].get("final_strategy", ""))
        
        # st.subheader("📋 Blackboard Progress")
        # for entry in dispaly_state["blackboard"]:
        #     st.markdown(entry)
        #     st.divider()

        final_text = "\n\n".join(dispaly_state.get("blackboard", [])[-5:])  # last few reports
        st.markdown(final_text)

        # Download Button
        st.download_button(
            label="📥 Download Full Progress",
            key=1,
            data=final_text,
            file_name="Mental_Loop_Full_Progress.md",
            mime="text/markdown"
        )

    # ====================== CHAT SECTION ======================
    st.subheader("💬 Chat with Your Finance Advisor")
    
    # Chat History
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_input := st.chat_input("Ask anything (e.g., change investment to 100k, get latest news, explain why...)"):
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, state, output_type = handle_chat_message(
                    user_message=user_input,
                    current_state=st.session_state.current_state,
                    finance_advisor=st.session_state.advisor,
                    llm=st.session_state.llm,
                    tavily_tool=st.session_state.tavily_tool
                )
                st.session_state.current_state = state
                st.markdown(response)
                # st.markdown(output_type)
                
                if output_type :

                    st.subheader("📌 Final Recommendation")
                    st.markdown(state["final_recommendation"].get("final_strategy", ""))
                    final_text = "\n\n".join(state.get("blackboard", [])[-5:])  # last few reports
                    # st.markdown(final_text)

                    # Download Button
                    st.download_button(
                        label="📥 Download Full Progress",
                        key=2,
                        data=final_text,
                        file_name="Mental_Loop_Full_Progress.md",
                        mime="text/markdown"
                    )
        
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.rerun()