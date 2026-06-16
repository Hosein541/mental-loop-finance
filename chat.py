import json
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from mental_loop_finance import fundamental_news_analyst_node

# class ChatIntent(BaseModel):
#     intent: str = Field(description="update_profile, ask_explanation, or general_question")
#     key: Optional[str] = Field(description="it can be : initial investment, time horizon, risk tolerance, monthly contribution, goals ")
#     value: Optional[Any] = None

class ChatIntent(BaseModel):
    intent: str = Field(
        description="""Choose one of the following intents:
        - update_profile: When user wants to change their personal information
        - ask_explanation: When user asks for explanation about current recommendation or reports
        - general_question: For casual conversation or other questions"""
    )
    key: Optional[str] = Field(
        description="Only used when intent is 'update_profile'. Valid keys: 'initial investment', 'time horizon', 'risk tolerance', 'monthly contribution', 'goals'",
        examples=["initial investment", "time horizon"]
    )
    value: Optional[Any] = Field(
        description="Only used when intent is 'update_profile'. The new value for the specified key. for keys that are numeric, value must be integer and dont mention year, $ and etc."
    )

def handle_chat_message(
    user_message: str,
    current_state: Dict,
    finance_advisor,      # compiled LangGraph
    llm,
    tavily_tool
) -> tuple:
    
    # # Intent Detection
    # prompt = ChatPromptTemplate.from_template(
    #     """You are a smart router. Detect user intent.

    #     User Message: {message}
    #     """
    # )

    prompt = ChatPromptTemplate.from_template(
    """You are an intelligent intent classifier for a Personal Finance AI Advisor.

    Your job is to analyze the user's message and return the correct intent with high accuracy.

    ### Available Intents and When to Use Them:

    1. **update_profile** → Use when user wants to change their personal/financial information.
       - Examples: "increase investment to 100000", "set time horizon to 7 years", "change risk to High", "monthly contribution 2000"

    2. **ask_explanation** → Use when user asks why, how, or explanation about current recommendation.
       - Examples: "why did you choose this allocation?", "explain the risk", "what does this mean?"

    3. **general_question** → Use for everything else (casual talk, greetings, general finance questions).


    User Message: {message}

    Respond with the correct structured intent.
    """
    )
    chain = prompt | llm.with_structured_output(ChatIntent)
    intent = chain.invoke({"message": user_message})

    response = ""
    output_type = False
    intent.key = intent.key.replace(" ", "_")
    print(f"intent:\t\t\t{intent.intent}")
    print(f"intent key:\t\t\t{intent.key}")
    print(f"intent value:\t\t\t{intent.value}\t\t{type(intent.value)}")

    if intent.intent == "update_profile" and intent.key:
        if intent.key in current_state["user_profile"]:
            old = current_state["user_profile"][intent.key]
            if type(old) == int:
                current_state["user_profile"][intent.key] = int(intent.value)
            else :
                current_state["user_profile"][intent.key] = intent.value
            response = f"✅ Updated **{intent.key}** from {old} → {intent.value}\n"
            
            # Auto Re-run
            response += "🔄 Re-running full analysis with updated profile..."
            response += "\n📍 **Updated results are displayed above** in this conversation."
            response += "\nYou can continue chatting or request further changes."
            
            current_state = rerun_full_analysis(current_state, finance_advisor)
            output_type = True
        else:
            response = f"❌ Unknown profile field: {intent.key}"

    # elif intent.intent == "refresh_news":
    #     response = "📰 Fetching latest market news...\n"
    #     # Call news analyst
    #     fundamental_news_analyst_node(current_state, llm, tavily_tool)  # assume this function exists
    #     response += "🔄 Re-running analysis with fresh news..."
    #     current_state = rerun_full_analysis(current_state, finance_advisor)

    # elif intent.intent == "rerun_analysis":
    #     response = "🔄 Re-running full financial analysis..."
    #     current_state = rerun_full_analysis(current_state, finance_advisor)

    elif intent.intent == "ask_explanation":
        response = explain_current_recommendation(current_state, llm, user_message)
        output_type = False

    else:
        # General chat
        general_prompt = ChatPromptTemplate.from_template(
            "You are a professional and friendly Personal Finance Advisor. Answer the user naturally.\n\nUser: {message}"
        )
        response = (general_prompt | llm.bind_tools([tavily_tool])).invoke({"message": user_message}).content[0]["text"]
        output_type = False
    
    print(f"output type:\t\t\t{output_type}")
    return response, current_state, output_type


def rerun_full_analysis(state: Dict, finance_advisor):
    """Re-run the entire advisor graph"""
    # Reset temporary fields
    state["simulation_results"] = None
    state["risk_evaluation"] = None
    state["final_recommendation"] = None
    
    result = finance_advisor.invoke(state)
    return result


def explain_current_recommendation(state: Dict, llm, question: str):
    prompt = ChatPromptTemplate.from_template(
        """Explain based on current analysis.

        User Question: {question}
        Current Recommendation: {rec}
        """
    )
    chain = prompt | llm
    return chain.invoke({
        "question": question,
        "rec": json.dumps(state.get("final_recommendation", {}), indent=2)
    }).content[0]["text"]