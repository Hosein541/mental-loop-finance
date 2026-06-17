import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

# Create images folder if not exists
os.makedirs("images", exist_ok=True)

def save_plot(fig, filename: str):
    """Save plot as image and return markdown image link"""
    filepath = f"images/{filename}"
    # Delete if already exists
    if os.path.exists(filepath):
        os.remove(filepath)
    
    fig.write_image(filepath, width=800, height=500, scale=2)
    return f"![{filename}]({filepath})"


def create_monte_carlo_distribution(sim_result: dict, initial_investment: float):
    """Monte Carlo Distribution Histogram"""
    mean_val = sim_result.get("mean_final_value", initial_investment * 1.3)
    std_val = (mean_val - sim_result.get("worst_5_percent", mean_val * 0.8)) / 1.65
    
    # Generate sample data for visualization
    np.random.seed(42)
    simulated_values = np.random.normal(mean_val, std_val, 10000)
    
    fig = px.histogram(
        x=simulated_values,
        nbins=80,
        title="Monte Carlo Simulation Distribution",
        labels={"x": "Final Portfolio Value ($)"},
        color_discrete_sequence=["#636EFA"]
    )
    
    # Add vertical lines
    fig.add_vline(x=mean_val, line_dash="dash", line_color="green", 
                  annotation_text="Expected Value", annotation_position="top right")
    fig.add_vline(x=sim_result.get("worst_5_percent", mean_val*0.85), line_dash="dash", line_color="red",
                  annotation_text="Worst 5%", annotation_position="top left")
    
    fig.update_layout(
        height=450,
        showlegend=False,
        xaxis_title="Final Portfolio Value after Time Horizon",
        yaxis_title="Frequency (out of simulations)"
    )
    
    img_md = save_plot(fig, "monte_carlo_distribution.png")
    return fig, img_md


def create_allocation_pie(allocation: dict):
    """Asset Allocation Pie Chart"""
    labels = list(allocation.keys())
    values = list(allocation.values())
    
    fig = px.pie(
        names=labels,
        values=values,
        title="Recommended Asset Allocation",
        color_discrete_sequence=px.colors.sequential.Blues_r
    )
    fig.update_traces(textinfo='percent+label')
    fig.update_layout(height=400)

    img_md = save_plot(fig, "asset_allocation.png")
    return fig, img_md


def create_growth_projection(sim_result: dict, initial: float, years: int):
    """Simple Growth Projection"""
    mean = sim_result.get("mean_final_value", initial * 1.3)
    worst = sim_result.get("worst_5_percent", initial * 0.9)
    best = sim_result.get("best_5_percent", initial * 1.8) if "best_5_percent" in sim_result else mean * 1.5
    
    years_list = list(range(years + 1))
    base = [initial * (1.07 ** y) for y in years_list]  # 7% average growth assumption
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=years_list, y=base, mode='lines+markers', name='Base Assumption', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=[years], y=[mean], mode='markers', name='Expected (Monte Carlo)', marker=dict(size=12, color='green')))
    fig.add_trace(go.Scatter(x=[years], y=[worst], mode='markers', name='Worst 5%', marker=dict(size=10, color='red')))
    
    fig.update_layout(
        title="Portfolio Growth Projection",
        xaxis_title="Years",
        yaxis_title="Portfolio Value ($)",
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )
    img_md = save_plot(fig, "growth_projection.png")
    return fig, img_md