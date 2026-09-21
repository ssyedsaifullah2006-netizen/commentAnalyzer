"""
Topic visualization functions.
"""
import pandas as pd
import plotly.graph_objects as go
from typing import Any

LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    margin=dict(l=40, r=40, t=40, b=40),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

def topic_scatter(topic_info: pd.DataFrame, topic_model: Any = None) -> go.Figure:
    """2D scatter plot of topics using bubble size = topic size, color = topic ID."""
    if topic_info.empty:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    if topic_model and hasattr(topic_model, 'visualize_topics'):
        try:
            return topic_model.visualize_topics()
        except Exception:
            pass
            
    fig = go.Figure(layout=LAYOUT_DEFAULTS)
    
    if 'Topic' in topic_info.columns and 'Count' in topic_info.columns:
        valid_topics = topic_info[topic_info['Topic'] != -1]
        
        fig.add_trace(go.Scatter(
            x=valid_topics['Topic'],
            y=[1] * len(valid_topics), # Dummy y-axis if no reduction is given
            mode='markers+text',
            marker=dict(
                size=valid_topics['Count'],
                sizemode='area',
                sizeref=2.*max(valid_topics['Count'])/(40.**2),
                sizemin=4,
                color=valid_topics['Topic'],
                colorscale='Viridis',
                showscale=True
            ),
            text=valid_topics.get('Name', valid_topics['Topic']),
            textposition="top center"
        ))
        
        fig.update_layout(title="Topic Scatter (Topic ID vs Dummy Axis)")
        
    return fig

def topic_bar_chart(topic_info: pd.DataFrame) -> go.Figure:
    """Bar chart showing top topics by size."""
    if topic_info.empty or 'Topic' not in topic_info.columns or 'Count' not in topic_info.columns:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    # Filter out outlier topic -1
    valid_topics = topic_info[topic_info['Topic'] != -1].copy()
    if valid_topics.empty:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    top_topics = valid_topics.sort_values(by='Count', ascending=False).head(10)
    top_topics = top_topics.sort_values(by='Count', ascending=True) # For horizontal chart
    
    names = top_topics.get('Name', top_topics['Topic'].astype(str))
    
    fig = go.Figure(go.Bar(
        x=top_topics['Count'],
        y=names,
        orientation='h'
    ))
    fig.update_layout(
        title="Top Topics by Size", 
        xaxis_title="Number of Documents", 
        yaxis_title="Topic",
        **LAYOUT_DEFAULTS
    )
    return fig

def topic_hierarchy(topic_model: Any) -> go.Figure:
    """Hierarchical topic visualization if BERTopic supports it."""
    try:
        if topic_model and hasattr(topic_model, 'visualize_hierarchy'):
            return topic_model.visualize_hierarchy()
    except Exception:
        pass
        
    fig = go.Figure(layout=LAYOUT_DEFAULTS)
    fig.update_layout(title="Topic Hierarchy Unavailable")
    return fig
