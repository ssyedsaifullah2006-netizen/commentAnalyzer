"""
Plotly chart functions for comment analysis.
Professional, modern color palette and clean typography.
"""
import pandas as pd
import plotly.graph_objects as go
from typing import List, Dict, Any

# Modern dark theme layout defaults
LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=12, color="#94a3b8"),
    title_font=dict(family="Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, sans-serif", size=14, color="#f1f5f9"),
    margin=dict(l=30, r=30, t=40, b=30),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
)

def sentiment_pie_chart(distribution: Dict[str, int]) -> go.Figure:
    """Pie chart of positive/negative/neutral with modern palette."""
    if not distribution:
        return go.Figure(layout=LAYOUT_DEFAULTS)
    
    labels = [k.title() for k in distribution.keys()]
    values = list(distribution.values())
    
    color_map = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#64748b"}
    colors = [color_map.get(label, "#3b82f6") for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels, 
        values=values, 
        marker=dict(colors=colors, line=dict(color="#111827", width=2)),
        hole=0.45
    )])
    fig.update_layout(title="Sentiment Breakdown", **LAYOUT_DEFAULTS)
    return fig

def sentiment_bar_chart(distribution: Dict[str, int]) -> go.Figure:
    """Horizontal bar chart of sentiment counts."""
    if not distribution:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    labels = [k.title() for k in distribution.keys()]
    values = list(distribution.values())
    
    color_map = {"Positive": "#10b981", "Negative": "#ef4444", "Neutral": "#64748b"}
    colors = [color_map.get(label, "#3b82f6") for label in labels]
    
    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation='h',
        marker=dict(color=colors, line=dict(color="#1f2937", width=1))
    ))
    fig.update_layout(
        title="Sentiment Volume", 
        xaxis=dict(title="Volume", gridcolor="#1f2937"),
        yaxis=dict(title="Sentiment", gridcolor="#1f2937"),
        **LAYOUT_DEFAULTS
    )
    return fig

def emotion_radar_chart(emotion_distribution: Dict[str, float]) -> go.Figure:
    """Radar/spider chart of emotions."""
    if not emotion_distribution:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    categories = [k.title() for k in emotion_distribution.keys()]
    values = list(emotion_distribution.values())
    
    categories.append(categories[0])
    values.append(values[0])
    
    fig = go.Figure(data=go.Scatterpolar(
      r=values,
      theta=categories,
      fill='toself',
      fillcolor='rgba(59, 130, 246, 0.2)',
      line=dict(color='#3b82f6', width=2),
      name='Emotions'
    ))
    
    fig.update_layout(
      polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(visible=True, range=[0, max(values) if values else 1], gridcolor="#1f2937", linecolor="#1f2937"),
        angularaxis=dict(gridcolor="#1f2937", linecolor="#1f2937")
      ),
      title="Emotional Distribution Profile",
      **LAYOUT_DEFAULTS
    )
    return fig

def emotion_bar_chart(emotion_distribution: Dict[str, int]) -> go.Figure:
    """Bar chart of emotions."""
    if not emotion_distribution:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    categories = [k.title() for k in emotion_distribution.keys()]
    values = list(emotion_distribution.values())
    
    fig = go.Figure(data=[
        go.Bar(name='Emotions', x=categories, y=values, marker_color="#3b82f6")
    ])
    fig.update_layout(
        title="Emotion Frequency", 
        xaxis=dict(title="Emotion", gridcolor="#1f2937"),
        yaxis=dict(title="Count", gridcolor="#1f2937"),
        **LAYOUT_DEFAULTS
    )
    return fig

def toxicity_gauge(avg_score: float) -> go.Figure:
    """Gauge/indicator chart showing overall toxicity level."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = avg_score,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Average Toxicity Index", 'font': {'size': 13, 'color': '#94a3b8'}},
        gauge = {
            'axis': {'range': [0, 1], 'tickcolor': '#64748b'},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps': [
                {'range': [0, 0.3], 'color': "rgba(16, 185, 129, 0.6)"},
                {'range': [0.3, 0.7], 'color': "rgba(245, 158, 11, 0.6)"},
                {'range': [0.7, 1.0], 'color': "rgba(239, 68, 68, 0.6)"}
            ],
            'threshold': {
                'line': {'color': "#f87171", 'width': 3},
                'thickness': 0.75,
                'value': avg_score
            }
        }
    ))
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig

def comment_length_histogram(comments: List[str]) -> go.Figure:
    """Distribution of comment lengths."""
    if not comments:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    lengths = [len(c) for c in comments]
    fig = go.Figure(data=[go.Histogram(x=lengths, nbinsx=30, marker_color="#6366f1")])
    fig.update_layout(
        title="Comment Length Distribution", 
        xaxis=dict(title="Character Count", gridcolor="#1f2937"),
        yaxis=dict(title="Frequency", gridcolor="#1f2937"),
        **LAYOUT_DEFAULTS
    )
    return fig

def top_commenters_chart(comments_df: pd.DataFrame) -> go.Figure:
    """Bar chart of most active commenters."""
    if comments_df.empty or 'author' not in comments_df.columns:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    top_authors = comments_df['author'].value_counts().head(10)
    fig = go.Figure(go.Bar(
        x=top_authors.values,
        y=top_authors.index,
        orientation='h',
        marker_color="#3b82f6"
    ))
    fig.update_layout(
        title="Top Commenters", 
        xaxis=dict(title="Number of Comments", gridcolor="#1f2937"), 
        yaxis=dict(title="Author", categoryorder='total ascending', gridcolor="#1f2937"),
        **LAYOUT_DEFAULTS
    )
    return fig

def engagement_chart(comments_df: pd.DataFrame) -> go.Figure:
    """Scatter plot of likes vs comment length."""
    if comments_df.empty or 'text' not in comments_df.columns or 'likes' not in comments_df.columns:
        return go.Figure(layout=LAYOUT_DEFAULTS)
        
    lengths = comments_df['text'].str.len()
    likes = comments_df['likes']
    
    fig = go.Figure(data=go.Scatter(
        x=lengths,
        y=likes,
        mode='markers',
        marker=dict(size=7, color="#3b82f6", opacity=0.7)
    ))
    fig.update_layout(
        title="Engagement vs Length", 
        xaxis=dict(title="Character Count", gridcolor="#1f2937"), 
        yaxis=dict(title="Likes", gridcolor="#1f2937"), 
        **LAYOUT_DEFAULTS
    )
    return fig
