"""
Utility functions for AI Text Detector
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from io import StringIO

class Utils:
    @staticmethod
    def set_page_config():
        """Set Streamlit page configuration"""
        st.set_page_config(
            page_title="AI Text Detector",
            page_icon="🤖",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def apply_custom_css():
        """Apply custom CSS for dark theme"""
        st.markdown("""
        <style>
        .main {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        
        .stApp {
            background-color: #0E1117;
        }
        
        .ai-highlight {
            background-color: #FF6B6B;
            color: white;
            padding: 2px 4px;
            border-radius: 3px;
            font-weight: bold;
        }
        
        .confidence-high {
            color: #FF4444;
            font-weight: bold;
        }
        
        .confidence-medium {
            color: #FFA500;
            font-weight: bold;
        }
        
        .confidence-low {
            color: #4CAF50;
            font-weight: bold;
        }
        
        .metric-card {
            background-color: #262730;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #FF6B6B;
        }
        
        .prediction-card {
            background-color: #262730;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 0.5rem 0;
            border: 1px solid #444;
        }
        
        /* Sidebar styling */
        .css-1d391kg {
            background-color: #262730;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #FF6B6B;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.5rem 1rem;
            font-weight: bold;
            transition: all 0.3s;
        }
        
        .stButton > button:hover {
            background-color: #FF5252;
            transform: translateY(-2px);
        }
        
        /* Text area styling */
        .stTextArea > div > div > textarea {
            background-color: #262730;
            color: #FAFAFA;
            border: 1px solid #444;
            border-radius: 0.5rem;
        }
        
        /* Progress bar styling */
        .stProgress > div > div > div > div {
            background-color: #FF6B6B;
        }
        
        /* Alert styling */
        .stAlert {
            background-color: #262730;
            border: 1px solid #444;
        }
        </style>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_confidence_gauge(ai_probability):
        """Create a gauge chart for AI confidence"""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = ai_probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Persentase Teks AI (%)"},
            delta = {'reference': 50},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgray"},
                    {'range': [30, 70], 'color': "yellow"},
                    {'range': [70, 100], 'color': "red"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 70
                }
            }
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_prediction_history_chart(predictions):
        """Create a chart showing prediction history"""
        if not predictions:
            return None
        
        df = pd.DataFrame(predictions)
        df['created_at'] = pd.to_datetime(df['created_at'])
        df['ai_percentage'] = df['ai_probability'] * 100
        
        fig = px.scatter(
            df, 
            x='created_at', 
            y='ai_percentage',
            color='is_ai_generated',
            title='Riwayat Prediksi AI',
            labels={
                'created_at': 'Waktu',
                'ai_percentage': 'Persentase AI (%)',
                'is_ai_generated': 'Teks AI'
            },
            color_discrete_map={True: '#FF6B6B', False: '#4CAF50'}
        )
        
        fig.add_hline(y=60, line_dash="dash", line_color="orange", 
                     annotation_text="Threshold AI (60%)")
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"},
            xaxis={'gridcolor': '#444'},
            yaxis={'gridcolor': '#444'}
        )
        
        return fig
    
    @staticmethod  
    def highlight_ai_text(text, highlighted_parts):
        """Highlight AI-generated parts in text"""
        if not highlighted_parts:
            return text
        
        highlighted_text = text
        # Sort by position to avoid overlapping highlights
        for part in sorted(highlighted_parts, key=lambda x: len(x['text']), reverse=True):
            part_text = part['text']
            probability = part['probability']
            
            highlight_html = f'<span class="ai-highlight" title="AI Confidence: {probability:.1%}">{part_text}</span>'
            highlighted_text = highlighted_text.replace(part_text, highlight_html)
        
        return highlighted_text
    
    @staticmethod
    def format_confidence_level(confidence_level, ai_probability):
        """Format confidence level with appropriate styling"""
        percentage = f"{ai_probability:.1%}"
        
        if confidence_level == 'high':
            return f'<span class="confidence-high">TINGGI ({percentage})</span>'
        elif confidence_level == 'medium':
            return f'<span class="confidence-medium">SEDANG ({percentage})</span>'
        else:
            return f'<span class="confidence-low">RENDAH ({percentage})</span>'
    
    @staticmethod
    def create_download_data(prediction_result, input_text):
        """Create downloadable data from prediction result"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'input_text': input_text,
            'ai_probability': prediction_result['ai_probability'],
            'is_ai_generated': prediction_result['is_ai_generated'],
            'confidence_level': prediction_result['confidence_level'],
            'highlighted_parts': prediction_result['highlighted_parts'],
            'total_chunks': prediction_result.get('total_chunks', 1),
            'chunk_predictions': prediction_result.get('chunk_predictions', [])
        }
        
        return json.dumps(data, indent=2, ensure_ascii=False)
    
    @staticmethod
    def create_stats_visualization(user_stats):
        """Create visualization for user statistics"""
        if user_stats['total_predictions'] == 0:
            return None
        
        # Pie chart for AI vs Human predictions
        labels = ['Teks AI', 'Teks Manusia']
        values = [user_stats['ai_predictions'], user_stats['human_predictions']]
        colors = ['#FF6B6B', '#4CAF50']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values,
            marker_colors=colors,
            hole=.3
        )])
        
        fig.update_layout(
            title="Distribusi Prediksi",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={'color': "white"}
        )
        
        return fig
    
    @staticmethod
    def display_prediction_card(prediction):
        """Display a prediction in card format"""
        with st.container():
            st.markdown('<div class="prediction-card">', unsafe_allow_html=True)
            
            col1, col2 = st.columns([3, 1])
            
            with col1:
                # Truncate text for display
                display_text = prediction['input_text'][:200] + "..." if len(prediction['input_text']) > 200 else prediction['input_text']
                st.write(f"**Teks:** {display_text}")
                
                # Confidence level
                confidence_html = Utils.format_confidence_level(
                    'high' if prediction['ai_probability'] > 0.85 else 'medium' if prediction['ai_probability'] > 0.7 else 'low',
                    prediction['ai_probability']
                )
                st.markdown(f"**Tingkat Kepercayaan:** {confidence_html}", unsafe_allow_html=True)
            
            with col2:
                st.metric(
                    "AI Probability",
                    f"{prediction['ai_probability']:.1%}",
                    delta=f"{'AI' if prediction['is_ai_generated'] else 'Human'}"
                )
                st.caption(f"📅 {prediction['created_at']}")
            
            st.markdown('</div>', unsafe_allow_html=True)
    
    @staticmethod
    def display_chunk_analysis(chunk_predictions):
        """Display detailed chunk analysis"""
        if not chunk_predictions:
            return
        
        st.subheader("📊 Analisis Per Bagian Teks")
        
        for i, chunk in enumerate(chunk_predictions):
            with st.expander(f"Bagian {i+1} - AI: {chunk['ai_probability']:.1%}"):
                st.write(f"**Teks:** {chunk['text']}")
                st.progress(chunk['ai_probability'])
                
                if chunk['is_ai']:
                    st.error(f"🚨 Bagian ini kemungkinan dibuat oleh AI ({chunk['ai_probability']:.1%})")
                else:
                    st.success(f"✅ Bagian ini kemungkinan dibuat oleh manusia ({chunk['ai_probability']:.1%})")
    
    @staticmethod
    def export_predictions_to_csv(predictions):
        """Export predictions to CSV format"""
        if not predictions:
            return None
        
        # Flatten predictions for CSV
        csv_data = []
        for pred in predictions:
            csv_data.append({
                'Timestamp': pred['created_at'],
                'Input_Text': pred['input_text'][:100] + "..." if len(pred['input_text']) > 100 else pred['input_text'],
                'AI_Probability': pred['ai_probability'],
                'Is_AI_Generated': pred['is_ai_generated'],
                'Highlighted_Parts_Count': len(pred['highlighted_parts'])
            })
        
        df = pd.DataFrame(csv_data)
        return df.to_csv(index=False)
