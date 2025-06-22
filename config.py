"""
Configuration file for AI Text Detector
"""

import os
import streamlit as st

class Config:
    # Model settings - Updated for Hugging Face
    @staticmethod
    def get_model_name():
        """Get model name from secrets or environment"""
        try:
            # Try to get from Streamlit secrets first
            return st.secrets["huggingface"]["model_name"]
        except:
            # Fallback to environment variable
            return os.getenv("HF_MODEL_NAME", None)
    
    @staticmethod
    def get_hf_token():
        """Get Hugging Face token from secrets or environment"""
        try:
            # Try to get from Streamlit secrets first
            return st.secrets["huggingface"]["token"]
        except:
            # Fallback to environment variable
            return os.getenv("HF_TOKEN", None)
    
    # Model configuration
    MODEL_PATH = get_model_name()  # This will be the HF repo name
    BASE_MODEL_NAME = "indobenchmark/indobert-base-p2"  # Keep as fallback
    MAX_LENGTH = 512
    USE_HUGGINGFACE_HUB = True  # New flag to indicate we're using HF Hub
    
    # Thresholds
    AI_THRESHOLD = 0.7  # 70% confidence untuk menentukan teks AI
    HIGH_CONFIDENCE_THRESHOLD = 0.85  # 85% untuk confidence tinggi
    
    # Database
    DATABASE_PATH = "database/users.db"
    
    # UI Settings
    APP_TITLE = "🤖 Detector Teks AI Indonesia"
    APP_DESCRIPTION = "Sistem deteksi teks yang dibuat oleh AI menggunakan IndoBERT + LoRA"
    
    # Styling
    DARK_THEME = {
        "primary_color": "#FF6B6B",
        "background_color": "#0E1117",
        "secondary_background_color": "#262730",
        "text_color": "#FAFAFA"
    }
    
    # Logging
    LOG_LEVEL = "INFO"
    LOG_FILE = "logs/app.log"
    
    # Session settings
    SESSION_TIMEOUT = 3600  # 1 hour in seconds