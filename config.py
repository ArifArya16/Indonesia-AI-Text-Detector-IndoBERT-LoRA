"""
Configuration file for AI Text Detector - FIXED VERSION (No Debug Output)
"""

import os
import streamlit as st
import logging

class Config:
    # Model configuration constants
    BASE_MODEL_NAME = "indobenchmark/indobert-base-p2"
    MAX_LENGTH = 512
    USE_HUGGINGFACE_HUB = True
    
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
    LOG_LEVEL = "ERROR"  # Changed from INFO to ERROR to reduce output
    LOG_FILE = "logs/app.log"
    
    # Session settings
    SESSION_TIMEOUT = 3600  # 1 hour in seconds
    
    # Model settings - Updated for silent operation
    @staticmethod
    def get_model_name():
        """Get model name from secrets or environment (silent operation)"""
        try:
            # Try to get from Streamlit secrets first
            if hasattr(st, 'secrets') and 'huggingface' in st.secrets:
                return st.secrets["huggingface"]["model_name"]
        except Exception:
            pass  # Silent fail
        
        # Fallback to environment variable
        return os.getenv("HF_MODEL_NAME", "Aureonn/indobert-ai-detector-private")
    
    @staticmethod
    def get_hf_token():
        """Get Hugging Face token from secrets or environment (silent operation)"""
        try:
            # Try to get from Streamlit secrets first
            if hasattr(st, 'secrets') and 'huggingface' in st.secrets:
                return st.secrets["huggingface"]["token"]
        except Exception:
            pass  # Silent fail
        
        # Fallback to environment variable
        return os.getenv("HF_TOKEN", None)
    
    @staticmethod
    def validate_config():
        """Validate that all required configuration is available (silent)"""
        model_name = Config.get_model_name()
        hf_token = Config.get_hf_token()
        
        if not model_name:
            return False, "Model name tidak ditemukan"
        
        if not hf_token:
            return False, "Hugging Face token tidak ditemukan"
        
        # Validate token format
        if not hf_token.startswith('hf_'):
            return False, "Format Hugging Face token tidak valid"
        
        return True, "Konfigurasi valid"
    
    @staticmethod
    def get_model_config():
        """Get complete model configuration"""
        return {
            'model_name': Config.get_model_name(),
            'token': Config.get_hf_token(),
            'max_length': Config.MAX_LENGTH,
            'use_auth_token': True,
            'trust_remote_code': True
        }
