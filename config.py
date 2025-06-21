"""
Configuration file for AI Text Detector
"""

import os

class Config:
    # Model settings
    MODEL_PATH = "indobert_ai_detector"
    BASE_MODEL_NAME = "indobenchmark/indobert-base-p1"
    MAX_LENGTH = 512
    
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