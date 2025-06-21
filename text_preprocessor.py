"""
Text preprocessing functions for AI Text Detector
"""

import re
import string

class TextPreprocessor:
    def __init__(self):
        self.unnecessary_symbols = ['@', '#', '$', '^', '&', '*', '(', ')', 
                                '[', ']', '{', '}', '|', '\\', ':', ';', 
                                '<', '>', '?', '/', '~', '`']
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning function
        """
        if not text or not isinstance(text, str):
            return ""
        
        # Convert to string and strip
        text = str(text).strip()
        
        # 1. Ubah % menjadi " persen"
        text = re.sub(r'(\d+)%', r'\1 persen', text)
        text = re.sub(r'%', ' persen', text)
        
        # 2. Hapus spasi berlebihan
        text = re.sub(r'\s+', ' ', text)
        
        # 3. Hapus simbol yang tidak penting
        for symbol in self.unnecessary_symbols:
            text = text.replace(symbol, '')
        
        # 4. Escape quotes (opsional - untuk menghindari error)
        text = text.replace('"', '\\"')
        text = text.replace("'", "\\'")
        
        # 5. Ubah \n menjadi spasi
        text = text.replace('\n', ' ')
        text = text.replace('\r', ' ')
        text = text.replace('\t', ' ')
        
        # 6. Hapus spasi berlebihan lagi setelah cleaning
        text = re.sub(r'\s+', ' ', text)
        
        # 7. Hapus spasi di awal dan akhir
        text = text.strip()
        
        return text
    
    def split_into_sentences(self, text):
        """
        Split text into sentences for sentence-level analysis
        """
        # Simple sentence splitting - bisa diperbaiki dengan library NLP
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences
    
    def split_into_chunks(self, text, max_length=512):
        """
        Split text into chunks that fit model's max_length
        """
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            # Estimasi token length (rough approximation)
            word_tokens = len(word.split()) + 1
            
            if current_length + word_tokens > max_length - 2:  # -2 for [CLS] and [SEP]
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = [word]
                    current_length = word_tokens
                else:
                    # Single word too long, truncate
                    chunks.append(word[:max_length-2])
                    current_chunk = []
                    current_length = 0
            else:
                current_chunk.append(word)
                current_length += word_tokens
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def preprocess_for_model(self, text):
        """
        Full preprocessing pipeline for model input
        """
        # Clean text
        cleaned_text = self.clean_text(text)
        
        # Split into chunks if too long
        chunks = self.split_into_chunks(cleaned_text)
        
        return chunks, cleaned_text