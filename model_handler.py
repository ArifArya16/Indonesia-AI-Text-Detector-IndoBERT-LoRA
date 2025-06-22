"""
Model handler for AI Text Detector - Hugging Face Only Version
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login
import numpy as np
from config import Config
from text_preprocessor import TextPreprocessor
import logging
import traceback

class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.loaded = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def authenticate_huggingface(self):
        """Authenticate with Hugging Face using token from secrets"""
        try:
            hf_token = Config.get_hf_token()
            if not hf_token:
                self.logger.error("No Hugging Face token found in configuration")
                return False
            
            # Validate token format
            if not hf_token.startswith('hf_'):
                self.logger.error("Invalid Hugging Face token format")
                return False
            
            login(token=hf_token, add_to_git_credential=False)
            self.logger.info("✅ Successfully authenticated with Hugging Face")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to authenticate with Hugging Face: {str(e)}")
            return False
    
    @st.cache_resource
    def load_model(_self):
        """Load the model from Hugging Face Hub"""
        try:
            # Validate configuration first
            model_name = Config.get_model_name()
            hf_token = Config.get_hf_token()
            
            if not model_name:
                raise Exception("Model name tidak ditemukan dalam konfigurasi")
            
            if not hf_token:
                raise Exception("Hugging Face token tidak ditemukan")
            
            _self.logger.info(f"🔄 Loading model from Hugging Face Hub: {model_name}")
            
            # Authenticate with Hugging Face
            if not _self.authenticate_huggingface():
                raise Exception("Failed to authenticate with Hugging Face")
            
            # Load tokenizer with progress indication
            with st.spinner("🔤 Loading tokenizer..."):
                _self.logger.info("Loading tokenizer...")
                _self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    use_auth_token=hf_token,
                    trust_remote_code=True
                )
                st.success("✅ Tokenizer loaded successfully")
            
            # Load model with progress indication
            with st.spinner("🤖 Loading model..."):
                _self.logger.info("Loading model...")
                _self.model = AutoModelForSequenceClassification.from_pretrained(
                    model_name,
                    num_labels=2,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    use_auth_token=hf_token,
                    trust_remote_code=True
                )
                
                # Move to device and set to eval mode
                _self.model = _self.model.to(_self.device)
                _self.model.eval()
                st.success("✅ Model loaded successfully")
            
            _self.loaded = True
            _self.logger.info(f"✅ Model loaded successfully on {_self.device}")
            
            # Display model info
            st.info(f"📋 Model: {model_name}")
            st.info(f"💻 Device: {_self.device}")
            
            return _self
            
        except Exception as e:
            error_msg = str(e)
            _self.logger.error(f"❌ Error loading model: {error_msg}")
            
            # Provide specific error guidance in Streamlit
            st.error(f"❌ Gagal memuat model: {error_msg}")
            
            if "401" in error_msg or "authentication" in error_msg.lower():
                st.error("🔐 **Error Autentikasi:**")
                st.write("- Pastikan Hugging Face token valid")
                st.write("- Periksa apakah token memiliki akses ke model private")
                st.write("- Coba regenerate token di Hugging Face")
                
            elif "404" in error_msg or "not found" in error_msg.lower():
                st.error("🔍 **Model Tidak Ditemukan:**")
                st.write("- Periksa nama model di konfigurasi")
                st.write("- Pastikan model tersedia di Hugging Face")
                st.write(f"- Model yang dicari: {Config.get_model_name()}")
                
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                st.error("🌐 **Masalah Koneksi:**")
                st.write("- Periksa koneksi internet")
                st.write("- Coba lagi dalam beberapa menit")
            
            # Show detailed error in expander
            with st.expander("🔧 Detail Error (untuk debugging)"):
                st.code(traceback.format_exc())
            
            # Suggest solutions
            st.info("💡 **Solusi yang bisa dicoba:**")
            st.write("1. Refresh halaman dan coba lagi")
            st.write("2. Periksa konfigurasi Hugging Face token")
            st.write("3. Pastikan model masih tersedia")
            
            raise Exception(f"Failed to load model: {error_msg}")
    
    def cleanup_model(self):
        """Clean up model to free memory"""
        try:
            if self.model:
                del self.model
                self.model = None
            if self.tokenizer:
                del self.tokenizer
                self.tokenizer = None
            
            # Clear CUDA cache if available
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            self.loaded = False
            self.logger.info("✅ Model cleaned up successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during model cleanup: {str(e)}")
    
    def is_model_loaded(self):
        """Check if model is properly loaded"""
        return self.loaded and self.model is not None and self.tokenizer is not None
    
    def predict_single_chunk(self, text_chunk):
        """Predict AI probability for a single text chunk"""
        if not self.is_model_loaded():
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if not text_chunk or not text_chunk.strip():
            return 0.0
        
        try:
            # Tokenize
            inputs = self.tokenizer(
                text_chunk,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=Config.MAX_LENGTH
            )
            
            # Move to device
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.nn.functional.softmax(logits, dim=-1)
                
                # Assuming label 1 is AI-generated
                ai_probability = probabilities[0][1].cpu().item()
            
            return ai_probability
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            return 0.0
    
    def predict_text(self, input_text):
        """
        Predict AI probability for input text
        Returns: dict with prediction results
        """
        if not input_text or not input_text.strip():
            return {
                'ai_probability': 0.0,
                'human_probability': 1.0,
                'is_ai_generated': False,
                'confidence_level': 'low',
                'prediction': 'Human',
                'highlighted_parts': [],
                'chunk_predictions': [],
                'total_chunks': 0,
                'cleaned_text': ''
            }
        
        if not self.is_model_loaded():
            raise ValueError("Model not loaded. Please reload the page or check configuration.")
        
        try:
            # Preprocess text
            chunks, cleaned_text = self.preprocessor.preprocess_for_model(input_text)
            
            chunk_predictions = []
            ai_probabilities = []
            
            # Predict each chunk
            for i, chunk in enumerate(chunks):
                try:
                    ai_prob = self.predict_single_chunk(chunk)
                    chunk_predictions.append({
                        'chunk_id': i,
                        'text': chunk,
                        'ai_probability': ai_prob,
                        'human_probability': 1.0 - ai_prob,
                        'is_ai': ai_prob > Config.AI_THRESHOLD
                    })
                    ai_probabilities.append(ai_prob)
                    
                except Exception as e:
                    self.logger.error(f"Error predicting chunk {i}: {str(e)}")
                    chunk_predictions.append({
                        'chunk_id': i,
                        'text': chunk,
                        'ai_probability': 0.0,
                        'human_probability': 1.0,
                        'is_ai': False,
                        'error': str(e)
                    })
                    ai_probabilities.append(0.0)
            
            # Calculate overall AI probability (weighted average by chunk length)
            if ai_probabilities:
                chunk_lengths = [len(chunk.split()) for chunk in chunks]
                total_length = sum(chunk_lengths)
                
                if total_length > 0:
                    weighted_ai_prob = sum(
                        prob * length for prob, length in zip(ai_probabilities, chunk_lengths)
                    ) / total_length
                else:
                    weighted_ai_prob = np.mean(ai_probabilities)
            else:
                weighted_ai_prob = 0.0
            
            # Calculate human probability
            human_probability = 1.0 - weighted_ai_prob
            
            # Determine if text is AI-generated
            is_ai_generated = weighted_ai_prob > Config.AI_THRESHOLD
            prediction = 'AI' if is_ai_generated else 'Human'
            
            # Determine confidence level
            confidence_score = max(weighted_ai_prob, human_probability)
            if confidence_score > Config.HIGH_CONFIDENCE_THRESHOLD:
                confidence_level = 'high'
            elif confidence_score > Config.AI_THRESHOLD:
                confidence_level = 'medium'
            else:
                confidence_level = 'low'
            
            # Generate highlighted parts (chunks that are likely AI)
            highlighted_parts = []
            for chunk_pred in chunk_predictions:
                if chunk_pred.get('ai_probability', 0) > Config.AI_THRESHOLD:
                    highlighted_parts.append({
                        'text': chunk_pred['text'],
                        'probability': chunk_pred['ai_probability'],
                        'chunk_id': chunk_pred['chunk_id']
                    })
            
            return {
                'ai_probability': weighted_ai_prob,
                'human_probability': human_probability,
                'is_ai_generated': is_ai_generated,
                'prediction': prediction,
                'confidence_level': confidence_level,
                'highlighted_parts': highlighted_parts,
                'chunk_predictions': chunk_predictions,
                'cleaned_text': cleaned_text,
                'total_chunks': len(chunks)
            }
            
        except Exception as e:
            self.logger.error(f"Error in predict_text: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_sentence_level_predictions(self, input_text):
        """
        Get sentence-level predictions for more granular highlighting
        """
        if not self.is_model_loaded():
            raise ValueError("Model not loaded")
        
        sentences = self.preprocessor.split_into_sentences(input_text)
        sentence_predictions = []
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                try:
                    ai_prob = self.predict_single_chunk(sentence)
                    sentence_predictions.append({
                        'sentence_id': i,
                        'text': sentence,
                        'ai_probability': ai_prob,
                        'human_probability': 1.0 - ai_prob,
                        'is_ai': ai_prob > Config.AI_THRESHOLD
                    })
                except Exception as e:
                    self.logger.error(f"Error predicting sentence {i}: {str(e)}")
                    sentence_predictions.append({
                        'sentence_id': i,
                        'text': sentence,
                        'ai_probability': 0.0,
                        'human_probability': 1.0,
                        'is_ai': False,
                        'error': str(e)
                    })
        
        return sentence_predictions

# Global model handler instance
@st.cache_resource
def get_model_handler():
    """Get or create model handler instance"""
    handler = ModelHandler()
    return handler

def load_model_with_error_handling():
    """Helper function to load model with proper error handling"""
    try:
        handler = get_model_handler()
        if not handler.is_model_loaded():
            handler.load_model()
        return handler
    except Exception as e:
        st.error(f"❌ Gagal memuat model: {str(e)}")
        return None
