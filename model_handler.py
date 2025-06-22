"""
Model handler dengan debug authentication yang lebih detail
"""

import torch
import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from huggingface_hub import login, HfApi
import numpy as np
from config import Config
from text_preprocessor import TextPreprocessor
import logging
import traceback
import requests

class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.loaded = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def debug_hf_config(self):
        """Debug Hugging Face configuration"""
        st.write("🔍 **Debugging Hugging Face Configuration:**")
        
        # Check token
        hf_token = Config.get_hf_token()
        model_name = Config.get_model_name()
        
        if hf_token:
            st.write(f"✅ Token found: {hf_token[:10]}...")
            st.write(f"📝 Token length: {len(hf_token)}")
            st.write(f"🔑 Token format valid: {hf_token.startswith('hf_')}")
        else:
            st.write("❌ No token found")
            return False
        
        if model_name:
            st.write(f"✅ Model name: {model_name}")
        else:
            st.write("❌ No model name found")
            return False
        
        return True
    
    def test_hf_api_access(self):
        """Test Hugging Face API access"""
        try:
            hf_token = Config.get_hf_token()
            model_name = Config.get_model_name()
            
            st.write("🧪 **Testing API Access:**")
            
            # Test 1: Basic API test
            try:
                api = HfApi(token=hf_token)
                user_info = api.whoami()
                st.write(f"✅ Token valid for user: {user_info.get('name', 'Unknown')}")
            except Exception as e:
                st.write(f"❌ Token validation failed: {str(e)}")
                return False
            
            # Test 2: Model access test
            try:
                model_info = api.model_info(model_name)
                st.write(f"✅ Model accessible: {model_info.modelId}")
                st.write(f"🏷️ Model private: {model_info.private}")
            except Exception as e:
                st.write(f"❌ Model access failed: {str(e)}")
                return False
            
            # Test 3: Direct API call
            try:
                headers = {"Authorization": f"Bearer {hf_token}"}
                url = f"https://huggingface.co/api/models/{model_name}"
                response = requests.get(url, headers=headers, timeout=10)
                
                if response.status_code == 200:
                    st.write("✅ Direct API call successful")
                elif response.status_code == 401:
                    st.write("❌ Unauthorized - check token permissions")
                    return False
                elif response.status_code == 404:
                    st.write("❌ Model not found")
                    return False
                else:
                    st.write(f"❌ API error: {response.status_code}")
                    return False
            except Exception as e:
                st.write(f"❌ Direct API test failed: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            st.write(f"❌ API test failed: {str(e)}")
            return False
    
    def authenticate_huggingface(self):
        """Authenticate with Hugging Face using token from secrets"""
        try:
            hf_token = Config.get_hf_token()
            
            if not hf_token:
                self.logger.error("No Hugging Face token found in configuration")
                st.error("❌ Hugging Face token tidak ditemukan")
                return False
            
            # Validate token format
            if not hf_token.startswith('hf_'):
                self.logger.error("Invalid Hugging Face token format")
                st.error("❌ Format Hugging Face token tidak valid (harus dimulai dengan 'hf_')")
                return False
            
            # Test token first
            try:
                api = HfApi(token=hf_token)
                user_info = api.whoami()
                st.write(f"✅ Token valid untuk user: {user_info.get('name', 'Unknown')}")
            except Exception as e:
                st.error(f"❌ Token tidak valid: {str(e)}")
                return False
            
            # Login to Hugging Face
            login(token=hf_token, add_to_git_credential=False)
            self.logger.info("✅ Successfully authenticated with Hugging Face")
            st.success("✅ Berhasil login ke Hugging Face")
            return True
            
        except Exception as e:
            error_msg = str(e)
            self.logger.error(f"❌ Failed to authenticate with Hugging Face: {error_msg}")
            st.error(f"❌ Gagal login ke Hugging Face: {error_msg}")
            
            # Provide specific guidance
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                st.error("🔐 **Token tidak valid atau expired**")
                st.write("- Cek token di Hugging Face Settings")
                st.write("- Generate token baru jika perlu")
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                st.error("🚫 **Tidak ada akses ke model**")
                st.write("- Pastikan token punya akses ke model private")
                st.write("- Hubungi owner model untuk akses")
            
            return False
    
    @st.cache_resource
    def load_model(_self):
        """Load the model from Hugging Face Hub with detailed debugging"""
        try:
            # Debug configuration
            if not _self.debug_hf_config():
                raise Exception("Konfigurasi tidak valid")
            
            # Test API access
            if not _self.test_hf_api_access():
                raise Exception("API access test failed")
            
            model_name = Config.get_model_name()
            hf_token = Config.get_hf_token()
            
            _self.logger.info(f"🔄 Loading model from Hugging Face Hub: {model_name}")
            
            # Authenticate with Hugging Face
            if not _self.authenticate_huggingface():
                raise Exception("Authentication failed")
            
            # Load tokenizer with detailed error handling
            try:
                with st.spinner("🔤 Loading tokenizer..."):
                    _self.logger.info("Loading tokenizer...")
                    _self.tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        token=hf_token,  # Use 'token' instead of 'use_auth_token'
                        trust_remote_code=True
                    )
                    st.success("✅ Tokenizer loaded successfully")
            except Exception as e:
                st.error(f"❌ Gagal load tokenizer: {str(e)}")
                raise e
            
            # Load model with detailed error handling
            try:
                with st.spinner("🤖 Loading model..."):
                    _self.logger.info("Loading model...")
                    _self.model = AutoModelForSequenceClassification.from_pretrained(
                        model_name,
                        num_labels=2,
                        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                        token=hf_token,  # Use 'token' instead of 'use_auth_token'
                        trust_remote_code=True
                    )
                    
                    # Move to device and set to eval mode
                    _self.model = _self.model.to(_self.device)
                    _self.model.eval()
                    st.success("✅ Model loaded successfully")
            except Exception as e:
                st.error(f"❌ Gagal load model: {str(e)}")
                raise e
            
            _self.loaded = True
            _self.logger.info(f"✅ Model loaded successfully on {_self.device}")
            
            # Display model info
            st.info(f"📋 Model: {model_name}")
            st.info(f"💻 Device: {_self.device}")
            st.info(f"🎯 Model Labels: {_self.model.config.num_labels}")
            
            return _self
            
        except Exception as e:
            error_msg = str(e)
            _self.logger.error(f"❌ Error loading model: {error_msg}")
            
            st.error(f"❌ Gagal memuat model: {error_msg}")
            
            # Show detailed error in expander
            with st.expander("🔧 Detail Error (untuk debugging)"):
                st.code(traceback.format_exc())
            
            # Suggest solutions
            st.info("💡 **Solusi yang bisa dicoba:**")
            st.write("1. Periksa token Hugging Face di secrets")
            st.write("2. Pastikan token memiliki akses ke model")
            st.write("3. Coba generate token baru")
            st.write("4. Hubungi owner model jika private")
            
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
        """Predict AI probability for input text"""
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
            
            # Calculate overall AI probability
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
            
            # Determine prediction
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
            
            # Generate highlighted parts
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
