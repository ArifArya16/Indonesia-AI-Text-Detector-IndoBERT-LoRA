"""
Model handler for AI Text Detector
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np
from config import Config
from text_preprocessor import TextPreprocessor
import logging

class ModelHandler:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = None
        self.model = None
        self.preprocessor = TextPreprocessor()
        self.loaded = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def load_model(self):
        """Load the fine-tuned model with LoRA adapters"""
        try:
            self.logger.info("Loading tokenizer...")
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(Config.MODEL_PATH)
            
            self.logger.info("Loading base model...")
            # Load base model
            base_model = AutoModelForSequenceClassification.from_pretrained(
                Config.BASE_MODEL_NAME,
                num_labels=2,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            self.logger.info("Loading LoRA adapters...")
            # Load model with LoRA adapters
            self.model = PeftModel.from_pretrained(base_model, Config.MODEL_PATH)
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.loaded = True
            self.logger.info(f"Model loaded successfully on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise e
    
    def predict_single_chunk(self, text_chunk):
        """Predict AI probability for a single text chunk"""
        if not self.loaded:
            raise ValueError("Model not loaded. Call load_model() first.")
        
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
    
    def predict_text(self, input_text):
        """
        Predict AI probability for input text
        Returns: dict with prediction results
        """
        if not input_text or not input_text.strip():
            return {
                'ai_probability': 0.0,
                'is_ai_generated': False,
                'confidence_level': 'low',
                'highlighted_parts': [],
                'chunk_predictions': []
            }
        
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
                    'is_ai': ai_prob > Config.AI_THRESHOLD
                })
                ai_probabilities.append(ai_prob)
            except Exception as e:
                self.logger.error(f"Error predicting chunk {i}: {str(e)}")
                chunk_predictions.append({
                    'chunk_id': i,
                    'text': chunk,
                    'ai_probability': 0.0,
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
        
        # Determine if text is AI-generated
        is_ai_generated = weighted_ai_prob > Config.AI_THRESHOLD
        
        # Determine confidence level
        if weighted_ai_prob > Config.HIGH_CONFIDENCE_THRESHOLD:
            confidence_level = 'high'
        elif weighted_ai_prob > Config.AI_THRESHOLD:
            confidence_level = 'medium'
        else:
            confidence_level = 'low'
        
        # Generate highlighted parts (chunks that are likely AI)
        highlighted_parts = []
        for chunk_pred in chunk_predictions:
            if chunk_pred['ai_probability'] > Config.AI_THRESHOLD:
                highlighted_parts.append({
                    'text': chunk_pred['text'],
                    'probability': chunk_pred['ai_probability'],
                    'chunk_id': chunk_pred['chunk_id']
                })
        
        return {
            'ai_probability': weighted_ai_prob,
            'is_ai_generated': is_ai_generated,
            'confidence_level': confidence_level,
            'highlighted_parts': highlighted_parts,
            'chunk_predictions': chunk_predictions,
            'cleaned_text': cleaned_text,
            'total_chunks': len(chunks)
        }
    
    def get_sentence_level_predictions(self, input_text):
        """
        Get sentence-level predictions for more granular highlighting
        """
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
                        'is_ai': ai_prob > Config.AI_THRESHOLD
                    })
                except Exception as e:
                    self.logger.error(f"Error predicting sentence {i}: {str(e)}")
                    sentence_predictions.append({
                        'sentence_id': i,
                        'text': sentence,
                        'ai_probability': 0.0,
                        'is_ai': False,
                        'error': str(e)
                    })
        
        return sentence_predictions