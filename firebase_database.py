"""
Firebase Database operations for AI Text Detector
"""

import firebase_admin
from firebase_admin import credentials, firestore
import bcrypt
import json
from datetime import datetime, timedelta
import streamlit as st
from config import Config

class FirebaseDatabase:
    def __init__(self):
        self.init_firebase()
        self.db = firestore.client()
    
    def init_firebase(self):
        """Initialize Firebase connection"""
        if not firebase_admin._apps:
            # Ambil credentials dari Streamlit secrets
            try:
                # Firebase config dari Streamlit secrets
                firebase_secrets = st.secrets['firebase']
                
                # Create credentials dari secrets
                cred = credentials.Certificate({
                    "type": firebase_secrets["type"],
                    "project_id": firebase_secrets["project_id"],
                    "private_key_id": firebase_secrets["private_key_id"],
                    "private_key": firebase_secrets["private_key"].replace('\\n', '\n'),
                    "client_email": firebase_secrets["client_email"],
                    "client_id": firebase_secrets["client_id"],
                    "auth_uri": firebase_secrets["auth_uri"],
                    "token_uri": firebase_secrets["token_uri"],
                    "auth_provider_x509_cert_url": firebase_secrets["auth_provider_x509_cert_url"],
                    "client_x509_cert_url": firebase_secrets["client_x509_cert_url"]
                })
                
                firebase_admin.initialize_app(cred)
            except Exception as e:
                st.error(f"Error initializing Firebase: {e}")
                raise e
        
        # Create default users
    
    def create_admin_user(self):
        """Create default admin user"""
        try:
            # Check if admin exists
            users_ref = self.db.collection('users')
            query = users_ref.where('username', '==', 'arifaryaaureon1603').limit(1)
            docs = query.get()
            
            if not docs:
                # Create admin user
                password_hash = bcrypt.hashpw('arif_ganteng'.encode('utf-8'), bcrypt.gensalt())
                admin_data = {
                    'username': 'arifaryaaureon1603',
                    'email': 'admin@detector.ai',
                    'password_hash': password_hash,
                    'role': 'admin',
                    'created_at': datetime.now(),
                    'last_login': None,
                    'is_active': True
                }
                self.db.collection('users').add(admin_data)
                
        except Exception as e:
            print(f"Error creating admin user: {e}")
    
    def create_guest_user(self):
        """Create default guest account"""
        try:
            # Check if guest exists
            users_ref = self.db.collection('users')
            query = users_ref.where('username', '==', 'Guest').limit(1)
            docs = query.get()
            
            if not docs:
                # Create guest user
                password_hash = bcrypt.hashpw('Guest123'.encode('utf-8'), bcrypt.gensalt())
                guest_data = {
                    'username': 'Guest',
                    'email': 'Guest@gmail.com',
                    'password_hash': password_hash,
                    'role': 'user',
                    'created_at': datetime.now(),
                    'last_login': None,
                    'is_active': True
                }
                self.db.collection('users').add(guest_data)
                
        except Exception as e:
            print(f"Error creating Guest Account: {e}")
    
    def get_user_role(self, user_id):
        """Get user role"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict().get('role', 'user')
            return 'user'
        except Exception as e:
            print(f"Error getting user role: {e}")
            return 'user'
    
    # ADMIN METHODS
    def get_all_users(self, limit=100):
        """Get all users for admin"""
        try:
            users_ref = self.db.collection('users').order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
            users = users_ref.get()
            docs = [user for user in users if user.get('username') != 'Guest']
            users = []
            for doc in docs:
                user_data = doc.to_dict()
                
                # Count predictions for this user
                predictions_count = len(self.db.collection('predictions').where('user_id', '==', doc.id).get())
                
                user = {
                    'id': doc.id,
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'role': user_data.get('role'),
                    'created_at': user_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('created_at') else '',
                    'last_login': user_data.get('last_login').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('last_login') else None,
                    'is_active': user_data.get('is_active'),
                    'total_predictions': predictions_count
                }
                users.append(user)
            
            return users
        except Exception as e:
            print(f"Error getting all users: {e}")
            return []
    
    def get_all_predictions(self, limit=100):
        """Get all predictions for admin"""
        try:
            predictions_ref = self.db.collection('predictions').order_by('created_at', direction=firestore.Query.DESCENDING).limit(limit)
            docs = predictions_ref.get()
            
            predictions = []
            for doc in docs:
                pred_data = doc.to_dict()
                
                # Get username
                user_doc = self.db.collection('users').document(pred_data.get('user_id')).get()
                username = user_doc.to_dict().get('username') if user_doc.exists else 'Unknown'
                
                pred = {
                    'id': doc.id,
                    'user_id': pred_data.get('user_id'),
                    'username': username,
                    'input_text': pred_data.get('input_text'),
                    'ai_probability': pred_data.get('ai_probability'),
                    'is_ai_generated': pred_data.get('is_ai_generated'),
                    'created_at': pred_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if pred_data.get('created_at') else ''
                }
                predictions.append(pred)
            
            return predictions
        except Exception as e:
            print(f"Error getting all predictions: {e}")
            return []
    
    def toggle_user_status(self, user_id):
        """Toggle user active status"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                current_status = doc.to_dict().get('is_active', True)
                doc_ref.update({'is_active': not current_status})
        except Exception as e:
            print(f"Error toggling user status: {e}")
    
    def delete_user(self, user_id):
        """Delete user and their predictions"""
        try:
            # Delete user's predictions first
            predictions_ref = self.db.collection('predictions').where('user_id', '==', user_id)
            predictions = predictions_ref.get()
            
            for pred in predictions:
                pred.reference.delete()
            
            # Delete user
            self.db.collection('users').document(user_id).delete()
            
        except Exception as e:
            print(f"Error deleting user: {e}")
    
    def get_system_stats(self):
        """Get system-wide statistics"""
        try:
            # Total users (excluding admins)
            users_ref = self.db.collection('users').where('role', '==', 'user')
            users = users_ref.get()
            filtered_users = [user for user in users if user.get('username') != 'Guest']
            total_users = len(filtered_users)
            
            # Active users (logged in last 30 days)
            active_users_ref = self.db.collection('users').where('role', '==', 'user').where('is_active', '==', True)
            users = active_users_ref.get()
            filtered_users_active = [user for user in users if user.get('username') != 'Guest']
            active_users = len(filtered_users_active)
            
            # Total predictions
            predictions_ref = self.db.collection('predictions')
            all_predictions = predictions_ref.get()
            total_predictions = len(all_predictions)
            
            # AI vs Human predictions
            ai_predictions = len([p for p in all_predictions if p.to_dict().get('is_ai_generated')])
            
            # Recent activity (last 7 days)
            seven_days_ago = datetime.now().replace(day=datetime.now().day-7)
            recent_predictions_ref = predictions_ref.where('created_at', '>=', seven_days_ago)
            recent_predictions = len(recent_predictions_ref.get())
            
            return {
                'total_users': total_users,
                'active_users': active_users,
                'total_predictions': total_predictions,
                'ai_predictions': ai_predictions,
                'human_predictions': total_predictions - ai_predictions,
                'recent_predictions': recent_predictions
            }
        except Exception as e:
            print(f"Error getting system stats: {e}")
            return {
                'total_users': 0,
                'active_users': 0,
                'total_predictions': 0,
                'ai_predictions': 0,
                'human_predictions': 0,
                'recent_predictions': 0
            }
    
    def search_users(self, query):
        """Search users by username or email"""
        try:
            # Firebase doesn't support OR queries directly, so we need to do two separate queries
            users_by_username = self.db.collection('users').where('username', '>=', query).where('username', '<=', query + '\uf8ff').get()
            users_by_email = self.db.collection('users').where('email', '>=', query).where('email', '<=', query + '\uf8ff').get()
            
            # Combine results and remove duplicates
            all_users = {}
            
            for doc in users_by_username:
                user_data = doc.to_dict()
                predictions_count = len(self.db.collection('predictions').where('user_id', '==', doc.id).get())
                
                all_users[doc.id] = {
                    'id': doc.id,
                    'username': user_data.get('username'),
                    'email': user_data.get('email'),
                    'role': user_data.get('role'),
                    'created_at': user_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('created_at') else '',
                    'last_login': user_data.get('last_login').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('last_login') else None,
                    'is_active': user_data.get('is_active'),
                    'total_predictions': predictions_count
                }
            
            for doc in users_by_email:
                if doc.id not in all_users:
                    user_data = doc.to_dict()
                    predictions_count = len(self.db.collection('predictions').where('user_id', '==', doc.id).get())
                    
                    all_users[doc.id] = {
                        'id': doc.id,
                        'username': user_data.get('username'),
                        'email': user_data.get('email'),
                        'role': user_data.get('role'),
                        'created_at': user_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('created_at') else '',
                        'last_login': user_data.get('last_login').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('last_login') else None,
                        'is_active': user_data.get('is_active'),
                        'total_predictions': predictions_count
                    }
            
            return list(all_users.values())
        except Exception as e:
            print(f"Error searching users: {e}")
            return []
    
    def create_user(self, username, email, password):
        """Create a new user with proper validation (case-insensitive)"""
        try:
            users_ref = self.db.collection('users')
            
            # Check username (case-insensitive) - get all users and check manually
            all_users = users_ref.get()
            for user_doc in all_users:
                existing_username = user_doc.to_dict().get('username', '')
                if existing_username.lower() == username.lower():
                    print(f"Username '{username}' already exists (case-insensitive)")
                    return None  # Username already exists
            
            # Check email (case-insensitive)
            for user_doc in all_users:
                existing_email = user_doc.to_dict().get('email', '')
                if existing_email.lower() == email.lower():
                    print(f"Email '{email}' already exists (case-insensitive)")
                    return None  # Email already exists
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            # Create user data (store original case but add lowercase fields for easier querying)
            user_data = {
                'username': username,  # Store original case
                'username_lower': username.lower(),  # For case-insensitive queries
                'email': email.lower(),  # Store email in lowercase
                'password_hash': password_hash,
                'role': 'user',
                'created_at': datetime.now(),
                'last_login': None,
                'is_active': True
            }
            
            # Add user to database
            doc_ref = self.db.collection('users').add(user_data)
            print(f"User created successfully with ID: {doc_ref[1].id}")
            return doc_ref[1].id  # Return document ID
            
        except Exception as e:
            print(f"Error creating user: {e}")
            st.error(f"Error creating user: {e}")
            return None
    
    def authenticate_user(self, username, password):
        """Authenticate user login with case-insensitive username"""
        try:
            if not username or not password:
                return None, "Username dan password harus diisi"
            
            users_ref = self.db.collection('users')
            # Get all users and check manually for case-insensitive match
            all_users = users_ref.get()
            
            user_doc = None
            for doc in all_users:
                user_data = doc.to_dict()
                existing_username = user_data.get('username', '')
                if existing_username.lower() == username.lower():
                    user_doc = doc
                    break
            
            if not user_doc:
                print(f"User '{username}' not found")
                return None, "Username atau password salah"
            
            user_data = user_doc.to_dict()
            
            # Check if account is active
            if not user_data.get('is_active', True):
                print(f"Account '{username}' is deactivated")
                return None, "Akun Anda telah dinonaktifkan"
            
            # Verify password
            password_hash = user_data.get('password_hash')
            if not password_hash:
                print(f"No password hash found for user '{username}'")
                return None, "Error sistem, silakan hubungi administrator"
            
            if bcrypt.checkpw(password.encode('utf-8'), password_hash):
                # Update last login
                self.update_last_login(user_doc.id)
                user_role = user_data.get('role', 'user')
                actual_username = user_data.get('username')  # Get the original case username
                print(f"User '{actual_username}' authenticated successfully with role '{user_role}'")
                return user_doc.id, user_role
            else:
                print(f"Invalid password for user '{username}'")
                return None, "Username atau password salah"
            
        except Exception as e:
            print(f"Error authenticating user: {e}")
            return None, f"Error sistem: {e}"
    
    def get_actual_username(self, user_id):
        """Get the actual username (with original case) from user ID"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                return doc.to_dict().get('username')
            return None
        except Exception as e:
            print(f"Error getting actual username: {e}")
            return None
    
    def check_email_exists(self, email):
        """Check if email already exists (case-insensitive)"""
        try:
            users_ref = self.db.collection('users')
            all_users = users_ref.get()
            
            for user_doc in all_users:
                existing_email = user_doc.to_dict().get('email', '')
                if existing_email.lower() == email.lower():
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking email: {e}")
            return True  # Return True to be safe

    
    def check_username_exists(self, username):
        """Check if username already exists (case-insensitive)"""
        try:
            users_ref = self.db.collection('users')
            all_users = users_ref.get()
            
            for user_doc in all_users:
                existing_username = user_doc.to_dict().get('username', '')
                if existing_username.lower() == username.lower():
                    return True
            
            return False
        except Exception as e:
            print(f"Error checking username: {e}")
            return True  # Return True to be safe
    
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc_ref.update({'last_login': datetime.now()})
        except Exception as e:
            print(f"Error updating last login: {e}")
    
    def get_user_info(self, user_id):
        """Get user information"""
        try:
            doc_ref = self.db.collection('users').document(user_id)
            doc = doc_ref.get()
            
            if doc.exists:
                user_data = doc.to_dict()
                return (
                    user_data.get('username'),
                    user_data.get('email'),
                    user_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('created_at') else '',
                    user_data.get('last_login').strftime('%Y-%m-%d %H:%M:%S') if user_data.get('last_login') else None
                )
            return None
        except Exception as e:
            print(f"Error getting user info: {e}")
            return None
    
    def save_prediction(self, user_id, input_text, ai_probability, is_ai_generated, highlighted_parts):
        """Save prediction result"""
        try:
            prediction_data = {
                'user_id': user_id,
                'input_text': input_text,
                'ai_probability': ai_probability,
                'is_ai_generated': is_ai_generated,
                'highlighted_parts': highlighted_parts,  # Already JSON serializable
                'created_at': datetime.now()
            }
            
            doc_ref = self.db.collection('predictions').add(prediction_data)
            return doc_ref[1].id  # Return document ID
            
        except Exception as e:
            print(f"Error saving prediction: {e}")
            return None
    
    def get_user_predictions(self, user_id, limit=50):
        """Get user's prediction history"""
        try:
            # Option 1: Simple query without ordering (then sort in Python)
            predictions_ref = self.db.collection('predictions').where('user_id', '==', user_id).limit(limit * 2)  # Get more to account for sorting
            docs = predictions_ref.get()
            
            predictions = []
            for doc in docs:
                pred_data = doc.to_dict()
                pred = {
                    'id': doc.id,
                    'input_text': pred_data.get('input_text'),
                    'ai_probability': pred_data.get('ai_probability'),
                    'is_ai_generated': pred_data.get('is_ai_generated'),
                    'highlighted_parts': pred_data.get('highlighted_parts', []),
                    'created_at': pred_data.get('created_at'),
                    'created_at_str': pred_data.get('created_at').strftime('%Y-%m-%d %H:%M:%S') if pred_data.get('created_at') else ''
                }
                predictions.append(pred)
            
            # Sort by created_at in Python (descending order)
            predictions.sort(key=lambda x: x['created_at'] if x['created_at'] else datetime.min, reverse=True)
            
            # Remove the datetime object from final result and limit
            final_predictions = []
            for pred in predictions[:limit]:
                final_pred = {
                    'id': pred['id'],
                    'input_text': pred['input_text'],
                    'ai_probability': pred['ai_probability'],
                    'is_ai_generated': pred['is_ai_generated'],
                    'highlighted_parts': pred['highlighted_parts'],
                    'created_at': pred['created_at_str']
                }
                final_predictions.append(final_pred)
            
            return final_predictions
            
        except Exception as e:
            print(f"Error getting user predictions: {e}")
            return []
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        try:
            predictions_ref = self.db.collection('predictions').where('user_id', '==', user_id)
            predictions = predictions_ref.get()
            
            total_predictions = len(predictions)
            ai_predictions = len([p for p in predictions if p.to_dict().get('is_ai_generated')])
            
            # Calculate average AI probability
            if total_predictions > 0:
                total_prob = sum([p.to_dict().get('ai_probability', 0) for p in predictions])
                avg_ai_prob = total_prob / total_predictions
            else:
                avg_ai_prob = 0
            
            return {
                'total_predictions': total_predictions,
                'ai_predictions': ai_predictions,
                'human_predictions': total_predictions - ai_predictions,
                'avg_ai_probability': avg_ai_prob
            }
        except Exception as e:
            print(f"Error getting user stats: {e}")
            return {
                'total_predictions': 0,
                'ai_predictions': 0,
                'human_predictions': 0,
                'avg_ai_probability': 0
            }
