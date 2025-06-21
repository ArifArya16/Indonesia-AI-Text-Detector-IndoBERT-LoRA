"""
Database operations for AI Text Detector
"""

import sqlite3
import bcrypt
import json
from datetime import datetime
import os
from config import Config

class Database:
    def __init__(self):
        self.db_path = Config.DATABASE_PATH
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables if they don't exist"""
        # Create database directory if it doesn't exist
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table - UBAH YANG INI untuk menambah kolom role
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role TEXT DEFAULT 'user',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP,
                is_active BOOLEAN DEFAULT 1
            )
        ''')
        
        # Predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                input_text TEXT NOT NULL,
                ai_probability REAL NOT NULL,
                is_ai_generated BOOLEAN NOT NULL,
                highlighted_parts TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        # Create admin user if not exists
        self.create_admin_user()
        self.create_guest_user()
        
        conn.commit()
        conn.close()
    
    def create_admin_user(self):
        """Create default admin user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if admin exists
            cursor.execute('SELECT id FROM users WHERE username = ?', ('arifaryaaureon1603',))
            if not cursor.fetchone():
                # Create admin user
                password_hash = bcrypt.hashpw('arif_ganteng'.encode('utf-8'), bcrypt.gensalt())
                cursor.execute('''
                    INSERT INTO users (username, email, password_hash, role)
                    VALUES (?, ?, ?, ?)
                ''', ('arifaryaaureon1603', 'admin@detector.ai', password_hash, 'admin'))
                conn.commit()
            
            conn.close()
        except Exception as e:
            print(f"Error creating admin user: {e}")
            
    def create_guest_user(self):
            """Create default guest account"""
            try:
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                
                # Check if admin exists
                cursor.execute('SELECT id FROM users WHERE username = ?', ('Guest',))
                if not cursor.fetchone():
                    # Create admin user
                    password_hash = bcrypt.hashpw('Guest123'.encode('utf-8'), bcrypt.gensalt())
                    cursor.execute('''
                        INSERT INTO users (username, email, password_hash, role)
                        VALUES (?, ?, ?, ?)
                    ''', ('Guest', 'Guest@gmail.com', password_hash, 'user'))
                    conn.commit()
                
                conn.close()
            except Exception as e:
                print(f"Error creating Guest Account: {e}")
    
    def get_user_role(self, user_id):
        """Get user role"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT role FROM users WHERE id = ?', (user_id,))
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result else 'user'
    
    # ADMIN METHODS
    def get_all_users(self, limit=100):
        """Get all users for admin"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, role, created_at, last_login, is_active,
                (SELECT COUNT(*) FROM predictions WHERE user_id = users.id) as total_predictions
            FROM users 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        users = []
        for row in results:
            user = {
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'role': row[3],
                'created_at': row[4],
                'last_login': row[5],
                'is_active': row[6],
                'total_predictions': row[7]
            }
            users.append(user)
        
        return users
    
    def get_all_predictions(self, limit=100):
        """Get all predictions for admin"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT p.id, p.user_id, u.username, p.input_text, p.ai_probability, 
                p.is_ai_generated, p.created_at
            FROM predictions p
            JOIN users u ON p.user_id = u.id
            ORDER BY p.created_at DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in results:
            pred = {
                'id': row[0],
                'user_id': row[1],
                'username': row[2],
                'input_text': row[3],
                'ai_probability': row[4],
                'is_ai_generated': row[5],
                'created_at': row[6]
            }
            predictions.append(pred)
        
        return predictions
    
    def toggle_user_status(self, user_id):
        """Toggle user active status"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('UPDATE users SET is_active = NOT is_active WHERE id = ?', (user_id,))
        conn.commit()
        conn.close()
    
    def delete_user(self, user_id):
        """Delete user and their predictions"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Delete user's predictions first
        cursor.execute('DELETE FROM predictions WHERE user_id = ?', (user_id,))
        # Delete user
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        
        conn.commit()
        conn.close()
    
    def get_system_stats(self):
        """Get system-wide statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute('SELECT COUNT(*) FROM users WHERE role = "user"')
        total_users = cursor.fetchone()[0]
        
        # Active users (logged in last 30 days)
        cursor.execute('''
            SELECT COUNT(*) FROM users 
            WHERE role = "user" AND last_login >= datetime('now', '-30 days')
        ''')
        active_users = cursor.fetchone()[0]
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_predictions = cursor.fetchone()[0]
        
        # AI vs Human predictions
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE is_ai_generated = 1')
        ai_predictions = cursor.fetchone()[0]
        
        # Recent activity (last 7 days)
        cursor.execute('''
            SELECT COUNT(*) FROM predictions 
            WHERE created_at >= datetime('now', '-7 days')
        ''')
        recent_predictions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_users': total_users,
            'active_users': active_users,
            'total_predictions': total_predictions,
            'ai_predictions': ai_predictions,
            'human_predictions': total_predictions - ai_predictions,
            'recent_predictions': recent_predictions
        }
    
    def search_users(self, query):
        """Search users by username or email"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, username, email, role, created_at, last_login, is_active,
                (SELECT COUNT(*) FROM predictions WHERE user_id = users.id) as total_predictions
            FROM users 
            WHERE username LIKE ? OR email LIKE ?
            ORDER BY created_at DESC
        ''', (f'%{query}%', f'%{query}%'))
        
        results = cursor.fetchall()
        conn.close()
        
        users = []
        for row in results:
            user = {
                'id': row[0],
                'username': row[1],
                'email': row[2],
                'role': row[3],
                'created_at': row[4],
                'last_login': row[5],
                'is_active': row[6],
                'total_predictions': row[7]
            }
            users.append(user)
        
        return users
    
    def create_user(self, username, email, password):
        """Create a new user"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Hash password
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            
            cursor.execute('''
                INSERT INTO users (username, email, password_hash)
                VALUES (?, ?, ?)
            ''', (username, email, password_hash))
            
            conn.commit()
            user_id = cursor.lastrowid
            conn.close()
            return user_id
        except sqlite3.IntegrityError:
            return None
    
    def authenticate_user(self, username, password):
        """Authenticate user login"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id, password_hash, role, is_active FROM users WHERE username = ?
        ''', (username,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            user_id, password_hash,role,is_active = result
            if bcrypt.checkpw(password.encode('utf-8'), password_hash):
                self.update_last_login(user_id)
                return user_id,role
        return None
    
    def update_last_login(self, user_id):
        """Update user's last login timestamp"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE users SET last_login = CURRENT_TIMESTAMP WHERE id = ?
        ''', (user_id,))
        
        conn.commit()
        conn.close()
    
    def get_user_info(self, user_id):
        """Get user information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT username, email, created_at, last_login FROM users WHERE id = ?
        ''', (user_id,))
        
        result = cursor.fetchone()
        conn.close()
        return result
    
    def save_prediction(self, user_id, input_text, ai_probability, is_ai_generated, highlighted_parts):
        """Save prediction result"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO predictions (user_id, input_text, ai_probability, is_ai_generated, highlighted_parts)
            VALUES (?, ?, ?, ?, ?)
        ''', (user_id, input_text, ai_probability, is_ai_generated, json.dumps(highlighted_parts)))
        
        conn.commit()
        prediction_id = cursor.lastrowid
        conn.close()
        return prediction_id
    
    def get_user_predictions(self, user_id, limit=50):
        """Get user's prediction history"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT id, input_text, ai_probability, is_ai_generated, highlighted_parts, created_at
            FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        results = cursor.fetchall()
        conn.close()
        
        # Convert to list of dictionaries
        predictions = []
        for row in results:
            pred = {
                'id': row[0],
                'input_text': row[1],
                'ai_probability': row[2],
                'is_ai_generated': row[3],
                'highlighted_parts': json.loads(row[4]) if row[4] else [],
                'created_at': row[5]
            }
            predictions.append(pred)
        
        return predictions
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Total predictions
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ?', (user_id,))
        total_predictions = cursor.fetchone()[0]
        
        # AI predictions
        cursor.execute('SELECT COUNT(*) FROM predictions WHERE user_id = ? AND is_ai_generated = 1', (user_id,))
        ai_predictions = cursor.fetchone()[0]
        
        # Average AI probability
        cursor.execute('SELECT AVG(ai_probability) FROM predictions WHERE user_id = ?', (user_id,))
        avg_ai_prob = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            'total_predictions': total_predictions,
            'ai_predictions': ai_predictions,
            'human_predictions': total_predictions - ai_predictions,
            'avg_ai_probability': avg_ai_prob
        }