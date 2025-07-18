"""
Main Streamlit Application for AI Text Detector
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
import logging
import os

# Import custom modules
from config import Config
from auth import Auth
from firebase_database import FirebaseDatabase as Database
from model_handler import ModelHandler
from utils import Utils

# Configure logging
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(Config.LOG_FILE),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class AITextDetectorApp:
    def __init__(self):
        Utils.set_page_config()
        Utils.apply_custom_css()
        
        self.auth = Auth()
        self.db = Database()
        self.model_handler = None
        
        # Initialize session states yang diperlukan
        if 'show_detailed_analysis' not in st.session_state:
            st.session_state.show_detailed_analysis = False
        if 'analisis_text' not in st.session_state:
            st.session_state.analisis_text = None
        if 'analisis_text_history' not in st.session_state:
            st.session_state.analisis_text_history = None
        if 'login' not in st.session_state:
            st.session_state.login = False
        
        # Initialize session state
        self.auth.init_session_state()
        
        # Initialize model handler
        self.init_model()
    
    def init_model(self):
        """Initialize model handler - FIXED VERSION"""
        if 'model_handler' not in st.session_state:
            try:
                with st.spinner("🔄 Memuat model AI..."):
                    self.model_handler = ModelHandler()
                    # Load model and store in session state
                    loaded_handler = self.model_handler.load_model()
                    st.session_state.model_handler = loaded_handler
                    logger.info("Model loaded and cached successfully")
                    return True
            except Exception as e:
                st.error(f"❌ Gagal memuat model: {str(e)}")
                logger.error(f"Failed to load model: {str(e)}")
                # Set fallback state
                st.session_state.model_handler = None
                return False
        else:
            self.model_handler = st.session_state.model_handler
            if self.model_handler and hasattr(self.model_handler, 'loaded') and self.model_handler.loaded:
                return True
            else:
                # Model handler exists but not loaded, reload
                try:
                    with st.spinner("🔄 Memuat ulang model..."):
                        if self.model_handler:
                            self.model_handler.cleanup_model()
                        self.model_handler = ModelHandler()
                        loaded_handler = self.model_handler.load_model()
                        st.session_state.model_handler = loaded_handler
                        return True
                except Exception as e:
                    st.error(f"❌ Gagal memuat ulang model: {str(e)}")
                    st.session_state.model_handler = None
                    return False
    
    def main_interface(self):
        """Main application interface"""
        # Header
        page = False
        st.title("🤖 Detector Teks AI Indonesia")
        if st.session_state.authenticated == True:
            role_badge = "👑 Admin" if self.auth.is_admin() else "👤 User"
            st.markdown(f"**Selamat datang, {self.auth.get_current_username()}!** ({role_badge})")
        
        # Sidebar
        with st.sidebar:
            if st.session_state.authenticated == True:
                st.markdown("### 📋 Menu")
                
                # Menu options based on role
                menu_options = ["📊 Dashboard", "🔍 Deteksi Teks", "📈 Riwayat", "👤 Profil"]
                if self.auth.is_admin():
                    menu_options.append("⚙️ Admin Panel")
                
                page = st.selectbox(
                "Pilih Halaman:",
                menu_options,
                key="page_selector"
                )
                
                st.markdown("---")
                
                # User stats
                if not self.auth.is_admin():
                    user_stats = self.db.get_user_stats(self.auth.get_current_user_id())
                    st.markdown("### 📈 Statistik Anda")
                    st.metric("Total Prediksi", user_stats['total_predictions'])
                    st.metric("Teks AI Terdeteksi", user_stats['ai_predictions'])
                    st.metric("Rata-rata AI Score", f"{user_stats['avg_ai_probability']:.1%}")
                else:
                    # Admin stats
                    system_stats = self.db.get_system_stats()
                    st.markdown("### 🏛️ Statistik Sistem")
                    st.metric("Total User", system_stats['total_users'])
                    st.metric("Total Prediksi", system_stats['total_predictions'])
                    st.metric("Aktifitas 7 Hari", system_stats['recent_predictions'])
                
                st.markdown("---")
                
                if st.button("🚪 Logout", use_container_width=True):
                    self.auth.logout()
            else:
                st.markdown("Selamat datang di Detector Teks AI Indonesia\n\n Login untuk mendapatkan fitur lengkapnya")
                
                st.markdown("---")
                
                if st.button("🚪 Login", use_container_width=True):
                    st.session_state.login = True
                    st.rerun()
                    
        # Route to appropriate page
        if page == "🔍 Deteksi Teks":
            self.detection_page()
        elif page == "📊 Dashboard":
            self.dashboard_page()
            st.session_state.analisis_text = None
            st.session_state.analisis_text_history = None
        elif page == "📈 Riwayat":
            self.history_page()
            st.session_state.analisis_text = None
            st.session_state.analisis_text_history = None
        elif page == "👤 Profil":
            self.profile_page()
            st.session_state.analisis_text = None
            st.session_state.analisis_text_history = None
        elif page == "⚙️ Admin Panel" and self.auth.is_admin():
            self.admin_panel()
            st.session_state.analisis_text = None
            st.session_state.analisis_text_history = None
        else:
            self.detection_page()

    def admin_panel(self):
        """Admin panel page"""
        if not self.auth.is_admin():
            st.error("❌ Akses ditolak. Hanya admin yang dapat mengakses halaman ini.")
            return
        
        st.header("⚙️ Admin Panel")
        
        # Admin tabs
        tab1, tab2, tab3, tab4 = st.tabs(["👥 Kelola User", "📊 Statistik Sistem", "📋 Semua Prediksi", "🔍 Pencarian"])
        
        with tab1:
            self.admin_manage_users()
        
        with tab2:
            self.admin_system_stats()
        
        with tab3:
            self.admin_all_predictions()
        
        with tab4:
            self.admin_search()

    def admin_manage_users(self):
        """Admin user management"""
        st.subheader("👥 Kelola Pengguna")
        
        # Get all users
        users = self.db.get_all_users(200)
        
        if not users:
            st.info("Tidak ada pengguna yang ditemukan.")
            return
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_role = st.selectbox("Filter Role:", ["Semua", "User", "Admin"])
        
        with col2:
            filter_status = st.selectbox("Filter Status:", ["Semua", "Aktif", "Nonaktif"])
        
        with col3:
            sort_by = st.selectbox("Urutkan:", ["Terbaru", "Username", "Total Prediksi"])
        
        # Filter users
        filtered_users = users
        
        if filter_role != "Semua":
            filtered_users = [u for u in filtered_users if u['role'] == filter_role.lower()]
        
        if filter_status != "Semua":
            is_active = filter_status == "Aktif"
            filtered_users = [u for u in filtered_users if u['is_active'] == is_active]
        
        # Sort users
        if sort_by == "Username":
            filtered_users = sorted(filtered_users, key=lambda x: x['username'])
        elif sort_by == "Total Prediksi":
            filtered_users = sorted(filtered_users, key=lambda x: x['total_predictions'], reverse=True)
        
        st.markdown(f"**Menampilkan {len(filtered_users)} dari {len(users)} pengguna**")
        
        # Display users in cards
        for user in filtered_users:
            with st.container():
                col1, col2, col3, col4 = st.columns([2, 1, 1, 2])
                
                with col1:
                    status_icon = "🟢" if user['is_active'] else "🔴"
                    role_icon = "👑" if user['role'] == 'admin' else "👤"
                    st.markdown(f"""
                    **{role_icon} {user['username']}** {status_icon}
                    
                    📧 {user['email']}
                    
                    📅 Bergabung: {user['created_at'][:10]}
                    """)
                
                with col2:
                    st.metric("Prediksi", user['total_predictions'])
                
                with col3:
                    st.markdown("**Last Login:**")
                    st.text(user['last_login'][:10] if user['last_login'] else "Belum pernah")
                
                with col4:
                    if user['role'] != 'admin':  # Don't allow admin to modify other admins
                        col4_1, col4_2 = st.columns(2)
                        
                        with col4_1:
                            status_text = "Nonaktifkan" if user['is_active'] else "Aktifkan"
                            if st.button(f"🔄 {status_text}", key=f"toggle_{user['id']}"):
                                self.db.toggle_user_status(user['id'])
                                st.rerun()
                        
                        with col4_2:
                            if st.button("🗑️ Hapus", key=f"delete_{user['id']}", type="secondary"):
                                if st.session_state.get(f"confirm_delete_{user['id']}", False):
                                    self.db.delete_user(user['id'])
                                    st.success(f"User {user['username']} berhasil dihapus!")
                                    st.rerun()
                                else:
                                    st.session_state[f"confirm_delete_{user['id']}"] = True
                                    st.warning("Klik sekali lagi untuk konfirmasi penghapusan!")
                
                st.markdown("---")

    def admin_system_stats(self):
        """Admin system statistics"""
        st.subheader("📊 Statistik Sistem")
        
        stats = self.db.get_system_stats()
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total User", stats['total_users'])
        
        with col2:
            st.metric("User Aktif", stats['active_users'])
        
        with col3:
            st.metric("Total Prediksi", stats['total_predictions'])
        
        with col4:
            st.metric("Aktivitas 7 Hari", stats['recent_predictions'])
        
        # Additional metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Prediksi AI", stats['ai_predictions'])
            if stats['total_predictions'] > 0:
                ai_percentage = (stats['ai_predictions'] / stats['total_predictions']) * 100
                st.caption(f"{ai_percentage:.1f}% dari total prediksi")
        
        with col2:
            st.metric("Prediksi Manusia", stats['human_predictions'])
            if stats['total_predictions'] > 0:
                human_percentage = (stats['human_predictions'] / stats['total_predictions']) * 100
                st.caption(f"{human_percentage:.1f}% dari total prediksi")

    def admin_all_predictions(self):
        """Admin view all predictions"""
        st.subheader("📋 Semua Prediksi")
        
        predictions = self.db.get_all_predictions(200)
        
        if not predictions:
            st.info("Tidak ada prediksi yang ditemukan.")
            return
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            filter_type = st.selectbox("Filter Tipe:", ["Semua", "AI", "Manusia"])
        
        with col2:
            # Get unique usernames for filter
            filter_user = st.selectbox("Filter User:", ["Semua","Tanpa Guest"] )
        
        with col3:
            limit = st.selectbox("Tampilkan:", [25, 50, 100, 200])
        
        # Filter predictions
        filtered_predictions = predictions
        
        if filter_type != "Semua":
            is_ai = filter_type == "AI"
            filtered_predictions = [p for p in filtered_predictions if p['is_ai_generated'] == is_ai]
            filtered_predictions = [p for p in filtered_predictions if p['is_ai_generated'] == is_ai]
        
        if filter_user != "Semua":
            filtered_predictions = [p for p in filtered_predictions if p['username'] != "Guest"]
        
        filtered_predictionss = filtered_predictions[:limit]
        st.markdown(f"**Menampilkan {len(filtered_predictionss)} prediksi**")
        
        if st.button("📥 Export ke CSV"):
                csv_data = Utils.export_predictions_to_csv(filtered_predictionss)
                if csv_data:
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
        # Display predictions
        for pred in filtered_predictions:
            with st.expander(f"🤖 {pred['username']} - {pred['created_at'][:16]} ({'AI' if pred['is_ai_generated'] else 'Manusia'})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("**Teks Input:**")
                    preview = pred['input_text'][:200]
                    if len(pred['input_text']) > 200:
                        preview += "..."
                    st.text(preview)
                
                with col2:
                    st.metric("AI Probability", f"{pred['ai_probability']:.1%}")
                    st.text(f"User: {pred['username']}")
                    st.text(f"ID: {pred['id']}")

    def admin_search(self):
        """Admin search functionality"""
        st.subheader("🔍 Pencarian")
        
        # Search users
        st.markdown("### 👥 Cari Pengguna")
        search_query = st.text_input("Cari berdasarkan username atau email:")
        
        if search_query:
            users = self.db.search_users(search_query)
            
            if users:
                st.markdown(f"**Ditemukan {len(users)} pengguna:**")
                
                for user in users:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        status_icon = "🟢" if user['is_active'] else "🔴"
                        role_icon = "👑" if user['role'] == 'admin' else "👤"
                        st.markdown(f"{role_icon} **{user['username']}** {status_icon}")
                        st.caption(f"📧 {user['email']}")
                    
                    with col2:
                        st.metric("Prediksi", user['total_predictions'])
                    
                    with col3:
                        st.text(f"Bergabung: {user['created_at'][:10]}")
                    
                    st.markdown("---")
            else:
                st.info("Tidak ada pengguna yang ditemukan.")

    def detection_page(self):
        """Text detection page - FIXED VERSION"""
        st.header("🔍 Analisis Teks AI")
        
        # Check if model is available
        if not self.model_handler or not getattr(self.model_handler, 'loaded', False):
            st.error("❌ Model tidak tersedia. Silakan muat ulang halaman.")
            if st.button("🔄 Muat Ulang Model"):
                # Clear cache and reload
                st.cache_resource.clear()
                if 'model_handler' in st.session_state:
                    del st.session_state.model_handler
                st.rerun()
            return
        
        # Input text area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            input_text = st.text_area(
                "Masukkan teks untuk dianalisis:",
                height=300,
                placeholder="Ketik atau paste teks di sini...",
                help="Masukkan teks bahasa Indonesia yang ingin Anda periksa",
                max_chars=5000  # Batasi input untuk mencegah memory issue
            )
            
            # Validate input length
            if input_text and len(input_text.split()) < 10:
                st.warning("⚠️ Untuk hasil yang lebih akurat, masukkan teks minimal 10 kata.")
            
            analyze_button = st.button("🔬 Analisis Teks", type="primary", use_container_width=True)
        
        with col2:
            st.markdown("### 💡 Tips Penggunaan")
            st.info("""
            **Untuk hasil terbaik:**
            - Gunakan teks bahasa Indonesia
            - Minimal 50 kata untuk analisis akurat
            - Maksimal 5000 karakter
            - Teks yang koheren dan lengkap
            """)
            
            st.markdown("### 📊 Interpretasi Hasil")
            st.markdown("""
            - **0-30%**: Sangat mungkin teks manusia
            - **30-60%**: Tidak pasti
            - **60-100%**: Sangat mungkin teks AI
            """)
        
        # Analysis results
        if analyze_button and input_text.strip():
            if not self.model_handler or not getattr(self.model_handler, 'loaded', False):
                st.error("❌ Model belum dimuat. Silakan muat ulang halaman.")
                return
            
            # Validate input
            if len(input_text) > 5000:
                st.error("❌ Teks terlalu panjang. Maksimal 5000 karakter.")
                return
            
            with st.spinner("🔄 Menganalisis teks..."):
                try:
                    # Get prediction with timeout handling
                    result = self.model_handler.predict_text(input_text)
                    st.session_state.analisis_text = result
                    
                    # Check for errors in result
                    if 'error' in result:
                        st.error(f"❌ Error dalam analisis: {result['error']}")
                        return
                
                except Exception as e:
                    st.error(f"❌ Terjadi kesalahan saat analisis: {str(e)}")
                    logger.error(f"Prediction error: {str(e)}")
                    
                    # Try to cleanup and reload model
                    try:
                        if self.model_handler:
                            self.model_handler.cleanup_model()
                        st.cache_resource.clear()
                        if 'model_handler' in st.session_state:
                            del st.session_state.model_handler
                        st.warning("Model akan dimuat ulang pada refresh berikutnya.")
                    except:
                        pass
                    return
            
        # Display results
        if st.session_state.analisis_text != None:
            st.markdown("---")
            st.header("📋 Hasil Analisis")
            
            # Main result
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Probabilitas AI",
                    f"{st.session_state.analisis_text['ai_probability']:.1%}",
                    delta="AI" if st.session_state.analisis_text['is_ai_generated'] else "Manusia",
                    delta_color="inverse" if st.session_state.analisis_text['is_ai_generated'] else "normal"
                )
            
            with col2:
                confidence_color = "🔴" if st.session_state.analisis_text['confidence_level'] == 'high' else "🟠" if st.session_state.analisis_text['confidence_level'] == 'medium' else "🟢"
                st.metric(
                    "Persentase Teks AI",
                    f"{confidence_color} {st.session_state.analisis_text['confidence_level'].upper()}"
                )
            
            with col3:
                st.metric(
                    "Total Bagian",
                    st.session_state.analisis_text['total_chunks']
                )
            
            # Confidence gauge
            col1, col2 = st.columns([1, 1])
            
            with col1:
                gauge_fig = Utils.create_confidence_gauge(st.session_state.analisis_text['ai_probability'])
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col2:
                # Prediction interpretation
                if st.session_state.analisis_text['is_ai_generated']:
                    st.error(f"🚨 **TEKS KEMUNGKINAN DIBUAT AI** ({st.session_state.analisis_text['ai_probability']:.1%})")
                    st.markdown("Teks ini memiliki karakteristik yang mirip dengan hasil generasi AI.")
                else:
                    st.success(f"✅ **TEKS KEMUNGKINAN DIBUAT MANUSIA **")
                    st.markdown("Teks ini memiliki karakteristik penulisan manusia.")
                
                # Highlighted text with character limit
                if st.session_state.analisis_text['highlighted_parts']:
                    st.markdown("### 🖍️ Bagian yang Dicurigai AI:")
                    
                    # Initialize session state for show full highlighted text
                    if 'show_full_highlighted' not in st.session_state:
                        st.session_state.show_full_highlighted = False
                    
                    highlighted_text = Utils.highlight_ai_text(input_text, st.session_state.analisis_text['highlighted_parts'])
                    
                    # Character limit for highlighted text (without HTML tags)
                    HIGHLIGHT_LIMIT = 500  # Adjust this value as needed
                    
                    # Remove HTML tags to count actual text length
                    import re
                    clean_text = re.sub(r'<[^>]+>', '', highlighted_text)
                    
                    if len(clean_text) > HIGHLIGHT_LIMIT and not st.session_state.show_full_highlighted:
                        # Show limited version with natural "Baca Selengkapnya" flow
                        # Find a good cutoff point (try to cut at word boundary)
                        limited_highlighted = highlighted_text[:HIGHLIGHT_LIMIT]
                        
                        # Try to cut at last complete word
                        last_space = limited_highlighted.rfind(' ')
                        if last_space > HIGHLIGHT_LIMIT - 100:  # If space is reasonably close to limit
                            limited_highlighted = limited_highlighted[:last_space]
                        
                        # Create natural flow text with "Baca Selengkapnya" 
                        display_text = f"{limited_highlighted}... [Baca Selengkapnya]"
                        
                        # Add custom CSS for link-like styling
                        st.markdown("""
                        <style>
                        .highlight-text {
                            line-height: 1.6;
                        }
                        .read-more-text {
                            color: #1f77b4;
                            font-weight: 600;
                            text-decoration: none;
                            font-size: 14px;
                            cursor: pointer;
                            margin-left: 8px;
                            padding: 2px 8px;
                            border-radius: 4px;
                            background-color: rgba(31, 119, 180, 0.1);
                            border: 1px solid rgba(31, 119, 180, 0.2);
                            transition: all 0.2s ease;
                        }
                        .read-more-text:hover {
                            background-color: rgba(31, 119, 180, 0.2);
                            border-color: rgba(31, 119, 180, 0.4);
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Display the truncated highlighted text
                        st.markdown(f'<div class="highlight-text">{limited_highlighted}...</div>', unsafe_allow_html=True)
                        
                        # Clean, minimal "Baca Selengkapnya" button styled like a link
                        if st.button("📖 Baca Selengkapnya", key="show_full_highlight", 
                                    help="Tampilkan seluruh bagian yang dicurigai AI"):
                            st.session_state.show_full_highlighted = True
                            st.rerun()
                            
                    else:
                        # Show full text
                        st.markdown(highlighted_text, unsafe_allow_html=True)
                        
                        # Show clean "Tampilkan Lebih Sedikit" if text was previously expanded
                        if st.session_state.show_full_highlighted and len(clean_text) > HIGHLIGHT_LIMIT:
                            st.markdown("---")  # Add separator
                            
                            if st.button("📄 Tampilkan Lebih Sedikit", key="hide_full_highlight",
                                        help="Sembunyikan sebagian teks yang dicurigai AI"):
                                st.session_state.show_full_highlighted = False
                                st.rerun()
                    
            # Detailed analysis
            show_detail = st.checkbox(
                "📊 Tampilkan Analisis Detail", 
                value=st.session_state.show_detailed_analysis,
                key="detailed_analysis_checkbox"
            )
            
            # Update session state
            st.session_state.show_detailed_analysis = show_detail
            
            # Tampilkan analisis detail jika checkbox dicentang
            if st.session_state.show_detailed_analysis:
                if st.session_state.authenticated == True:
                    Utils.display_chunk_analysis(st.session_state.analisis_text['chunk_predictions'])
                else:
                    st.markdown("Anda Harus Login Terlebih Dahulu")
                    if st.button("🚪  Login", use_container_width=True):
                        st.session_state.login = True
                        st.rerun()
        
            if st.session_state.authenticated == False:
            # Save result
                if st.session_state.analisis_text_history != st.session_state.analisis_text:
                    prediction_id = self.db.save_prediction(
                        "2",
                        input_text,
                        st.session_state.analisis_text['ai_probability'],
                        st.session_state.analisis_text['is_ai_generated'],
                        st.session_state.analisis_text['highlighted_parts']
                    )
                    st.session_state.analisis_text_history = st.session_state.analisis_text
            elif st.session_state.authenticated == True:
                if st.session_state.analisis_text_history != st.session_state.analisis_text:
                    # Save result
                    prediction_id = self.db.save_prediction(
                        self.auth.get_current_user_id(),
                        input_text,
                        st.session_state.analisis_text['ai_probability'],
                        st.session_state.analisis_text['is_ai_generated'],
                        st.session_state.analisis_text['highlighted_parts']
                    )
                    st.session_state.analisis_text_history = st.session_state.analisis_text
                
                # Download result
                col1, col2 = st.columns(2)
                with col1:
                    download_data = Utils.create_download_data(st.session_state.analisis_text, input_text)
                    st.download_button(
                        "📥 Download Hasil (JSON)",
                        download_data,
                        file_name=f"ai_detection_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
                
                # with col2:
                #     if st.button("💾 Simpan ke Riwayat"):
                #         st.success(f"✅ Hasil disimpan dengan ID: {prediction_id}")
    
    def dashboard_page(self):
        if st.session_state.authenticated == True:
            """Dashboard page with statistics and visualizations"""
            st.header("📊 Dashboard")
            
            if self.auth.is_admin():
                # Admin dashboard
                stats = self.db.get_system_stats()
                
                st.markdown("### 🏛️ Dashboard Admin")
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total User", stats['total_users'])
                
                with col2:
                    st.metric("User Aktif", stats['active_users'])
                
                with col3:
                    st.metric("Total Prediksi", stats['total_predictions'])
                
                with col4:
                    st.metric("Aktivitas 7 Hari", stats['recent_predictions'])
                
                # Recent users
                st.markdown("### 👥 Pengguna Terbaru")
                recent_users = self.db.get_all_users(5)
                
                for user in recent_users:
                    if user['role'] != 'admin':
                        status_icon = "🟢" if user['is_active'] else "🔴"
                        st.markdown(f"👤 **{user['username']}** {status_icon} - {user['total_predictions']} prediksi")
            
            else:
                # Regular user dashboard
                user_id = self.auth.get_current_user_id()
                user_stats = self.db.get_user_stats(user_id)
                predictions = self.db.get_user_predictions(user_id, limit=100)
                
                # Key metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Analisis", user_stats['total_predictions'])
                
                with col2:
                    st.metric("Teks AI", user_stats['ai_predictions'])
                
                with col3:
                    st.metric("Teks Manusia", user_stats['human_predictions'])
                
                with col4:
                    st.metric("Rata-rata AI Score", f"{user_stats['avg_ai_probability']:.1%}")
                
                if predictions:
                    # Charts
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Pie chart
                        stats_fig = Utils.create_stats_visualization(user_stats)
                        if stats_fig:
                            st.plotly_chart(stats_fig, use_container_width=True)
                    
                    with col2:
                        # History chart
                        history_fig = Utils.create_prediction_history_chart(predictions)
                        if history_fig:
                            st.plotly_chart(history_fig, use_container_width=True)
                    
                    # Recent predictions
                    st.subheader("🕒 Prediksi Terbaru")
                    recent_predictions = predictions[:5]
                    
                    for pred in recent_predictions:
                        Utils.display_prediction_card(pred)
                
                else:
                    st.info("📝 Belum ada data prediksi. Mulai dengan menganalisis teks pertama Anda!")
        else:
            self.main_interface()
    
    def history_page(self):
        if st.session_state.authenticated == True:
            """History page showing all user predictions"""
            st.header("📈 Riwayat Prediksi")
            
            user_id = self.auth.get_current_user_id()
            predictions = self.db.get_user_predictions(user_id, limit=100)
            
            if not predictions:
                st.info("📝 Belum ada riwayat prediksi.")
                return
            
            # Filters
            col1, col2, col3 = st.columns(3)
            
            with col1:
                filter_type = st.selectbox(
                    "Filter berdasarkan:",
                    ["Semua", "Teks AI", "Teks Manusia"]
                )
            
            with col2:
                sort_order = st.selectbox(
                    "Urutkan:",
                    ["Terbaru", "Terlama", "AI Score Tertinggi", "AI Score Terendah"]
                )
            
            with col3:
                show_limit = st.selectbox(
                    "Tampilkan:",
                    [10, 25, 50, 100]
                )
            
            # Filter and sort predictions
            filtered_predictions = predictions
            
            if filter_type == "Teks AI":
                filtered_predictions = [p for p in predictions if p['is_ai_generated']]
            elif filter_type == "Teks Manusia":
                filtered_predictions = [p for p in predictions if not p['is_ai_generated']]
            
            # Sort
            if sort_order == "Terlama":
                filtered_predictions = sorted(filtered_predictions, key=lambda x: x['created_at'])
            elif sort_order == "AI Score Tertinggi":
                filtered_predictions = sorted(filtered_predictions, key=lambda x: x['ai_probability'], reverse=True)
            elif sort_order == "AI Score Terendah":
                filtered_predictions = sorted(filtered_predictions, key=lambda x: x['ai_probability'])
            
            # Limit results
            filtered_predictions = filtered_predictions[:show_limit]
            
            # Export button
            if st.button("📥 Export ke CSV"):
                csv_data = Utils.export_predictions_to_csv(filtered_predictions)
                if csv_data:
                    st.download_button(
                        "Download CSV",
                        csv_data,
                        file_name=f"prediction_history_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
            
            # Display predictions
            st.markdown(f"**Menampilkan {len(filtered_predictions)} dari {len(predictions)} prediksi**")
            
            for pred in filtered_predictions:
                Utils.display_prediction_card(pred)
        else:
            main_interface()
    
    def profile_page(self):
        if st.session_state.authenticated == True:
            """User profile page"""
            st.header("👤 Profil Pengguna")
            
            user_id = self.auth.get_current_user_id()
            user_info = self.db.get_user_info(user_id)
            user_stats = self.db.get_user_stats(user_id)
            
            if user_info:
                username, email, created_at, last_login = user_info
                
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.markdown("### 📋 Informasi Akun")
                    st.info(f"""
                    **Username:** {username}
                    **Email:** {email}
                    **Bergabung:** {created_at}
                    **Login Terakhir:** {last_login or 'Belum pernah'}
                    """)
                
                with col2:
                    st.markdown("### 📊 Statistik Penggunaan")
                    
                    col2_1, col2_2 = st.columns(2)
                    
                    with col2_1:
                        st.metric("Total Analisis", user_stats['total_predictions'])
                        st.metric("Teks AI Terdeteksi", user_stats['ai_predictions'])
                    
                    with col2_2:
                        st.metric("Teks Manusia", user_stats['human_predictions'])
                        st.metric("Rata-rata AI Score", f"{user_stats['avg_ai_probability']:.1%}")
            
            # # Settings
            # st.markdown("---")
            # st.subheader("⚙️ Pengaturan")
            
            # if st.button("🗑️ Hapus Semua Riwayat", type="secondary"):
            #     if st.confirm("Apakah Anda yakin ingin menghapus semua riwayat prediksi?"):
            #         # Implementation would require additional database method
            #         st.warning("Fitur ini akan segera tersedia.")
        else:
            main_interface()
    
    def run(self):
        """Main application runner"""
        try:
            if st.session_state.login == True:
                self.auth.authentication_page()
            else :
                self.main_interface()
        
        except Exception as e:
            st.error(f"❌ Terjadi kesalahan aplikasi: {str(e)}")
            logger.error(f"Application error: {str(e)}")

def main():
    """Main function to run the app"""
    # Create necessary directories
    os.makedirs("database", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    app = AITextDetectorApp()
    app.run()

if __name__ == "__main__":
    main()
