import os
import logging
from typing import Dict, Any, Optional
import json

import streamlit as st
from supabase import create_client, Client
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Get Supabase credentials
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

class Auth:
    """Authentication handler for the Legal AI application."""
    
    def __init__(self):
        """Initialize the authentication module with Supabase client."""
        try:
            if not SUPABASE_URL or not SUPABASE_KEY:
                logger.error("Supabase URL and Key must be set in environment variables.")
                raise ValueError("Supabase URL and Key must be set in environment variables.")
            
            self.client: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
            logger.info("Auth module initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Auth module: {e}")
            raise
        
    def register_user(self, email: str, password: str, full_name: str) -> Dict[str, Any]:
        """Register a new user and save their details to the custom_users table."""
        try:
            # Validate password
            if len(password) < 6:
                return {
                    "success": False,
                    "message": "Password should be at least 6 characters."
                }
            
            # Register with Supabase Auth
            logger.info(f"Attempting to register user with email: {email}")
            response = self.client.auth.sign_up({
                "email": email,
                "password": password,
                "options": {
                    "data": {
                        "full_name": full_name
                    }
                }
            })
            
            if not response.user:
                logger.error("Auth registration returned no user")
                return {
                    "success": False,
                    "message": "Registration failed - no user returned from auth service."
                }
                
            user_id = response.user.id
            logger.info(f"User registered successfully in auth with ID: {user_id}")
            
            # Check if we should skip database user creation (like in main.py)
            skip_db_creation = os.getenv("SKIP_DB_USER_CREATION", "false").lower() == "true"
            
            if not skip_db_creation:
                # Insert into custom_users table
                try:
                    logger.info(f"Inserting user into custom_users table: {user_id}")
                    self.client.table("custom_users").insert({
                        "id": user_id,
                        "email": email,
                        "full_name": full_name,
                        "role": "user",
                        "created_at": "now()",
                        "last_login": None
                    }).execute()
                    logger.info(f"Successfully inserted user into custom_users table: {user_id}")
                except Exception as db_error:
                    logger.error(f"Error saving user to custom_users table: {db_error}")
                    # Continue anyway - user is registered in auth
                    logger.info("Continuing with registration as user exists in auth system")
            else:
                logger.info("Skipping database user creation due to SKIP_DB_USER_CREATION environment variable")
            
            return {
                "success": True,
                "user_id": user_id,
                "message": "User registered successfully."
            }
        except Exception as e:
            logger.error(f"Error registering user: {e}")
            return {
                "success": False,
                "message": f"Registration error: {str(e)}"
            }
    
    def login_user(self, email: str, password: str) -> Dict[str, Any]:
        """Authenticate a user with email and password."""
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            
            if response.user:
                # Get additional user data from custom_users table
                user_data = self.get_user_profile(response.user.id)
                
                # If user doesn't exist in custom_users table, create it
                if not user_data:
                    try:
                        logger.info(f"User {response.user.id} exists in auth but not in custom_users table. Creating entry.")
                        self.client.table("custom_users").insert({
                            "id": response.user.id,
                            "email": response.user.email,
                            "full_name": response.user.user_metadata.get('full_name', ''),
                            "role": "user"
                        }).execute()
                        
                        # Fetch the newly created user data
                        user_data = self.get_user_profile(response.user.id)
                    except Exception as e:
                        logger.error(f"Error creating user record during login: {e}")
                        # Use basic user data
                        user_data = {
                            "full_name": response.user.user_metadata.get('full_name', ''),
                            "role": "user"
                        }
                
                # Update last login time
                try:
                    self.client.table("custom_users").update({
                        "last_login": "now()"
                    }).eq("id", response.user.id).execute()
                except Exception as e:
                    logger.error(f"Error updating last login time: {e}")
                
                return {
                    "success": True,
                    "user": {
                        "id": response.user.id,
                        "email": response.user.email,
                        "full_name": user_data.get("full_name", ""),
                        "role": user_data.get("role", "user")
                    },
                    "session": response.session
                }
            else:
                return {
                    "success": False,
                    "message": "Invalid credentials."
                }
        except Exception as e:
            logger.error(f"Error logging in: {e}")
            return {
                "success": False,
                "message": f"Login error: {str(e)}"
            }
    
    def logout_user(self) -> Dict[str, Any]:
        """Log out the current user."""
        try:
            self.client.auth.sign_out()
            return {
                "success": True,
                "message": "Logged out successfully."
            }
        except Exception as e:
            logger.error(f"Error logging out: {e}")
            return {
                "success": False,
                "message": f"Logout error: {str(e)}"
            }
    
    def get_user_profile(self, user_id: str) -> Dict[str, Any]:
        """Get a user's full profile from the custom_users table."""
        try:
            # Use the RPC function if available
            try:
                response = self.client.rpc('get_current_user_profile', {}).execute()
                if response.data:
                    if isinstance(response.data, list) and len(response.data) > 0:
                        return response.data[0]
                    elif isinstance(response.data, dict):
                        return response.data
            except Exception as rpc_error:
                logger.warning(f"RPC get_current_user_profile failed: {rpc_error}, trying direct query")
            
            # Fallback: direct query to custom_users table
            response = self.client.table("custom_users").select("*").eq("id", user_id).single().execute()
            
            if response.data:
                return response.data
            return {}
        except Exception as e:
            logger.error(f"Error fetching user profile: {e}")
            return {}
    
    def is_admin(self, user_id: str) -> bool:
        """Check if a user has admin role."""
        user_data = self.get_user_profile(user_id)
        return user_data.get("role") == "admin"
    
    def has_project_access(self, user_id: str, project_id: str, access_level: str = "read") -> bool:
        """Check if a user has specified access to a project."""
        try:
            # First check if user is admin (admins have all access)
            if self.is_admin(user_id):
                return True
                
            # Check project access
            response = self.client.table("user_project_access")\
                .select("access_level")\
                .eq("user_id", user_id)\
                .eq("project_id", project_id)\
                .single()\
                .execute()
                
            if response.data:
                user_access = response.data["access_level"]
                
                # Check permission level
                if access_level == "read":
                    return True  # All access levels include read
                elif access_level == "write" and user_access in ["write", "admin"]:
                    return True
                elif access_level == "admin" and user_access == "admin":
                    return True
            
            return False
        except Exception as e:
            logger.error(f"Error checking project access: {e}")
            return False

    def fetch_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Fetch a user by email from the custom_users table."""
        try:
            response = self.client.table("custom_users").select("*").eq("email", email).single().execute()
            if response.data:
                return response.data
            return None
        except Exception as e:
            logger.error(f"Error fetching user record: {e}")
            return None

# Initialize auth in session state
def init_auth():
    """Initialize authentication in the session state."""
    if "auth" not in st.session_state:
        st.session_state.auth = Auth()

# Authentication UI components
def login_form():
    """Display login form and handle login logic."""
    init_auth()
    
    with st.form("login_form"):
        st.subheader("Login")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        submit = st.form_submit_button("Login")
        
        if submit:
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                try:
                    result = st.session_state.auth.login_user(email, password)
                    if result["success"]:
                        st.session_state.user = result["user"]
                        st.session_state.authenticated = True
                        st.session_state.access_token = result["session"].access_token if result.get("session") else None
                        st.session_state.auth_state_changed = True  # Set flag to reinitialize app
                        st.success("Login successful!")
                        st.rerun()
                    else:
                        st.error(f"Login failed: {result['message']}")
                except Exception as e:
                    st.error(f"An error occurred during login: {str(e)}")
                    logger.error(f"Login form error: {e}", exc_info=True)

def register_form():
    """Display registration form and handle registration logic."""
    init_auth()
    
    with st.form("register_form"):
        st.subheader("Register")
        full_name = st.text_input("Full Name")
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        password_confirm = st.text_input("Confirm Password", type="password")
        submit = st.form_submit_button("Register")
        
        if submit:
            if not email or not password or not full_name:
                st.error("Please fill out all fields.")
            elif password != password_confirm:
                st.error("Passwords do not match.")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters long.")
            else:
                with st.spinner("Creating your account..."):
                    try:
                        # Set environment variable to skip DB user creation during tests
                        # or when database issues are detected (same as main.py)
                        import os
                        os.environ["SKIP_DB_USER_CREATION"] = "true"
                        
                        # Try authentication registration only
                        result = st.session_state.auth.register_user(email, password, full_name)
                        
                        if result["success"]:
                            st.success("Registration successful! Please log in.")
                            st.session_state.show_login = True
                            st.rerun()
                        else:
                            st.error(result["message"])
                    except Exception as e:
                        st.error(f"Registration error: {str(e)}")
                        logger.error(f"Registration error: {e}", exc_info=True)

def logout():
    """Logout the current user."""
    init_auth()
    result = st.session_state.auth.logout_user()
    if result["success"]:
        # Clear session state
        st.session_state.authenticated = False
        st.session_state.user = None
        st.session_state.access_token = None
        return True
    return False

def auth_required(func):
    """Decorator to require authentication for certain functions."""
    def wrapper(*args, **kwargs):
        if not st.session_state.get("authenticated", False):
            st.error("Authentication required.")
            st.stop()
        return func(*args, **kwargs)
    return wrapper

__all__ = [
    'Auth',
    'init_auth',
    'login_form',
    'register_form',
    'logout',
    'auth_required',
]
