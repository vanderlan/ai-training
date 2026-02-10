import sqlite3
from datetime import datetime
from typing import Optional, List
import random
import string


class Database:
    def __init__(self, db_path: str = "urls.db"):
        self.db_path = db_path
        self.init_db()
    
    def get_connection(self):
        """Create a database connection"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_db(self):
        """Initialize the database schema"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_url TEXT NOT NULL,
                short_code TEXT UNIQUE NOT NULL,
                clicks INTEGER DEFAULT 0,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
    
    def generate_short_code(self, length: int = 6) -> str:
        """Generate a random short code"""
        characters = string.ascii_letters + string.digits
        return ''.join(random.choice(characters) for _ in range(length))
    
    def create_short_url(self, original_url: str, custom_alias: Optional[str] = None) -> dict:
        """Create a shortened URL"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Use custom alias or generate a short code
        short_code = custom_alias if custom_alias else self.generate_short_code()
        
        # Check if short code already exists
        cursor.execute("SELECT * FROM urls WHERE short_code = ?", (short_code,))
        if cursor.fetchone():
            conn.close()
            if custom_alias:
                raise ValueError(f"Custom alias '{custom_alias}' is already taken")
            # If auto-generated code conflicts, try again
            return self.create_short_url(original_url, None)
        
        # Insert the new URL
        cursor.execute(
            "INSERT INTO urls (original_url, short_code) VALUES (?, ?)",
            (original_url, short_code)
        )
        conn.commit()
        
        # Fetch the created record
        url_id = cursor.lastrowid
        cursor.execute("SELECT * FROM urls WHERE id = ?", (url_id,))
        result = cursor.fetchone()
        conn.close()
        
        return dict(result)
    
    def get_url_by_code(self, short_code: str) -> Optional[dict]:
        """Get original URL by short code"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM urls WHERE short_code = ?", (short_code,))
        result = cursor.fetchone()
        conn.close()
        
        return dict(result) if result else None
    
    def increment_clicks(self, short_code: str):
        """Increment click count for a URL"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE urls SET clicks = clicks + 1 WHERE short_code = ?",
            (short_code,)
        )
        conn.commit()
        conn.close()
    
    def get_all_urls(self) -> List[dict]:
        """Get all URLs (for admin/stats)"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM urls ORDER BY created_at DESC")
        results = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in results]
    
    def get_stats(self, short_code: str) -> Optional[dict]:
        """Get statistics for a specific URL"""
        return self.get_url_by_code(short_code)
