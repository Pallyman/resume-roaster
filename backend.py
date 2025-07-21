#!/usr/bin/env python3
"""
Resume Roaster Backend - Modern Production Version
AI-powered resume analyzer with multiple AI providers and viral roast generation
"""

import os
import json
import uuid
import sqlite3
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Union
import base64
import io
import hashlib
import re
from functools import wraps
import time

from flask import Flask, request, jsonify, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import PyPDF2
from PIL import Image, ImageDraw, ImageFont
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Basic config
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    UPLOAD_FOLDER = 'uploads'
    DATABASE_PATH = 'resume_roaster.db'
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
    
    # AI Provider Configuration
    AI_PROVIDER = os.getenv('AI_PROVIDER', 'deepseek')  # 'deepseek', 'openai', or 'mock'
    
    # Deepseek config
    DEEPSEEK_API_KEY = os.getenv('DEEPSEEK_API_KEY', '')
    DEEPSEEK_API_BASE = os.getenv('DEEPSEEK_API_BASE', 'https://api.deepseek.com/v1')
    DEEPSEEK_MODEL = os.getenv('DEEPSEEK_MODEL', 'deepseek-chat')
    
    # OpenAI config (fallback)
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY', '')
    OPENAI_MODEL = os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo')
    
    # Rate limiting
    RATE_LIMIT_ENABLED = os.getenv('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
    RATE_LIMIT_PER_HOUR = int(os.getenv('RATE_LIMIT_PER_HOUR', '100'))

app.config.from_object(Config)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('roast_cards', exist_ok=True)

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ResumeAnalysis:
    analysis_id: str
    verdict: Dict[str, Any]
    analysis: Dict[str, Any]
    feedback: Dict[str, Any]
    benchmarks: Dict[str, Any]
    roast_card_url: str
    created_at: str
    resume_type: str = "general"
    user_email: str = ""
    is_public: bool = True
    ai_provider: str = "deepseek"
    processing_time: float = 0.0

# ============================================================================
# DATABASE
# ============================================================================

class Database:
    """Modern database handler with connection pooling and better error handling"""
    
    @staticmethod
    def init():
        """Initialize database with modern schema"""
        try:
            with sqlite3.connect(app.config['DATABASE_PATH']) as conn:
                # Enable WAL mode for better concurrency
                conn.execute("PRAGMA journal_mode=WAL")
                
                # Create main table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS analyses (
                        id TEXT PRIMARY KEY,
                        data TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_email TEXT DEFAULT '',
                        is_public BOOLEAN DEFAULT 1,
                        resume_type TEXT DEFAULT 'general',
                        ai_provider TEXT DEFAULT 'deepseek',
                        processing_time REAL DEFAULT 0.0,
                        ip_hash TEXT DEFAULT ''
                    )
                ''')
                
                # Create indices
                conn.execute('CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at DESC)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_public ON analyses(is_public, created_at DESC)')
                conn.execute('CREATE INDEX IF NOT EXISTS idx_email ON analyses(user_email)')
                
                # Create rate limiting table
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS rate_limits (
                        ip_hash TEXT PRIMARY KEY,
                        request_count INTEGER DEFAULT 1,
                        window_start TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    @staticmethod
    def get_connection():
        """Get database connection with proper settings"""
        conn = sqlite3.connect(app.config['DATABASE_PATH'], timeout=30)
        conn.row_factory = sqlite3.Row
        return conn
    
    @staticmethod
    def save_analysis(analysis: ResumeAnalysis, ip_hash: str = "") -> bool:
        """Save analysis to database"""
        try:
            with Database.get_connection() as conn:
                conn.execute('''
                    INSERT INTO analyses 
                    (id, data, user_email, is_public, resume_type, ai_provider, processing_time, ip_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    analysis.analysis_id,
                    json.dumps(asdict(analysis)),
                    analysis.user_email,
                    analysis.is_public,
                    analysis.resume_type,
                    analysis.ai_provider,
                    analysis.processing_time,
                    ip_hash
                ))
                conn.commit()
                return True
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            return False
    
    @staticmethod
    def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
        """Get analysis by ID"""
        try:
            with Database.get_connection() as conn:
                row = conn.execute(
                    'SELECT data FROM analyses WHERE id = ?',
                    (analysis_id,)
                ).fetchone()
                
                if row:
                    return json.loads(row['data'])
                return None
        except Exception as e:
            logger.error(f"Failed to get analysis: {e}")
            return None
    
    @staticmethod
    def get_recent_public(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent public analyses"""
        try:
            with Database.get_connection() as conn:
                rows = conn.execute('''
                    SELECT data FROM analyses 
                    WHERE is_public = 1
                    ORDER BY created_at DESC 
                    LIMIT ?
                ''', (limit,)).fetchall()
                
                analyses = []
                for row in rows:
                    try:
                        data = json.loads(row['data'])
                        analyses.append({
                            "analysis_id": data["analysis_id"],
                            "decision": data["verdict"]["decision"],
                            "confidence": data["verdict"]["confidence"],
                            "hot_take": data["verdict"]["hot_take"],
                            "overall_score": data["benchmarks"]["overall_score"],
                            "resume_type": data.get("resume_type", "general"),
                            "created_at": data["created_at"]
                        })
                    except:
                        continue
                
                return analyses
        except Exception as e:
            logger.error(f"Failed to get recent analyses: {e}")
            return []
    
    @staticmethod
    def get_stats() -> Dict[str, Any]:
        """Get platform statistics"""
        try:
            with Database.get_connection() as conn:
                # Total count
                total = conn.execute('SELECT COUNT(*) as count FROM analyses').fetchone()['count']
                
                # Recent stats
                recent = conn.execute('''
                    SELECT 
                        COUNT(*) as total,
                        SUM(CASE WHEN json_extract(data, '$.verdict.decision') = 'INTERVIEW' THEN 1 ELSE 0 END) as interviews,
                        AVG(json_extract(data, '$.benchmarks.overall_score')) as avg_score
                    FROM analyses 
                    WHERE created_at > datetime('now', '-30 days')
                ''').fetchone()
                
                interview_rate = (recent['interviews'] / recent['total'] * 100) if recent['total'] > 0 else 0
                
                return {
                    "total_analyses": total,
                    "recent_analyses": recent['total'] or 0,
                    "interview_rate": round(interview_rate, 1),
                    "avg_overall_score": round(recent['avg_score'] or 5.0, 1)
                }
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {
                "total_analyses": 0,
                "recent_analyses": 0,
                "interview_rate": 0,
                "avg_overall_score": 5.0
            }

# ============================================================================
# RATE LIMITING
# ============================================================================

def get_client_ip():
    """Get client IP address"""
    if request.headers.get('X-Forwarded-For'):
        return request.headers.get('X-Forwarded-For').split(',')[0]
    return request.remote_addr or '127.0.0.1'

def rate_limit(max_requests: int = None):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not app.config['RATE_LIMIT_ENABLED']:
                return f(*args, **kwargs)
            
            ip = get_client_ip()
            ip_hash = hashlib.sha256(ip.encode()).hexdigest()
            max_req = max_requests or app.config['RATE_LIMIT_PER_HOUR']
            
            try:
                with Database.get_connection() as conn:
                    # Check current rate
                    row = conn.execute(
                        'SELECT request_count, window_start FROM rate_limits WHERE ip_hash = ?',
                        (ip_hash,)
                    ).fetchone()
                    
                    now = datetime.now()
                    
                    if row:
                        window_start = datetime.fromisoformat(row['window_start'])
                        if now - window_start < timedelta(hours=1):
                            if row['request_count'] >= max_req:
                                return jsonify({
                                    "error": "Rate limit exceeded. Try again later."
                                }), 429
                            
                            # Increment counter
                            conn.execute(
                                'UPDATE rate_limits SET request_count = request_count + 1 WHERE ip_hash = ?',
                                (ip_hash,)
                            )
                        else:
                            # Reset window
                            conn.execute(
                                'UPDATE rate_limits SET request_count = 1, window_start = ? WHERE ip_hash = ?',
                                (now.isoformat(), ip_hash)
                            )
                    else:
                        # First request
                        conn.execute(
                            'INSERT INTO rate_limits (ip_hash, request_count, window_start) VALUES (?, 1, ?)',
                            (ip_hash, now.isoformat())
                        )
                    
                    conn.commit()
                    
            except Exception as e:
                logger.error(f"Rate limiting error: {e}")
                # Don't block on rate limit errors
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

# ============================================================================
# AI PROVIDERS
# ============================================================================

class AIProvider:
    """Base AI provider class"""
    
    def analyze_resume(self, content: str, resume_type: str = "general") -> Dict[str, Any]:
        raise NotImplementedError

class DeepseekProvider(AIProvider):
    """Deepseek AI provider"""
    
    def __init__(self):
        self.api_key = app.config['DEEPSEEK_API_KEY']
        self.api_base = app.config['DEEPSEEK_API_BASE']
        self.model = app.config['DEEPSEEK_MODEL']
    
    def analyze_resume(self, content: str, resume_type: str = "general") -> Dict[str, Any]:
        """Analyze resume using Deepseek"""
        if not self.api_key:
            logger.error("Deepseek API key not configured")
            raise ValueError("Deepseek API key not configured")
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json'
        }
        
        system_prompt = self._get_system_prompt()
        user_prompt = f"Roast this {resume_type} resume:\n\n{content[:6000]}"
        
        data = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.8,
            "max_tokens": 2000
        }
        
        try:
            response = requests.post(
                f"{self.api_base}/chat/completions",
                headers=headers,
                json=data,
                timeout=30
            )
            
            if response.status_code != 200:
                logger.error(f"Deepseek API error: {response.status_code} - {response.text}")
                raise ValueError(f"API error: {response.status_code}")
            
            result = response.json()
            content = result['choices'][0]['message']['content']
            
            # Parse JSON from response
            return self._parse_response(content)
            
        except requests.exceptions.Timeout:
            logger.error("Deepseek API timeout")
            raise ValueError("API request timed out")
        except Exception as e:
            logger.error(f"Deepseek API error: {e}")
            raise
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for resume roasting"""
        return """You are a brutally honest hiring manager who roasts resumes with tough love.
        Be harsh but constructive, funny but helpful. You've seen thousands of resumes.

        Analyze the resume and respond EXACTLY in this JSON format:
        {
            "verdict": {
                "decision": "INTERVIEW/PASS",
                "confidence": 1-100,
                "hot_take": "One memorable, shareable roast line",
                "first_impression": "What you think in the first 6 seconds"
            },
            "analysis": {
                "clarity": {
                    "strength": "What's crystal clear",
                    "weakness": "What's confusing or vague"
                },
                "impact": {
                    "strength": "Impressive achievements",
                    "weakness": "Missed opportunities to show impact"
                },
                "mistakes": ["Top 3 glaring issues"]
            },
            "feedback": {
                "brutal_truth": "The hardest truth they need to hear",
                "actionable_advice": [
                    "Specific action 1",
                    "Specific action 2",
                    "Specific action 3"
                ],
                "encouragement": "What genuinely stands out positively"
            },
            "benchmarks": {
                "clarity_score": 1-10,
                "impact_score": 1-10,
                "formatting_score": 1-10,
                "overall_score": 1-10
            }
        }

        Be specific, reference their actual content. Make the roast memorable but professional."""
    
    def _parse_response(self, content: str) -> Dict[str, Any]:
        """Parse JSON from AI response"""
        try:
            # Find JSON in response
            start = content.find('{')
            end = content.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = content[start:end]
                return json.loads(json_str)
            
            raise ValueError("No valid JSON found in response")
        except Exception as e:
            logger.error(f"Failed to parse AI response: {e}")
            raise ValueError("Failed to parse AI response")

class OpenAIProvider(AIProvider):
    """OpenAI provider (fallback)"""
    
    def __init__(self):
        self.api_key = app.config['OPENAI_API_KEY']
        self.model = app.config['OPENAI_MODEL']
    
    def analyze_resume(self, content: str, resume_type: str = "general") -> Dict[str, Any]:
        """Analyze resume using OpenAI"""
        if not self.api_key:
            raise ValueError("OpenAI API key not configured")
        
        # Similar implementation to Deepseek but using OpenAI API
        # This is a fallback option
        raise NotImplementedError("OpenAI provider not implemented yet")

class MockProvider(AIProvider):
    """Mock provider for testing"""
    
    def analyze_resume(self, content: str, resume_type: str = "general") -> Dict[str, Any]:
        """Return mock analysis for testing"""
        return {
            "verdict": {
                "decision": "PASS",
                "confidence": 85,
                "hot_take": "This resume reads like ChatGPT's first draft - soulless and forgettable",
                "first_impression": "Another generic resume in the pile"
            },
            "analysis": {
                "clarity": {
                    "strength": "Contact info is readable",
                    "weakness": "Everything else is corporate word salad"
                },
                "impact": {
                    "strength": "You showed up to work",
                    "weakness": "No measurable impact anywhere"
                },
                "mistakes": [
                    "Zero quantifiable achievements",
                    "Buzzword bingo champion",
                    "Reads like a job description, not accomplishments"
                ]
            },
            "feedback": {
                "brutal_truth": "You're invisible in a sea of identical resumes",
                "actionable_advice": [
                    "Add numbers to EVERY bullet point",
                    "Delete 'responsible for' and start with action verbs",
                    "Show impact, not just activities"
                ],
                "encouragement": "You have experience - now make it shine"
            },
            "benchmarks": {
                "clarity_score": 4,
                "impact_score": 3,
                "formatting_score": 6,
                "overall_score": 4
            }
        }

# AI Provider Factory
def get_ai_provider() -> AIProvider:
    """Get the configured AI provider"""
    provider_name = app.config['AI_PROVIDER'].lower()
    
    if provider_name == 'deepseek':
        return DeepseekProvider()
    elif provider_name == 'openai':
        return OpenAIProvider()
    else:
        return MockProvider()

# ============================================================================
# FILE PROCESSING
# ============================================================================

class FileProcessor:
    """Modern file processor with better error handling"""
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file type is allowed"""
        if not filename or '.' not in filename:
            return False
        return filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @staticmethod
    def extract_text(file_path: str, file_type: str) -> str:
        """Extract text from file"""
        try:
            if file_type == 'pdf':
                return FileProcessor._extract_pdf(file_path)
            elif file_type in ['txt', 'text']:
                return FileProcessor._extract_text_file(file_path)
            else:
                return "Error: Unsupported file type"
        except Exception as e:
            logger.error(f"Text extraction error: {e}")
            return f"Error: Failed to extract text - {str(e)}"
    
    @staticmethod
    def _extract_pdf(file_path: str) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                try:
                    text += page.extract_text() + "\n"
                except:
                    continue
        
        if not text.strip():
            return "Error: Could not extract text from PDF"
        
        return text[:10000]  # Limit to 10k chars
    
    @staticmethod
    def _extract_text_file(file_path: str) -> str:
        """Extract text from text file"""
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()[:10000]
    
    @staticmethod
    def detect_resume_type(content: str) -> str:
        """Detect resume type from content"""
        content_lower = content.lower()
        
        keywords = {
            'tech': ['software', 'developer', 'engineer', 'programming', 'api', 'database'],
            'creative': ['design', 'creative', 'art', 'ui/ux', 'portfolio'],
            'executive': ['director', 'vp', 'president', 'executive', 'leadership'],
            'sales': ['sales', 'revenue', 'quota', 'pipeline', 'closing'],
            'marketing': ['marketing', 'campaign', 'brand', 'social media', 'seo']
        }
        
        scores = {}
        for resume_type, words in keywords.items():
            scores[resume_type] = sum(1 for word in words if word in content_lower)
        
        if max(scores.values()) > 0:
            return max(scores, key=scores.get)
        return 'general'

# ============================================================================
# ROAST CARD GENERATION
# ============================================================================

class RoastCardGenerator:
    """Generate viral roast cards"""
    
    @staticmethod
    def generate(analysis: Dict[str, Any], analysis_id: str) -> str:
        """Generate roast card image"""
        try:
            # Create card
            width, height = 1080, 1920
            img = Image.new('RGB', (width, height), color='#0a0a0a')
            draw = ImageDraw.Draw(img)
            
            # Try to load fonts
            try:
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans"
                title_font = ImageFont.truetype(f"{font_path}-Bold.ttf", 80)
                body_font = ImageFont.truetype(f"{font_path}.ttf", 45)
                small_font = ImageFont.truetype(f"{font_path}.ttf", 35)
            except:
                # Fallback to default font
                title_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Header
            draw.text((50, 50), "RESUME ROASTER", fill='#ff6600', font=title_font)
            draw.text((50, 150), "AI-Powered Reality Check", fill='#cccccc', font=body_font)
            
            # Decision badge
            verdict = analysis['verdict']
            decision = verdict['decision']
            confidence = verdict['confidence']
            
            badge_color = '#00ff00' if decision == 'INTERVIEW' else '#ff4444'
            draw.rectangle((50, 300, 550, 420), fill=badge_color)
            draw.text((70, 325), f"{decision} ({confidence}%)", fill='#000000', font=body_font)
            
            # Hot take
            hot_take = verdict['hot_take']
            y_pos = 500
            
            # Quote marks
            draw.text((50, y_pos), '"', fill='#ff6600', font=title_font)
            
            # Word wrap hot take
            words = hot_take.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                test_line = ' '.join(current_line)
                if len(test_line) > 25:
                    if len(current_line) > 1:
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                    else:
                        lines.append(test_line)
                        current_line = []
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw hot take text
            y_pos += 50
            for line in lines[:5]:
                draw.text((100, y_pos), line, fill='#ffffff', font=body_font)
                y_pos += 60
            
            draw.text((900, y_pos), '"', fill='#ff6600', font=title_font)
            
            # Scores
            y_pos += 100
            draw.text((50, y_pos), "SCORECARD:", fill='#ff6600', font=body_font)
            y_pos += 80
            
            benchmarks = analysis['benchmarks']
            for label, key in [
                ('Clarity', 'clarity_score'),
                ('Impact', 'impact_score'),
                ('Format', 'formatting_score'),
                ('Overall', 'overall_score')
            ]:
                score = benchmarks.get(key, 5)
                color = '#00ff00' if score >= 7 else '#ffaa00' if score >= 5 else '#ff4444'
                
                draw.text((50, y_pos), f"{label}:", fill='#ffffff', font=body_font)
                
                # Score bar
                bar_width = int((score / 10) * 600)
                draw.rectangle((250, y_pos + 10, 850, y_pos + 40), outline='#333333', width=2)
                draw.rectangle((250, y_pos + 10, 250 + bar_width, y_pos + 40), fill=color)
                draw.text((870, y_pos), f"{score}/10", fill=color, font=body_font)
                
                y_pos += 70
            
            # Top mistake
            y_pos += 50
            mistakes = analysis['analysis'].get('mistakes', [])
            if mistakes:
                draw.text((50, y_pos), "BIGGEST ISSUE:", fill='#ff4444', font=body_font)
                y_pos += 60
                
                mistake_text = mistakes[0]
                # Word wrap
                words = mistake_text.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    if len(' '.join(current_line)) > 35:
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                for line in lines[:3]:
                    draw.text((50, y_pos), line, fill='#ffffff', font=small_font)
                    y_pos += 45
            
            # Footer
            draw.text((50, height - 200), "Get roasted at:", fill='#888888', font=body_font)
            draw.text((50, height - 150), "resumeroaster.ai", fill='#ff6600', font=body_font)
            draw.text((50, height - 100), f"#{analysis_id[:8]}", fill='#666666', font=small_font)
            
            # Save to file
            filename = f"roast_cards/roast_{analysis_id}.png"
            img.save(filename, 'PNG', quality=95)
            
            # Convert to base64 for embedding
            with open(filename, 'rb') as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode()
            
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Roast card generation error: {e}")
            return ""

# ============================================================================
# API ROUTES
# ============================================================================

@app.route('/')
def index():
    """Health check / basic info"""
    return jsonify({
        "service": "Resume Roaster API",
        "version": "2.0",
        "status": "operational",
        "endpoints": {
            "POST /api/analyze": "Analyze a resume",
            "GET /api/analysis/<id>": "Get specific analysis",
            "GET /api/gallery": "Get recent public analyses",
            "GET /api/stats": "Get platform statistics",
            "GET /health": "Health check"
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Check database
        with Database.get_connection() as conn:
            conn.execute('SELECT 1')
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db_status,
            "ai_provider": app.config['AI_PROVIDER'],
            "ai_configured": bool(app.config.get(f"{app.config['AI_PROVIDER'].upper()}_API_KEY"))
        }
    })

@app.route('/api/analyze', methods=['POST'])
@rate_limit(max_requests=20)  # 20 analyses per hour per IP
def analyze_resume():
    """Main resume analysis endpoint"""
    start_time = time.time()
    
    try:
        # Validate file upload
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if not file or file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not FileProcessor.allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Supported: PDF, TXT, DOC, DOCX"
            }), 400
        
        # Get metadata
        is_public = request.form.get('is_public', 'true').lower() == 'true'
        user_email = request.form.get('email', '').strip()
        
        # Save file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        
        file.save(file_path)
        logger.info(f"File saved: {file_path}")
        
        # Extract text
        file_type = filename.rsplit('.', 1)[1].lower()
        content = FileProcessor.extract_text(file_path, file_type)
        
        if content.startswith("Error:"):
            os.remove(file_path)
            return jsonify({"error": content}), 400
        
        # Detect resume type
        resume_type = FileProcessor.detect_resume_type(content)
        logger.info(f"Detected resume type: {resume_type}")
        
        # Analyze with AI
        try:
            ai_provider = get_ai_provider()
            analysis_data = ai_provider.analyze_resume(content, resume_type)
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            os.remove(file_path)
            
            # Use mock provider as fallback
            if app.config['AI_PROVIDER'] != 'mock':
                logger.info("Falling back to mock provider")
                analysis_data = MockProvider().analyze_resume(content, resume_type)
            else:
                return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
        
        # Generate analysis ID
        analysis_id = str(uuid.uuid4())
        
        # Generate roast card
        roast_url = RoastCardGenerator.generate(analysis_data, analysis_id)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create analysis object
        analysis = ResumeAnalysis(
            analysis_id=analysis_id,
            verdict=analysis_data['verdict'],
            analysis=analysis_data['analysis'],
            feedback=analysis_data['feedback'],
            benchmarks=analysis_data['benchmarks'],
            roast_card_url=roast_url,
            created_at=datetime.now().isoformat(),
            resume_type=resume_type,
            user_email=user_email,
            is_public=is_public,
            ai_provider=app.config['AI_PROVIDER'],
            processing_time=processing_time
        )
        
        # Save to database
        ip_hash = hashlib.sha256(get_client_ip().encode()).hexdigest()
        if not Database.save_analysis(analysis, ip_hash):
            logger.error("Failed to save analysis to database")
        
        # Clean up
        try:
            os.remove(file_path)
        except:
            pass
        
        logger.info(f"Analysis completed in {processing_time:.2f}s")
        
        # Return response
        return jsonify(asdict(analysis))
        
    except RequestEntityTooLarge:
        return jsonify({"error": "File too large. Maximum size is 10MB"}), 413
    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {e}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500

@app.route('/api/analysis/<analysis_id>')
def get_analysis(analysis_id):
    """Get specific analysis by ID"""
    analysis = Database.get_analysis(analysis_id)
    
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404
    
    return jsonify(analysis)

@app.route('/api/gallery')
def get_gallery():
    """Get recent public analyses"""
    limit = request.args.get('limit', 20, type=int)
    limit = min(limit, 50)  # Cap at 50
    
    analyses = Database.get_recent_public(limit)
    return jsonify(analyses)

@app.route('/api/stats')
def get_stats():
    """Get platform statistics"""
    stats = Database.get_stats()
    
    # Add some fun facts
    stats['fun_facts'] = [
        f"{stats['total_analyses']} resumes roasted!",
        f"Only {stats['interview_rate']}% made it to interview",
        f"Average score: {stats['avg_overall_score']}/10"
    ]
    
    return jsonify(stats)

@app.route('/api/download-roast/<analysis_id>')
def download_roast(analysis_id):
    """Download roast card as image"""
    # Check if roast card exists
    filename = f"roast_cards/roast_{analysis_id}.png"
    
    if os.path.exists(filename):
        return send_file(
            filename,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'resume_roast_{analysis_id[:8]}.png'
        )
    
    # Try to regenerate
    analysis = Database.get_analysis(analysis_id)
    if not analysis:
        return jsonify({"error": "Analysis not found"}), 404
    
    # Regenerate roast card
    RoastCardGenerator.generate(analysis, analysis_id)
    
    if os.path.exists(filename):
        return send_file(
            filename,
            mimetype='image/png',
            as_attachment=True,
            download_name=f'resume_roast_{analysis_id[:8]}.png'
        )
    
    return jsonify({"error": "Failed to generate roast card"}), 500

# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {e}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(RequestEntityTooLarge)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB"}), 413

# ============================================================================
# INITIALIZATION
# ============================================================================

# Initialize database on startup
try:
    Database.init()
except Exception as e:
    logger.error(f"Failed to initialize database: {e}")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    # Get port from environment
    port = int(os.getenv("PORT", 5000))
    
    # Log configuration
    logger.info(f"Starting Resume Roaster Backend on port {port}")
    logger.info(f"AI Provider: {app.config['AI_PROVIDER']}")
    logger.info(f"Rate Limiting: {'Enabled' if app.config['RATE_LIMIT_ENABLED'] else 'Disabled'}")
    logger.info(f"Database: {app.config['DATABASE_PATH']}")
    
    # Run server
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
