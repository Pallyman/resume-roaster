#!/usr/bin/env python3
"""
Resume Roaster Backend API - Production Ready Version
AI-powered resume analyzer with brutal honesty and viral meme generation
"""

import os
import json
import uuid
import sqlite3
import logging
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List
import base64
import io
from PIL import Image, ImageDraw, ImageFont
import statistics
import traceback
import re

from flask import Flask, request, jsonify, render_template, send_from_directory, send_file, make_response
from flask_cors import CORS
from werkzeug.utils import secure_filename
import PyPDF2
import openai

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
    template_folder="templates",
    static_folder="static"
)

# Configure CORS for production
CORS(app, resources={
    r"/api/*": {
        "origins": ["*"],  # In production, specify your frontend domain
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Configuration
class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    AI_MODEL = os.getenv('AI_MODEL', 'gpt-3.5-turbo')
    MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10MB max file size
    UPLOAD_FOLDER = 'uploads'
    DATABASE_PATH = 'resume_roaster.db'
    ALLOWED_EXTENSIONS = {'pdf', 'txt', 'doc', 'docx'}
    
app.config.from_object(Config)

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('templates', exist_ok=True)
os.makedirs('static', exist_ok=True)

# Data models
@dataclass
class ResumeAnalysis:
    verdict: Dict[str, Any]
    analysis: Dict[str, Any]
    feedback: Dict[str, Any]
    benchmarks: Dict[str, Any]
    analysis_id: str
    created_at: str
    roast_card_url: str = ""
    user_email: str = ""
    is_public: bool = True
    resume_type: str = "general"  # general, tech, creative, executive

# Database initialization
def init_database():
    """Initialize SQLite database with proper schema"""
    try:
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        
        # Create analyses table
        conn.execute('''
            CREATE TABLE IF NOT EXISTS analyses (
                id TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_email TEXT DEFAULT '',
                is_public BOOLEAN DEFAULT 1,
                resume_type TEXT DEFAULT 'general',
                industry TEXT DEFAULT '',
                experience_level TEXT DEFAULT ''
            )
        ''')
        
        # Create index for faster queries
        conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_created_at ON analyses(created_at DESC)
        ''')
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise

# AI Analysis
class ResumeAnalyzer:
    """Handles all AI analysis operations for resumes"""
    
    @staticmethod
    def get_roast_prompt() -> str:
        """Returns the system prompt for resume roasting"""
        return """You are a brutally honest hiring manager who's seen thousands of resumes. 
        You've hired for FAANG companies and startups alike. Your job is to roast resumes 
        with tough love - be harsh but constructive, funny but helpful.

        Analyze the resume and respond EXACTLY in this JSON format:
        {
            "verdict": {
                "decision": "INTERVIEW/PASS",
                "confidence": 1-100,
                "hot_take": "One memorable, roast-style line that's shareable but professional",
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
                "brutal_truth": "The hardest truth they need to hear (be funny but constructive)",
                "actionable_advice": [
                    "Specific action item 1",
                    "Specific action item 2",
                    "Specific action item 3"
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

        Common roast themes:
        - Generic objectives that say nothing
        - Buzzword soup ("synergistic", "leverage", "utilize")
        - Responsibilities instead of achievements
        - No quantifiable impact
        - Poor formatting that hurts readability
        - Spelling/grammar errors
        - Too long (nobody reads page 3)
        - Clichés ("hard worker", "team player", "fast learner")
        
        Be specific, not generic. Reference actual content from their resume."""
    
    @staticmethod
    def analyze_resume(content: str, resume_type: str = "general") -> Dict[str, Any]:
        """Analyze resume content using OpenAI"""
        try:
            if not app.config['OPENAI_API_KEY']:
                logger.error("OpenAI API key not configured")
                return ResumeAnalyzer._get_fallback_analysis()
            
            openai.api_key = app.config['OPENAI_API_KEY']
            
            logger.info(f"Attempting OpenAI API call for {resume_type} resume")
            
            response = openai.ChatCompletion.create(
                model=app.config['AI_MODEL'],
                messages=[
                    {"role": "system", "content": ResumeAnalyzer.get_roast_prompt()},
                    {"role": "user", "content": f"Roast this {resume_type} resume:\n\n{content[:6000]}"}
                ],
                temperature=0.8,  # Higher temp for more creative roasts
                max_tokens=2000
            )
            
            logger.info("OpenAI API call successful")
            
            # Parse JSON response
            analysis_text = response.choices[0].message.content
            
            # Extract JSON
            start = analysis_text.find('{')
            end = analysis_text.rfind('}') + 1
            
            if start != -1 and end > start:
                json_str = analysis_text[start:end]
                analysis_data = json.loads(json_str)
                
                # Validate structure
                required_keys = ['verdict', 'analysis', 'feedback', 'benchmarks']
                if all(key in analysis_data for key in required_keys):
                    logger.info("Analysis data parsed successfully")
                    return analysis_data
            
            logger.warning("Failed to parse AI response, using fallback")
            return ResumeAnalyzer._get_fallback_analysis()
            
        except openai.error.AuthenticationError as e:
            logger.error(f"OpenAI Authentication Error: {e}")
            return ResumeAnalyzer._get_fallback_analysis("Authentication failed - check API key")
        except openai.error.RateLimitError as e:
            logger.error(f"OpenAI Rate Limit Error: {e}")
            return ResumeAnalyzer._get_fallback_analysis("Rate limit exceeded - try again later")
        except Exception as e:
            logger.error(f"AI analysis error: {str(e)}")
            logger.error(traceback.format_exc())
            return ResumeAnalyzer._get_fallback_analysis()
    
    @staticmethod
    def _get_fallback_analysis(error_msg: str = None) -> Dict[str, Any]:
        """Returns a fallback analysis when AI fails"""
        hot_take = error_msg if error_msg else "Your resume reads like it was written by a robot for robots"
        
        return {
            "verdict": {
                "decision": "PASS",
                "confidence": 75,
                "hot_take": hot_take,
                "first_impression": "Generic and forgettable - needs serious work"
            },
            "analysis": {
                "clarity": {
                    "strength": "Contact information is present",
                    "weakness": "Everything else is vague corporate speak"
                },
                "impact": {
                    "strength": "You have work experience",
                    "weakness": "No quantifiable achievements or impact shown"
                },
                "mistakes": [
                    "Too many buzzwords, not enough substance",
                    "No numbers or metrics to prove impact",
                    "Generic objective statement that says nothing"
                ]
            },
            "feedback": {
                "brutal_truth": "This resume could belong to literally anyone - there's nothing memorable here",
                "actionable_advice": [
                    "Replace every responsibility with a quantified achievement",
                    "Delete the objective and write a punchy summary instead",
                    "Cut it down to 1 page - nobody's reading your life story"
                ],
                "encouragement": "You have experience - now make it shine with specifics and impact"
            },
            "benchmarks": {
                "clarity_score": 4,
                "impact_score": 3,
                "formatting_score": 5,
                "overall_score": 4
            }
        }

# File Processing
class FileProcessor:
    """Handles file upload and content extraction"""
    
    @staticmethod
    def allowed_file(filename: str) -> bool:
        """Check if file extension is allowed"""
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']
    
    @staticmethod
    def extract_pdf_text(file_path: str) -> str:
        """Extract text content from PDF file"""
        try:
            text = ""
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        text += page.extract_text() + "\n"
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num}: {e}")
            
            if not text.strip():
                return "Error: Could not extract text from PDF. The file might be image-based or corrupted."
            
            return text[:10000]  # Limit to first 10k chars
            
        except Exception as e:
            logger.error(f"PDF extraction error: {e}")
            return f"Error extracting PDF: {str(e)}"
    
    @staticmethod
    def extract_text_file(file_path: str) -> str:
        """Extract content from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                return file.read()[:10000]
        except Exception as e:
            logger.error(f"Text file extraction error: {e}")
            return f"Error reading text file: {str(e)}"
    
    @staticmethod
    def detect_resume_type(content: str) -> str:
        """Detect the type of resume based on content"""
        content_lower = content.lower()
        
        # Tech keywords
        tech_keywords = ['software', 'developer', 'engineer', 'programming', 'coding', 
                        'javascript', 'python', 'java', 'react', 'api', 'database']
        
        # Creative keywords
        creative_keywords = ['design', 'creative', 'art', 'graphic', 'ui/ux', 'portfolio',
                           'adobe', 'figma', 'sketch', 'illustration']
        
        # Executive keywords
        exec_keywords = ['director', 'vp', 'president', 'executive', 'strategic', 
                        'leadership', 'c-suite', 'board', 'revenue growth']
        
        tech_count = sum(1 for keyword in tech_keywords if keyword in content_lower)
        creative_count = sum(1 for keyword in creative_keywords if keyword in content_lower)
        exec_count = sum(1 for keyword in exec_keywords if keyword in content_lower)
        
        if tech_count > max(creative_count, exec_count):
            return "tech"
        elif creative_count > max(tech_count, exec_count):
            return "creative"
        elif exec_count > 0:
            return "executive"
        else:
            return "general"

# Roast Card Generation
class RoastCardGenerator:
    """Generates viral roast cards from analysis"""
    
    @staticmethod
    def generate_roast_card(analysis_data: Dict[str, Any], analysis_id: str) -> str:
        """Generate shareable roast card image"""
        try:
            # Create card (Instagram story size)
            width, height = 1080, 1920
            img = Image.new('RGB', (width, height), color='#0a0a0a')
            draw = ImageDraw.Draw(img)
            
            # Use default font (production would use custom fonts)
            try:
                title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 80)
                subtitle_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 60)
                body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 45)
                small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 35)
            except:
                # Fallback to default
                title_font = ImageFont.load_default()
                subtitle_font = ImageFont.load_default()
                body_font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # Resume Roaster branding
            draw.text((50, 50), "Resume Roaster", fill='#ff6600', font=title_font)
            draw.text((50, 150), "AI Resume Analysis", fill='#ffffff', font=subtitle_font)
            
            # Decision badge
            decision = analysis_data['verdict']['decision']
            confidence = analysis_data['verdict']['confidence']
            badge_color = '#00ff00' if decision == 'INTERVIEW' else '#ff4444'
            
            # Draw decision badge
            badge_y = 300
            draw.rectangle((50, badge_y, 550, badge_y + 120), fill=badge_color)
            draw.text((70, badge_y + 25), f"{decision} ({confidence}%)", 
                     fill='#000000', font=subtitle_font)
            
            # Hot take (main content)
            hot_take = analysis_data['verdict']['hot_take']
            y_pos = 500
            
            # Add quote marks
            draw.text((50, y_pos - 50), '"', fill='#ff6600', font=title_font)
            
            # Word wrap
            words = hot_take.split()
            lines = []
            current_line = []
            
            for word in words:
                current_line.append(word)
                if len(' '.join(current_line)) > 25:
                    lines.append(' '.join(current_line[:-1]))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Draw hot take
            for line in lines[:5]:  # Max 5 lines
                draw.text((100, y_pos), line, fill='#ffffff', font=body_font)
                y_pos += 80
            
            draw.text((900, y_pos), '"', fill='#ff6600', font=title_font)
            
            # Scores section
            y_pos += 150
            draw.text((50, y_pos), "ROAST SCORES:", fill='#ff6600', font=subtitle_font)
            y_pos += 100
            
            scores = analysis_data['benchmarks']
            score_items = [
                ('Clarity', scores.get('clarity_score', 0)),
                ('Impact', scores.get('impact_score', 0)),
                ('Format', scores.get('formatting_score', 0)),
                ('Overall', scores.get('overall_score', 0))
            ]
            
            for label, score in score_items:
                # Draw score bar
                bar_width = int((score / 10) * 600)
                bar_color = '#00ff00' if score >= 7 else '#ffaa00' if score >= 5 else '#ff4444'
                
                draw.text((50, y_pos), f"{label}:", fill='#ffffff', font=body_font)
                draw.rectangle((250, y_pos + 10, 850, y_pos + 40), outline='#333333', width=2)
                draw.rectangle((250, y_pos + 10, 250 + bar_width, y_pos + 40), fill=bar_color)
                draw.text((870, y_pos), f"{score}/10", fill=bar_color, font=body_font)
                y_pos += 80
            
            # Top mistake
            y_pos += 50
            if analysis_data['analysis']['mistakes']:
                draw.text((50, y_pos), "BIGGEST MISTAKE:", fill='#ff4444', font=small_font)
                y_pos += 50
                mistake = analysis_data['analysis']['mistakes'][0]
                
                # Word wrap mistake
                words = mistake.split()
                lines = []
                current_line = []
                
                for word in words:
                    current_line.append(word)
                    if len(' '.join(current_line)) > 35:
                        lines.append(' '.join(current_line[:-1]))
                        current_line = [word]
                
                if current_line:
                    lines.append(' '.join(current_line))
                
                for line in lines[:2]:
                    draw.text((50, y_pos), line, fill='#ffffff', font=small_font)
                    y_pos += 40
            
            # Footer
            draw.text((50, height - 200), "Get your resume roasted:", fill='#888888', font=body_font)
            draw.text((50, height - 150), "resumeroaster.ai", fill='#ff6600', font=subtitle_font)
            draw.text((50, height - 100), f"#{analysis_id[:8]}", fill='#666666', font=body_font)
            
            # Convert to base64
            img_bytes = io.BytesIO()
            img.save(img_bytes, format='PNG', quality=95)
            img_bytes.seek(0)
            
            img_base64 = base64.b64encode(img_bytes.getvalue()).decode()
            return f"data:image/png;base64,{img_base64}"
            
        except Exception as e:
            logger.error(f"Roast card generation error: {e}")
            return ""

# Database Operations
class Database:
    """Handles all database operations"""
    
    @staticmethod
    def save_analysis(analysis_data: Dict[str, Any], 
                     resume_type: str = "general",
                     user_email: str = "",
                     is_public: bool = True) -> ResumeAnalysis:
        """Save analysis to database"""
        analysis_id = str(uuid.uuid4())
        
        # Generate roast card
        roast_url = RoastCardGenerator.generate_roast_card(analysis_data, analysis_id)
        
        # Create analysis object
        analysis = ResumeAnalysis(
            verdict=analysis_data["verdict"],
            analysis=analysis_data["analysis"],
            feedback=analysis_data["feedback"],
            benchmarks=analysis_data["benchmarks"],
            analysis_id=analysis_id,
            created_at=datetime.now().isoformat(),
            roast_card_url=roast_url,
            user_email=user_email,
            is_public=is_public,
            resume_type=resume_type
        )
        
        # Save to database
        try:
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            conn.execute(
                """INSERT INTO analyses 
                   (id, data, user_email, is_public, resume_type) 
                   VALUES (?, ?, ?, ?, ?)""",
                (analysis_id, json.dumps(asdict(analysis)), user_email, is_public, resume_type)
            )
            conn.commit()
            conn.close()
            
            logger.info(f"Analysis saved successfully with ID: {analysis_id}")
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to save analysis: {e}")
            raise
    
    @staticmethod
    def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve analysis by ID"""
        try:
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.execute(
                'SELECT data FROM analyses WHERE id = ?',
                (analysis_id,)
            )
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return json.loads(row[0])
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to get analysis: {e}")
            return None
    
    @staticmethod
    def get_public_analyses(limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent public analyses for gallery"""
        try:
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            cursor = conn.execute('''
                SELECT data FROM analyses 
                WHERE is_public = 1
                ORDER BY created_at DESC 
                LIMIT ?
            ''', (limit,))
            
            analyses = []
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[0])
                    analyses.append({
                        "decision": data["verdict"]["decision"],
                        "confidence": data["verdict"]["confidence"],
                        "hot_take": data["verdict"]["hot_take"],
                        "overall_score": data["benchmarks"]["overall_score"],
                        "resume_type": data.get("resume_type", "general"),
                        "created_at": data["created_at"]
                    })
                except:
                    continue
            
            conn.close()
            return analyses
            
        except Exception as e:
            logger.error(f"Failed to get public analyses: {e}")
            return []
    
    @staticmethod
    def get_statistics() -> Dict[str, Any]:
        """Calculate platform statistics"""
        try:
            conn = sqlite3.connect(app.config['DATABASE_PATH'])
            
            # Total analyses
            cursor = conn.execute('SELECT COUNT(*) FROM analyses')
            total_analyses = cursor.fetchone()[0]
            
            # Recent analyses for stats
            cursor = conn.execute('''
                SELECT data FROM analyses 
                WHERE created_at > datetime('now', '-30 days')
                ORDER BY created_at DESC 
                LIMIT 500
            ''')
            
            scores = {
                'clarity': [],
                'impact': [],
                'formatting': [],
                'overall': []
            }
            
            interview_count = 0
            total_count = 0
            common_mistakes = {}
            
            for row in cursor.fetchall():
                try:
                    data = json.loads(row[0])
                    benchmarks = data.get('benchmarks', {})
                    
                    scores['clarity'].append(benchmarks.get('clarity_score', 5))
                    scores['impact'].append(benchmarks.get('impact_score', 5))
                    scores['formatting'].append(benchmarks.get('formatting_score', 5))
                    scores['overall'].append(benchmarks.get('overall_score', 5))
                    
                    if data['verdict']['decision'] == 'INTERVIEW':
                        interview_count += 1
                    total_count += 1
                    
                    # Track common mistakes
                    for mistake in data['analysis'].get('mistakes', []):
                        if mistake in common_mistakes:
                            common_mistakes[mistake] += 1
                        else:
                            common_mistakes[mistake] = 1
                    
                except:
                    continue
            
            conn.close()
            
            # Calculate averages
            def safe_mean(lst):
                return round(statistics.mean(lst), 1) if lst else 5.0
            
            # Get top mistakes
            sorted_mistakes = sorted(common_mistakes.items(), key=lambda x: x[1], reverse=True)
            top_mistakes = [mistake for mistake, _ in sorted_mistakes[:5]]
            
            return {
                "total_analyses": total_analyses,
                "interview_rate": round((interview_count / total_count * 100), 1) if total_count > 0 else 0,
                "avg_scores": {
                    "clarity": safe_mean(scores['clarity']),
                    "impact": safe_mean(scores['impact']),
                    "formatting": safe_mean(scores['formatting']),
                    "overall": safe_mean(scores['overall'])
                },
                "top_mistakes": top_mistakes
            }
            
        except Exception as e:
            logger.error(f"Failed to calculate statistics: {e}")
            return {
                "total_analyses": 0,
                "interview_rate": 0,
                "avg_scores": {
                    "clarity": 5.0,
                    "impact": 5.0,
                    "formatting": 5.0,
                    "overall": 5.0
                },
                "top_mistakes": []
            }

# API Routes
@app.route('/')
def index():
    """Serve the main application"""
    try:
        # Try to serve index.html
        if os.path.exists('index.html'):
            return send_file('index.html')
        elif os.path.exists(os.path.join(app.template_folder, 'index.html')):
            return render_template('index.html')
        else:
            # Return a basic page if index.html not found
            return """
            <!DOCTYPE html>
            <html>
            <head>
                <title>Resume Roaster - Backend Running</title>
                <style>
                    body { font-family: Arial, sans-serif; padding: 40px; background: #0a0a0a; color: white; }
                    h1 { color: #ff6600; }
                    .info { background: #1a1a1a; padding: 20px; border-radius: 8px; margin: 20px 0; }
                </style>
            </head>
            <body>
                <h1>Resume Roaster Backend Running</h1>
                <div class="info">
                    <h2>API Endpoints Available:</h2>
                    <ul>
                        <li>POST /api/analyze - Analyze a resume</li>
                        <li>GET /api/gallery - Get public analyses</li>
                        <li>GET /api/stats - Get platform statistics</li>
                        <li>GET /api/analysis/{id} - Get specific analysis</li>
                        <li>GET /health - Health check</li>
                    </ul>
                </div>
            </body>
            </html>
            """
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({"error": "Failed to serve frontend"}), 500

@app.route('/api/analyze', methods=['POST'])
def analyze_resume():
    """Main endpoint for resume analysis"""
    try:
        # Validate request
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        if not FileProcessor.allowed_file(file.filename):
            return jsonify({
                "error": "Invalid file type. Please upload a PDF or text file."
            }), 400
        
        # Get form data
        is_public = request.form.get('is_public', 'true').lower() == 'true'
        user_email = request.form.get('email', '')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{timestamp}_{filename}")
        
        try:
            file.save(file_path)
            logger.info(f"File saved successfully: {file_path}")
        except Exception as e:
            logger.error(f"Failed to save file: {e}")
            return jsonify({"error": f"Failed to save uploaded file: {str(e)}"}), 500
        
        # Extract content
        file_ext = filename.lower().split('.')[-1]
        
        try:
            if file_ext == 'pdf':
                content = FileProcessor.extract_pdf_text(file_path)
            else:
                content = FileProcessor.extract_text_file(file_path)
            
            logger.info(f"Extracted {len(content)} characters from {file_ext} file")
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            os.remove(file_path)
            return jsonify({"error": f"Failed to extract content: {str(e)}"}), 500
        
        # Check for extraction errors
        if content.startswith("Error"):
            os.remove(file_path)
            return jsonify({"error": content}), 400
        
        # Detect resume type
        resume_type = FileProcessor.detect_resume_type(content)
        logger.info(f"Detected resume type: {resume_type}")
        
        # Analyze with AI
        try:
            analysis_data = ResumeAnalyzer.analyze_resume(content, resume_type)
            logger.info("AI analysis completed")
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            os.remove(file_path)
            return jsonify({"error": f"Analysis failed: {str(e)}"}), 500
        
        # Save analysis
        try:
            analysis = Database.save_analysis(
                analysis_data,
                resume_type,
                user_email,
                is_public
            )
            logger.info(f"Analysis saved with ID: {analysis.analysis_id}")
        except Exception as e:
            logger.error(f"Database save failed: {e}")
            os.remove(file_path)
            return jsonify({"error": f"Failed to save analysis: {str(e)}"}), 500
        
        # Clean up file
        try:
            os.remove(file_path)
        except Exception as e:
            logger.warning(f"Failed to delete file: {e}")
        
        # Prepare response
        response_data = asdict(analysis)
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"Unexpected error in analyze endpoint: {e}")
        logger.error(traceback.format_exc())
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

@app.route('/api/gallery', methods=['GET'])
def get_gallery():
    """Get public resume analyses for gallery"""
    try:
        analyses = Database.get_public_analyses()
        return jsonify(analyses)
    except Exception as e:
        logger.error(f"Gallery endpoint error: {e}")
        return jsonify({"error": "Failed to load gallery"}), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get platform statistics"""
    try:
        stats = Database.get_statistics()
        
        # Add some fun stats
        stats['fun_facts'] = [
            f"{stats['total_analyses']} resumes roasted and counting!",
            f"Only {stats['interview_rate']}% made it to interview stage",
            f"Average clarity score: {stats['avg_scores']['clarity']}/10",
            "Most common mistake: Too many buzzwords"
        ]
        
        return jsonify(stats)
        
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}")
        return jsonify({"error": "Failed to load statistics"}), 500

@app.route('/api/analysis/<analysis_id>', methods=['GET'])
def get_analysis(analysis_id):
    """Get specific analysis by ID"""
    try:
        analysis = Database.get_analysis(analysis_id)
        
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404
        
        return jsonify(analysis)
        
    except Exception as e:
        logger.error(f"Get analysis error: {e}")
        return jsonify({"error": "Failed to retrieve analysis"}), 500

@app.route('/api/roast-card/<analysis_id>', methods=['GET'])
def get_roast_card(analysis_id):
    """Get roast card for specific analysis"""
    try:
        analysis = Database.get_analysis(analysis_id)
        
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404
        
        roast_url = analysis.get('roast_card_url', '')
        
        if not roast_url:
            # Regenerate if missing
            roast_url = RoastCardGenerator.generate_roast_card(analysis, analysis_id)
        
        return jsonify({"roast_url": roast_url})
        
    except Exception as e:
        logger.error(f"Get roast card error: {e}")
        return jsonify({"error": "Failed to get roast card"}), 500

@app.route('/api/download-roast/<analysis_id>', methods=['GET'])
def download_roast(analysis_id):
    """Download roast card as image file"""
    try:
        analysis = Database.get_analysis(analysis_id)
        
        if not analysis:
            return jsonify({"error": "Analysis not found"}), 404
        
        # Generate fresh roast card
        roast_base64 = RoastCardGenerator.generate_roast_card(analysis, analysis_id)
        
        # Convert base64 to image
        if roast_base64.startswith('data:image/png;base64,'):
            image_data = base64.b64decode(roast_base64.split(',')[1])
            
            # Create response with image
            response = make_response(image_data)
            response.headers['Content-Type'] = 'image/png'
            response.headers['Content-Disposition'] = f'attachment; filename=resume_roast_{analysis_id[:8]}.png'
            
            return response
        else:
            return jsonify({"error": "Failed to generate roast card"}), 500
            
    except Exception as e:
        logger.error(f"Download roast error: {e}")
        return jsonify({"error": "Failed to download roast card"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check database connection
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        conn.execute('SELECT 1')
        conn.close()
        db_status = "healthy"
    except:
        db_status = "unhealthy"
    
    return jsonify({
        "status": "healthy",
        "service": "Resume Roaster API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "database": db_status,
            "ai_enabled": bool(app.config['OPENAI_API_KEY']),
            "file_upload": True
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large. Maximum size is 10MB"}), 413

# Initialize and run
if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Get port from environment
    port = int(os.getenv("PORT", 5000))
    
    # Log startup information
    logger.info(f"Starting Resume Roaster Backend on port {port}")
    logger.info(f"OpenAI API Key configured: {bool(app.config.get('OPENAI_API_KEY'))}")
    logger.info(f"Database path: {app.config['DATABASE_PATH']}")
    
    # Run server
    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )
