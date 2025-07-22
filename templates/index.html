<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Resume Roaster - Get Your Resume Roasted by AI</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0a;
            color: #ffffff;
            line-height: 1.6;
            overflow-x: hidden;
        }
        
        /* Header */
        header {
            padding: 20px 0;
            text-align: center;
            background: rgba(0, 0, 0, 0.5);
            backdrop-filter: blur(10px);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        
        .logo {
            font-size: 2em;
            font-weight: bold;
            background: linear-gradient(135deg, #ff6600 0%, #ff9900 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        /* Hero Section */
        .hero {
            text-align: center;
            padding: 80px 20px;
            max-width: 800px;
            margin: 0 auto;
        }
        
        .hero h1 {
            font-size: 3em;
            margin-bottom: 20px;
            background: linear-gradient(135deg, #ff6600 0%, #ff9900 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        
        .hero p {
            font-size: 1.2em;
            color: #ccc;
            margin-bottom: 40px;
        }
        
        /* Upload Section */
        .upload-section {
            max-width: 600px;
            margin: 0 auto;
            padding: 40px;
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .file-upload-wrapper {
            position: relative;
            overflow: hidden;
            display: inline-block;
            width: 100%;
        }
        
        .file-upload-input {
            position: absolute;
            font-size: 100px;
            opacity: 0;
            right: 0;
            top: 0;
            cursor: pointer;
        }
        
        .file-upload-label {
            display: block;
            padding: 20px;
            background: #ff6600;
            color: white;
            border-radius: 10px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: bold;
            font-size: 1.1em;
        }
        
        .file-upload-label:hover {
            background: #ff8800;
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(255, 102, 0, 0.3);
        }
        
        .file-info {
            margin-top: 20px;
            padding: 15px;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            display: none;
        }
        
        .analyze-button {
            width: 100%;
            padding: 15px;
            background: #00cc44;
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s ease;
            margin-top: 20px;
            display: none;
        }
        
        .analyze-button:hover {
            background: #00ff55;
            transform: translateY(-2px);
        }
        
        .analyze-button:disabled {
            background: #666;
            cursor: not-allowed;
            transform: none;
        }
        
        /* Loading State */
        .loading {
            display: none;
            text-align: center;
            padding: 40px;
        }
        
        .spinner {
            border: 4px solid #333;
            border-top: 4px solid #ff6600;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px;
        }
        
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        /* Results Section */
        .results-section {
            display: none;
            max-width: 1000px;
            margin: 40px auto;
            padding: 0 20px;
        }
        
        /* Verdict Card */
        .verdict-card {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-radius: 20px;
            padding: 40px;
            margin-bottom: 30px;
            text-align: center;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .verdict-decision {
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
        }
        
        .verdict-decision.pass {
            color: #ff4444;
        }
        
        .verdict-decision.interview {
            color: #00ff00;
        }
        
        .confidence-badge {
            display: inline-block;
            padding: 10px 20px;
            background: rgba(255, 102, 0, 0.2);
            border: 2px solid #ff6600;
            border-radius: 20px;
            color: #ff6600;
            font-weight: bold;
            margin-bottom: 20px;
        }
        
        .hot-take {
            font-size: 1.5em;
            font-style: italic;
            color: #fff;
            margin: 20px 0;
            padding: 20px;
            background: rgba(255, 102, 0, 0.1);
            border-left: 4px solid #ff6600;
            border-radius: 10px;
        }
        
        /* Scores Grid */
        .scores-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin: 30px 0;
        }
        
        .score-card {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .score-card:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .score-label {
            font-size: 0.9em;
            color: #999;
            margin-bottom: 10px;
        }
        
        .score-value {
            font-size: 2.5em;
            font-weight: bold;
            color: #ff6600;
        }
        
        .score-bar {
            width: 100%;
            height: 8px;
            background: #333;
            border-radius: 4px;
            margin-top: 10px;
            overflow: hidden;
        }
        
        .score-fill {
            height: 100%;
            background: #ff6600;
            transition: width 1s ease;
        }
        
        /* Feedback Section */
        .feedback-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-radius: 20px;
            padding: 40px;
            margin: 30px 0;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .feedback-title {
            font-size: 2em;
            color: #ff6600;
            margin-bottom: 20px;
        }
        
        .brutal-truth {
            font-size: 1.3em;
            color: #ff4444;
            margin-bottom: 30px;
            padding: 20px;
            background: rgba(255, 68, 68, 0.1);
            border-radius: 10px;
            border-left: 4px solid #ff4444;
        }
        
        .advice-list {
            list-style: none;
            margin: 20px 0;
        }
        
        .advice-item {
            padding: 15px;
            margin: 10px 0;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 10px;
            border-left: 4px solid #00cc44;
            transition: all 0.3s ease;
        }
        
        .advice-item:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .encouragement {
            font-size: 1.1em;
            color: #00cc44;
            margin-top: 20px;
            padding: 20px;
            background: rgba(0, 204, 68, 0.1);
            border-radius: 10px;
        }
        
        /* Enhanced styles for detailed critique section */
        .detailed-critique-section {
            background: linear-gradient(135deg, #1a1a1a 0%, #2d2d2d 100%);
            border-radius: 20px;
            padding: 40px;
            margin-top: 40px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        }
        
        .critique-header {
            display: flex;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 2px solid #ff6600;
        }
        
        .critique-icon {
            font-size: 48px;
            margin-right: 20px;
        }
        
        .critique-title {
            font-size: 32px;
            font-weight: 700;
            color: #ff6600;
            margin: 0;
        }
        
        .critique-subtitle {
            font-size: 16px;
            color: #999;
            margin-top: 5px;
        }
        
        .critique-section {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 15px;
            padding: 25px;
            margin-bottom: 20px;
            border-left: 4px solid #ff6600;
            transition: all 0.3s ease;
        }
        
        .critique-section:hover {
            transform: translateX(5px);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .critique-section-title {
            font-size: 20px;
            font-weight: 600;
            color: #ff6600;
            margin-bottom: 15px;
            display: flex;
            align-items: center;
        }
        
        .critique-section-icon {
            margin-right: 10px;
            font-size: 24px;
        }
        
        .critique-text {
            font-size: 16px;
            line-height: 1.8;
            color: #e0e0e0;
            text-align: justify;
        }
        
        .critique-navigation {
            display: flex;
            gap: 10px;
            margin-bottom: 30px;
            flex-wrap: wrap;
        }
        
        .critique-nav-button {
            padding: 10px 20px;
            background: rgba(255, 102, 0, 0.2);
            border: 2px solid #ff6600;
            border-radius: 25px;
            color: #ff6600;
            cursor: pointer;
            transition: all 0.3s ease;
            font-weight: 600;
            border: none;
            font-size: 14px;
        }
        
        .critique-nav-button:hover,
        .critique-nav-button.active {
            background: #ff6600;
            color: #000;
            transform: translateY(-2px);
        }
        
        .critique-summary-box {
            background: linear-gradient(135deg, rgba(255, 102, 0, 0.1) 0%, rgba(255, 102, 0, 0.05) 100%);
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            border: 2px solid rgba(255, 102, 0, 0.3);
        }
        
        .critique-summary-text {
            font-size: 18px;
            line-height: 1.8;
            color: #ffffff;
            font-weight: 500;
        }
        
        /* Roast Card */
        .roast-card-section {
            text-align: center;
            margin: 40px 0;
        }
        
        .roast-card-title {
            font-size: 2em;
            color: #ff6600;
            margin-bottom: 20px;
        }
        
        .roast-card-image {
            max-width: 100%;
            border-radius: 20px;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
        }
        
        .share-buttons {
            margin-top: 30px;
            display: flex;
            gap: 15px;
            justify-content: center;
            flex-wrap: wrap;
        }
        
        .share-button {
            padding: 12px 24px;
            background: rgba(255, 102, 0, 0.2);
            border: 2px solid #ff6600;
            border-radius: 25px;
            color: #ff6600;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .share-button:hover {
            background: #ff6600;
            color: #000;
            transform: translateY(-2px);
        }
        
        /* Gallery */
        .gallery-section {
            max-width: 1200px;
            margin: 60px auto;
            padding: 0 20px;
        }
        
        .gallery-title {
            font-size: 2.5em;
            text-align: center;
            color: #ff6600;
            margin-bottom: 40px;
        }
        
        .gallery-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
            gap: 20px;
        }
        
        .gallery-item {
            background: rgba(255, 255, 255, 0.05);
            padding: 20px;
            border-radius: 15px;
            transition: all 0.3s ease;
            cursor: pointer;
        }
        
        .gallery-item:hover {
            transform: translateY(-5px);
            background: rgba(255, 255, 255, 0.08);
        }
        
        .gallery-hot-take {
            font-style: italic;
            color: #fff;
            margin-bottom: 10px;
        }
        
        .gallery-score {
            color: #ff6600;
            font-weight: bold;
        }
        
        /* Error Message */
        .error-message {
            background: rgba(255, 68, 68, 0.1);
            border: 2px solid #ff4444;
            color: #ff4444;
            padding: 20px;
            border-radius: 10px;
            margin: 20px 0;
            text-align: center;
            display: none;
        }
        
        /* Responsive Design */
        @media (max-width: 768px) {
            .hero h1 {
                font-size: 2em;
            }
            
            .scores-grid {
                grid-template-columns: 1fr 1fr;
            }
            
            .critique-header {
                flex-direction: column;
                text-align: center;
            }
            
            .critique-icon {
                margin-right: 0;
                margin-bottom: 10px;
            }
            
            .detailed-critique-section {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <header>
        <div class="logo">RESUME ROASTER 🔥</div>
    </header>
    
    <section class="hero">
        <h1>Get Your Resume Roasted</h1>
        <p>Upload your resume and get brutally honest AI feedback that actually helps you land interviews.</p>
    </section>
    
    <section class="upload-section">
        <div class="file-upload-wrapper">
            <input type="file" id="fileInput" class="file-upload-input" accept=".pdf,.txt,.doc,.docx">
            <label for="fileInput" class="file-upload-label">
                📄 Choose Your Resume (PDF, TXT, DOC, DOCX)
            </label>
        </div>
        
        <div id="fileInfo" class="file-info">
            <strong>Selected:</strong> <span id="fileName"></span>
        </div>
        
        <button id="analyzeButton" class="analyze-button" onclick="analyzeResume()">
            🔥 Roast My Resume!
        </button>
        
        <div id="loading" class="loading">
            <div class="spinner"></div>
            <p>Analyzing your resume...</p>
        </div>
        
        <div id="error" class="error-message"></div>
    </section>
    
    <section id="results" class="results-section">
        <!-- Results will be dynamically inserted here -->
    </section>
    
    <section id="gallery" class="gallery-section">
        <h2 class="gallery-title">Recent Roasts</h2>
        <div id="galleryGrid" class="gallery-grid">
            <!-- Gallery items will be loaded here -->
        </div>
    </section>
    
    <script>
        // File handling
        document.getElementById('fileInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                document.getElementById('fileName').textContent = file.name;
                document.getElementById('fileInfo').style.display = 'block';
                document.getElementById('analyzeButton').style.display = 'block';
            }
        });
        
        // Main analysis function
        async function analyzeResume() {
            const fileInput = document.getElementById('fileInput');
            const file = fileInput.files[0];
            
            if (!file) {
                showError('Please select a resume file');
                return;
            }
            
            // Show loading state
            document.getElementById('loading').style.display = 'block';
            document.getElementById('analyzeButton').disabled = true;
            document.getElementById('error').style.display = 'none';
            document.getElementById('results').style.display = 'none';
            
            // Prepare form data
            const formData = new FormData();
            formData.append('file', file);
            formData.append('is_public', 'true');
            
            try {
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                const data = await response.json();
                
                if (!response.ok) {
                    throw new Error(data.error || 'Analysis failed');
                }
                
                // Display results
                displayAnalysis(data);
                
            } catch (error) {
                showError(error.message || 'Failed to analyze resume. Please try again.');
            } finally {
                document.getElementById('loading').style.display = 'none';
                document.getElementById('analyzeButton').disabled = false;
            }
        }
        
        // Display analysis results
        function displayAnalysis(analysis) {
            const resultsSection = document.getElementById('results');
            resultsSection.style.display = 'block';
            resultsSection.innerHTML = '';
            
            // Add verdict
            resultsSection.innerHTML += displayVerdict(analysis);
            
            // Add scores
            resultsSection.innerHTML += displayScores(analysis);
            
            // Add feedback
            resultsSection.innerHTML += displayFeedback(analysis);
            
            // Add detailed critique if available
            if (analysis.detailed_critique) {
                resultsSection.innerHTML += displayDetailedCritique(analysis);
            }
            
            // Add roast card
            resultsSection.innerHTML += displayRoastCard(analysis);
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        }
        
        // Display verdict section
        function displayVerdict(analysis) {
            const verdict = analysis.verdict;
            const decisionClass = verdict.decision === 'INTERVIEW' ? 'interview' : 'pass';
            
            return `
                <div class="verdict-card">
                    <div class="verdict-decision ${decisionClass}">
                        ${verdict.decision}
                    </div>
                    <div class="confidence-badge">
                        ${verdict.confidence}% Confidence
                    </div>
                    <div class="hot-take">
                        "${verdict.hot_take}"
                    </div>
                    <p style="color: #999; margin-top: 10px;">
                        First impression: ${verdict.first_impression}
                    </p>
                </div>
            `;
        }
        
        // Display scores
        function displayScores(analysis) {
            const benchmarks = analysis.benchmarks;
            
            return `
                <div class="scores-grid">
                    <div class="score-card">
                        <div class="score-label">Clarity</div>
                        <div class="score-value">${benchmarks.clarity_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${benchmarks.clarity_score * 10}%"></div>
                        </div>
                    </div>
                    <div class="score-card">
                        <div class="score-label">Impact</div>
                        <div class="score-value">${benchmarks.impact_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${benchmarks.impact_score * 10}%"></div>
                        </div>
                    </div>
                    <div class="score-card">
                        <div class="score-label">Formatting</div>
                        <div class="score-value">${benchmarks.formatting_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${benchmarks.formatting_score * 10}%"></div>
                        </div>
                    </div>
                    <div class="score-card">
                        <div class="score-label">Overall</div>
                        <div class="score-value">${benchmarks.overall_score}/10</div>
                        <div class="score-bar">
                            <div class="score-fill" style="width: ${benchmarks.overall_score * 10}%"></div>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Display feedback
        function displayFeedback(analysis) {
            const feedback = analysis.feedback;
            
            let adviceHtml = '';
            feedback.actionable_advice.forEach(advice => {
                adviceHtml += `<li class="advice-item">${advice}</li>`;
            });
            
            return `
                <div class="feedback-section">
                    <h2 class="feedback-title">Your Personalized Feedback</h2>
                    
                    <div class="brutal-truth">
                        💣 Brutal Truth: ${feedback.brutal_truth}
                    </div>
                    
                    <h3 style="color: #00cc44; margin-bottom: 15px;">✅ Action Items:</h3>
                    <ul class="advice-list">
                        ${adviceHtml}
                    </ul>
                    
                    <div class="encouragement">
                        💚 ${feedback.encouragement}
                    </div>
                </div>
            `;
        }
        
        // Display detailed critique
        function displayDetailedCritique(analysis) {
            if (!analysis.detailed_critique) {
                return '';
            }
            
            const critique = analysis.detailed_critique;
            
            return `
                <div class="detailed-critique-section">
                    <div class="critique-header">
                        <div class="critique-icon">📋</div>
                        <div>
                            <h2 class="critique-title">Detailed Analysis</h2>
                            <p class="critique-subtitle">Your comprehensive resume critique</p>
                        </div>
                    </div>
                    
                    <!-- Summary -->
                    <div class="critique-summary-box">
                        <p class="critique-summary-text">${critique.summary_paragraph || 'Analysis summary not available.'}</p>
                    </div>
                    
                    <!-- Navigation buttons -->
                    <div class="critique-navigation">
                        <button class="critique-nav-button active" onclick="showCritiqueSection('all')">
                            Show All
                        </button>
                        <button class="critique-nav-button" onclick="showCritiqueSection('experience')">
                            Experience
                        </button>
                        <button class="critique-nav-button" onclick="showCritiqueSection('skills')">
                            Skills
                        </button>
                        <button class="critique-nav-button" onclick="showCritiqueSection('formatting')">
                            Formatting
                        </button>
                        <button class="critique-nav-button" onclick="showCritiqueSection('strategy')">
                            Strategy
                        </button>
                    </div>
                    
                    <!-- Experience Critique -->
                    <div class="critique-section" data-section="experience">
                        <h3 class="critique-section-title">
                            <span class="critique-section-icon">💼</span>
                            Experience Analysis
                        </h3>
                        <p class="critique-text">${critique.experience_critique || 'Experience analysis not available.'}</p>
                    </div>
                    
                    <!-- Skills Critique -->
                    <div class="critique-section" data-section="skills">
                        <h3 class="critique-section-title">
                            <span class="critique-section-icon">🛠️</span>
                            Skills Assessment
                        </h3>
                        <p class="critique-text">${critique.skills_critique || 'Skills analysis not available.'}</p>
                    </div>
                    
                    <!-- Formatting Critique -->
                    <div class="critique-section" data-section="formatting">
                        <h3 class="critique-section-title">
                            <span class="critique-section-icon">🎨</span>
                            Formatting & Design
                        </h3>
                        <p class="critique-text">${critique.formatting_critique || 'Formatting analysis not available.'}</p>
                    </div>
                    
                    <!-- Strategy Critique -->
                    <div class="critique-section" data-section="strategy">
                        <h3 class="critique-section-title">
                            <span class="critique-section-icon">🎯</span>
                            Strategic Positioning
                        </h3>
                        <p class="critique-text">${critique.strategy_critique || 'Strategy analysis not available.'}</p>
                    </div>
                </div>
            `;
        }
        
        // Navigation function for critique sections
        function showCritiqueSection(section) {
            const sections = document.querySelectorAll('.critique-section');
            const buttons = document.querySelectorAll('.critique-nav-button');
            
            // Remove active class from all buttons
            buttons.forEach(btn => btn.classList.remove('active'));
            
            if (section === 'all') {
                // Show all sections
                sections.forEach(sec => sec.style.display = 'block');
                buttons[0].classList.add('active');
            } else {
                // Show only selected section
                sections.forEach(sec => {
                    if (sec.getAttribute('data-section') === section) {
                        sec.style.display = 'block';
                    } else {
                        sec.style.display = 'none';
                    }
                });
                
                // Add active class to clicked button
                event.target.classList.add('active');
            }
        }
        
        // Display roast card
        function displayRoastCard(analysis) {
            if (!analysis.roast_card_url) {
                return '';
            }
            
            return `
                <div class="roast-card-section">
                    <h2 class="roast-card-title">Your Roast Card</h2>
                    <img src="${analysis.roast_card_url}" alt="Resume Roast Card" class="roast-card-image">
                    
                    <div class="share-buttons">
                        <a href="#" onclick="downloadRoastCard('${analysis.analysis_id}')" class="share-button">
                            📥 Download
                        </a>
                        <a href="#" onclick="shareOnTwitter('${analysis.verdict.hot_take}')" class="share-button">
                            🐦 Share on Twitter
                        </a>
                        <a href="#" onclick="shareOnLinkedIn('${analysis.analysis_id}')" class="share-button">
                            💼 Share on LinkedIn
                        </a>
                    </div>
                </div>
            `;
        }
        
        // Helper functions
        function showError(message) {
            const errorDiv = document.getElementById('error');
            errorDiv.textContent = message;
            errorDiv.style.display = 'block';
        }
        
        function downloadRoastCard(analysisId) {
            window.open(`/api/download-roast/${analysisId}`, '_blank');
        }
        
        function shareOnTwitter(hotTake) {
            const text = encodeURIComponent(`Just got my resume roasted by AI: "${hotTake}" 🔥 Try it at resumeroaster.ai`);
            window.open(`https://twitter.com/intent/tweet?text=${text}`, '_blank');
        }
        
        function shareOnLinkedIn(analysisId) {
            const url = encodeURIComponent(`https://resumeroaster.ai/analysis/${analysisId}`);
            window.open(`https://www.linkedin.com/sharing/share-offsite/?url=${url}`, '_blank');
        }
        
        // Load gallery on page load
        async function loadGallery() {
            try {
                const response = await fetch('/api/gallery');
                const analyses = await response.json();
                
                const galleryGrid = document.getElementById('galleryGrid');
                galleryGrid.innerHTML = '';
                
                analyses.slice(0, 6).forEach(item => {
                    const galleryItem = document.createElement('div');
                    galleryItem.className = 'gallery-item';
                    galleryItem.innerHTML = `
                        <div class="gallery-hot-take">"${item.hot_take}"</div>
                        <div class="gallery-score">Score: ${item.overall_score}/10</div>
                        <div style="color: #666; font-size: 0.9em; margin-top: 10px;">
                            ${item.decision} • ${new Date(item.created_at).toLocaleDateString()}
                        </div>
                    `;
                    galleryItem.onclick = () => window.location.href = `/analysis/${item.analysis_id}`;
                    galleryGrid.appendChild(galleryItem);
                });
            } catch (error) {
                console.error('Failed to load gallery:', error);
            }
        }
        
        // Load gallery when page loads
        window.addEventListener('load', loadGallery);
    </script>
</body>
</html>
