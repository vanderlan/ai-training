from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import RedirectResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from src.models import URLCreate, URLResponse, URLStats
from src.database import Database
import os
from datetime import datetime

# Initialize FastAPI app
app = FastAPI(
    title="URL Shortener API",
    description="A production-ready URL shortening service built with FastAPI",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize database
db = Database()

# Get base URL from environment or use default
BASE_URL = os.getenv("BASE_URL", "https://url-shortener-vanderlan.vercel.app")


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend HTML"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>URL Shortener - Taller Training</title>
        <style>
            * {
                margin: 0;
                padding: 0;
                box-sizing: border-box;
            }
            
            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 20px;
            }
            
            .container {
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);
                padding: 40px;
                max-width: 600px;
                width: 100%;
            }
            
            h1 {
                color: #333;
                margin-bottom: 10px;
                font-size: 2rem;
            }
            
            .subtitle {
                color: #666;
                margin-bottom: 30px;
                font-size: 0.9rem;
            }
            
            .form-group {
                margin-bottom: 20px;
            }
            
            label {
                display: block;
                color: #555;
                margin-bottom: 8px;
                font-weight: 500;
            }
            
            input {
                width: 100%;
                padding: 12px 15px;
                border: 2px solid #e0e0e0;
                border-radius: 8px;
                font-size: 1rem;
                transition: border-color 0.3s;
            }
            
            input:focus {
                outline: none;
                border-color: #667eea;
            }
            
            button {
                width: 100%;
                padding: 14px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 8px;
                font-size: 1rem;
                font-weight: 600;
                cursor: pointer;
                transition: transform 0.2s, box-shadow 0.2s;
            }
            
            button:hover {
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(102, 126, 234, 0.4);
            }
            
            button:active {
                transform: translateY(0);
            }
            
            .result {
                margin-top: 30px;
                padding: 20px;
                background: #f8f9fa;
                border-radius: 8px;
                display: none;
            }
            
            .result.show {
                display: block;
            }
            
            .short-url {
                display: flex;
                align-items: center;
                gap: 10px;
                margin: 15px 0;
            }
            
            .short-url input {
                flex: 1;
                background: white;
            }
            
            .copy-btn {
                padding: 12px 20px;
                width: auto;
                background: #28a745;
                min-width: 80px;
            }
            
            .copy-btn:hover {
                background: #218838;
            }
            
            .stats {
                margin-top: 15px;
                padding-top: 15px;
                border-top: 1px solid #dee2e6;
                color: #666;
                font-size: 0.9rem;
            }
            
            .error {
                color: #dc3545;
                margin-top: 10px;
                padding: 10px;
                background: #f8d7da;
                border-radius: 5px;
                display: none;
            }
            
            .error.show {
                display: block;
            }
            
            .loading {
                display: none;
                text-align: center;
                color: #667eea;
            }
            
            .footer {
                margin-top: 30px;
                text-align: center;
                color: #999;
                font-size: 0.85rem;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔗 URL Shortener</h1>
            <p class="subtitle">Module 1 Project - Taller AI Training</p>
            
            <form id="urlForm">
                <div class="form-group">
                    <label for="url">Long URL *</label>
                    <input 
                        type="url" 
                        id="url" 
                        name="url" 
                        placeholder="https://example.com/very/long/url"
                        required
                    >
                </div>
                
                <div class="form-group">
                    <label for="custom_alias">Custom Alias (optional)</label>
                    <input 
                        type="text" 
                        id="custom_alias" 
                        name="custom_alias" 
                        placeholder="my-custom-link"
                        pattern="[a-zA-Z0-9_-]{3,20}"
                        title="3-20 characters: letters, numbers, dash, underscore"
                    >
                </div>
                
                <button type="submit">Shorten URL</button>
            </form>
            
            <div class="loading">⏳ Creating short URL...</div>
            <div class="error" id="error"></div>
            
            <div class="result" id="result">
                <strong>✅ Short URL created!</strong>
                <div class="short-url">
                    <input type="text" id="shortUrl" readonly>
                    <button class="copy-btn" onclick="copyToClipboard()">Copy</button>
                </div>
                <div class="stats" id="stats"></div>
            </div>
            
            <div class="footer">
                Built with FastAPI • <a href="/docs" target="_blank" style="color: #667eea;">API Docs</a>
            </div>
        </div>
        
        <script>
            const form = document.getElementById('urlForm');
            const result = document.getElementById('result');
            const error = document.getElementById('error');
            const loading = document.querySelector('.loading');
            
            form.addEventListener('submit', async (e) => {
                e.preventDefault();
                
                const url = document.getElementById('url').value;
                const customAlias = document.getElementById('custom_alias').value;
                
                // Hide previous results
                result.classList.remove('show');
                error.classList.remove('show');
                loading.style.display = 'block';
                
                try {
                    const response = await fetch('/api/shorten', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            url: url,
                            custom_alias: customAlias || null
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (!response.ok) {
                        throw new Error(data.detail || 'Failed to create short URL');
                    }
                    
                    // Show result
                    document.getElementById('shortUrl').value = data.short_url;
                    document.getElementById('stats').innerHTML = `
                        <div><strong>Code:</strong> ${data.short_code}</div>
                        <div><strong>Clicks:</strong> ${data.clicks}</div>
                    `;
                    result.classList.add('show');
                    
                    // Reset form
                    form.reset();
                    
                } catch (err) {
                    error.textContent = err.message;
                    error.classList.add('show');
                } finally {
                    loading.style.display = 'none';
                }
            });
            
            function copyToClipboard() {
                const shortUrl = document.getElementById('shortUrl');
                shortUrl.select();
                document.execCommand('copy');
                
                const btn = event.target;
                const originalText = btn.textContent;
                btn.textContent = '✓ Copied!';
                btn.style.background = '#218838';
                
                setTimeout(() => {
                    btn.textContent = originalText;
                    btn.style.background = '';
                }, 2000);
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)


@app.post("/api/shorten", response_model=URLResponse)
async def create_short_url(url_data: URLCreate):
    """Create a shortened URL"""
    try:
        # Create short URL in database
        url_record = db.create_short_url(
            str(url_data.url), 
            url_data.custom_alias
        )
        
        # Build response
        return URLResponse(
            id=url_record["id"],
            original_url=url_record["original_url"],
            short_code=url_record["short_code"],
            short_url=f"{BASE_URL}/{url_record['short_code']}",
            clicks=url_record["clicks"],
            created_at=datetime.fromisoformat(url_record["created_at"])
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create short URL")


@app.get("/{short_code}")
async def redirect_to_url(short_code: str):
    """Redirect short code to original URL"""
    url_record = db.get_url_by_code(short_code)
    
    if not url_record:
        raise HTTPException(status_code=404, detail="Short URL not found")
    
    # Increment click count
    db.increment_clicks(short_code)
    
    # Redirect to original URL
    return RedirectResponse(url=url_record["original_url"], status_code=301)


@app.get("/api/stats/{short_code}", response_model=URLStats)
async def get_url_stats(short_code: str):
    """Get statistics for a shortened URL"""
    url_record = db.get_url_by_code(short_code)
    
    if not url_record:
        raise HTTPException(status_code=404, detail="Short URL not found")
    
    return URLStats(
        original_url=url_record["original_url"],
        short_code=url_record["short_code"],
        short_url=f"{BASE_URL}/{url_record['short_code']}",
        clicks=url_record["clicks"],
        created_at=datetime.fromisoformat(url_record["created_at"])
    )


@app.get("/api/urls")
async def list_all_urls():
    """List all shortened URLs (admin endpoint)"""
    urls = db.get_all_urls()
    
    return {
        "total": len(urls),
        "urls": [
            {
                "id": url["id"],
                "original_url": url["original_url"],
                "short_code": url["short_code"],
                "short_url": f"{BASE_URL}/{url['short_code']}",
                "clicks": url["clicks"],
                "created_at": url["created_at"]
            }
            for url in urls
        ]
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
