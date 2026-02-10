# URL Shortener

**Module 1 Project: GenAI Foundations & Vibe Coding**

## 🎯 Project Overview

Build a URL shortening service that takes long URLs and creates short, memorable aliases. This project introduces AI-first development methodology using tools like Claude Code, Cursor, and Copilot.

## 📋 Requirements

### Core Features
- [ ] Shorten long URLs to compact codes
- [ ] Redirect short codes to original URLs
- [ ] Track click statistics
- [ ] Basic validation and error handling

### Optional Features
- [ ] Custom aliases
- [ ] Expiration dates
- [ ] Analytics dashboard
- [ ] QR code generation

## 🛠️ Tech Stack

- **Backend:** Python 3.11 + FastAPI
- **Database:** SQLite (production-ready, upgradeable to PostgreSQL)
- **Frontend:** Vanilla HTML/CSS/JavaScript (embedded)
- **Deployment:** Docker + Railway

## 📁 Project Structure

```
url-shortener/
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── railway.json          # Railway deployment config
├── src/
│   ├── __init__.py
│   ├── main.py           # FastAPI application
│   ├── models.py         # Pydantic models
│   └── database.py       # Database operations
└── tests/
    └── test_api.py       # API tests
```

## 🚀 Getting Started

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install production dependencies
pip install -r requirements.txt

# Install dev dependencies (optional, for testing)
pip install -r requirements-dev.txt
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings (optional)
```

### 4. Run the Application

```bash
# Start the server
uvicorn src.main:app --reload

# Or run directly
python -m src.main
```

The application will be available at:
- **Web Interface:** http://localhost:8000
- **API Docs:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

## 🧪 Testing

```bash
# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## 📊 API Endpoints

### Create Short URL
```http
POST /api/shorten
Content-Type: application/json

{
  "url": "https://example.com/very/long/url",
  "custom_alias": "mylink"  // optional
}
```

**Response:**
```json
{
  "id": 1,
  "original_url": "https://example.com/very/long/url",
  "short_code": "mylink",
  "short_url": "http://localhost:8000/mylink",
  "clicks": 0,
  "created_at": "2026-02-10T10:30:00"
}
```

### Redirect to Original URL
```http
GET /{short_code}
```
Returns 301 redirect to original URL.

### Get URL Statistics
```http
GET /api/stats/{short_code}
```

### List All URLs (Admin)
```http
GET /api/urls
```

### Health Check
```http
GET /health
```

## 📊 Learning Objectives

- Practice AI-first development workflow
- Understand basic web service architecture
- Implement CRUD operations
- Handle URL validation and storage
- Deploy a working application

## 🎓 Key Concepts

- **URL Encoding:** Converting long URLs to short codes
- **Database Design:** Storing and retrieving URL mappings
- **HTTP Redirects:** Implementing 301/302 redirects
- **Basic Analytics:** Tracking usage patterns

## 📝 Notes

Document your learnings, challenges, and solutions here as you progress through the project.

## 🚢 Deployment

### Deploy to Railway

1. **Create Railway Account**
   - Sign up at [railway.app](https://railway.app)
   - Connect your GitHub account

2. **Deploy from GitHub**
   ```bash
   # Push your code to GitHub
   git add .
   git commit -m "Add URL shortener"
   git push origin main
   ```

3. **Create New Project in Railway**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway will auto-detect the Dockerfile

4. **Configure Environment Variables**
   - Add `BASE_URL` variable with your Railway URL
   - Example: `https://your-app.up.railway.app`

5. **Deploy!**
   - Railway will build and deploy automatically
   - Get your public URL from the dashboard

### Alternative: Deploy to Vercel (Serverless)

For serverless deployment, you'll need to adapt the code slightly. Railway is recommended for this project.

### Local Docker Deployment

```bash
# Build image
docker build -t url-shortener .

# Run container
docker run -p 8000:8000 url-shortener
```

## ✅ Features Implemented

- [x] Shorten long URLs to compact codes
- [x] Redirect short codes to original URLs
- [x] Track click statistics
- [x] Basic validation and error handling
- [x] Custom aliases
- [x] Beautiful web interface
- [x] RESTful API with documentation
- [x] SQLite database
- [x] Docker support
- [x] Production-ready deployment config

## 🎯 Next Steps & Enhancements

Want to take this further? Try adding:
- [ ] Expiration dates for URLs
- [ ] Analytics dashboard with charts
- [ ] QR code generation
- [ ] User authentication
- [ ] Rate limiting
- [ ] Custom domains
- [ ] URL preview before redirect
- [ ] Bulk URL shortening
- [ ] API key authentication
- [ ] PostgreSQL for production scale

## 📝 Notes

### What I Learned
- FastAPI basics and automatic API documentation
- Pydantic models for validation
- SQLite database operations
- HTTP redirects (301 vs 302)
- RESTful API design patterns
- Docker containerization
- Production deployment considerations

### Challenges & Solutions
- **Challenge:** Ensuring unique short codes
  - **Solution:** Check database before insertion, regenerate if collision
  
- **Challenge:** Handling invalid URLs
  - **Solution:** Use Pydantic's HttpUrl type for automatic validation

- **Challenge:** Tracking clicks reliably
  - **Solution:** Increment counter in database during redirect

## 🚢 Deployment

- [ ] Choose deployment platform (Vercel, Railway, Render, etc.)
- [ ] Configure environment variables
- [ ] Deploy and test
- [ ] Share your live URL!

## 📚 Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Models](https://docs.pydantic.dev/)
- [URL Shortening Algorithms](https://en.wikipedia.org/wiki/URL_shortening)
- [HTTP Status Codes](https://developer.mozilla.org/en-US/docs/Web/HTTP/Status)
- [Railway Deployment Guide](https://docs.railway.app/)
- [SQLite Tutorial](https://www.sqlitetutorial.net/)

---

**Part of Taller AI Training Program - Module 1** | [View Live Demo](#) | [API Docs](http://localhost:8000/docs)
