# Deployment Guide

## Vercel vs Railway: Which to Choose?

### ✅ Vercel (Serverless)
**Pros:**
- Free tier is generous
- Automatic HTTPS
- Fast deployments
- Great for demos

**Cons:**
- SQLite database resets on each deployment
- 10-second function timeout
- Not ideal for persistent data

**Best for:** Quick demos, testing, serverless workloads

### ✅ Railway (Persistent Server)
**Pros:**
- Persistent database
- Long-running processes
- PostgreSQL support
- Better for production

**Cons:**
- Free tier is limited ($5 credit)
- Requires credit card after trial

**Best for:** Production apps, persistent data, real use cases

---

## Quick Deploy to Vercel

### Step 1: Prepare Your Code

Make sure you have these files:
- ✅ `vercel.json` (configuration)
- ✅ `api/index.py` (serverless entry point)
- ✅ `requirements.txt` (dependencies)

### Step 2: Deploy via Vercel CLI

```bash
# Install Vercel CLI (first time only)
npm install -g vercel

# Login
vercel login

# Deploy from project root
cd module-1-genai-foundations/url-shortener
vercel

# For production deployment
vercel --prod
```

### Step 3: Deploy via Vercel Dashboard (Alternative)

1. Go to [vercel.com](https://vercel.com)
2. Click "New Project"
3. Import from GitHub
4. Select your repository
5. Configure:
   - **Framework Preset:** Other
   - **Root Directory:** `module-1-genai-foundations/url-shortener`
   - **Build Command:** (leave empty)
   - **Output Directory:** (leave empty)
6. Add Environment Variable:
   - `BASE_URL` = `https://your-app.vercel.app` (you'll get this after first deploy)
7. Click "Deploy"

### Step 4: Update BASE_URL

After first deployment:
1. Copy your Vercel URL (e.g., `https://url-shortener-abc123.vercel.app`)
2. Go to Project Settings → Environment Variables
3. Update `BASE_URL` with your Vercel URL
4. Redeploy

---

## Deploy to Railway (Recommended)

### Step 1: Sign Up & Connect GitHub

1. Go to [railway.app](https://railway.app)
2. Sign up with GitHub
3. Authorize Railway

### Step 2: Create New Project

1. Click "New Project"
2. Select "Deploy from GitHub repo"
3. Choose your repository
4. Select `module-1-genai-foundations/url-shortener`

### Step 3: Configure

Railway auto-detects the Dockerfile. If not:
- Set **Start Command:** `uvicorn src.main:app --host 0.0.0.0 --port $PORT`

### Step 4: Add Environment Variables

In Railway dashboard:
1. Go to Variables tab
2. Add:
   - `BASE_URL` = Your Railway URL (provided after deployment)
   - `PORT` = `8000` (or leave empty, Railway sets it)

### Step 5: Deploy!

Railway automatically builds and deploys. Get your public URL from the Settings.

### Step 6: Upgrade to PostgreSQL (Optional)

For production scale:
1. In Railway, click "New" → "Database" → "PostgreSQL"
2. Connect database to your service
3. Update code to use PostgreSQL connection string

---

## Troubleshooting

### Vercel: 404 Not Found
- ✅ **Fixed!** Make sure `vercel.json` and `api/index.py` exist
- Check file paths are correct
- Ensure `api/` folder is at project root

### Railway: Build Fails
- Check Dockerfile syntax
- Verify requirements.txt has all dependencies
- Check logs in Railway dashboard

### Database Not Persisting (Vercel)
- Expected behavior - Vercel is serverless
- Data resets between deployments
- Use Railway or add external database (Supabase, Neon, PlanetScale)

### CORS Errors
- Add your Vercel/Railway URL to CORS allowed origins in `main.py`

---

## Production Checklist

Before going to production:

- [ ] Switch from SQLite to PostgreSQL
- [ ] Add rate limiting
- [ ] Implement caching
- [ ] Add authentication for admin endpoints
- [ ] Set up monitoring (Sentry, LogRocket)
- [ ] Configure custom domain
- [ ] Add HTTPS (automatic on Vercel/Railway)
- [ ] Set up backups
- [ ] Add error tracking
- [ ] Implement input sanitization

---

## Cost Estimates

| Platform | Free Tier | Paid Tier |
|----------|-----------|-----------|
| **Vercel** | Unlimited deployments, 100GB bandwidth | $20/mo (Pro) |
| **Railway** | $5 credit (500 hours) | $5/mo per service |
| **Database** | Supabase free tier (500MB) | $25/mo (Pro) |

**Recommendation for learning:** Use Vercel for demos, Railway for serious projects.
