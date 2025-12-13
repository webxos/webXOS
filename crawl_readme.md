# WEBXOS Web Crawler v2.0

A terminal-style web crawler with server-side backend to bypass same-origin policy restrictions.

## Features

- 🕷️ **Terminal UI Interface** - Cyberpunk-style terminal interface
- 🔄 **Server-Side Crawling** - Bypasses CORS and same-origin policy
- 📊 **Real-time Statistics** - Live updates of crawling progress
- 🔍 **Depth Control** - Control crawl depth (1-5 levels)
- 📈 **Queue Management** - Smart URL queue with progress tracking
- 💾 **Export Results** - Export to Markdown or copy to clipboard
- 🌐 **Netlify Ready** - Deploy as serverless functions
- 🚀 **Local Development** - Full local development support

## Quick Start

### Option 1: Deploy to Netlify (Easiest)

[![Deploy to Netlify](https://www.netlify.com/img/deploy/button.svg)](https://app.netlify.com/start/deploy?repository=https://github.com/yourusername/webxos-crawler)

1. Click the "Deploy to Netlify" button above
2. Connect your GitHub repository
3. Netlify will automatically deploy the application

### Option 2: Local Development

```bash
# Clone the repository
git clone https://github.com/yourusername/webxos-crawler.git
cd webxos-crawler

# Install dependencies
npm install

# Start local server
npm start

# Open your browser to http://localhost:3000/crawl.html