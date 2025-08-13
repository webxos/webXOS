Vial MCP Controller Setup
Overview
The Vial MCP Controller manages AI vials with a web interface and backend API. This guide covers setup and troubleshooting.
Prerequisites

Python 3.8+
pip
Web browser
Optional: MongoDB for future database integration

Setup

Clone Repository:
git clone <repository-url>
cd vial-mcp-project


Install Dependencies:
pip install -r requirements.txt


Configure Environment:Create .env:
echo "JWT_SECRET=your-secret-key" > .env
echo "API_HOST=0.0.0.0" >> .env
echo "API_PORT=8000" >> .env
echo "VIAL_VERSION=2.8" >> .env


Run Backend:
python mock_backend.py


Serve Frontend:
python -m http.server 8080

Access http://localhost:8080/vial.html.

Run Tests:
python -m unittest vial/tests/test_mock_backend.py


Monitor Services (optional):
pip install psutil
python db/monitor_agent.py



Troubleshooting

NetworkError: Unable to connect to http://localhost:8000/api/health:
Verify backend: curl http://localhost:8000/api/health.
Check port conflicts: netstat -tuln | grep 8000 (Linux) or netstat -an | findstr 8000 (Windows).
Ensure mock_backend.py is running.


style.css 404:
Verify static/style.css exists and is accessible at http://localhost:8080/static/style.css.


Logs Missing:
Ensure db/errorlog.md is writable: chmod 666 db/errorlog.md.



API Documentation
See docs/api.markdown for endpoint details.
