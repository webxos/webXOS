Database Module Documentation
Overview
The /db/ folder contains database-related files for the Vial MCP Controller project, managing MongoDB initialization, NLP models, and library agent synchronization for Nomic, CogniTALLMware, LLMware, and Jina AI vials. This module integrates with the wallet system (webxos_wallet.py) and the Inception Gateway for seamless data operations.
Files

dbreadme.md: This documentation file.
dockerfile: Docker configuration for the database service.
mcp_db_init.py: Initializes MongoDB collections for wallet, sync, and audit logs.
nlp_model.py: NLP model for query enhancement using transformers.
requirements.txt: Python dependencies for the database module.
server.py: FastAPI server for database operations.
wait-for-it.sh: Utility script for service dependency management.
errorlog.md: Logs errors for database operations.
library_agent.py: Manages library interactions for the four vials.
nomic_sw.js, llmware_sw.js, jina_sw.js, cognitallmware_sw.js: Service workers for caching vial responses.
global_mcp_agents.py: Synchronizes library operations across vials.
translator_agent.py: Handles language translation with PyTorch and LangChain.
library_sync.py: Synchronizes library data across vials.
test_library_agent.py, test_translator_agent.py, test_library_sync.py, test_monitor_agent.py, test_auth_sync.py, test_cache_manager.py, test_performance_metrics.py, test_config_validator.py: Unit tests for respective components.
library_config.json: Configuration settings for library agents.
monitor_agent.py: Monitors library performance.
auth_sync.py: Synchronizes authentication with vials.
cache_manager.py: Manages caching for library responses.
audit_log.py: Logs user actions for auditing.
performance_metrics.py: Collects performance metrics for vials.
config_validator.py: Validates library configuration.
health_check.py: Checks health of database endpoints.

Setup

Ensure MongoDB is running on localhost:27017.
Install dependencies: pip install -r db/requirements.txt.
Run mcp_db_init.py to initialize MongoDB collections.
Start the database server: python db/server.py.
Use wait-for-it.sh to ensure service dependencies are ready.

Error Handling
Errors are logged to db/errorlog.md with timestamps to avoid redundancy. Check this file for debugging issues related to database operations.
Integration

Wallet: All scripts update the wallet via webxos_wallet.py, appending transactions and incrementing webxos balance.
Inception Gateway: Endpoints defined in library_config.json are used for library and translation tasks.
Frontend: Interacts with vial.html and chatbot.html for user interface.

For detailed API documentation, see /docs/api.markdown.
