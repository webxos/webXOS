from dotenv import load_dotenv
import os

load_dotenv()

DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://neondb_owner:npg_EzPpBWkGdm69@ep-sparkling-thunder-aetjtveu-pooler.c-2.us-east-2.aws.neon.tech/neondb?sslmode=require')
STACK_AUTH_CLIENT_ID = os.getenv('NEXT_PUBLIC_STACK_PROJECT_ID', '142ad169-5d57-4be3-bf41-6f3cd0a9ae1d')
STACK_AUTH_PUBLISHABLE_KEY = os.getenv('NEXT_PUBLIC_STACK_PUBLISHABLE_CLIENT_KEY', 'pck_hxr6cjqdg9cy62sd97ad2j24w1b5nsgdsbt74r9vw69t0')
STACK_AUTH_SECRET_KEY = os.getenv('STACK_SECRET_SERVER_KEY', 'ssk_jg4mmhab0d0ga2krj1sskadmnkaagcxy7nxwbaeagkbjg')
JWT_SECRET_KEY = os.getenv('JWT_SECRET_KEY', 'your_jwt_secret_key')
API_BASE_URL = os.getenv('API_BASE
