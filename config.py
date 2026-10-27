from dotenv import load_dotenv
import os

load_dotenv(dotenv_path="dev.env")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
# Used for streaming chat when the main model is rate-limited (HTTP 429); separate quota on Groq.
GROQ_FALLBACK_MODEL = os.getenv("GROQ_FALLBACK_MODEL", "openai/gpt-oss-20b")

POSTGRES_USER = os.getenv("POSTGRES_USER", "")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD", "")
POSTGRES_DB = os.getenv("POSTGRES_DB", "")
POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

# HMAC key for signing JWTs. Set a long random value in any real deployment;
# the default only exists so local dev keeps working (and old tokens stay valid).
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "your-secret-key")

# Optional: POSTed {"text": ..., "content": ...} whenever a chatbot visitor leaves
# contact details (works with Slack/Discord incoming webhooks). Leave unset to skip.
LEAD_WEBHOOK_URL = os.getenv("LEAD_WEBHOOK_URL")

# Whose CV the chatbot answers about at plain /chatbot (no ?u=<username>) --
# the link already shared with recruiters, so it must stay the same.
CHATBOT_DEFAULT_USERNAME = os.getenv("CHATBOT_DEFAULT_USERNAME", "gaurav66")
