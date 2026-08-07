from dotenv import load_dotenv
import os

load_dotenv(env_file="env.env")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")