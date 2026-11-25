"""
config.py -- loads the .env file so EVERY module sees the same settings.
Each module does `import config` as its first project import.
(.env is searched in this folder and its parents. Real environment variables win.)
"""
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:          # python-dotenv not installed: use normal environment variables
    pass
