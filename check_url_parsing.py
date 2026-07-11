"""
Diagnostic: shows exactly how your DATABASE_URL is being parsed,
without ever printing the password itself.

Run: uv run python check_url_parsing.py
"""

from urllib.parse import urlparse

from src.config import get_settings

settings = get_settings()
parsed = urlparse(settings.database_url)

print("scheme:  ", parsed.scheme)
print("username:", parsed.username)
print("password length:", len(parsed.password) if parsed.password else 0)
print("hostname:", repr(parsed.hostname))
print("port:    ", parsed.port)
print("path:    ", parsed.path)

# Also show the raw netloc so we can see if anything looks mangled
print("\nraw netloc (password masked):")
if parsed.password:
    masked = settings.database_url.replace(parsed.password, "***MASKED***")
    print(masked)
else:
    print(settings.database_url)