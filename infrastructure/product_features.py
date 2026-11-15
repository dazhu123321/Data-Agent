"""Feature policy for the local 3C retail analysis application."""

# Keep ordinary analysis and the business knowledge base available, without
# exposing or executing analysis Skills (including automatic matching).
CHAT_SKILLS_ENABLED = False
