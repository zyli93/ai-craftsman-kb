# Source Discovery Prompt
# Used by: backend/ai_craftsman_kb/source_discovery.py (task_42)

Given the following list of URLs and text snippets found on a webpage, identify which ones
are likely to be high-quality AI/ML content sources worth following regularly.

For each promising source, return:
- "url": the normalized URL (string)
- "source_type": one of: substack, rss, youtube, reddit, arxiv, devto, website
- "reason": why this looks like a good source (max 15 words)
- "confidence": float from 0.0 to 1.0

Links found:
{links}

Return a JSON array. Omit links that are clearly not content sources (ads, navigation, etc.).
Return only valid JSON. No explanation outside the JSON array.
