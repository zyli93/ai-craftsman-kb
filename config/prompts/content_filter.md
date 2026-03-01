# Content Filter Prompt
# Used by: backend/ai_craftsman_kb/ingestors/content_filter.py (task_07)

You are filtering content for an AI/ML knowledge base.

Rate the following article's relevance to AI, machine learning, and software engineering topics.

Return a JSON object with:
- "relevant": true or false
- "score": float from 0.0 to 1.0 (1.0 = highly relevant)
- "reason": one-sentence explanation (max 20 words)

Title: {title}
Excerpt: {excerpt}

Return only valid JSON. No explanation outside the JSON object.
