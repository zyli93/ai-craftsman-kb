# Entity Extraction Prompt
# Used by: backend/ai_craftsman_kb/processing/entity_extractor.py (task_22)

Extract named entities from the following text. Return a JSON array of objects.

Each object must have:
- "name": the canonical entity name (string)
- "type": one of: person, company, technology, event, book, paper, product
- "mentions": number of times this entity appears (integer)

Rules:
- Only include entities that are clearly named and unambiguous
- Normalize variants to a single canonical form (e.g., "GPT-4" not "gpt4" or "GPT4")
- Exclude common words and generic references
- Exclude the author's own name unless they are being discussed as a subject

Text:
{text}

Return only valid JSON. No explanation.
