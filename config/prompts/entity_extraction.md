Extract named entities from the following text.

For each entity found, provide:
- name: the canonical/most common name
- type: one of [person, company, technology, event, book, paper, product]
- context: a short phrase showing how it's mentioned (max 100 chars)

Rules:
- Only include clearly and explicitly mentioned entities
- Use canonical names (e.g. "Meta" not "Facebook, Inc.")
- Prefer specific names over generic terms
- If no entities found, return an empty array

Return ONLY a JSON array with no other text:
[{"name": "...", "type": "...", "context": "..."}, ...]

Text:
{content}
