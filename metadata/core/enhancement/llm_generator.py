"""
LLM-powered metadata enrichment module.

Sends enriched ColumnProfile data to an LLM (OpenAI / Anthropic /
local Ollama) and receives structured metadata back:

    - description   : plain-English explanation of the column
    - tags          : domain tags (e.g. 'PII', 'financial', 'temporal')
    - business_name : human-friendly display name
    - notes         : any data quality concerns or usage caveats

Prompts are assembled from the ColumnProfile context. Responses are
parsed from JSON — the LLM is instructed to respond in JSON only.

Configure the LLM backend in Django settings:
    LLM_BACKEND = 'openai'  # or 'anthropic' | 'ollama'
    LLM_MODEL   = 'gpt-4o-mini'
"""
