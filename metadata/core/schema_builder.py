"""
JSON Schema builder — final stage of the Metadata Extraction Pipeline.

Takes the enriched metadata list produced by LLMMetadataGenerator and
serialises it into a valid JSON Schema (draft-07) document, ready for
storage in the database or return via the REST API.

Output structure:
    {
      '$schema': 'http://json-schema.org/draft-07/schema#',
      'title':   '<dataset name>',
      'type':    'object',
      'properties': { <column_name>: { type, description, ... } }
    }
"""
