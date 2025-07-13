from ks_config import config, logger, openai_client
from typing import Optional, Dict, List, Any
import hashlib
import time
from ks_classes import Document  
from datetime import datetime
import re
import json

# Robust JSON parsing utilities
def extract_json_from_text(text: str, expected_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract JSON from potentially malformed text with multiple fallback strategies.
    """
    if not text or not text.strip():
        return {}

    # Strategy 1: Try to find complete JSON objects
    json_patterns = [
        r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}',  # Nested objects
        r'\[[^\[\]]*(?:\[[^\[\]]*\][^\[\]]*)*\]',  # Arrays
        r'\{.*?\}',  # Simple objects
    ]

    for pattern in json_patterns:
        matches = re.findall(pattern, text, re.DOTALL)
        for match in matches:
            try:
                data = json.loads(match)
                # Validate expected keys if provided
                if expected_keys and isinstance(data, dict):
                    if all(key in data for key in expected_keys):
                        return data
                elif not expected_keys:
                    return data
            except json.JSONDecodeError:
                continue

    # Strategy 2: Try to fix common JSON issues
    cleaned_text = text.strip()
    if cleaned_text.startswith('```json'):
        cleaned_text = cleaned_text[7:]
    if cleaned_text.endswith('```'):
        cleaned_text = cleaned_text[:-3]

    try:
        return json.loads(cleaned_text)
    except json.JSONDecodeError:
        pass

    # Strategy 3: Return empty structure based on expected keys
    if expected_keys:
        return {key: [] if key.endswith('s') else "" for key in expected_keys}

    logger.warning("Failed to extract JSON from text")
    return {}

def validate_json_structure(data: Dict[str, Any], required_keys: List[str], optional_keys: List[str] = None) -> bool:
    """
    Validate JSON structure against expected schema.
    """
    if not isinstance(data, dict):
        return False

    # Check required keys
    for key in required_keys:
        if key not in data:
            logger.warning(f"Missing required key: {key}")
            return False

    # Log unexpected keys
    all_expected = set(required_keys + (optional_keys or []))
    unexpected = set(data.keys()) - all_expected
    if unexpected:
        logger.debug(f"Unexpected keys found: {unexpected}")

    return True


# Input validation utilities
def validate_content_length(content: str, max_length: int = None) -> str:
    """
    Validate and potentially truncate content to safe length.
    """
    max_length = max_length or config.max_content_length

    if not content:
        return ""

    if len(content) > max_length:
        logger.warning(f"Content too long ({len(content)} chars), truncating to {max_length}")
        return content[:max_length]

    return content
	
def call_openai_robust(
    prompt: str,
    max_tokens: int = None,
    temperature: float = None,
    system_message: Optional[str] = None,
    model: Optional[str] = None,
    expected_json_keys: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Robust OpenAI API call with comprehensive error handling and validation.

    Args:
        prompt: User prompt
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        system_message: System message for context
        model: Model to use (defaults to config)
        expected_json_keys: Expected keys in JSON response for validation

    Returns:
        Dict containing 'content' and 'metadata' keys
    """
    if not openai_client:
        raise RuntimeError("OpenAI client not initialized")

    # Use defaults from config
    max_tokens = max_tokens or config.max_tokens_default
    temperature = temperature or config.temperature_default
    model = model or config.openai_model

    # Validate input lengths
    if len(prompt) > config.max_prompt_length:
        logger.warning(f"Prompt too long ({len(prompt)} chars), truncating to {config.max_prompt_length}")
        prompt = prompt[:config.max_prompt_length]

    # Prepare messages
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})

    try:
        logger.debug(f"Calling OpenAI API with {len(prompt)} char prompt, model: {model}")

        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        content = response.choices[0].message.content

        if not content:
            raise ValueError("Empty response from OpenAI")

        logger.debug(f"OpenAI response received: {len(content)} characters")

        # Parse JSON if expected
        parsed_json = {}
        if expected_json_keys:
            parsed_json = extract_json_from_text(content, expected_json_keys)
            if not validate_json_structure(parsed_json, expected_json_keys):
                logger.warning("JSON response validation failed")

        return {
            "content": content,
            "parsed_json": parsed_json,
            "metadata": {
                "model": model,
                "tokens_used": response.usage.total_tokens if response.usage else 0,
                "finish_reason": response.choices[0].finish_reason,
                "timestamp": datetime.now().isoformat()
            }
        }

    except Exception as e:
        logger.error(f"OpenAI API call failed: {str(e)}")
        # Re-raise for backoff decorator to handle
        raise

def chunk_text(text, max_chars=30000, overlap=1000):
    """Splits large text into chunks for LLM processing."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        # Try to split at a line break for neatness
        if end < len(text):
            next_break = text.rfind('\n', start, end)
            if next_break > start:
                end = next_break
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap  # overlap slightly for context
    return chunks

def process_document_with_chunking(document, extractor):
    """
    Handles chunking for large documents transparently.
    Calls extractor.extract_knowledge_nodes(doc_chunk) per chunk.
    Merges and returns all nodes.
    """
    max_chunk_size = config.max_prompt_length - 2000  # leave headroom for prompt text
    overlap = 1000  # configurable if desired

    if len(document.content) <= max_chunk_size:
        # Normal case: one chunk
        return extractor.extract_knowledge_nodes(document)
    else:
        chunks = chunk_text(document.content, max_chars=max_chunk_size, overlap=overlap)
        all_nodes = []
        for i, chunk in enumerate(chunks):
            # Tell the LLM what chunk this is
            chunk_intro = (
                f"(Chunk {i+1}/{len(chunks)} of file '{document.filename}')\n"
                "This is a partial input. Do not summarize or repeat nodes from previous chunks."
            )
            # Wrap chunk as a Document
            doc_chunk = Document(
                filename=document.filename,
                doc_type=document.doc_type,
                content=chunk_intro + "\n" + chunk,
                frequency=document.frequency,
                metadata=document.metadata,
                timestamp=document.timestamp,
                id=document.id
            )
            nodes = extractor.extract_knowledge_nodes(doc_chunk)
            all_nodes.extend(nodes)
        return all_nodes
		
def generate_secure_id(prefix: str, content: str) -> str:
    """
    Generate secure, deterministic ID from content.
    """
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]
    timestamp = int(time.time() * 1000) % 1000000
    return f"{prefix}_{hash_hex}_{timestamp}"
	
