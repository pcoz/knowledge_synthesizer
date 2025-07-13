# ===============================
# SECTION 1: Imports, Logging, and Configuration
# ===============================

import os
import sys
import json
import re
import time
import logging
from typing import Dict, List, Any, Optional, Set, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import hashlib
from enum import Enum
import threading
from functools import wraps
import backoff
from collections import defaultdict, deque

# Import core configuration and infrastructure objects from ks_config.py:
from ks_config import config, logger, openai_client, RateLimiter

# Import helper utilities
from ks_helpers import (
    extract_json_from_text,
    validate_json_structure,
    validate_content_length,
    call_openai_robust,
    chunk_text,
    process_document_with_chunking,
    generate_secure_id,
)

# Import all core data classes from ks_classes.py:
from ks_classes import (
    Document,
    KnowledgeNode,
    FoundationLevel
)


# ===============================
# SECTION 2: Enhanced Data Validation
# ===============================

class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass

def validate_node_structure(node_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and sanitizes node data structure.
    
    Args:
        node_data: Raw node data from LLM
        
    Returns:
        Validated and sanitized node data
        
    Raises:
        ValidationError: If node structure is invalid
    """
    required_fields = ['title', 'content']
    optional_fields = ['index', 'level', 'foundation_hint', 'info_type', 'parents', 'children', 'tags']
    
    # Check required fields
    for field in required_fields:
        if field not in node_data or not node_data[field]:
            raise ValidationError(f"Missing required field: {field}")
    
    # Sanitize and validate fields
    sanitized = {}
    
    # Title validation
    title = str(node_data['title']).strip()
    if not title or len(title) > 200:
        raise ValidationError(f"Invalid title: must be 1-200 characters")
    sanitized['title'] = title
    
    # Content validation
    content = str(node_data['content']).strip()
    if not content:
        raise ValidationError("Content cannot be empty")
    sanitized['content'] = content[:config.max_content_length] if hasattr(config, 'max_content_length') else content
    
    # Index validation
    index = node_data.get('index')
    if index is not None:
        try:
            sanitized['index'] = int(index)
        except (ValueError, TypeError):
            logger.warning(f"Invalid index value: {index}, will be auto-assigned")
    
    # Level validation
    level = node_data.get('level', 'atomic')
    valid_levels = ['pillar', 'core', 'supporting', 'detail', 'atomic']
    if level.lower() not in valid_levels:
        logger.warning(f"Invalid level: {level}, defaulting to 'atomic'")
        level = 'atomic'
    sanitized['level'] = level.lower()
    
    # Info type validation
    info_type = node_data.get('info_type', 'mixed')
    valid_info_types = ['conceptual', 'procedural', 'mixed']
    if info_type.lower() not in valid_info_types:
        logger.warning(f"Invalid info_type: {info_type}, defaulting to 'mixed'")
        info_type = 'mixed'
    sanitized['info_type'] = info_type.lower()
    
    # Parents/children validation
    for field in ['parents', 'children']:
        values = node_data.get(field, [])
        if not isinstance(values, list):
            logger.warning(f"Invalid {field} format, expected list, got {type(values)}")
            values = []
        sanitized[field] = [int(v) for v in values if isinstance(v, (int, str)) and str(v).isdigit()]
    
    # Tags validation
    tags = node_data.get('tags', [])
    if not isinstance(tags, list):
        tags = []
    sanitized['tags'] = [str(tag).strip() for tag in tags if tag]
    
    # Foundation hint
    sanitized['foundation_hint'] = node_data.get('foundation_hint', node_data.get('level', 'auto'))
    
    return sanitized


# ===============================
# SECTION 3: Enhanced AI Knowledge Extraction
# ===============================

class AIKnowledgeExtractor:
    def __init__(self, openai_client, openai_model: str):
        self.openai_client = openai_client
        self.model = openai_model
        self.extraction_stats = {
            'total_attempts': 0,
            'successful_extractions': 0,
            'failed_extractions': 0,
            'retry_count': 0
        }

    def extract_knowledge_nodes(self, document: Document, chunk_intro: str = "") -> List[Dict[str, Any]]:
        """
        Enhanced knowledge extraction with better error handling and validation.
        """
        self.extraction_stats['total_attempts'] += 1
        
        # Improved prompt with clearer instructions
        base_prompt = self._build_extraction_prompt(document, chunk_intro)
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                # Get stricter prompt for retry attempts
                prompt = self._get_prompt_for_attempt(base_prompt, attempt)
                
                result = call_openai_robust(
                    prompt=prompt,
                    max_tokens=2000,  # Increased token limit
                    temperature=0.1,  # Lower temperature for more consistent output
                    system_message="You are a precise knowledge extraction system. Always output valid JSON.",
                    model=self.model
                )
                
                content = result.get("content", "").strip()
                
                # Enhanced JSON extraction
                nodes = self._extract_and_validate_json(content, attempt, key="nodes")
                
                if nodes is not None:
                    # Validate each node
                    validated_nodes = []
                    for i, node in enumerate(nodes):
                        try:
                            validated_node = validate_node_structure(node)
                            validated_nodes.append(validated_node)
                        except ValidationError as e:
                            logger.warning(f"Node {i} validation failed: {e}")
                            continue
                    
                    if validated_nodes:
                        self.extraction_stats['successful_extractions'] += 1
                        logger.info(f"Successfully extracted {len(validated_nodes)} valid nodes from {document.filename}")
                        return validated_nodes
                
                self.extraction_stats['retry_count'] += 1
                logger.warning(f"Attempt {attempt + 1} failed for {document.filename}, retrying...")
                
            except Exception as ex:
                logger.error(f"Attempt {attempt + 1} failed with exception: {ex}")
                self.extraction_stats['retry_count'] += 1
                continue
        
        # All attempts failed
        self.extraction_stats['failed_extractions'] += 1
        logger.error(f"Failed to extract knowledge nodes from {document.filename} after {max_attempts} attempts")
        return []

    def _build_extraction_prompt(self, document: Document, chunk_intro: str = "") -> str:
        """Build the base extraction prompt with improved clarity."""
        return (
            f"{chunk_intro}\n" if chunk_intro else ""
        ) + f"""
You are an expert knowledge extraction system. Extract the knowledge structure from the following document as a hierarchical graph.

EXTRACTION RULES:
1. Break content into atomic, self-contained knowledge units
2. Each node must have a clear title (1-200 chars) and complete content
3. Classify info_type as: 'conceptual', 'procedural', or 'mixed'
4. Build parent-child relationships based on conceptual hierarchy
5. Assign level: 'pillar' (highest), 'core', 'supporting', 'detail', or 'atomic' (lowest)

OUTPUT FORMAT - JSON ONLY:
{{
  "doc_id": "{document.id}",
  "filename": "{document.filename}",
  "nodes": [
    {{
      "index": 0,
      "title": "Clear, descriptive title",
      "content": "Complete content or explanation",
      "level": "pillar|core|supporting|detail|atomic",
      "info_type": "conceptual|procedural|mixed",
      "parents": [list of parent indices],
      "children": [list of child indices],
      "tags": ["relevant", "keywords"]
    }}
  ]
}}

DOCUMENT CONTENT:
{validate_content_length(document.content, getattr(config, 'max_content_length', 8000))}

OUTPUT ONLY THE JSON - NO EXPLANATIONS OR MARKDOWN:
"""

    def _get_prompt_for_attempt(self, base_prompt: str, attempt: int) -> str:
        """Get progressively stricter prompts for retry attempts."""
        if attempt == 0:
            return base_prompt
        elif attempt == 1:
            return base_prompt + "\n\nIMPORTANT: Output ONLY valid JSON. No markdown, no explanations, no code blocks."
        else:
            return """
OUTPUT ONLY VALID JSON WITH THIS EXACT STRUCTURE:
{"doc_id": "...", "filename": "...", "nodes": [...]}

NO MARKDOWN. NO EXPLANATIONS. NO CODE BLOCKS. ONLY JSON.

""" + base_prompt

    def _extract_and_validate_json(self, content: str, attempt: int, key: str = "nodes") -> Optional[List[Dict[str, Any]]]:
        """
        Enhanced JSON extraction with multiple fallback methods, configurable key (e.g. 'themes', 'nodes', 'nuggets').
        """
        # Try to parse as plain JSON
        try:
            data = json.loads(content)
            if key in data:
                return data[key]
        except Exception:
            pass
    
        # Try markdown code block extraction
        code_block_pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
        matches = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)
        for match in matches:
            try:
                data = json.loads(match)
                if key in data:
                    return data[key]
            except Exception:
                continue
    
        # Try to extract object with target key
        json_pattern = r'\{[\s\S]*?"' + re.escape(key) + r'"\s*:\s*\[[\s\S]*?\][\s\S]*?\}'
        matches = re.findall(json_pattern, content)
        for match in matches:
            try:
                data = json.loads(match)
                if key in data:
                    return data[key]
            except Exception:
                continue
    
        # Try to extract just the array for the key
        nodes_pattern = r'"' + re.escape(key) + r'"\s*:\s*(\[[\s\S]*?\])'
        matches = re.findall(nodes_pattern, content)
        for match in matches:
            try:
                return json.loads(match)
            except Exception:
                continue
    
        logger.warning(f"Failed to extract JSON key '{key}' from LLM output on attempt {attempt + 1}")
        return None


    def nodes_to_graph(self, nodes: List[Dict[str, Any]], document: Document, graph: "KnowledgeGraph") -> None:
        """
        Enhanced node-to-graph conversion with better error handling and cycle detection.
        """
        if not nodes:
            logger.warning(f"No nodes to process for document {document.filename}")
            return
        
        # Enhanced index normalization
        index_to_id = self._normalize_and_map_indices(nodes, document)
        
        # Create nodes with enhanced validation
        created_nodes = self._create_validated_nodes(nodes, index_to_id, document, graph)
        
        # Link nodes with cycle detection
        self._link_nodes_safely(nodes, index_to_id, graph)
        
        logger.info(f"Successfully added {len(created_nodes)} nodes to graph from {document.filename}")

    def _normalize_and_map_indices(self, nodes: List[Dict[str, Any]], document: Document) -> Dict[int, str]:
        """Normalize indices and create node ID mappings."""
        used_indices: Set[int] = set()
        index_to_id: Dict[int, str] = {}
        
        # First pass: collect existing valid indices
        for node in nodes:
            if isinstance(node.get('index'), int) and node['index'] >= 0:
                used_indices.add(node['index'])
        
        # Second pass: assign missing indices
        next_available = 0
        for node in nodes:
            if not isinstance(node.get('index'), int) or node['index'] < 0:
                while next_available in used_indices:
                    next_available += 1
                node['index'] = next_available
                used_indices.add(next_available)
                next_available += 1
        
        # Create node IDs
        for node in nodes:
            idx = node['index']
            level = node.get('level', 'atomic')
            prefix = {
                'pillar': 'pillar',
                'core': 'core', 
                'supporting': 'supp',
                'detail': 'det',
                'atomic': 'atom'
            }.get(level, 'atom')
            
            # Create more unique IDs
            base_content = f"{document.filename}:{node.get('title', '')}:{node.get('content', '')[:100]}"
            index_to_id[idx] = generate_secure_id(prefix, base_content)
        
        return index_to_id

    def _create_validated_nodes(self, nodes: List[Dict[str, Any]], index_to_id: Dict[int, str], 
                               document: Document, graph: "KnowledgeGraph") -> List[str]:
        """Create validated KnowledgeNode objects."""
        created_nodes = []
        
        for node in nodes:
            try:
                node_id = index_to_id[node['index']]
                
                # Check for duplicates
                if graph.get_node(node_id):
                    logger.warning(f"Duplicate node ID {node_id}, merging data")
                    existing = graph.get_node(node_id)
                    existing.add_doc_ref(document.id)
                    existing.tags.update(node.get('tags', []))
                    continue
                
                # Map foundation level
                foundation = self._map_foundation_level(node.get('foundation_hint', node.get('level', 'auto')))
                
                # Create new node
                knowledge_node = KnowledgeNode(
                    id=node_id,
                    title=node['title'],
                    content=node['content'],
                    node_type=node.get('level', 'atomic'),
                    foundation=foundation,
                    info_type=node.get('info_type', 'mixed'),
                    doc_refs={document.id},
                    parents=set(),  # Will be populated in linking phase
                    children=set(),  # Will be populated in linking phase
                    tags=set(node.get('tags', [])),
                    metadata={
                        'source_doc': document.filename,
                        'source_doc_type': document.doc_type,
                        'extraction_timestamp': datetime.now().isoformat(),
                        'synthesis_covered': False,
                        'synthesis_coverage_level': 'none',
                        'synthesis_section_id': None,
                        'last_processing_status': 'extracted',
                        'validation_passed': True
                    }
                )
                
                graph.add_node(knowledge_node)
                created_nodes.append(node_id)
                
            except Exception as e:
                logger.error(f"Failed to create node {node.get('index', 'unknown')}: {e}")
                continue
        
        return created_nodes

    def _map_foundation_level(self, level_hint: str) -> Union[FoundationLevel, str]:
        """Map level hint to FoundationLevel enum."""
        if isinstance(level_hint, str):
            if level_hint.isdigit():
                try:
                    return FoundationLevel(int(level_hint))
                except ValueError:
                    pass
            
            level_map = {
                'pillar': FoundationLevel.PILLAR,
                'core': FoundationLevel.CORE,
                'supporting': FoundationLevel.SUPPORTING,
                'detail': FoundationLevel.DETAIL,
                'atomic': FoundationLevel.ATOMIC
            }
            return level_map.get(level_hint.lower(), 'auto')
        
        return 'auto'

    def _link_nodes_safely(self, nodes: List[Dict[str, Any]], index_to_id: Dict[int, str], graph: "KnowledgeGraph") -> None:
        """Link nodes with cycle detection and validation."""
        for node in nodes:
            try:
                node_id = index_to_id[node['index']]
                
                # Link parents
                for parent_idx in node.get('parents', []):
                    if parent_idx in index_to_id:
                        parent_id = index_to_id[parent_idx]
                        if parent_id != node_id:  # Avoid self-reference
                            graph.link_parent_child(parent_id, node_id)
                
                # Link children
                for child_idx in node.get('children', []):
                    if child_idx in index_to_id:
                        child_id = index_to_id[child_idx]
                        if child_id != node_id:  # Avoid self-reference
                            graph.link_parent_child(node_id, child_id)
                            
            except Exception as e:
                logger.error(f"Failed to link node {node.get('index', 'unknown')}: {e}")
                continue

    def get_extraction_stats(self) -> Dict[str, Any]:
        """Get extraction statistics."""
        return self.extraction_stats.copy()

    def extract_themes(self, document: Document) -> List[Dict[str, Any]]:
        prompt = f"""
        Identify the main themes present in the following document. For each theme, provide:
          - a clear, descriptive title (unique among themes)
          - a level of fundamentalness ('pillar', 'core', 'supporting', 'detail', or 'atomic')

        Do NOT output any parent or child relationships.
        Focus ONLY on the inherent conceptual granularity—higher-level (more fundamental) themes should be marked as 'pillar', mid-level as 'core' or 'supporting', and details as 'detail' or 'atomic'.

        DOCUMENT CONTENT:
        {validate_content_length(document.content, getattr(config, 'max_content_length', 8000))}

        OUTPUT FORMAT - JSON ONLY:
        {{
          "themes": [
            {{
              "title": "Theme title",
              "level": "pillar|core|supporting|detail|atomic"
            }}
          ]
        }}

        OUTPUT ONLY THE JSON - NO EXPLANATIONS OR MARKDOWN.
        """
    
        print("\n==================\n[DEBUG] AI Prompt for THEME EXTRACTION:\n==================\n")
        print(prompt)
        print("\n==================\n")
    
        result = call_openai_robust(
            prompt=prompt,
            max_tokens=1500,
            temperature=0.1,
            system_message="You are a precise theme identification system. Always output valid JSON.",
            model=self.model
        )
    
        content = result.get("content", "").strip()
        
        print("\n==================\n[DEBUG] AI Response for THEME EXTRACTION:\n==================\n")
        print(content)
        print("\n==================\n")
        
        themes_data = self._extract_and_validate_json(content, attempt=0, key="themes")
        
        if not themes_data:
            logger.error(f"Failed to extract themes from {document.filename}")
            return []
        
        # If _extract_and_validate_json returns a dict with 'themes', use it. If it returns a list, return that.
        if isinstance(themes_data, dict) and "themes" in themes_data:
            return themes_data["themes"]
        if isinstance(themes_data, list):
            return themes_data
        logger.error(f"Unexpected themes_data format from {document.filename}: {type(themes_data)}")
        return []


    def extract_theme_nuggets(self, document: Document, theme: Dict[str, str]) -> List[Dict[str, Any]]:
        prompt = f"""
        For the following theme titled '{theme["title"]}', extract all distinct, self-contained pieces of knowledge—facts, explanations, key insights, procedures, or crucial details—from the document below that directly support, exemplify, or elaborate this theme.

        Each knowledge item should:
        - Have a clear, concise title (1-200 characters; unique among items for this theme)
        - Include the full, standalone explanation, statement, or instruction
        - Be classified as 'conceptual', 'procedural', or 'mixed' (for 'info_type')
        - Include 1 or more relevant tags

        Do NOT output any 'parents' or 'children' field.

        DOCUMENT CONTENT:
        {validate_content_length(document.content, getattr(config, 'max_content_length', 8000))}

        OUTPUT FORMAT - JSON ONLY:
        {{
          "theme": "{theme["title"]}",
          "nuggets": [
            {{
              "title": "Short descriptive title",
              "content": "Full, self-contained fact, explanation, or instruction",
              "info_type": "conceptual|procedural|mixed",
              "tags": ["relevant", "keywords"]
            }}
          ]
        }}

        OUTPUT ONLY THE JSON - NO EXPLANATIONS OR MARKDOWN.
        """
    
        print("\n==================\n[DEBUG] AI Prompt for NUGGET EXTRACTION (theme: '{}'):\n==================\n".format(theme["title"]))
        print(prompt)
        print("\n==================\n")
    
        result = call_openai_robust(
            prompt=prompt,
            max_tokens=2000,
            temperature=0.1,
            system_message="You are a precise knowledge extraction system. Always output valid JSON.",
            model=self.model
        )
    
        content = result.get("content", "").strip()
    
        print("\n==================\n[DEBUG] AI Response for NUGGET EXTRACTION (theme: '{}'):\n==================\n".format(theme["title"]))
        print(content)
        print("\n==================\n")
    
        nuggets_data = self._extract_and_validate_json(content, attempt=0, key="nuggets")
    
        if not nuggets_data:
            logger.error(f"Failed to extract nuggets for theme '{theme['title']}' from {document.filename}")
            return []
    
        if isinstance(nuggets_data, dict) and "nuggets" in nuggets_data:
            return nuggets_data["nuggets"]
        if isinstance(nuggets_data, list):
            return nuggets_data
        logger.error(f"Unexpected nuggets_data format for theme '{theme['title']}' from {document.filename}: {type(nuggets_data)}")
        return []


    
    
# ===============================
# SECTION 4: Enhanced Knowledge Graph
# ===============================

class KnowledgeGraph:
    """
    Enhanced knowledge graph with cycle detection, validation, and performance optimizations.
    """

    def __init__(self):
        self.nodes: Dict[str, KnowledgeNode] = {}
        self._adjacency_cache: Dict[str, Dict[str, Set[str]]] = {
            'children': {},
            'parents': {},
            'ancestors': {},
            'descendants': {}
        }
        self._cache_valid = False
        self._metrics = {
            'nodes_added': 0,
            'links_created': 0,
            'cycles_detected': 0,
            'cache_hits': 0,
            'cache_misses': 0
        }

    def add_node(self, node: KnowledgeNode) -> bool:
        """
        Add a node to the graph with validation.
        
        Returns:
            bool: True if node was added, False if it already existed
        """
        if node.id in self.nodes:
            # Instead of warning, merge the document references
            existing_node = self.nodes[node.id]
            existing_node.doc_refs.update(node.doc_refs)
            existing_node.tags.update(node.tags)
            logger.debug(f"Node {node.id!r} already exists, merged references")
            return False
        
        self.nodes[node.id] = node
        self._invalidate_cache()
        self._metrics['nodes_added'] += 1
        return True
    
    def link_parent_child(self, parent_id: str, child_id: str) -> bool:
        """
        Link two nodes with cycle detection and self-link prevention.
        
        Returns:
            bool: True if link was created, False if it would create a cycle or self-link
        """
        # Prevent self-linking
        if parent_id == child_id:
            logger.debug(f"Skipping self-link for node {parent_id}")
            return False
        
        if parent_id not in self.nodes or child_id not in self.nodes:
            logger.error(f"Cannot link: {parent_id!r} or {child_id!r} not found in graph")
            return False
        
        # Check if link already exists
        if child_id in self.nodes[parent_id].children:
            logger.debug(f"Link already exists: {parent_id} -> {child_id}")
            return False
        
        # Check for cycle
        if self._would_create_cycle(parent_id, child_id):
            logger.debug(f"Cycle would be created: cannot link {parent_id} -> {child_id}")
            self._metrics['cycles_detected'] += 1
            return False
        
        # Create the link
        self.nodes[parent_id].add_child(child_id)
        self.nodes[child_id].add_parent(parent_id)
        self._invalidate_cache()
        self._metrics['links_created'] += 1
        return True

    def _would_create_cycle(self, parent_id: str, child_id: str) -> bool:
        """Check if adding a link would create a cycle using BFS traversal."""
        # If child_id is already an ancestor of parent_id, adding this link would create a cycle
        # Use BFS to find if there's a path from child_id to parent_id
        visited = set()
        queue = deque([child_id])
        
        while queue:
            current = queue.popleft()
            if current == parent_id:
                return True  # Found a path, would create cycle
            
            if current in visited:
                continue
            visited.add(current)
            
            # Add children to queue (following the direction of existing links)
            node = self.nodes.get(current)
            if node:
                for child in node.children:
                    if child not in visited:
                        queue.append(child)
        
        return False  # No path found, safe to add link
    
    def _get_ancestors_cached(self, node_id: str) -> Set[str]:
        """Get ancestors with caching."""
        if not self._cache_valid:
            self._rebuild_cache()
        
        if node_id in self._adjacency_cache['ancestors']:
            self._metrics['cache_hits'] += 1
            return self._adjacency_cache['ancestors'][node_id]
        
        self._metrics['cache_misses'] += 1
        ancestors = self._compute_ancestors(node_id)
        self._adjacency_cache['ancestors'][node_id] = ancestors
        return ancestors

    def _compute_ancestors(self, node_id: str) -> Set[str]:
        """Compute all ancestors of a node."""
        ancestors = set()
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            node = self.nodes.get(current)
            if node:
                for parent_id in node.parents:
                    if parent_id not in ancestors:
                        ancestors.add(parent_id)
                        queue.append(parent_id)
        
        return ancestors

    def _rebuild_cache(self):
        """Rebuild adjacency cache."""
        self._adjacency_cache = {
            'children': {},
            'parents': {},
            'ancestors': {},
            'descendants': {}
        }
        
        # Pre-compute for all nodes
        for node_id in self.nodes:
            self._adjacency_cache['ancestors'][node_id] = self._compute_ancestors(node_id)
            self._adjacency_cache['descendants'][node_id] = self._compute_descendants(node_id)
        
        self._cache_valid = True

    def _compute_descendants(self, node_id: str) -> Set[str]:
        """Compute all descendants of a node."""
        descendants = set()
        visited = set()
        queue = deque([node_id])
        
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            
            node = self.nodes.get(current)
            if node:
                for child_id in node.children:
                    if child_id not in descendants:
                        descendants.add(child_id)
                        queue.append(child_id)
        
        return descendants

    def _invalidate_cache(self):
        """Invalidate adjacency cache."""
        self._cache_valid = False

    def get_node(self, node_id: str) -> Optional[KnowledgeNode]:
        """Get a node by ID."""
        return self.nodes.get(node_id)

    def get_children(self, node_id: str) -> List[str]:
        """Get immediate children of a node."""
        node = self.get_node(node_id)
        return list(node.children) if node else []

    def get_parents(self, node_id: str) -> List[str]:
        """Get immediate parents of a node."""
        node = self.get_node(node_id)
        return list(node.parents) if node else []

    def get_roots(self) -> List[KnowledgeNode]:
        """Get all root nodes (nodes with no parents)."""
        return [n for n in self.nodes.values() if not n.parents]

    def get_leaves(self) -> List[KnowledgeNode]:
        """Get all leaf nodes (nodes with no children)."""
        return [n for n in self.nodes.values() if not n.children]

    def walk_subtree(self, node_id: str, max_depth: int = 6) -> List[KnowledgeNode]:
        """Walk subtree with cycle detection and depth limiting."""
        if not self._cache_valid:
            self._rebuild_cache()
        
        visited = set()
        results = []
        queue = deque([(node_id, 0)])
        
        while queue:
            nid, depth = queue.popleft()
            
            if nid in visited or depth > max_depth:
                if depth > max_depth:
                    logger.debug(f"Max depth {max_depth} exceeded at node {nid}")
                continue
                
            visited.add(nid)
            node = self.get_node(nid)
            
            if node:
                results.append(node)
                for child_id in node.children:
                    if child_id not in visited:
                        queue.append((child_id, depth + 1))
        
        return results

    def get_ancestors(self, node_id: str, max_depth: int = 6) -> List[KnowledgeNode]:
        """Get all ancestors with depth limiting."""
        ancestors = self._get_ancestors_cached(node_id)
        return [self.nodes[aid] for aid in ancestors if aid in self.nodes][:max_depth]

    def find_by_tag(self, tag: str) -> List[KnowledgeNode]:
        """Find nodes by tag."""
        return [node for node in self.nodes.values() if tag in node.tags]

    def find_by_doc_ref(self, doc_id: str) -> List[KnowledgeNode]:
        """Find nodes by document reference."""
        return [node for node in self.nodes.values() if doc_id in node.doc_refs]

    def get_uncovered_nodes(self) -> List[KnowledgeNode]:
        """Get nodes not covered in synthesis."""
        return [
            node for node in self.nodes.values()
            if not node.metadata.get('synthesis_covered', False)
        ]

    def compute_foundation_levels(self) -> None:
        """Enhanced foundation level computation with better scoring."""
        if not self.nodes:
            return
        
        total_nodes = len(self.nodes)
        total_documents = len({doc for n in self.nodes.values() for doc in n.doc_refs}) or 1
        
        # Rebuild cache for accurate metrics
        if not self._cache_valid:
            self._rebuild_cache()
        
        scores: Dict[str, float] = {}
        
        for node_id, node in self.nodes.items():
            # Skip nodes with manually set foundation levels
            if node.foundation != 'auto':
                continue
            
            # Compute multiple scoring factors
            descendants = self._adjacency_cache['descendants'].get(node_id, set())
            structural_score = len(descendants) / total_nodes
            
            # Connectivity score (normalized)
            connectivity_score = min((len(node.children) + len(node.parents)) / 10, 1.0)
            
            # Cross-document score
            cross_doc_score = len(node.doc_refs) / total_documents
            
            # Content richness score
            content_score = min(len(node.content) / 1000, 1.0)
            
            # Weighted combination
            raw_score = (
                0.40 * structural_score +
                0.25 * cross_doc_score +
                0.20 * connectivity_score +
                0.15 * content_score
            )
            
            scores[node_id] = max(0.0, min(raw_score, 1.0))
        
        # Assign foundation levels based on scores
        for node_id, score in scores.items():
            node = self.nodes[node_id]
            if score >= 0.80:
                node.foundation = FoundationLevel.PILLAR
            elif score >= 0.60:
                node.foundation = FoundationLevel.CORE
            elif score >= 0.40:
                node.foundation = FoundationLevel.SUPPORTING
            elif score >= 0.20:
                node.foundation = FoundationLevel.DETAIL
            else:
                node.foundation = FoundationLevel.ATOMIC
        
        logger.info(f"Computed foundation levels for {len(scores)} nodes")

    def ensure_roots(self) -> None:
        """Log the current number of root nodes (nodes with no parents). No artificial promotions."""
        logger.info(f"Graph has {len(self.get_roots())} root nodes")
    

    def get_metrics(self) -> Dict[str, Any]:
        """Get graph metrics."""
        metrics = self._metrics.copy()
        metrics.update({
            'total_nodes': len(self.nodes),
            'total_roots': len(self.get_roots()),
            'total_leaves': len(self.get_leaves()),
            'cache_valid': self._cache_valid
        })
        return metrics

    def validate_graph_integrity(self) -> Dict[str, Any]:
        """Validate graph integrity and return report."""
        issues = []

        # Check for orphaned references
        all_referenced_ids = set()
        for node in self.nodes.values():
            all_referenced_ids.update(node.parents)
            all_referenced_ids.update(node.children)

        orphaned_refs = all_referenced_ids - set(self.nodes.keys())
        if orphaned_refs:
            issues.append(f"Orphaned references detected: {orphaned_refs}")

        # Check for inconsistent bidirectional links
        for node_id, node in self.nodes.items():
            for child_id in node.children:
                child_node = self.nodes.get(child_id)
                if child_node and node_id not in child_node.parents:
                    issues.append(f"Inconsistent child link: {node_id} → {child_id}")

            for parent_id in node.parents:
                parent_node = self.nodes.get(parent_id)
                if parent_node and node_id not in parent_node.children:
                    issues.append(f"Inconsistent parent link: {parent_id} → {node_id}")

        # Check for disconnected nodes
        for node_id, node in self.nodes.items():
            if not node.parents and not node.children:
                issues.append(f"Disconnected node: {node_id}")

        # Check for cycles (just to ensure, even though cycles should be prevented)
        for node_id in self.nodes:
            if node_id in self._get_ancestors_cached(node_id):
                issues.append(f"Cycle detected involving node: {node_id}")

        integrity_report = {
            'is_valid': len(issues) == 0,
            'issues': issues,
            'total_nodes': len(self.nodes),
            'total_roots': len(self.get_roots()),
            'total_leaves': len(self.get_leaves()),
            'orphaned_references': list(orphaned_refs),
            'disconnected_nodes': [node_id for node_id, node in self.nodes.items() if not node.parents and not node.children]
        }

        if integrity_report['is_valid']:
            logger.info("Graph integrity validation passed with no issues.")
        else:
            logger.warning(f"Graph integrity validation found {len(issues)} issues.")

        return integrity_report

    def __repr__(self):
        return (f"<KnowledgeGraph: {len(self.nodes)} nodes, "
                f"{len(self.get_roots())} roots, {len(self.get_leaves())} leaves, "
                f"cache_valid={self._cache_valid}>")

# ===============================
# SECTION 5: Knowledge Graph Builder
# ===============================

class KnowledgeGraphBuilder:
    """
    Builds a KnowledgeGraph from a list of Document objects using enhanced extraction.
    """

    def __init__(self, openai_client, openai_model: str = "gpt-4o"):
        self.openai = openai_client
        self.model = openai_model
        self.extractor = AIKnowledgeExtractor(self.openai, self.model)

    def create_graph_from_documents(self, documents: List[Document]) -> KnowledgeGraph:
        kg = KnowledgeGraph()
        logger.info("Starting phased knowledge extraction from documents...")
    
        all_theme_nodes = []
        all_nugget_nodes = []
        
        # Track used IDs to ensure uniqueness
        used_ids = set()
        
        for doc in documents:
            try:
                # Phase 1: Extract themes (no parents, just levels)
                themes = self.extractor.extract_themes(doc)
                logger.info(f"[{doc.filename}] Extracted {len(themes)} themes")
                for theme_idx, theme in enumerate(themes):
                    theme_node = {
                        'title': theme['title'],
                        'content': f"(Theme node for '{theme['title']}')",
                        'level': theme['level'],
                        'info_type': 'conceptual',
                        'tags': [],
                        'doc_id': doc.id,
                        'theme_idx': theme_idx,
                        # Ignore any parents field, we will set relationships later
                    }
                    all_theme_nodes.append((doc, theme_node))
    
                # Phase 2: Extract nuggets per theme (no parents field)
                for theme_idx, theme in enumerate(themes):
                    nuggets = self.extractor.extract_theme_nuggets(doc, theme)
                    logger.info(f"[{doc.filename}] Extracted {len(nuggets)} nuggets for theme '{theme['title']}'")
                    for nugget_idx, nugget in enumerate(nuggets):
                        node = {
                            'title': nugget['title'],
                            'content': nugget['content'],
                            'level': theme['level'],
                            'info_type': nugget['info_type'],
                            'tags': nugget['tags'],
                            'doc_id': doc.id,
                            'theme_idx': theme_idx,
                            'nugget_idx': nugget_idx,
                            # No parents; relationship will be set below
                        }
                        all_nugget_nodes.append((doc, node))
            except Exception as ex:
                logger.error(f"[{doc.filename}] Error processing document: {ex}")
                continue
    
        # Build all nodes, mapping title to node ID
        node_objs = []
    
        # 1. Theme nodes
        for doc, node_data in all_theme_nodes:
            # Create unique ID using document ID, theme index, and content hash
            unique_content = f"{doc.id}:{node_data['theme_idx']}:{node_data['title']}"
            node_id = generate_secure_id("theme", unique_content)
            
            # Ensure uniqueness
            counter = 0
            original_id = node_id
            while node_id in used_ids:
                counter += 1
                node_id = f"{original_id}_{counter}"
            used_ids.add(node_id)
            
            knowledge_node = KnowledgeNode(
                id=node_id,
                title=node_data['title'],
                content=node_data['content'],
                node_type=node_data['level'],
                foundation=node_data['level'],
                info_type=node_data['info_type'],
                doc_refs={doc.id},
                parents=set(),    # Populated by level linking below
                children=set(),
                tags=set(node_data['tags']),
                metadata={'source_doc': doc.filename, 'node_type': 'theme'}
            )
            kg.add_node(knowledge_node)
            node_objs.append(knowledge_node)
    
        # 2. Nugget nodes (each nugget shares its theme's level)
        for doc, node_data in all_nugget_nodes:
            # Create unique ID using document ID, theme index, nugget index, and content hash
            unique_content = f"{doc.id}:{node_data['theme_idx']}:{node_data['nugget_idx']}:{node_data['title']}:{node_data['content'][:100]}"
            node_id = generate_secure_id("nugget", unique_content)
            
            # Ensure uniqueness
            counter = 0
            original_id = node_id
            while node_id in used_ids:
                counter += 1
                node_id = f"{original_id}_{counter}"
            used_ids.add(node_id)
            
            knowledge_node = KnowledgeNode(
                id=node_id,
                title=node_data['title'],
                content=node_data['content'],
                node_type=node_data['level'],
                foundation=node_data['level'],
                info_type=node_data['info_type'],
                doc_refs={doc.id},
                parents=set(),
                children=set(),
                tags=set(node_data['tags']),
                metadata={'source_doc': doc.filename, 'node_type': 'nugget'}
            )
            kg.add_node(knowledge_node)
            node_objs.append(knowledge_node)
    
        # Group nodes by level (for strict level hierarchy linking)
        level_order = ['pillar', 'core', 'supporting', 'detail', 'atomic']
        level_to_nodes = {level: [] for level in level_order}
        for node in node_objs:
            lvl = getattr(node, 'node_type', 'atomic')
            if lvl not in level_to_nodes:
                lvl = 'atomic'
            level_to_nodes[lvl].append(node)
    
        # Link nodes: each lower-level node gets *all* nodes at next-higher level as parent
        # But avoid self-links and excessive linking
        for i in range(1, len(level_order)):
            higher_level = level_order[i-1]
            lower_level = level_order[i]
            
            # Only link if we have nodes at both levels
            if level_to_nodes[higher_level] and level_to_nodes[lower_level]:
                # Limit linking to avoid excessive connections
                max_parents_per_child = min(3, len(level_to_nodes[higher_level]))
                
                for child in level_to_nodes[lower_level]:
                    # Link to first few parents at higher level (not all)
                    for parent in level_to_nodes[higher_level][:max_parents_per_child]:
                        if parent.id != child.id:  # Avoid self-links
                            kg.link_parent_child(parent.id, child.id)
    
        kg.compute_foundation_levels()
        kg.ensure_roots()
        logger.info(f"Knowledge graph built successfully: {len(kg.nodes)} nodes")
        return kg
    



# ===============================
# SECTION 6: Knowledge Synthesizer
# ===============================

class KnowledgeSynthesizer:
    """
    Synthesizes structured knowledge output from the KnowledgeGraph with coherent document structure.
    """

    def __init__(self, kg: KnowledgeGraph, openai_client, openai_model: str = "gpt-4o"):
        self.kg = kg
        self.openai = openai_client
        self.model = openai_model
        self.document_structure = {}
        self.synthesis_content = {}

    def synthesize_coherent_knowledge(
        self, 
        output_format: str = "comprehensive_guide", 
        writing_style: str = None, 
        structural_mode: str = "flat"
    ) -> str:
        """
        Creates a coherent, structured synthesis of the knowledge graph, routed by output_format.
        """
        logger.info(f"Starting knowledge synthesis with output_format: {output_format}")
    
        # Add all your new output modes here:
        if output_format == "faq":
            return self._generate_faq_output(writing_style)
        elif output_format == "quick_reference":
            return self._generate_quick_reference_output(writing_style)
        elif output_format == "executive_summary":
            return self._generate_executive_summary_output(writing_style)
        elif output_format == "best_practices_checklist":
            return self._generate_checklist_output(writing_style)
        # You can add more modes if you like!
    
        # Fallback to standard (hierarchical) synthesis
        structure_plan = self._analyze_and_plan_structure()
        introduction = self._generate_comprehensive_introduction(structure_plan, writing_style)
        if structural_mode == "structural":
            section_contents = self._generate_structured_content_graphwise(structure_plan, writing_style)
        else:
            section_contents = self._generate_structured_content(structure_plan, writing_style)
        return self._assemble_final_document(introduction, section_contents, structure_plan, getattr(self, 'title', None))
    
    
    def _generate_faq_output(self, writing_style=None) -> str:
        """
        Synthesizes the knowledge graph into a high-quality FAQ format using the LLM for grouping and natural question formation.
        """
        # Gather all key nodes, sort by importance
        faq_nodes = sorted(self.kg.nodes.values(), key=lambda n: getattr(n, 'foundation', FoundationLevel.ATOMIC))
        # Prepare the FAQ prompt
        qas = []
        for node in faq_nodes:
            # Try to turn the title into a question, fall back to 'What is...'
            q = self._faqify_question(node.title)
            a = node.content.strip()
            qas.append({'question': q, 'answer': a})
        # Compose an LLM prompt for a beautiful FAQ
        base_prompt = (
            "Given the following list of knowledge items (each with a proposed FAQ question and its answer), "
            "generate a well-structured, readable FAQ section in Markdown. "
            "For each, ensure the question is natural and the answer is clear. "
            "Group very similar or redundant questions together if possible. "
            "Format each entry as:\n\n"
            "**Q:** ...\n\n**A:** ...\n\n"
            "If possible, sort questions from basic to advanced."
            "\n\nFAQ ENTRIES:\n"
        )
        for qa in qas:
            base_prompt += f"\nQ: {qa['question']}\nA: {qa['answer']}\n"
    
        result = call_openai_robust(
            prompt=base_prompt,
            max_tokens=3500,
            temperature=0.18,
            system_message="You are a world-class technical FAQ writer. Output only the Markdown FAQ section.",
            model=self.model,
        )
        return "# Frequently Asked Questions\n\n" + result.get("content", "").strip()
    
    def _faqify_question(self, title: str) -> str:
        """
        Turns a node title into a more natural FAQ-style question.
        """
        t = title.strip()
        # Try to avoid repeating 'What is...' if title already starts like a question
        if re.match(r"^(what|how|why|when|who|where)\b", t.lower()):
            return t if t.endswith("?") else t + "?"
        # If it's about a process, use "How does X work?"
        if any(x in t.lower() for x in ("process", "step", "procedure", "method", "flow")):
            return f"How does {t} work?"
        # Else default: "What is X?"
        return f"What is {t}?"
    
    def _generate_quick_reference_output(self, writing_style=None) -> str:
        """
        Synthesizes the knowledge graph into a quick reference table/cheat sheet.
        """
        ref_nodes = sorted(self.kg.nodes.values(), key=lambda n: getattr(n, 'foundation', FoundationLevel.ATOMIC))
        # Make a Markdown table (title | short summary)
        lines = [
            "# Quick Reference\n",
            "| Concept | Key Point |",
            "|---|---|",
        ]
        for node in ref_nodes:
            summary = node.content.strip().split("\n")[0][:200]
            lines.append(f"| **{node.title}** | {summary} |")
        return "\n".join(lines)

    def _generate_executive_summary_output(self, writing_style=None) -> str:
        """
        Synthesizes a concise executive summary from pillar/core nodes.
        """
        key_nodes = [
            n for n in self.kg.nodes.values() if getattr(n, 'foundation', None) in (FoundationLevel.PILLAR, FoundationLevel.CORE)
        ]
        # Ask the LLM for an executive summary
        core_points = [f"{n.title}: {n.content[:200]}" for n in key_nodes]
        prompt = (
            "Write a highly concise, compelling executive summary covering the following key points. "
            "Do not copy the points verbatim; synthesize them into 1-3 paragraphs of prose. "
            "Audience is a busy executive; no technical jargon. Format as Markdown.\n\n"
            "KEY POINTS:\n" + "\n".join(core_points)
        )
        result = call_openai_robust(
            prompt=prompt,
            max_tokens=700,
            temperature=0.18,
            system_message="You are a world-class executive summary writer. Output only the summary in Markdown.",
            model=self.model,
        )
        return "# Executive Summary\n\n" + result.get("content", "").strip()
    
    def _generate_checklist_output(self, writing_style=None) -> str:
        """
        Synthesizes a best practices checklist from all supporting/detail/atomic nodes.
        """
        checklist_nodes = [
            n for n in self.kg.nodes.values()
            if getattr(n, 'foundation', None) in (FoundationLevel.SUPPORTING, FoundationLevel.DETAIL, FoundationLevel.ATOMIC)
        ]
        items = []
        for node in checklist_nodes:
            # Use content's first line or main actionable
            item = f"- [ ] {node.title}: {node.content.splitlines()[0][:120]}"
            items.append(item)
        return "# Best Practices Checklist\n\n" + "\n".join(items)
    

    def _analyze_and_plan_structure(self) -> Dict[str, Any]:
        """
        Analyzes the knowledge graph to determine optimal document structure.
        """
        # Get all nodes organized by foundation level
        nodes_by_level = self._organize_nodes_by_level()
        
        # Get thematic clusters
        thematic_clusters = self._identify_thematic_clusters()
        
        # Prepare structure analysis prompt
        structure_prompt = f"""
        Analyze this knowledge graph structure and design an optimal document organization:

        FOUNDATION LEVELS:
        - Pillar nodes ({len(nodes_by_level.get('pillar', []))}): {[n.title for n in nodes_by_level.get('pillar', [])][:10]}
        - Core nodes ({len(nodes_by_level.get('core', []))}): {[n.title for n in nodes_by_level.get('core', [])][:10]}
        - Supporting nodes ({len(nodes_by_level.get('supporting', []))}): {[n.title for n in nodes_by_level.get('supporting', [])][:10]}
        - Detail nodes ({len(nodes_by_level.get('detail', []))}): {[n.title for n in nodes_by_level.get('detail', [])][:10]}
        - Atomic nodes ({len(nodes_by_level.get('atomic', []))}): {[n.title for n in nodes_by_level.get('atomic', [])][:10]}

        THEMATIC CLUSTERS:
        {self._format_thematic_clusters(thematic_clusters)}

        Design a document structure that:
        1. Starts with a comprehensive introduction covering all major themes
        2. Organizes content hierarchically (major sections, subsections, details)
        3. Ensures logical flow from fundamental concepts to specific applications
        4. Integrates related concepts rather than treating them in isolation
        5. Provides appropriate depth progression

        OUTPUT FORMAT - JSON:
        {{
            "introduction_scope": "What should the introduction cover?",
            "main_sections": [
                {{
                    "title": "Section Title",
                    "scope": "What this section covers",
                    "foundation_level": "pillar|core|supporting",
                    "subsections": [
                        {{
                            "title": "Subsection Title",
                            "scope": "What this subsection covers",
                            "foundation_level": "core|supporting|detail",
                            "detail_integration": "How to integrate atomic-level details"
                        }}
                    ]
                }}
            ],
            "content_flow_strategy": "How should content flow between sections?"
        }}
        """

        result = call_openai_robust(
            prompt=structure_prompt,
            max_tokens=2000,
            temperature=0.2,
            system_message="You are an expert document architect. Always output valid JSON.",
            model=self.model
        )

        structure_data = self._extract_structure_plan(result.get("content", ""))
        if not structure_data:
            # Fallback to default structure
            structure_data = self._create_default_structure(nodes_by_level)

        return structure_data

    def _generate_comprehensive_introduction(self, structure_plan: Dict[str, Any], writing_style: str = None) -> str:
        """
        Generates a comprehensive introduction that provides overview of all root themes.
        """
        pillar_nodes = [n for n in self.kg.nodes.values() if getattr(n, 'foundation', 'auto') == FoundationLevel.PILLAR]
        core_nodes = [n for n in self.kg.nodes.values() if getattr(n, 'foundation', 'auto') == FoundationLevel.CORE]
        
        key_themes = []
        for node in pillar_nodes + core_nodes[:10]:
            key_themes.append(f"- {node.title}: {node.content[:200]}...")
    
        intro_prompt = f"""
    Create a comprehensive introduction that provides a broad overview of this knowledge domain.
    
    INTRODUCTION SCOPE: {structure_plan.get('introduction_scope', 'Provide comprehensive overview')}
    DESIRED WRITING STYLE: {writing_style or 'Default style'}
    
    KEY THEMES TO COVER:
    {chr(10).join(key_themes)}
    
    MAIN SECTIONS TO PREVIEW:
    {chr(10).join([f"- {section['title']}: {section['scope']}" for section in structure_plan.get('main_sections', [])])}
    
    Instructions:
    - Establish the domain and its importance
    - Provide conceptual framework for understanding the material
    - Preview all major themes and their relationships
    - Set expectations for the reader's journey through the content
    - Motivate why this knowledge matters
    - **Format your response as well-structured Markdown using headings, bold, lists, and code (or quote) blocks as appropriate.**
    - Write a compelling, comprehensive introduction (500-800 words) that serves as an intellectual roadmap.
    """
    
        result = call_openai_robust(
            prompt=intro_prompt,
            max_tokens=1000,
            temperature=0.3,
            system_message="You are an expert technical writer creating compelling introductions. Output in Markdown.",
            model=self.model
        )
    
        return result.get("content", "")
    

    def _generate_structured_content(self, structure_plan: Dict[str, Any], writing_style: str = None) -> Dict[str, str]:
        """
        Generates content for each section according to the structure plan.
        """
        section_contents = {}
        
        for section in structure_plan.get('main_sections', []):
            # Find nodes relevant to this section
            relevant_nodes = self._find_nodes_for_section(section)
            
            # Generate main section content
            section_content = self._generate_section_content(section, relevant_nodes, writing_style)
            section_contents[section['title']] = section_content
            
            # Generate subsection content
            for subsection in section.get('subsections', []):
                subsection_nodes = self._find_nodes_for_subsection(subsection, relevant_nodes)
                subsection_content = self._generate_subsection_content(subsection, subsection_nodes, writing_style)
                section_contents[f"{section['title']} - {subsection['title']}"] = subsection_content
    
        return section_contents
    
    def _generate_structured_content_graphwise(self, structure_plan: Dict[str, Any], writing_style: str = None) -> Dict[str, str]:
        """
        Generates content for each section according to the structure plan,
        using graph structure to guide content flow—starting from a unified umbrella node above all roots.
        """
        section_contents = {}
    
        for section in structure_plan.get('main_sections', []):
            # Generate an all-encompassing node as umbrella for root nodes
            umbrella_node = self._generate_all_encompassing_node(writing_style)
            roots = self._find_graph_roots_for_section(section)
            all_nodes = []
    
            # Use the umbrella as the new parent in the narrative structure
            if umbrella_node and roots:
                # Construct a pseudo KnowledgeNode for synthesis flow only
                from types import SimpleNamespace
                umbrella_knowledgenode = SimpleNamespace(
                    id="umbrella_node",
                    title=umbrella_node['title'],
                    content=umbrella_node['content'],
                    parents=[],
                    children=[n.id for n in roots]
                )
                all_nodes = [umbrella_knowledgenode] + roots
            else:
                all_nodes = roots
    
            section_content = self._synthesize_section_graphwise(section, all_nodes, writing_style)
            section_contents[section['title']] = section_content
    
            for subsection in section.get('subsections', []):
                relevant_nodes = self._find_graph_nodes_for_subsection(subsection, roots)
                subsection_content = self._synthesize_section_graphwise(subsection, relevant_nodes, writing_style, depth=2)
                section_contents[f"{section['title']} - {subsection['title']}"] = subsection_content
    
        return section_contents
    
    
    def _find_graph_roots_for_section(self, section: Dict[str, Any]) -> List[KnowledgeNode]:
        """Get root nodes for a section based on foundation level, using graph roots and not just flat filtering."""
        level = section.get('foundation_level', 'core')
        level_enum = {
            'pillar': FoundationLevel.PILLAR,
            'core': FoundationLevel.CORE,
            'supporting': FoundationLevel.SUPPORTING,
            'detail': FoundationLevel.DETAIL,
            'atomic': FoundationLevel.ATOMIC
        }.get(level, FoundationLevel.CORE)
    
        # Prefer nodes at this level that are also roots (no parents)
        roots = [
            n for n in self.kg.nodes.values()
            if n.foundation == level_enum and not n.parents
        ]
        if roots:
            return roots
        # Fallback: any nodes at this level
        return [n for n in self.kg.nodes.values() if n.foundation == level_enum]
    
    def _find_graph_nodes_for_subsection(self, subsection: Dict[str, Any], section_roots: List[KnowledgeNode]) -> List[KnowledgeNode]:
        """Get all children and descendants of section_roots matching the subsection's level."""
        desired_level = subsection.get('foundation_level', 'detail')
        level_enum = {
            'pillar': FoundationLevel.PILLAR,
            'core': FoundationLevel.CORE,
            'supporting': FoundationLevel.SUPPORTING,
            'detail': FoundationLevel.DETAIL,
            'atomic': FoundationLevel.ATOMIC
        }.get(desired_level, FoundationLevel.DETAIL)
        # Walk subtree from section_roots
        nodes = []
        seen = set()
        queue = list(section_roots)
        while queue:
            node = queue.pop(0)
            if node.id in seen:
                continue
            seen.add(node.id)
            if node.foundation == level_enum:
                nodes.append(node)
            queue.extend([self.kg.get_node(cid) for cid in node.children if self.kg.get_node(cid)])
        return nodes
    
    def _synthesize_section_graphwise(self, section: Dict[str, Any], nodes: List[KnowledgeNode], writing_style: str = None, depth: int = 1) -> str:
        """
        Synthesize content for a section/subsection based on graph structure.
        """
        # Compose a content tree for the AI, with relationships
        content_outline = []
        for node in nodes:
            children = [self.kg.get_node(cid) for cid in node.children if self.kg.get_node(cid)]
            child_titles = ', '.join(c.title for c in children)
            content_outline.append(
                f"Title: {node.title}\nContent: {node.content[:180]}...\nDirectly builds on: {', '.join([self.kg.get_node(pid).title for pid in node.parents if self.kg.get_node(pid)])}\n"
                f"Direct subtopics: {child_titles}\n"
            )
    
        prompt = f"""
    You are synthesizing a section titled "{section['title']}" using an explicit knowledge graph structure.
    
    SECTION SCOPE: {section.get('scope', 'N/A')}
    DESIRED WRITING STYLE: {writing_style or 'Default style'}
    FOUNDATION LEVEL: {section.get('foundation_level', 'core')}
    
    CONTENT OUTLINE (ordered by conceptual dependency, per graph structure):
    {chr(10).join(content_outline)}
    
    Instructions:
    - Start from high-level/root concepts, then progressively introduce dependencies and subtopics.
    - Explicitly guide the reader through the graph, noting why each concept follows from its parents or leads into its children.
    - When integrating each concept, explain its role in the structure and how it connects to what came before/after.
    - Use bullet points or diagrams if helpful to visualize structure.
    - Format as well-structured Markdown using headings, lists, bold, and links where appropriate.
    - Highlight cross-links or recurring dependencies.
    - For subtopics (children), introduce them in the order suggested by the graph and explain the rationale for their placement.
    
    Write {300 if depth > 1 else 600}–{600 if depth > 1 else 1200} words that make the graph structure apparent to the reader.
    """
        result = call_openai_robust(
            prompt=prompt,
            max_tokens=1500 if depth == 1 else 700,
            temperature=0.2,
            system_message="You are a structural knowledge synthesizer who uses explicit graph structure. Output in Markdown.",
            model=self.model
        )
        return result.get("content", "")
    

    def _generate_section_content(self, section: Dict[str, Any], relevant_nodes: List[KnowledgeNode], writing_style: str = None) -> str:
        """
        Generates synthesized content for a main section.
        """
        node_contents = []
        for node in relevant_nodes:
            node_contents.append(f"**{node.title}**: {node.content}")
    
        synthesis_prompt = f"""
    Synthesize content for the section: "{section['title']}"
    
    SECTION SCOPE: {section['scope']}
    FOUNDATION LEVEL: {section['foundation_level']}
    DESIRED WRITING STYLE: {writing_style or 'Default style'}
    
    RELEVANT KNOWLEDGE:
    {chr(10).join(node_contents)}
    
    Instructions:
    - Provide a clear conceptual framework for this topic
    - Integrate the knowledge pieces into a flowing narrative
    - Explain relationships between concepts
    - Build understanding progressively
    - Connect to broader themes in the document
    - **Format your response as well-structured Markdown using headings, bold, lists, and code (or quote) blocks as appropriate.**
    - Generate 800-1200 words of well-structured content with clear subheadings.
    """
    
        result = call_openai_robust(
            prompt=synthesis_prompt,
            max_tokens=1500,
            temperature=0.2,
            system_message="You are synthesizing knowledge into coherent sections. Output in Markdown.",
            model=self.model
        )
    
        return result.get("content", "")

    def _generate_subsection_content(self, subsection: Dict[str, Any], relevant_nodes: List[KnowledgeNode], writing_style: str = None) -> str:
        """
        Generates synthesized content for a subsection with integrated details.
        """
        detail_nodes = [n for n in relevant_nodes if getattr(n, 'foundation', 'auto') in [FoundationLevel.DETAIL, FoundationLevel.ATOMIC]]
        detail_content = []
        for node in detail_nodes:
            detail_content.append(f"- {node.title}: {node.content}")
    
        synthesis_prompt = f"""
    Create detailed content for subsection: "{subsection['title']}"
    
    SUBSECTION SCOPE: {subsection['scope']}
    DETAIL INTEGRATION STRATEGY: {subsection.get('detail_integration', 'Integrate seamlessly')}
    DESIRED WRITING STYLE: {writing_style or 'Default style'}
    
    DETAILED KNOWLEDGE TO INTEGRATE:
    {chr(10).join(detail_content)}
    
    Instructions:
    - Provide specific, actionable knowledge
    - Integrate atomic-level details naturally into the narrative
    - Maintain clear organization and flow
    - Include concrete examples and applications
    - Connect details back to broader principles
    - **Format your response as well-structured Markdown using headings, bold, lists, and code (or quote) blocks as appropriate.**
    - Generate 400-600 words of detailed, practical content.
    """
    
        result = call_openai_robust(
            prompt=synthesis_prompt,
            max_tokens=800,
            temperature=0.2,
            system_message="You are creating detailed subsections with integrated specifics. Output in Markdown.",
            model=self.model
        )
    
        return result.get("content", "")

    def _assemble_final_document(self, introduction: str, section_contents: Dict[str, str], structure_plan: Dict[str, Any], title: str = None) -> str:
        document_parts = []
    
        # Optional: Document Title
        if title:
            document_parts.append(f"# {title.replace('_', ' ')}\n\n")
    
        # Introduction section
        document_parts.append("## Introduction\n")
        document_parts.append(introduction.strip())
        document_parts.append("\n\n---\n\n")
    
        # Main Sections
        for section in structure_plan.get('main_sections', []):
            # Section Title
            document_parts.append(f"## {section['title']}\n")
            # Section Content
            section_content = section_contents.get(section['title'], '').strip()
            if section_content:
                document_parts.append(section_content)
                document_parts.append("\n\n---\n\n")
            
            # Subsections
            for subsection in section.get('subsections', []):
                document_parts.append(f"### {subsection['title']}\n")
                subsection_key = f"{section['title']} - {subsection['title']}"
                subsection_content = section_contents.get(subsection_key, '').strip()
                if subsection_content:
                    document_parts.append(subsection_content)
                    document_parts.append("\n\n---\n\n")
    
        # Remove any trailing horizontal rules for clean ending
        while document_parts and document_parts[-1].strip() == '---':
            document_parts.pop()
    
        # Join everything with a single newline to avoid accidental double-spacing
        return "".join(document_parts)
    
    
    def _organize_nodes_by_level(self) -> Dict[str, List[KnowledgeNode]]:
        """Organize nodes by foundation level."""
        levels = {
            'pillar': [],
            'core': [],
            'supporting': [],
            'detail': [],
            'atomic': []
        }
        
        for node in self.kg.nodes.values():
            foundation = getattr(node, 'foundation', 'auto')
            if foundation == FoundationLevel.PILLAR:
                levels['pillar'].append(node)
            elif foundation == FoundationLevel.CORE:
                levels['core'].append(node)
            elif foundation == FoundationLevel.SUPPORTING:
                levels['supporting'].append(node)
            elif foundation == FoundationLevel.DETAIL:
                levels['detail'].append(node)
            else:
                levels['atomic'].append(node)
        
        return levels

    def _identify_thematic_clusters(self) -> List[Dict[str, Any]]:
        """Identify clusters of thematically related nodes."""
        clusters = []
        processed = set()
        
        for node in self.kg.nodes.values():
            if node.id in processed:
                continue
                
            # Find nodes with similar tags or content themes
            cluster = self._find_thematic_cluster(node)
            if len(cluster) > 1:
                clusters.append({
                    'theme': cluster[0].title,
                    'nodes': cluster,
                    'size': len(cluster)
                })
                processed.update(n.id for n in cluster)
        
        return clusters

    def _find_thematic_cluster(self, seed_node: KnowledgeNode) -> List[KnowledgeNode]:
        """Find nodes thematically related to the seed node."""
        cluster = [seed_node]
        
        # Simple clustering based on shared tags and connected nodes
        for node in self.kg.nodes.values():
            if node.id != seed_node.id:
                # Check for tag overlap
                tag_overlap = len(seed_node.tags & node.tags)
                # Check for direct connection
                is_connected = (node.id in seed_node.children or 
                              node.id in seed_node.parents or
                              seed_node.id in node.children or
                              seed_node.id in node.parents)
                
                if tag_overlap > 0 or is_connected:
                    cluster.append(node)
        
        return cluster

    def _format_thematic_clusters(self, clusters: List[Dict[str, Any]]) -> str:
        """Format thematic clusters for display."""
        if not clusters:
            return "No clear thematic clusters identified."
        
        formatted = []
        for cluster in clusters[:10]:  # Limit to avoid overwhelming
            formatted.append(f"- {cluster['theme']} ({cluster['size']} nodes)")
        
        return "\n".join(formatted)

    def _extract_structure_plan(self, content: str) -> Optional[Dict[str, Any]]:
        """Extract structure plan from AI response."""
        try:
            # Try to parse as JSON
            data = json.loads(content)
            return data
        except:
            # Try to extract from markdown code block
            pattern = r'```(?:json)?\s*(\{.*?\})\s*```'
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    return json.loads(match)
                except:
                    continue
        
        return None

    def _create_default_structure(self, nodes_by_level: Dict[str, List[KnowledgeNode]]) -> Dict[str, Any]:
        """Create default structure if AI analysis fails."""
        return {
            "introduction_scope": "Provide comprehensive overview of all major themes and concepts",
            "main_sections": [
                {
                    "title": "Foundational Concepts",
                    "scope": "Core principles and pillar concepts",
                    "foundation_level": "pillar",
                    "subsections": [
                        {
                            "title": "Key Principles",
                            "scope": "Fundamental principles",
                            "foundation_level": "core",
                            "detail_integration": "Include specific examples and applications"
                        }
                    ]
                },
                {
                    "title": "Applied Knowledge",
                    "scope": "Practical applications and supporting concepts",
                    "foundation_level": "supporting",
                    "subsections": [
                        {
                            "title": "Implementation Details",
                            "scope": "Specific procedures and methods",
                            "foundation_level": "detail",
                            "detail_integration": "Provide step-by-step guidance"
                        }
                    ]
                }
            ],
            "content_flow_strategy": "Progress from fundamental concepts to specific applications"
        }

    def _find_nodes_for_section(self, section: Dict[str, Any]) -> List[KnowledgeNode]:
        """Find nodes relevant to a specific section."""
        foundation_level = section.get('foundation_level', 'core')
        
        # Map foundation level to enum
        level_map = {
            'pillar': FoundationLevel.PILLAR,
            'core': FoundationLevel.CORE,
            'supporting': FoundationLevel.SUPPORTING,
            'detail': FoundationLevel.DETAIL,
            'atomic': FoundationLevel.ATOMIC
        }
        
        target_level = level_map.get(foundation_level, FoundationLevel.CORE)
        
        # Find nodes at this level and related levels
        relevant_nodes = []
        for node in self.kg.nodes.values():
            node_level = getattr(node, 'foundation', 'auto')
            if node_level == target_level:
                relevant_nodes.append(node)
        
        return relevant_nodes

    def _find_nodes_for_subsection(self, subsection: Dict[str, Any], section_nodes: List[KnowledgeNode]) -> List[KnowledgeNode]:
        """Find nodes relevant to a specific subsection."""
        # Get children and related nodes of section nodes
        subsection_nodes = []
        
        for section_node in section_nodes:
            # Add direct children
            for child_id in section_node.children:
                child_node = self.kg.get_node(child_id)
                if child_node:
                    subsection_nodes.append(child_node)
            
            # Add nodes with similar tags
            for node in self.kg.nodes.values():
                if node not in subsection_nodes and node not in section_nodes:
                    if len(node.tags & section_node.tags) > 0:
                        subsection_nodes.append(node)
        
        return subsection_nodes
        
    def _generate_all_encompassing_node(self, writing_style: str = None) -> Dict[str, str]:
        """
        Uses AI to generate a single umbrella/generalization node covering all root nodes.
        Returns a dict with 'title' and 'content'.
        """
        roots = self.kg.get_roots()
        if not roots:
            return None
    
        root_titles = [n.title for n in roots]
        root_summaries = [f"- {n.title}: {n.content[:120]}..." for n in roots[:8]]
        prompt = f"""
    Given the following list of root knowledge nodes (the top-level ideas), synthesize a single all-encompassing conceptual generalization that unites and contextualizes all of them. 
    Provide:
    - A concise, meaningful, unique title (max 12 words)
    - A clear, substantial summary (2–3 paragraphs, 150-250 words), addressing how all these roots fit under this umbrella and why this generalization is necessary to understand the entire body of knowledge.
    
    Root node titles:
    {chr(10).join(root_titles)}
    
    Root summaries:
    {chr(10).join(root_summaries)}
    
    Style: {writing_style or 'Default style'}
    Output format:
    {{
      "title": "All-encompassing title",
      "content": "Cohesive, detailed summary as described"
    }}
    Output only the JSON—no markdown, no code blocks, no explanation.
    """
        result = call_openai_robust(
            prompt=prompt,
            max_tokens=400,
            temperature=0.15,
            system_message="You are a master conceptual summarizer. Output valid JSON only.",
            model=self.model
        )
        content = result.get("content", "").strip()
        try:
            data = json.loads(content)
            if "title" in data and "content" in data:
                return data
        except Exception:
            pass
        # fallback if parsing fails
        matches = re.findall(r'\{[\s\S]*\}', content)
        for m in matches:
            try:
                data = json.loads(m)
                if "title" in data and "content" in data:
                    return data
            except Exception:
                continue
        # As a last fallback, return a generic node
        return {
            "title": "General Overview",
            "content": "This document brings together all major themes and root concepts into a single unifying vision."
        }
    
    def identify_missing_steps_with_ai(self, ordered_nodes):
        """
        Given a list of ordered section objects (each with 'title' and 'content'), ask the AI if there are any missing conceptual steps
        in the progression from general to specific. Returns a list of gap descriptors with the indices of the nodes between which
        the gap occurs, and a brief description.
        """
        section_texts = ""
        for i, node in enumerate(ordered_nodes):
            section_texts += f"\n[{i}] {node.title}: {node.content[:200]}"
    
        prompt = f"""
    You are reviewing an educational document made up of the following sections, listed in order from general to specific.
    
    Please check if there are any important conceptual or explanatory steps missing between consecutive sections. For each place where you believe a new intermediate section should be inserted (to make the flow smoother and help readers understand), list:
    - The numbers of the two sections between which the step is missing
    - A short description (1-2 sentences) of what is missing or what should be explained
    
    Sections:
    {section_texts}
    
    Output JSON only, as a list like this:
    [
      {{"between_indices": [2,3], "description": "What needs to be explained here."}},
      ...
    ]
    (No markdown, no commentary, just the JSON array.)
    """
        result = call_openai_robust(
            prompt=prompt,
            max_tokens=800,
            temperature=0.15,
            system_message="You are an expert in educational content flow. Output JSON only.",
            model=self.model
        )
        content = result.get("content", "").strip()
        try:
            gaps = json.loads(content)
            assert isinstance(gaps, list)
            return gaps
        except Exception:
            # Fallback: try extracting JSON from any code block
            matches = re.findall(r'\[[\s\S]*\]', content)
            if matches:
                try:
                    gaps = json.loads(matches[0])
                    return gaps
                except Exception:
                    pass
        print("Failed to parse AI gap identification response.")
        return []
    

    def generate_bridging_sections_with_ai(self, ordered_nodes, gaps):
        """
        For each identified gap, call the AI to generate a bridging section between the two specified indices.
        Returns a list of dicts: { "insert_after": idx, "title": ..., "content": ... }
        """
        bridging_sections = []
        for gap in gaps:
            idx1, idx2 = gap.get('between_indices', [None, None])
            if idx1 is None or idx2 is None or idx1 < 0 or idx2 > len(ordered_nodes) or idx2 != idx1 + 1:
                continue  # Only support gaps between direct neighbors for now
    
            node_a = ordered_nodes[idx1]
            node_b = ordered_nodes[idx2]
    
            prompt = f"""
    You are improving an educational document.
    
    There is a gap in the flow of information between the following two sections:
    
    Section A (more general):
    Title: {node_a.title}
    Content: {node_a.content[:400]}
    
    Section B (more specific):
    Title: {node_b.title}
    Content: {node_b.content[:400]}
    
    Please write a new intermediate section to insert **between** A and B. It should:
    - Have a clear, descriptive title (1-200 characters)
    - Provide the necessary explanation, background, or conceptual link that connects A to B, making the progression smooth for readers.
    - Be detailed enough to stand alone as its own section.
    - Do not just summarize or repeat A and B.
    
    Output JSON only:
    {{
      "title": "Bridging Section Title",
      "content": "Complete bridging explanation."
    }}
    (No markdown, no commentary, just JSON.)
    """
            result = call_openai_robust(
                prompt=prompt,
                max_tokens=700,
                temperature=0.18,
                system_message="You are an educational content editor. Output valid JSON only.",
                model=self.model
            )
            content = result.get("content", "").strip()
            try:
                section = json.loads(content)
                bridging_sections.append({
                    "insert_after": idx1,
                    "title": section.get("title", "Bridging Section"),
                    "content": section.get("content", "")
                })
            except Exception:
                # Fallback: try extracting JSON object
                match = re.search(r'\{[\s\S]*\}', content)
                if match:
                    try:
                        section = json.loads(match.group())
                        bridging_sections.append({
                            "insert_after": idx1,
                            "title": section.get("title", "Bridging Section"),
                            "content": section.get("content", "")
                        })
                    except Exception:
                        pass
        return bridging_sections
    

def generate_smart_title_from_graph(kg: "KnowledgeGraph", openai_client, openai_model="gpt-4o"):
    """
    Generate a concise, filename-safe, meaningful title based on core/pillar themes of the knowledge graph.
    """
    core_themes = [n.title for n in kg.nodes.values() if getattr(n, "foundation", None) in ("pillar", "core")][:6]
    joined = "; ".join(core_themes)
    prompt = f"""
You are generating a filename-safe, succinct, and informative title for a synthesized knowledge document.
Themes to consider: {joined}
Rules:
- No more than 7 words.
- Do not include punctuation except dashes or underscores.
- Title should be understandable, professional, and make sense as a filename.
- Use Title_Case_With_Underscores or dashes (not spaces).

Only output the title. Do not write any explanations.
"""
    result = call_openai_robust(
        prompt=prompt,
        max_tokens=16,
        temperature=0.1,
        system_message="You are a filename and title generator. Output a filename-safe title only.",
        model=openai_model
    )
    raw = result.get("content", "").strip()
    # Final scrub: remove any stray quotes/punctuation
    cleaned = re.sub(r"[^a-zA-Z0-9_-]", "", raw.replace(" ", "_"))
    return cleaned[:80] or "Untitled"


# ===============================
# SECTION 7: Main Pipeline & Entry Point
# ===============================

def build_graph_and_synthesize(
    documents: List[Document],
    output_format: str,
    writing_style: str,
    structural_mode: str
):
    """
    Unified main pipeline that creates structured, coherent synthesis.

    Args:
        documents (List[Document]): List of Document objects to process.
        output_format (str): The desired format for the synthesized output.
        writing_style (str): The stylistic direction for the output.
        structural_mode (str): Either 'flat' or 'structural' synthesis mode.

    Returns:
        dict: Metrics and output file info, including the filename used.
    """
    logger.info("Initiating the enhanced knowledge synthesis pipeline...")

    # 1. Build knowledge graph from extracted document nodes
    kgb = KnowledgeGraphBuilder(openai_client, openai_model=config.openai_model)
    kg = kgb.create_graph_from_documents(documents)

    # 2. Validate the constructed knowledge graph for integrity
    integrity_report = kg.validate_graph_integrity()
    if not integrity_report['is_valid']:
        logger.warning(f"Graph integrity issues: {integrity_report['issues']}")

    # 3. Generate a smart AI-driven title based on graph content (themes)
    title = generate_smart_title_from_graph(kg, openai_client, openai_model=config.openai_model)
    logger.info(f"Generated document title: {title}")

    # 4. Synthesize a coherent document from the graph
    synthesizer = KnowledgeSynthesizer(kg, openai_client, openai_model=config.openai_model)
    synthesizer.title = title  # <-- add this line!
    synthesized_document = synthesizer.synthesize_coherent_knowledge(
        output_format=output_format,
        writing_style=writing_style,
        structural_mode=structural_mode
    )
    
    # 5. Build filename using title, all user choices, and timestamp
    short_style = writing_style.replace(" ", "_")[:24]
    short_format = output_format.replace(" ", "_")[:24]
    short_mode = structural_mode[:12]
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_base = f"{title}_{short_format}_{short_style}_{short_mode}_{now_str}".replace("__", "_")
    file_base = re.sub(r"[^a-zA-Z0-9_-]", "", file_base)
    output_path = Path(getattr(config, 'output_path', '.')) / f"{file_base}.md"

    # 6. Write to file and display
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(synthesized_document)

    print("=" * 80)
    print("SYNTHESIZED KNOWLEDGE DOCUMENT")
    print("=" * 80)
    print(synthesized_document)
    print("=" * 80)
    print(f"Output written to: {output_path}")
    logger.info(f"Synthesized document saved to: {output_path}")

    # 7. Return metrics and file info for further processing or tests
    return {
        'total_nodes': len(kg.nodes),
        'graph_integrity': integrity_report,
        'extraction_stats': kgb.extractor.get_extraction_stats(),
        'output_file': str(output_path),
        'ai_title': title
    }




# ===============================
# SECTION 8: Execution Entry Point
# ===============================
def select_writing_style():
    styles = [
        "Upbeat and energetic",
        "Factual and objective",
        "Conversational and friendly",
        "Concise and to-the-point",
        "Detailed and explanatory",
        "Storytelling/narrative",
        "Formal and academic",
        "Inspirational/motivational",
        "Humorous and light-hearted",
        "Authoritative and expert"
    ]
    print("\nWhat style would you like for the final output?")
    for idx, style in enumerate(styles, 1):
        print(f"{idx}. {style}")
    print("11. Enter your own custom style description")

    while True:
        choice = input("\nEnter the number (1-11) for your chosen style: ").strip()
        if not choice:
            print("You must select a style (press 1-11 or 11 for custom). Please try again.")
            continue
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= 10:
                return styles[num - 1]
            elif num == 11:
                custom = input("Enter your desired writing style description: ").strip()
                if not custom:
                    print("Custom style cannot be blank. Please enter a description.")
                    continue
                return custom
        print("Invalid input. Please enter a number between 1 and 11.")

def select_output_format():
    formats = [
        "comprehensive_guide",
        "quick_reference",
        "faq",
        "step_by_step_manual",
        "executive_summary",
        "annotated_outline",
        "best_practices_checklist",
        "case_studies",
        "problem_solution_playbook",
        "visual_roadmap"
    ]
    print("\nWhich output format would you like?")
    for idx, fmt in enumerate(formats, 1):
        # Make the display nice
        display_name = fmt.replace('_', ' ').title()
        print(f"{idx}. {display_name}")
    print("11. Enter a custom output format")

    while True:
        choice = input("\nEnter the number (1-11) for your chosen format: ").strip()
        if not choice:
            print("You must select a format (press 1-10, or 11 for custom). Please try again.")
            continue
        if choice.isdigit():
            num = int(choice)
            if 1 <= num <= 10:
                return formats[num - 1]
            elif num == 11:
                custom = input("Enter your desired output format: ").strip()
                if not custom:
                    print("Custom format cannot be blank. Please enter a description.")
                    continue
                return custom
        print("Invalid input. Please enter a number between 1 and 11.")

def select_structural_mode():
    print("\nHow should the document structure be used to guide synthesis?")
    print("1. Flat synthesis (like a textbook): Each section is treated independently, organized mostly by topic or foundation level. This is good for traditional guides or reference works.")
    print("2. Intelligent (structural) synthesis (like an executive report): Each section's narrative is shaped by the actual knowledge graph—conceptual dependencies, relationships, and flow are made explicit. Use this for executive summaries, strategy docs, or anything where 'how things fit together' is as important as the content itself.")
    while True:
        choice = input("Enter 1 for Flat or 2 for Intelligent synthesis: ").strip()
        if choice == "1":
            return "flat"
        elif choice == "2":
            return "structural"
        print("Invalid input. Please enter 1 or 2.")

def select_fill_gaps():
    print("\nFILLING IN GAPS (OPTIONAL):")
    print(
        "Would you like the system to automatically identify and fill in missing steps or 'gaps' in the logical flow of knowledge?\n"
        "If enabled, the AI will review the step-by-step progression from the most general ideas down to the specifics, "
        "and if it detects places where the explanation seems to 'jump' too far without sufficient bridging content, "
        "it will create and insert new explanatory nodes to make the knowledge flow smoother and more comprehensive."
    )
    print("Recommended for guides/tutorials, less necessary for simple references.")
    while True:
        resp = input("Enable gap filling? (y/n): ").strip().lower()
        if resp in ("y", "yes"):
            return True
        elif resp in ("n", "no"):
            return False
        print("Please enter y (yes) or n (no).")

# Linearize nodes for outline order (general → specific)
def get_outline_linear_order(kg):
    level_order = [
        ("pillar", FoundationLevel.PILLAR),
        ("core", FoundationLevel.CORE),
        ("supporting", FoundationLevel.SUPPORTING),
        ("detail", FoundationLevel.DETAIL),
        ("atomic", FoundationLevel.ATOMIC)
    ]
    outline = []
    seen = set()
    for name, lvl in level_order:
        for node in kg.nodes.values():
            if getattr(node, "foundation", None) == lvl and node.id not in seen:
                outline.append(node)
                seen.add(node.id)
    return outline

def apply_gap_filling(kg, synthesizer, fill_gaps):
    """
    Detect and fill logical gaps in the knowledge structure by generating bridging nodes.
    This mutates the graph and inserts new nodes if needed.
    """
    if not fill_gaps:
        return
    print("\nDetecting gaps and fleshing out the knowledge structure with bridging content where needed...")
    ordered_nodes = get_outline_linear_order(kg)
    gaps = synthesizer.identify_missing_steps_with_ai(ordered_nodes)
    if gaps:
        print(f"Detected {len(gaps)} gaps in the logical progression. Generating bridging sections...")
        bridging_sections = synthesizer.generate_bridging_sections_with_ai(ordered_nodes, gaps)
        print(f"Adding {len(bridging_sections)} new bridging sections to the structure.")
        for bridge in sorted(bridging_sections, key=lambda b: b["insert_after"], reverse=True):
            from ks_classes import KnowledgeNode, FoundationLevel
            from ks_helpers import generate_secure_id
            bridge_id = generate_secure_id("bridge", f"{bridge['title']}:{bridge['content'][:80]}")
            bridging_node = KnowledgeNode(
                id=bridge_id,
                title=bridge['title'],
                content=bridge['content'],
                node_type="supporting",
                foundation=FoundationLevel.SUPPORTING,
                info_type="conceptual",
                doc_refs=set(),
                parents=set(),
                children=set(),
                tags={"gap_filled"},
                metadata={
                    "source_doc": None,
                    "node_type": "gap_bridge",
                    "created_at": datetime.now().isoformat()
                }
            )
            kg.add_node(bridging_node)
            ordered_nodes.insert(bridge["insert_after"] + 1, bridging_node)
    else:
        print("No gaps detected! Structure appears logically complete.")


def main():
    documents_path = Path(config.corpus_path)
    text_extensions = [".txt", ".md", ".rst", ".json", ".yaml", ".yml", ".csv", ".tsv", ".toml", ".ini", ".log"]

    # 1. List corpus files (optional but nice for UX)
    corpus_files = [f for f in documents_path.iterdir() if f.is_file() and f.suffix.lower() in text_extensions]
    print(f"\nCorpus contains {len(corpus_files)} files:")
    for f in corpus_files:
        print(f"  - {f.name}")
    print()

    # 2. Ask for options BEFORE graph extraction
    writing_style = select_writing_style()
    output_format = select_output_format()
    structural_mode = select_structural_mode()
    fill_gaps = select_fill_gaps()

    # --- Load and build knowledge graph ONCE ---
    def load_and_build_kg():
        documents = []
        for file_path in corpus_files:
            try:
                with open(file_path, encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    filename=file_path.name,
                    content=content,
                    doc_type=f"text/{file_path.suffix[1:]}" if file_path.suffix else "text/plain",
                    metadata={"loaded_at": datetime.now().isoformat()}
                )
                documents.append(doc)
            except Exception as ex:
                logger.warning(f"Could not read file {file_path}: {ex}")
        print("\nBuilding knowledge graph from corpus (this will happen only once or after a rebuild)...")
        kgb = KnowledgeGraphBuilder(openai_client, openai_model=config.openai_model)
        kg = kgb.create_graph_from_documents(documents)
        integrity_report = kg.validate_graph_integrity()
        if not integrity_report['is_valid']:
            logger.warning(f"Graph integrity issues: {integrity_report['issues']}")
        return kg, documents

    kg, documents = load_and_build_kg()  # Only called ONCE unless rebuild
    synthesizer = KnowledgeSynthesizer(kg, openai_client, openai_model=config.openai_model)
    apply_gap_filling(kg, synthesizer, fill_gaps)

    while True:
        # 4. Generate and save synthesized document (NO graph rebuild here)
        title = generate_smart_title_from_graph(kg, openai_client, openai_model=config.openai_model)
        synthesizer.title = title
        synthesized_document = synthesizer.synthesize_coherent_knowledge(
            output_format=output_format,
            writing_style=writing_style,
            structural_mode=structural_mode
        )

        # Save and display output
        short_style = writing_style.replace(" ", "_")[:24]
        short_format = output_format.replace(" ", "_")[:24]
        short_mode = structural_mode[:12]
        now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_base = f"{title}_{short_format}_{short_style}_{short_mode}_{now_str}".replace("__", "_")
        file_base = re.sub(r"[^a-zA-Z0-9_-]", "", file_base)
        output_path = Path(getattr(config, 'output_path', '.')) / f"{file_base}.md"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(synthesized_document)

        print("=" * 80)
        print("SYNTHESIZED KNOWLEDGE DOCUMENT")
        print("=" * 80)
        print(synthesized_document)
        print("=" * 80)
        print(f"Output written to: {output_path}")
        logger.info(f"Synthesized document saved to: {output_path}")

        # 5. Offer options for regeneration
        print("\nWould you like to regenerate the output with different options?")
        print("1. Yes - change writing style")
        print("2. Yes - change output format")
        print("3. Yes - change structural mode")
        print("4. Yes - change ALL options")
        print("5. Rebuild knowledge graph from corpus (reload files)")
        print("6. No - exit")
        choice = input("Enter 1/2/3/4/5/6: ").strip()

        if choice == "1":
            writing_style = select_writing_style()
        elif choice == "2":
            output_format = select_output_format()
        elif choice == "3":
            structural_mode = select_structural_mode()
        elif choice == "4":
            writing_style = select_writing_style()
            output_format = select_output_format()
            structural_mode = select_structural_mode()
            fill_gaps = select_fill_gaps()
        elif choice == "5":
            # Optionally refresh file list in case corpus changed
            corpus_files = [f for f in documents_path.iterdir() if f.is_file() and f.suffix.lower() in text_extensions]
            print("File list reloaded. Graph will be rebuilt on next output.")
            kg, documents = load_and_build_kg()
            synthesizer = KnowledgeSynthesizer(kg, openai_client, openai_model=config.openai_model)
            apply_gap_filling(kg, synthesizer, fill_gaps)
        else:
            print("Exiting.")
            break

if __name__ == "__main__":
    main()
