from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set, Union
from datetime import datetime
import time           # ← needed by make_node_id
import re
import hashlib

# ------------------------------------------------------------------
# Five-tier foundation enum must live here (or in a tiny shared module)
# so KnowledgeNode can see it immediately and other modules can import it.
# ------------------------------------------------------------------
class FoundationLevel(Enum):
    PILLAR     = 5   # enterprise-wide linchpins
    CORE       = 4   # domain-central ideas
    SUPPORTING = 3   # mid-level patterns / procedures
    DETAIL     = 2   # fine-grained specs / examples
    ATOMIC     = 1   # single facts, config options

@dataclass
class Document:
    """
    Represents one source file in the corpus.
    """
    filename: str                  # relative path (or display name)
    doc_type: str                  # markdown | code | config | notes …
    content: str                   # raw text of the file
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    # id is injected automatically after init
    id: str = field(init=False)

    def __post_init__(self):
        """
        Generate a stable—but unique—ID once the instance is created.
        Format: <sanitized-filename>:<unix-timestamp>
        """
        sanitized = re.sub(r'[<>:"/\\|?*]', '_', self.filename)
        self.id = f"{sanitized}:{int(self.timestamp.timestamp())}"

    def __repr__(self) -> str:
        return f"<Document {self.filename!r} ({self.doc_type}) id={self.id}>"


@dataclass
class KnowledgeNode:
    """
    Represents a node in the knowledge graph.
    Can be a theme, requirement, spec, code unit, section, etc.
    Supports multi-parent and multi-child (DAG/folding) relationships.
    Tracks coverage and representation in the manual (for completeness queries).
    Now explicitly distinguishes between conceptual and technical information.
    """
    id: str
    title: str
    content: str
    node_type: str = "atomic"   # kept for legacy
    foundation: Union[FoundationLevel,str] = "auto"   # new field
    info_type: str = "mixed"    # "conceptual" | "technical" | "mixed"
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    doc_refs: Set[str] = field(default_factory=set)
    history: List[Dict[str, Any]] = field(default_factory=list)
    last_modified: datetime = field(default_factory=datetime.now)

    def add_child(self, child_id: str):
        if child_id and child_id != self.id:
            self.children.add(child_id)

    def add_parent(self, parent_id: str):
        if parent_id and parent_id != self.id:
            self.parents.add(parent_id)

    def add_tag(self, tag: str):
        if tag:
            self.tags.add(tag)

    def add_doc_ref(self, doc_id: str):
        if doc_id:
            self.doc_refs.add(doc_id)

    def update_content(self, new_content: str, meta: Optional[Dict[str, Any]] = None):
        self.history.append({
            "timestamp": datetime.now().isoformat(),
            "prev_content": self.content,
            "meta": meta or {}
        })
        self.content = new_content
        self.last_modified = datetime.now()

    def mark_manual_covered(self, manual_section_id: Optional[str] = None):
        """Marks this node as represented in the manual for completeness/QA."""
        self.metadata['manual_covered'] = True
        self.metadata['manual_section_id'] = manual_section_id or self.id
        self.metadata['last_covered_time'] = datetime.now().isoformat()

    def is_covered(self) -> bool:
        """Returns True if this node is covered in the manual."""
        return bool(self.metadata.get('manual_covered'))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "content": self.content,
            "node_type": self.node_type,
            "info_type": self.info_type,
            "parents": list(self.parents),
            "children": list(self.children),
            "tags": list(self.tags),
            "metadata": self.metadata,
            "doc_refs": list(self.doc_refs),
            "history": self.history,
            "last_modified": self.last_modified.isoformat(),
        }

    def __repr__(self):
        return f"<KnowledgeNode id={self.id!r} title={self.title!r} type={self.node_type!r} info_type={self.info_type!r} covered={self.is_covered()}>"

# Helper: Node ID generator (secure and deterministic, matches Section 1)
def make_node_id(prefix: str, content: str) -> str:
    hash_obj = hashlib.sha256(content.encode('utf-8'))
    hash_hex = hash_obj.hexdigest()[:12]
    timestamp = int(time.time() * 1000) % 1000000
    return f"{prefix}_{hash_hex}_{timestamp}"
