"""
Document processing utilities extracted from the Jupyter notebook.

This module contains helper functions for:
- Figure extraction and serialization
- Chunk preparation and embedding
- MongoDB persistence
- Metadata generation
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import numpy as np
from docling_core.transforms.chunker.hierarchical_chunker import (
    ChunkingDocSerializer,
    ChunkingSerializerProvider,
)
from docling_core.transforms.serializer.base import SerializationResult
from docling_core.transforms.serializer.common import create_ser_result
from docling_core.transforms.serializer.markdown import (
    MarkdownParams,
    MarkdownPictureSerializer,
)
from docling_core.types.doc.document import DoclingDocument, PictureItem
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError

# Constants
MIN_FIGURE_HEIGHT_PX = int(os.getenv('MIN_FIGURE_HEIGHT_PX', '120'))
BINARY_IMAGE_FORMAT = "png"
BINARY_MIME_TYPE = "image/png"
BINARY_FILE_EXTENSION = BINARY_IMAGE_FORMAT

EMBED_MAX_RETRIES = 3
EMBED_RETRY_BASE_DELAY = 2.0


class FigureReferencePictureSerializer(MarkdownPictureSerializer):
    """Render picture items as references instead of inline images."""

    def serialize(  # type: ignore[override]
        self,
        *,
        item: PictureItem,
        doc_serializer: ChunkingDocSerializer,
        doc: DoclingDocument,
        **kwargs: Any,
    ) -> SerializationResult:
        caption = ""
        if hasattr(item, "caption_text"):
            caption = (item.caption_text(doc=doc) or "").strip()
        text = f"[Figure {item.self_ref}]"
        if caption:
            text = f"{text} — {caption}"
        return create_ser_result(text=text, span_source=item)


class FigureReferenceSerializerProvider(ChunkingSerializerProvider):
    """Provide chunk serializers that keep figures as metadata references."""

    def get_serializer(self, doc: DoclingDocument) -> ChunkingDocSerializer:
        return ChunkingDocSerializer(
            doc=doc,
            params=MarkdownParams(image_placeholder=""),
            picture_serializer=FigureReferencePictureSerializer(),
        )


def slugify(value: str) -> str:
    """Convert a string to a URL-safe slug."""
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    value = value.strip("-")
    return value or "document"


def hash_path(path: Path) -> str:
    """Generate a short hash from a file path."""
    return hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:8]


def filter_figures_by_height(
    figure_map: dict[str, dict[str, Any]]
) -> dict[str, dict[str, Any]]:
    """Return a subset of figures whose height exceeds the minimum allowed threshold."""
    return {
        figure_id: figure
        for figure_id, figure in figure_map.items()
        if figure.get("height", 0) > MIN_FIGURE_HEIGHT_PX
    }


def extract_figures(
    doc: DoclingDocument, doc_id: str, image_root: Path
) -> dict[str, dict[str, Any]]:
    """
    Extract figures from a DoclingDocument and save them as PNG files.
    
    Args:
        doc: The DoclingDocument to extract figures from
        doc_id: Unique identifier for the document
        image_root: Root directory to save figure images
    
    Returns:
        Dictionary mapping figure IDs to figure metadata
    """
    image_root.mkdir(parents=True, exist_ok=True)
    figures: dict[str, dict[str, Any]] = {}
    
    for picture in getattr(doc, "pictures", []):
        if not isinstance(picture, PictureItem):
            continue

        image_ref = getattr(picture, "image", None)
        if image_ref is None or getattr(image_ref, "pil_image", None) is None:
            continue

        pil_image = image_ref.pil_image
        file_name = f"{slugify(picture.self_ref)}.png"
        image_path = image_root / file_name

        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        image_path.write_bytes(image_bytes)
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        width, height = pil_image.size

        caption = ""
        if hasattr(picture, "caption_text"):
            caption = (picture.caption_text(doc=doc) or "").strip()

        provenance_data = []
        for prov in getattr(picture, "prov", []) or []:
            prov_entry: dict[str, Any] = {}
            for attr in ("page_no", "page_id", "line_no"):
                if hasattr(prov, attr):
                    prov_entry[attr] = getattr(prov, attr)
            if prov_entry:
                provenance_data.append(prov_entry)

        figures[picture.self_ref] = {
            "figure_id": picture.self_ref,
            "doc_id": doc_id,
            "caption": caption or None,
            "image_path": str(image_path.resolve()),
            "image_b64": image_b64,
            "width": int(width),
            "height": int(height),
            "provenance": provenance_data,
        }

    return figures


def build_light_and_binary_maps(
    figure_map: dict[str, dict[str, Any]], *, image_root: Path
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    """Split a raw figure map into lightweight references and n8n-style binary entries."""
    figure_refs_light: dict[str, dict[str, Any]] = {}
    figure_binary_map: dict[str, dict[str, Any]] = {}
    
    for figure_id, figure in figure_map.items():
        rel_path = os.path.relpath(figure["image_path"], image_root)
        figure_refs_light[figure_id] = {
            "figure_id": figure_id,
            "doc_id": figure.get("doc_id"),
            "caption": figure.get("caption"),
            "image_path": rel_path,
            "width": figure.get("width"),
            "height": figure.get("height"),
            "provenance": figure.get("provenance", []),
        }
        
        file_name = Path(rel_path).name
        image_b64 = figure.get("image_b64")
        if image_b64:
            figure_binary_map[figure_id] = {
                "data": image_b64,
                "mimeType": BINARY_MIME_TYPE,
                "fileExtension": BINARY_FILE_EXTENSION,
                "fileName": file_name,
            }
    
    return figure_refs_light, figure_binary_map


def build_chunk_binary_for_refs(
    figure_refs: list[dict[str, Any]],
    figure_binary_map: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Build the per-chunk binary map containing only referenced figures."""
    chunk_binary: dict[str, dict[str, Any]] = {}
    for ref in figure_refs:
        figure_id = ref.get("figure_id")
        if not figure_id:
            continue
        binary_entry = figure_binary_map.get(figure_id)
        if not binary_entry:
            continue
        chunk_binary[figure_id] = dict(binary_entry)
    return chunk_binary


def figure_subset(
    doc_items: Iterable[Any],
    figure_refs_light: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract figure references from document items."""
    figure_refs: list[dict[str, Any]] = []
    for doc_item in doc_items:
        ref = None
        if isinstance(doc_item, PictureItem):
            ref = figure_refs_light.get(doc_item.self_ref)
        else:
            item_ref = getattr(doc_item, "self_ref", None)
            if item_ref:
                ref = figure_refs_light.get(item_ref)
        if not ref:
            continue
        figure_refs.append(dict(ref))
    return figure_refs


def prepare_chunk_records(
    *,
    doc: DoclingDocument,
    doc_id: str,
    source_path: Path,
    chunker,
    figure_refs_light: dict[str, dict[str, Any]],
    figure_binary_map: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Prepare chunk records from a document for embedding and storage.
    
    Args:
        doc: DoclingDocument to chunk
        doc_id: Unique document identifier
        source_path: Path to source PDF file
        chunker: HybridChunker instance
        figure_refs_light: Lightweight figure reference map
        figure_binary_map: Binary figure data map
    
    Returns:
        List of chunk record dictionaries
    """
    records: list[dict[str, Any]] = []
    chunks = list(chunker.chunk(dl_doc=doc))
    
    for idx, chunk in enumerate(chunks):
        enriched_text = chunker.contextualize(chunk=chunk)
        doc_items = getattr(chunk.meta, "doc_items", []) or []
        figure_refs = figure_subset(doc_items, figure_refs_light)
        
        record = {
            "chunk_id": f"{doc_id}::chunk-{idx:04d}",
            "document_id": doc_id,
            "chunk_index": idx,
            "text": enriched_text,
            "metadata": {
                "source_pdf": str(source_path),
                "pages": getattr(chunk.meta, "pages", []),
                "headings": getattr(chunk.meta, "headings", []),
                "doc_items": [getattr(item, "self_ref", None) for item in doc_items],
                "figure_refs": figure_refs,
            },
            "binary": build_chunk_binary_for_refs(figure_refs, figure_binary_map),
        }
        records.append(record)
    
    return records


def normalize_embeddings(raw_embeddings: Any, expected_length: int) -> list[list[float]]:
    """Normalize embedding response from VoyageAI into a list of vectors."""
    if raw_embeddings is None:
        return []

    def is_vector(candidate: Any) -> bool:
        return isinstance(candidate, (list, tuple)) and candidate and isinstance(candidate[0], (float, int))

    def flatten_embeddings(candidate: Any) -> list[list[float]]:
        if is_vector(candidate):
            return [[float(x) for x in candidate]]  # type: ignore[arg-type]
        if isinstance(candidate, list):
            vectors: list[list[float]] = []
            for item in candidate:
                vectors.extend(flatten_embeddings(item))
            return vectors
        raise ValueError(
            "Unexpected embedding payload type "
            f"{type(candidate)!r}; unable to flatten contextualized embeddings."
        )

    vectors = flatten_embeddings(raw_embeddings)
    if len(vectors) != expected_length:
        raise ValueError(
            "Unable to align contextualized embeddings with chunk outputs; "
            f"expected {expected_length} vectors, received {len(vectors)}."
        )
    return vectors


def embed_chunks(
    *,
    voyage_client,
    records: Sequence[dict[str, Any]],
    model_name: str,
    output_dimension: int | None,
    output_dtype: str,
) -> list[list[float]]:
    """
    Generate embeddings for chunk records using VoyageAI.
    
    Args:
        voyage_client: VoyageAI client instance
        records: List of chunk records to embed
        model_name: VoyageAI model identifier
        output_dimension: Optional output dimension for embeddings
        output_dtype: Output data type (e.g., 'float')
    
    Returns:
        List of embedding vectors
    """
    texts = [record["text"] for record in records]
    if not texts:
        return []

    total_chars = sum(len(text) for text in texts)
    
    for attempt in range(1, EMBED_MAX_RETRIES + 1):
        logging.info(
            "Embedding %d chunk(s) (%d chars) with Voyage model '%s' [attempt %d/%d]",
            len(texts),
            total_chars,
            model_name,
            attempt,
            EMBED_MAX_RETRIES,
        )
        try:
            response = voyage_client.contextualized_embed(
                inputs=[texts],
                model=model_name,
                input_type="document",
                output_dimension=output_dimension,
                output_dtype=output_dtype,
            )
            
            request_id = getattr(response, "request_id", None)
            if request_id:
                logging.info("Voyage embedding request succeeded (request_id=%s).", request_id)
            
            raw_embeddings = None
            if hasattr(response, "embeddings"):
                raw_embeddings = response.embeddings
            elif hasattr(response, "results"):
                raw_embeddings = response.results[0].embeddings  # type: ignore[index]
            
            return normalize_embeddings(raw_embeddings, len(records))
        
        except Exception as exc:
            if attempt == EMBED_MAX_RETRIES:
                logging.error(
                    "Embedding failed after %d attempt(s) for %d chunk(s): %s",
                    attempt,
                    len(texts),
                    exc,
                )
                raise
            
            backoff = EMBED_RETRY_BASE_DELAY * (2 ** (attempt - 1))
            logging.warning(
                "Embedding attempt %d/%d failed (%s). Retrying in %.1f seconds...",
                attempt,
                EMBED_MAX_RETRIES,
                exc,
                backoff,
            )
            time.sleep(backoff)

    raise RuntimeError("Unexpected embedding retry loop exit")


def attach_embeddings(
    records: list[dict[str, Any]], embeddings: Sequence[Sequence[float]]
) -> None:
    """Attach embedding vectors to chunk records."""
    if len(records) != len(embeddings):
        raise ValueError("Mismatch between records and embeddings length.")
    for record, embedding in zip(records, embeddings):
        record["embedding"] = np.asarray(embedding, dtype=np.float32).tolist()


def persist_records(collection: Collection, records: Sequence[dict[str, Any]], upsert: bool = True) -> None:
    """
    Persist chunk records to MongoDB collection.

    Args:
        collection: MongoDB collection
        records: List of chunk records to persist
        upsert: If True, update existing documents; if False, insert only
    """
    if not records:
        return

    if upsert:
        # Use bulk upsert operations based on chunk_id
        from pymongo import UpdateOne
        operations = [
            UpdateOne(
                {"chunk_id": record["chunk_id"]},
                {"$set": record},
                upsert=True
            )
            for record in records
        ]
        try:
            result = collection.bulk_write(operations, ordered=False)
            logging.info(
                "Upsert complete: %d inserted, %d modified",
                result.upserted_count,
                result.modified_count
            )
        except BulkWriteError as exc:
            logging.error("MongoDB bulk write encountered an error: %s", exc)
            raise
    else:
        # Original insert-only behavior
        try:
            collection.insert_many(records, ordered=False)
        except BulkWriteError as exc:
            logging.error("MongoDB bulk write encountered an error: %s", exc)
            raise


def write_metadata_doc(
    *,
    metadata_path: Path,
    doc_id: str,
    source_pdf: Path,
    figure_refs_light: dict[str, dict[str, Any]],
    chunk_count: int,
    voyage_model: str,
    conversion_strategy: str,
) -> None:
    """Write metadata document to JSON file."""
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_content = {
        "doc_id": doc_id,
        "source_pdf": str(source_pdf),
        "figure_inventory": [dict(fig) for fig in figure_refs_light.values()],
        "chunk_count": chunk_count,
        "voyage_model": voyage_model,
        "conversion_strategy": conversion_strategy,
    }
    metadata_path.write_text(json.dumps(metadata_content, indent=2), encoding="utf-8")
