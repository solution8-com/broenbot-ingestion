"""
Azure Function for PDF Ingestion with SharePoint Integration

This function processes PDF documents from SharePoint folders, extracts text and figures,
generates embeddings using VoyageAI, and stores chunks in MongoDB Atlas.
"""

import azure.functions as func
import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

# SharePoint/Microsoft Graph imports
from msal import ConfidentialClientApplication
from msgraph import GraphServiceClient
from azure.identity import ClientSecretCredential

# Azure Storage imports (optional)
from azure.storage.blob import BlobServiceClient

# Docling and processing imports
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from transformers import AutoTokenizer

# Database and embedding imports
from pymongo import MongoClient
from voyageai import Client as VoyageClient
import numpy as np

# Import helper functions (extracted from notebook)
from utils.document_processing import (
    extract_figures,
    filter_figures_by_height,
    build_light_and_binary_maps,
    prepare_chunk_records,
    embed_chunks,
    attach_embeddings,
    persist_records,
    write_metadata_doc,
    FigureReferenceSerializerProvider
)

# Initialize Azure Function App (ANONYMOUS at app level, API key checked in code)
app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

# Global configuration from environment
SHAREPOINT_TENANT_ID = os.getenv('SHAREPOINT_TENANT_ID')
SHAREPOINT_CLIENT_ID = os.getenv('SHAREPOINT_CLIENT_ID')
SHAREPOINT_CLIENT_SECRET = os.getenv('SHAREPOINT_CLIENT_SECRET')
SHAREPOINT_SITE_URL = os.getenv('SHAREPOINT_SITE_URL')
SHAREPOINT_FOLDER_PATH = os.getenv('SHAREPOINT_FOLDER_PATH', 'Shared Documents/PDFs')

VOYAGE_API_KEY = os.getenv('VOYAGE_API_KEY')
MONGODB_URI = os.getenv('MONGODB_URI')
MONGODB_DB = os.getenv('MONGODB_DB')
MONGODB_COLLECTION = os.getenv('MONGODB_COLLECTION')

TOKENIZER_MODEL = os.getenv('TOKENIZER_MODEL', 'voyageai/voyage-context-3')
VOYAGE_MODEL = os.getenv('VOYAGE_MODEL', 'voyage-context-3')
CHUNK_MAX_TOKENS = int(os.getenv('CHUNK_MAX_TOKENS', '768'))
CHUNK_MERGE_PEERS = os.getenv('CHUNK_MERGE_PEERS', 'true').lower() == 'true'
VOYAGE_OUTPUT_DIM = os.getenv('VOYAGE_OUTPUT_DIM')
VOYAGE_OUTPUT_DTYPE = os.getenv('VOYAGE_OUTPUT_DTYPE', 'float')

# API Security
API_SECRET_KEY = os.getenv('API_SECRET_KEY')


def verify_api_key(req: func.HttpRequest) -> bool:
    """Verify the API key from request header."""
    if not API_SECRET_KEY:
        logging.warning("API_SECRET_KEY not configured - skipping auth")
        return True

    provided_key = req.headers.get('X-API-Key') or req.headers.get('x-api-key')
    return provided_key == API_SECRET_KEY

if VOYAGE_OUTPUT_DIM and VOYAGE_OUTPUT_DIM.lower() != 'none':
    VOYAGE_OUTPUT_DIM = int(VOYAGE_OUTPUT_DIM)
else:
    VOYAGE_OUTPUT_DIM = None


def get_sharepoint_client() -> GraphServiceClient:
    """Initialize Microsoft Graph client for SharePoint access."""
    credential = ClientSecretCredential(
        tenant_id=SHAREPOINT_TENANT_ID,
        client_id=SHAREPOINT_CLIENT_ID,
        client_secret=SHAREPOINT_CLIENT_SECRET
    )
    scopes = ['https://graph.microsoft.com/.default']
    client = GraphServiceClient(credentials=credential, scopes=scopes)
    return client


async def list_sharepoint_files(folder_path: str) -> List[Dict[str, Any]]:
    """
    List PDF files in a SharePoint folder.
    
    Args:
        folder_path: Path to SharePoint folder (e.g., 'Shared Documents/PDFs')
    
    Returns:
        List of dictionaries with file metadata (name, download_url, size)
    """
    try:
        client = get_sharepoint_client()
        
        # Extract site domain and relative path from SHAREPOINT_SITE_URL
        site_url = SHAREPOINT_SITE_URL.replace('https://', '')
        parts = site_url.split('/')
        domain = parts[0]
        site_path = '/'.join(parts[2:]) if len(parts) > 2 else ''
        
        # Get site ID
        site = await client.sites.by_site_id(f"{domain}:/sites/{site_path}").get()
        
        # Get drive (document library)
        drives = await client.sites.by_site_id(site.id).drives.get()
        if not drives or not drives.value:
            raise ValueError("No drives found in SharePoint site")
        
        drive_id = drives.value[0].id
        
        # Navigate to folder and list items
        folder_items = await client.drives.by_drive_id(drive_id).items.by_drive_item_id(f"root:/{folder_path}:").children.get()
        
        pdf_files = []
        if folder_items and folder_items.value:
            for item in folder_items.value:
                if item.name.lower().endswith('.pdf'):
                    # Get download URL from additional_data (newer SDK) or direct attribute (older SDK)
                    download_url = None
                    if hasattr(item, 'additional_data') and item.additional_data:
                        download_url = item.additional_data.get('@microsoft.graph.downloadUrl')
                    if not download_url and hasattr(item, 'microsoft_graph_download_url'):
                        download_url = item.microsoft_graph_download_url

                    pdf_files.append({
                        'name': item.name,
                        'download_url': download_url,
                        'size': item.size,
                        'id': item.id
                    })
        
        return pdf_files
    
    except Exception as e:
        logging.error(f"Error listing SharePoint files: {str(e)}")
        raise


async def download_sharepoint_file(file_info: Dict[str, Any], temp_dir: Path) -> Path:
    """
    Download a file from SharePoint to local temporary directory.
    
    Args:
        file_info: Dictionary with file metadata including download_url
        temp_dir: Temporary directory to save the file
    
    Returns:
        Path to downloaded file
    """
    import requests
    
    try:
        # Get access token for download
        authority = f"https://login.microsoftonline.com/{SHAREPOINT_TENANT_ID}"
        app = ConfidentialClientApplication(
            SHAREPOINT_CLIENT_ID,
            authority=authority,
            client_credential=SHAREPOINT_CLIENT_SECRET
        )
        
        scopes = ['https://graph.microsoft.com/.default']
        result = app.acquire_token_for_client(scopes=scopes)
        
        if 'access_token' not in result:
            raise ValueError(f"Failed to acquire token: {result.get('error_description')}")
        
        # Download file
        headers = {'Authorization': f"Bearer {result['access_token']}"}
        response = requests.get(file_info['download_url'], headers=headers)
        response.raise_for_status()
        
        # Save to temp directory
        file_path = temp_dir / file_info['name']
        file_path.write_bytes(response.content)
        
        logging.info(f"Downloaded {file_info['name']} ({file_info['size']} bytes)")
        return file_path
    
    except Exception as e:
        logging.error(f"Error downloading file {file_info['name']}: {str(e)}")
        raise


def initialize_processing_components():
    """Initialize chunker, embedder, and database clients."""
    # Tokenizer and chunker
    hf_tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL)
    chunk_tokenizer = HuggingFaceTokenizer(
        tokenizer=hf_tokenizer,
        max_tokens=CHUNK_MAX_TOKENS
    )
    serializer_provider = FigureReferenceSerializerProvider()
    chunker = HybridChunker(
        tokenizer=chunk_tokenizer,
        merge_peers=CHUNK_MERGE_PEERS,
        serializer_provider=serializer_provider
    )
    
    # VoyageAI client
    voyage_client = VoyageClient(api_key=VOYAGE_API_KEY)
    
    # MongoDB client
    mongo_client = MongoClient(MONGODB_URI)
    mongo_collection = mongo_client[MONGODB_DB][MONGODB_COLLECTION]
    
    # Document converter
    pipeline_options = PdfPipelineOptions()
    pipeline_options.generate_picture_images = True
    pipeline_options.generate_page_images = False
    pipeline_options.do_picture_description = False
    
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
        }
    )
    
    return chunker, voyage_client, mongo_collection, converter


def process_single_pdf(
    pdf_path: Path,
    chunker,
    voyage_client,
    mongo_collection,
    converter,
    image_root: Path,
    metadata_root: Path
) -> Dict[str, Any]:
    """
    Process a single PDF file through the ingestion pipeline.

    Returns:
        Dictionary with processing results (doc_id, chunk_count, figure_count, etc.)
    """
    import hashlib
    import re

    def slugify(value: str) -> str:
        value = value.lower()
        value = re.sub(r"[^a-z0-9]+", "-", value)
        return value.strip("-") or "document"

    def hash_content(path: Path) -> str:
        """Generate SHA256 hash of file content (first 8 chars)."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()[:12]

    # Generate document ID based on content hash (not filename)
    # This ensures same content = same doc_id regardless of filename
    content_hash = hash_content(pdf_path)
    file_stem = slugify(pdf_path.stem)
    doc_id = f"{file_stem}-{content_hash}"
    
    logging.info(f"Processing {pdf_path.name} → {doc_id}")
    
    # Convert PDF
    result = converter.convert(source=str(pdf_path))
    doc = result.document
    
    # Extract figures
    figure_dir = image_root / doc_id
    figure_map = extract_figures(doc, doc_id, figure_dir)
    figure_map = filter_figures_by_height(figure_map)
    
    figure_refs_light, figure_binary_map = build_light_and_binary_maps(
        figure_map, image_root=image_root
    )
    
    # Prepare chunks
    records = prepare_chunk_records(
        doc=doc,
        doc_id=doc_id,
        source_path=pdf_path,
        chunker=chunker,
        figure_refs_light=figure_refs_light,
        figure_binary_map=figure_binary_map
    )
    
    # Generate embeddings
    if records:
        embeddings = embed_chunks(
            voyage_client=voyage_client,
            records=records,
            model_name=VOYAGE_MODEL,
            output_dimension=VOYAGE_OUTPUT_DIM,
            output_dtype=VOYAGE_OUTPUT_DTYPE
        )
        attach_embeddings(records, embeddings)
        
        # Persist to MongoDB
        persist_records(mongo_collection, records)
    
    # Write metadata
    metadata_path = metadata_root / f"{doc_id}.json"
    write_metadata_doc(
        metadata_path=metadata_path,
        doc_id=doc_id,
        source_pdf=pdf_path,
        figure_refs_light=figure_refs_light,
        chunk_count=len(records),
        voyage_model=VOYAGE_MODEL,
        conversion_strategy="docling"
    )
    
    return {
        'doc_id': doc_id,
        'chunk_count': len(records),
        'figure_count': len(figure_refs_light),
        'source_file': pdf_path.name,
        'strategy': 'docling'
    }


@app.route(route="ingest_sharepoint_folder", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
async def ingest_sharepoint_folder(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to ingest all PDFs from a SharePoint folder.

    Request body (JSON):
    {
        "folder_path": "Shared Documents/PDFs",  // Optional, uses env var if not provided
        "max_files": 10  // Optional, limits number of files to process
    }

    Response:
    {
        "status": "success",
        "processed": 5,
        "results": [...]
    }

    Headers:
        X-API-Key: your-secret-key
    """
    logging.info('Processing SharePoint folder ingestion request')

    # Verify API key
    if not verify_api_key(req):
        return func.HttpResponse(
            json.dumps({'error': 'Unauthorized - Invalid or missing API key'}),
            status_code=401,
            mimetype='application/json'
        )

    try:
        # Parse request body
        req_body = req.get_json()
        folder_path = req_body.get('folder_path', SHAREPOINT_FOLDER_PATH)
        max_files = req_body.get('max_files', None)
        
        # Validate configuration
        if not all([SHAREPOINT_TENANT_ID, SHAREPOINT_CLIENT_ID, SHAREPOINT_CLIENT_SECRET]):
            return func.HttpResponse(
                json.dumps({'error': 'SharePoint configuration missing'}),
                status_code=500,
                mimetype='application/json'
            )
        
        # List files from SharePoint
        pdf_files = await list_sharepoint_files(folder_path)
        
        if max_files:
            pdf_files = pdf_files[:max_files]
        
        if not pdf_files:
            return func.HttpResponse(
                json.dumps({'status': 'success', 'processed': 0, 'message': 'No PDF files found'}),
                status_code=200,
                mimetype='application/json'
            )
        
        # Initialize processing components
        chunker, voyage_client, mongo_collection, converter = initialize_processing_components()
        
        # Create temporary directory for processing
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_root = temp_path / 'images'
            metadata_root = temp_path / 'metadata'
            image_root.mkdir(parents=True, exist_ok=True)
            metadata_root.mkdir(parents=True, exist_ok=True)
            
            results = []
            
            # Process each PDF
            for file_info in pdf_files:
                try:
                    # Download file
                    pdf_path = await download_sharepoint_file(file_info, temp_path)
                    
                    # Process PDF
                    result = process_single_pdf(
                        pdf_path=pdf_path,
                        chunker=chunker,
                        voyage_client=voyage_client,
                        mongo_collection=mongo_collection,
                        converter=converter,
                        image_root=image_root,
                        metadata_root=metadata_root
                    )
                    
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Error processing {file_info['name']}: {str(e)}")
                    results.append({
                        'source_file': file_info['name'],
                        'error': str(e)
                    })
        
        return func.HttpResponse(
            json.dumps({
                'status': 'success',
                'processed': len(results),
                'results': results
            }, indent=2),
            status_code=200,
            mimetype='application/json'
        )
    
    except Exception as e:
        logging.error(f"Error in ingest_sharepoint_folder: {str(e)}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )


@app.route(route="ingest_single_pdf", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
async def ingest_single_pdf(req: func.HttpRequest) -> func.HttpResponse:
    """
    HTTP trigger to ingest a single PDF from SharePoint or Azure Blob Storage.

    Request body (JSON):
    {
        "source_type": "sharepoint",  // or "blob"
        "file_name": "document.pdf",  // For SharePoint
        "folder_path": "Shared Documents/PDFs",  // Optional for SharePoint
        "container_name": "pdfs",  // For Blob Storage
        "blob_name": "document.pdf"  // For Blob Storage
    }

    Response:
    {
        "status": "success",
        "doc_id": "...",
        "chunk_count": 10,
        "figure_count": 5
    }

    Headers:
        X-API-Key: your-secret-key
    """
    logging.info('Processing single PDF ingestion request')

    # Verify API key
    if not verify_api_key(req):
        return func.HttpResponse(
            json.dumps({'error': 'Unauthorized - Invalid or missing API key'}),
            status_code=401,
            mimetype='application/json'
        )

    try:
        # Parse request body
        req_body = req.get_json()
        source_type = req_body.get('source_type', 'sharepoint')
        
        # Initialize processing components
        chunker, voyage_client, mongo_collection, converter = initialize_processing_components()
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            image_root = temp_path / 'images'
            metadata_root = temp_path / 'metadata'
            image_root.mkdir(parents=True, exist_ok=True)
            metadata_root.mkdir(parents=True, exist_ok=True)
            
            # Download file based on source type
            if source_type == 'sharepoint':
                folder_path = req_body.get('folder_path', SHAREPOINT_FOLDER_PATH)
                file_name = req_body.get('file_name')
                
                if not file_name:
                    return func.HttpResponse(
                        json.dumps({'error': 'file_name is required for SharePoint source'}),
                        status_code=400,
                        mimetype='application/json'
                    )
                
                # List files and find the target
                pdf_files = await list_sharepoint_files(folder_path)
                file_info = next((f for f in pdf_files if f['name'] == file_name), None)
                
                if not file_info:
                    return func.HttpResponse(
                        json.dumps({'error': f'File {file_name} not found in {folder_path}'}),
                        status_code=404,
                        mimetype='application/json'
                    )
                
                pdf_path = await download_sharepoint_file(file_info, temp_path)
            
            elif source_type == 'blob':
                container_name = req_body.get('container_name')
                blob_name = req_body.get('blob_name')
                
                if not all([container_name, blob_name]):
                    return func.HttpResponse(
                        json.dumps({'error': 'container_name and blob_name required for blob source'}),
                        status_code=400,
                        mimetype='application/json'
                    )
                
                # Download from Azure Blob Storage
                connection_string = os.getenv('AZURE_STORAGE_CONNECTION_STRING')
                if not connection_string:
                    return func.HttpResponse(
                        json.dumps({'error': 'AZURE_STORAGE_CONNECTION_STRING not configured'}),
                        status_code=500,
                        mimetype='application/json'
                    )
                
                blob_service_client = BlobServiceClient.from_connection_string(connection_string)
                blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
                
                pdf_path = temp_path / blob_name
                with open(pdf_path, 'wb') as f:
                    download_stream = blob_client.download_blob()
                    f.write(download_stream.readall())
            
            else:
                return func.HttpResponse(
                    json.dumps({'error': f'Unsupported source_type: {source_type}'}),
                    status_code=400,
                    mimetype='application/json'
                )
            
            # Process the PDF
            result = process_single_pdf(
                pdf_path=pdf_path,
                chunker=chunker,
                voyage_client=voyage_client,
                mongo_collection=mongo_collection,
                converter=converter,
                image_root=image_root,
                metadata_root=metadata_root
            )
            
            return func.HttpResponse(
                json.dumps({
                    'status': 'success',
                    **result
                }, indent=2),
                status_code=200,
                mimetype='application/json'
            )
    
    except Exception as e:
        logging.error(f"Error in ingest_single_pdf: {str(e)}")
        return func.HttpResponse(
            json.dumps({'error': str(e)}),
            status_code=500,
            mimetype='application/json'
        )


@app.route(route="health", methods=["GET"], auth_level=func.AuthLevel.ANONYMOUS)
def health_check(req: func.HttpRequest) -> func.HttpResponse:
    """Simple health check endpoint (no auth required)."""
    return func.HttpResponse(
        json.dumps({'status': 'healthy', 'service': 'docling-ingest'}),
        status_code=200,
        mimetype='application/json'
    )
