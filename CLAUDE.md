# CLAUDE.md

This file provides guidance to Claude Code when working with this repository.

## Project Overview

Azure Container App that ingests PDF documents from SharePoint, processes them with Docling, generates embeddings with VoyageAI, and stores chunks in MongoDB Atlas for RAG (Retrieval-Augmented Generation).

**Architecture:**
```
SharePoint Folder → n8n (polls every 5 min) → HTTP POST → Azure Container App → MongoDB Atlas
```

**Flow:**
1. n8n polls SharePoint folder every 5 minutes for new/modified PDFs
2. n8n sends HTTP POST to Container App (with filename + folder path + API key)
3. Container App downloads PDF from SharePoint via Microsoft Graph API
4. Runs Docling (extract text + figures)
5. Chunks with HybridChunker (~768 tokens)
6. Generates embeddings via VoyageAI (voyage-context-3, 1024 dims)
7. Upserts to MongoDB based on chunk_id
8. Returns success/failure to n8n

**Key Components:**
- **Azure Container App**: `docling-ingest` in resource group `rg-broen-lab-ingestion`
- **Azure Container Registry**: `broenlabing.azurecr.io`
- **MongoDB Atlas**: `testCluster` → `broen-ingestion-test` → `broen-documents-test`
- **n8n**: Workflow polls SharePoint and triggers ingestion via HTTP

## Changes from Original Repo

- **Anonymous auth + API key check**: Azure Functions auth doesn't work well in Container Apps, so changed to anonymous at app level and added custom X-API-Key header validation in code instead

- **EnableWorkerIndexing flag**: Added to Dockerfile, required for Python v2 programming model to discover functions

- **Microsoft Graph SDK fixes**: Updated syntax for SDK compatibility (item_with_path and download_url attribute access changed in newer SDK versions)

- **Upsert logic**: Changed from insert to upsert using chunk_id as unique key - prevents duplicates when same PDF is re-processed

- **Content-based hashing**: Doc ID now uses SHA256 hash of PDF content instead of filename - same content = same ID regardless of what the file is named

## Document Naming Convention (for BROEN Lab)

**Option A - Keep filename exactly the same:**
- `TI-93.pdf` stays `TI-93.pdf` when updated
- We detect the file changed, delete old content, ingest new content

**Option B - Same filename with version suffix (preferred):**
- `TI-93-v1.pdf` → `TI-93-v2.pdf` → `TI-93-v3.pdf`
- We match on the base name (`TI-93`), delete the previous version, ingest the new one
- Benefit: BROEN keeps visible version history in their filenames

**New documents:** Completely new filenames are fine - just get added.

**Why filename matters:** If BROEN renames `TI-93-v1.pdf` to `TI-94-v1.pdf`, we can't know these are the same document. Both would exist in the knowledge base.

## What the Container App Does NOT Do

- Detect changes (n8n does that by checking modified dates)
- Compare old vs new content
- Delete obsolete chunks automatically
- Any "smart" dedup or versioning
- Link TIs to products
- Handle renamed/restructured documents

## Development Commands

### Local Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Build and Deploy
```bash
# In Azure Cloud Shell
cd ~/broenlab-ingestion
git pull origin main
az acr build --registry broenlabing --image docling-ingest:vX .
az containerapp update --name docling-ingest --resource-group rg-broen-lab-ingestion --image broenlabing.azurecr.io/docling-ingest:vX
```

### Test Health Endpoint
```bash
curl https://docling-ingest.thankfulbush-dcd4f38d.swedencentral.azurecontainerapps.io/api/health
```

## Environment Variables (Azure Container App)

Required environment variables configured in Azure:
- `SHAREPOINT_TENANT_ID` - Azure AD tenant ID
- `SHAREPOINT_CLIENT_ID` - App registration client ID
- `SHAREPOINT_CLIENT_SECRET` - App registration secret
- `SHAREPOINT_SITE_URL` - e.g., `https://tenant.sharepoint.com/sites/SiteName`
- `SHAREPOINT_FOLDER_PATH` - e.g., `Shared Documents/PDFs`
- `VOYAGE_API_KEY` - VoyageAI API key
- `MONGODB_URI` - MongoDB Atlas connection string
- `MONGODB_DB` - Database name (e.g., `broen-ingestion-test`)
- `MONGODB_COLLECTION` - Collection name (e.g., `broen-documents-test`)
- `API_SECRET_KEY` - API key for authenticating n8n requests

## API Endpoints

### GET /api/health
Health check endpoint (no authentication required).

### POST /api/ingest_single_pdf
Ingest a single PDF from SharePoint.

**Headers:**
- `X-API-Key`: API secret key

**Body:**
```json
{
  "source_type": "sharepoint",
  "file_name": "document.pdf",
  "folder_path": "BroenTest"
}
```

### POST /api/ingest_sharepoint_folder
Ingest all PDFs from a SharePoint folder.

**Headers:**
- `X-API-Key`: API secret key

**Body:**
```json
{
  "folder_path": "Shared Documents/PDFs",
  "max_files": 10
}
```

## File Structure

```
├── function_app.py              # Main Azure Functions code with HTTP triggers
├── utils/
│   ├── __init__.py
│   └── document_processing.py   # Helper functions for chunking, embedding, persistence
├── Dockerfile                   # Container image definition
├── requirements.txt             # Python dependencies
├── host.json                    # Azure Functions host configuration
└── .env.example                 # Environment variable template
```

## MongoDB Schema

Each chunk document contains:
```json
{
  "_id": "ObjectId",
  "chunk_id": "doc-name-hash::chunk-0001",
  "document_id": "doc-name-hash",
  "chunk_index": 0,
  "text": "Chunk text content...",
  "metadata": {
    "source_pdf": "/path/to/file.pdf",
    "pages": [1, 2],
    "headings": ["Section 1"],
    "figure_refs": [...]
  },
  "binary": {
    "figure-id": {
      "data": "base64...",
      "mimeType": "image/png",
      "fileName": "figure.png"
    }
  },
  "embedding": [0.123, 0.456, ...]  // 1024 dimensions
}
```

## Troubleshooting

### 404 - Functions not found
Ensure `AzureWebJobsFeatureFlags=EnableWorkerIndexing` is set in Dockerfile ENV.

### 401 - MongoDB authentication failed
Check `MONGODB_URI` has correct username/password (URL-encode special characters).

### 401 - API key invalid
Verify `X-API-Key` header in n8n matches `API_SECRET_KEY` in Container App env vars.

### Connection refused to localhost:27017
`MONGODB_URI` not set or incorrect. Should be Atlas connection string, not localhost.

## Links

- **GitHub**: https://github.com/Kasper-2904/broenlab-ingestion
- **Container App URL**: https://docling-ingest.thankfulbush-dcd4f38d.swedencentral.azurecontainerapps.io
- **Azure Resource Group**: `rg-broen-lab-ingestion`
