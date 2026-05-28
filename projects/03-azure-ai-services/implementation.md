# Project 03: Azure AI Document Intelligence — Implementation Guide

## Prerequisites
- Project 02 complete
- Azure account with $200 free credit (or free tier)
- Azure CLI installed and logged in (`az login`)
- `conda activate aiarch`
- Packages: `azure-ai-documentintelligence`, `azure-functions-core-tools`, `azure-cosmos`, `azure-storage-blob`

---

## Project Structure
```
03-azure-ai-services/
├── function_app/
│   ├── __init__.py
│   ├── function_app.py         # Azure Function entry point
│   ├── host.json
│   ├── local.settings.json     # Local dev config (not committed)
│   └── requirements.txt
├── document_processor.py       # Core extraction logic
├── cosmos_client.py            # Cosmos DB writes
├── batch_processor.py          # Parallel batch processing
├── test_local.py               # Local test script
├── sample_docs/
│   └── sample_invoice.pdf      # Sample document for testing
├── docs/
│   └── adr/
│       └── ADR-001-storage-selection.md
└── .env
```

---

## Phase 1: Azure Resource Provisioning

### Step 1.1 — Log in and set subscription
```bash
az login
az account list --output table
az account set --subscription "YOUR_SUBSCRIPTION_NAME_OR_ID"
```

### Step 1.2 — Create Resource Group
```bash
RESOURCE_GROUP="aiarch-docai-rg"
LOCATION="eastus"

az group create \
  --name $RESOURCE_GROUP \
  --location $LOCATION
```

### Step 1.3 — Create Document Intelligence resource
```bash
DOC_INTEL_NAME="aiarch-docintel"

az cognitiveservices account create \
  --name $DOC_INTEL_NAME \
  --resource-group $RESOURCE_GROUP \
  --kind FormRecognizer \
  --sku F0 \
  --location $LOCATION \
  --yes

# Get endpoint and key
DOC_INTEL_ENDPOINT=$(az cognitiveservices account show \
  --name $DOC_INTEL_NAME \
  --resource-group $RESOURCE_GROUP \
  --query properties.endpoint -o tsv)

DOC_INTEL_KEY=$(az cognitiveservices account keys list \
  --name $DOC_INTEL_NAME \
  --resource-group $RESOURCE_GROUP \
  --query key1 -o tsv)

echo "Endpoint: $DOC_INTEL_ENDPOINT"
echo "Key: $DOC_INTEL_KEY"
```

### Step 1.4 — Create Storage Account + Blob Container
```bash
STORAGE_ACCOUNT="aiarchdocstorage$RANDOM"

az storage account create \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --location $LOCATION \
  --sku Standard_LRS

STORAGE_CONNECTION=$(az storage account show-connection-string \
  --name $STORAGE_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query connectionString -o tsv)

az storage container create \
  --name "documents" \
  --connection-string "$STORAGE_CONNECTION"

echo "Storage connection: $STORAGE_CONNECTION"
```

### Step 1.5 — Create Cosmos DB
```bash
COSMOS_ACCOUNT="aiarch-cosmos-$RANDOM"

az cosmosdb create \
  --name $COSMOS_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --locations regionName=$LOCATION failoverPriority=0

az cosmosdb sql database create \
  --account-name $COSMOS_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --name "DocumentDB"

az cosmosdb sql container create \
  --account-name $COSMOS_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --database-name "DocumentDB" \
  --name "Extractions" \
  --partition-key-path "/documentType"

COSMOS_ENDPOINT=$(az cosmosdb show \
  --name $COSMOS_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query documentEndpoint -o tsv)

COSMOS_KEY=$(az cosmosdb keys list \
  --name $COSMOS_ACCOUNT \
  --resource-group $RESOURCE_GROUP \
  --query primaryMasterKey -o tsv)

echo "Cosmos Endpoint: $COSMOS_ENDPOINT"
echo "Cosmos Key: $COSMOS_KEY"
```

### Step 1.6 — Save all credentials to .env
```bash
cat > .env << EOF
DOC_INTEL_ENDPOINT=$DOC_INTEL_ENDPOINT
DOC_INTEL_KEY=$DOC_INTEL_KEY
STORAGE_CONNECTION_STRING=$STORAGE_CONNECTION
STORAGE_CONTAINER=documents
COSMOS_ENDPOINT=$COSMOS_ENDPOINT
COSMOS_KEY=$COSMOS_KEY
COSMOS_DATABASE=DocumentDB
COSMOS_CONTAINER=Extractions
EOF
```

---

## Phase 2: Document Processor

### Step 2.1 — Install dependencies
```bash
conda activate aiarch
pip install azure-ai-documentintelligence azure-cosmos azure-storage-blob python-dotenv
```

### Step 2.2 — Create document_processor.py

```python
# document_processor.py
import os
from datetime import datetime, timezone
from typing import Any
from dotenv import load_dotenv
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

load_dotenv()

ENDPOINT = os.environ["DOC_INTEL_ENDPOINT"]
KEY = os.environ["DOC_INTEL_KEY"]


def get_client() -> DocumentIntelligenceClient:
    return DocumentIntelligenceClient(
        endpoint=ENDPOINT,
        credential=AzureKeyCredential(KEY),
    )


def extract_invoice(document_url: str) -> dict[str, Any]:
    """Extract fields from an invoice using the prebuilt invoice model."""
    client = get_client()
    poller = client.begin_analyze_document(
        model_id="prebuilt-invoice",
        body=AnalyzeDocumentRequest(url_source=document_url),
    )
    result = poller.result()

    extracted = {
        "documentType": "invoice",
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUrl": document_url,
        "fields": {},
        "confidence": {},
    }

    for doc in result.documents:
        for field_name, field in doc.fields.items():
            if field is not None:
                extracted["fields"][field_name] = field.content or str(field.value)
                extracted["confidence"][field_name] = round(field.confidence or 0, 4)

    # Check acceptance criteria: confidence > 0.85
    avg_confidence = (
        sum(extracted["confidence"].values()) / len(extracted["confidence"])
        if extracted["confidence"] else 0
    )
    extracted["averageConfidence"] = round(avg_confidence, 4)
    extracted["meetsThreshold"] = avg_confidence > 0.85

    return extracted


def extract_receipt(document_url: str) -> dict[str, Any]:
    """Extract fields from a receipt."""
    client = get_client()
    poller = client.begin_analyze_document(
        model_id="prebuilt-receipt",
        body=AnalyzeDocumentRequest(url_source=document_url),
    )
    result = poller.result()

    extracted = {
        "documentType": "receipt",
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUrl": document_url,
        "fields": {},
        "items": [],
    }

    for doc in result.documents:
        for field_name, field in doc.fields.items():
            if field_name == "Items" and field is not None:
                for item in (field.value or []):
                    item_data = {}
                    for k, v in (item.value or {}).items():
                        item_data[k] = v.content if v else None
                    extracted["items"].append(item_data)
            elif field is not None:
                extracted["fields"][field_name] = field.content or str(field.value)

    return extracted


def extract_id_document(document_url: str) -> dict[str, Any]:
    """Extract fields from an ID document (driver's license, passport)."""
    client = get_client()
    poller = client.begin_analyze_document(
        model_id="prebuilt-idDocument",
        body=AnalyzeDocumentRequest(url_source=document_url),
    )
    result = poller.result()

    extracted = {
        "documentType": "idDocument",
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUrl": document_url,
        "fields": {},
    }

    for doc in result.documents:
        for field_name, field in doc.fields.items():
            if field is not None:
                extracted["fields"][field_name] = field.content or str(field.value)

    return extracted


def extract_layout(document_url: str) -> dict[str, Any]:
    """Extract layout including tables from any document."""
    client = get_client()
    poller = client.begin_analyze_document(
        model_id="prebuilt-layout",
        body=AnalyzeDocumentRequest(url_source=document_url),
    )
    result = poller.result()

    tables = []
    for table in (result.tables or []):
        table_data = {
            "rows": table.row_count,
            "columns": table.column_count,
            "cells": [
                {
                    "row": cell.row_index,
                    "col": cell.column_index,
                    "content": cell.content,
                }
                for cell in table.cells
            ],
        }
        tables.append(table_data)

    return {
        "documentType": "layout",
        "extractedAt": datetime.now(timezone.utc).isoformat(),
        "sourceUrl": document_url,
        "pageCount": len(result.pages) if result.pages else 0,
        "tables": tables,
        "tableCount": len(tables),
    }


def process_document(document_url: str, doc_type: str = "invoice") -> dict[str, Any]:
    """Route to correct extractor based on document type."""
    handlers = {
        "invoice":    extract_invoice,
        "receipt":    extract_receipt,
        "idDocument": extract_id_document,
        "layout":     extract_layout,
    }
    handler = handlers.get(doc_type, extract_layout)
    return handler(document_url)
```

---

## Phase 3: Cosmos DB Client

### Step 3.1 — Create cosmos_client.py

```python
# cosmos_client.py
import os
import uuid
from datetime import datetime, timezone
from dotenv import load_dotenv
from azure.cosmos import CosmosClient, PartitionKey
from azure.cosmos.exceptions import CosmosResourceExistsError

load_dotenv()

COSMOS_ENDPOINT = os.environ["COSMOS_ENDPOINT"]
COSMOS_KEY = os.environ["COSMOS_KEY"]
DATABASE_NAME = os.environ.get("COSMOS_DATABASE", "DocumentDB")
CONTAINER_NAME = os.environ.get("COSMOS_CONTAINER", "Extractions")


def get_container():
    client = CosmosClient(COSMOS_ENDPOINT, credential=COSMOS_KEY)
    database = client.get_database_client(DATABASE_NAME)
    container = database.get_container_client(CONTAINER_NAME)
    return container


def save_extraction(extraction: dict) -> str:
    """Save an extraction result to Cosmos DB. Returns the item id."""
    container = get_container()

    item = {
        "id": str(uuid.uuid4()),
        "documentType": extraction.get("documentType", "unknown"),
        "extractedAt": extraction.get("extractedAt", datetime.now(timezone.utc).isoformat()),
        **extraction,
    }

    container.upsert_item(item)
    print(f"Saved to Cosmos DB: {item['id']} (type={item['documentType']})")
    return item["id"]


def get_extractions_by_type(doc_type: str) -> list[dict]:
    """Query Cosmos DB for all extractions of a given type."""
    container = get_container()
    query = f"SELECT * FROM c WHERE c.documentType = '{doc_type}'"
    items = list(container.query_items(query=query, partition_key=doc_type))
    return items
```

---

## Phase 4: Batch Processor

### Step 4.1 — Create batch_processor.py

```python
# batch_processor.py
import os
import concurrent.futures
from typing import List, Tuple
from document_processor import process_document
from cosmos_client import save_extraction

MAX_WORKERS = 5  # Acceptance criteria: handle 5 concurrent docs


def process_single(doc_url: str, doc_type: str) -> Tuple[str, dict, str]:
    """Process one document and save to Cosmos DB. Returns (url, result, cosmos_id)."""
    try:
        result = process_document(doc_url, doc_type)
        cosmos_id = save_extraction(result)
        return doc_url, result, cosmos_id
    except Exception as e:
        return doc_url, {"error": str(e), "documentType": doc_type}, None


def process_batch(documents: List[Tuple[str, str]]) -> List[dict]:
    """
    Process a batch of documents concurrently.
    
    Args:
        documents: List of (url, doc_type) tuples
    
    Returns:
        List of extraction results
    """
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single, url, dtype): (url, dtype)
            for url, dtype in documents
        }

        for future in concurrent.futures.as_completed(futures):
            url, dtype = futures[future]
            try:
                doc_url, result, cosmos_id = future.result()
                print(f"[OK] {dtype}: {doc_url[:60]}... -> Cosmos ID: {cosmos_id}")
                results.append({"url": doc_url, "type": dtype, "result": result, "cosmosId": cosmos_id})
            except Exception as e:
                print(f"[ERROR] {dtype}: {url[:60]}... -> {e}")
                results.append({"url": url, "type": dtype, "error": str(e)})

    return results


if __name__ == "__main__":
    # Test batch with publicly available sample documents
    sample_docs = [
        ("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf", "invoice"),
        ("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-receipt.png", "receipt"),
        ("https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf", "layout"),
    ]

    print(f"Processing {len(sample_docs)} documents with {MAX_WORKERS} concurrent workers...")
    results = process_batch(sample_docs)
    print(f"\nCompleted: {len([r for r in results if 'error' not in r])}/{len(results)} successful")
```

---

## Phase 5: Azure Function App

### Step 5.1 — Install Azure Functions Core Tools
```bash
brew tap azure/functions
brew install azure-functions-core-tools@4
func --version
```

### Step 5.2 — Create function_app/ directory
```bash
mkdir -p function_app
cd function_app
func init --python
```

### Step 5.3 — Create function_app/function_app.py

```python
# function_app/function_app.py
import json
import logging
import azure.functions as func
import sys
import os

# Add parent directory to path to reuse our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from document_processor import process_document
from cosmos_client import save_extraction
from batch_processor import process_batch

app = func.FunctionApp()


@app.route(route="process", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def process_document_function(req: func.HttpRequest) -> func.HttpResponse:
    """Process a single document."""
    logging.info("Processing document request")

    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON body", status_code=400)

    doc_url = body.get("url")
    doc_type = body.get("type", "invoice")

    if not doc_url:
        return func.HttpResponse(
            json.dumps({"error": "Missing 'url' in request body"}),
            status_code=400,
            mimetype="application/json",
        )

    try:
        result = process_document(doc_url, doc_type)
        cosmos_id = save_extraction(result)
        result["cosmosId"] = cosmos_id

        return func.HttpResponse(
            json.dumps(result, default=str),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as e:
        logging.error(f"Processing error: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )


@app.route(route="batch", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def process_batch_function(req: func.HttpRequest) -> func.HttpResponse:
    """Process a batch of up to 5 documents concurrently."""
    try:
        body = req.get_json()
    except ValueError:
        return func.HttpResponse("Invalid JSON body", status_code=400)

    documents = body.get("documents", [])
    if not documents or len(documents) > 10:
        return func.HttpResponse(
            json.dumps({"error": "Provide 1-10 documents"}),
            status_code=400,
            mimetype="application/json",
        )

    doc_tuples = [(d["url"], d.get("type", "invoice")) for d in documents]
    results = process_batch(doc_tuples)

    return func.HttpResponse(
        json.dumps({"results": results, "processed": len(results)}, default=str),
        status_code=200,
        mimetype="application/json",
    )
```

### Step 5.4 — Create function_app/local.settings.json
```bash
cat > function_app/local.settings.json << 'EOF'
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "DOC_INTEL_ENDPOINT": "YOUR_ENDPOINT",
    "DOC_INTEL_KEY": "YOUR_KEY",
    "COSMOS_ENDPOINT": "YOUR_COSMOS_ENDPOINT",
    "COSMOS_KEY": "YOUR_COSMOS_KEY",
    "COSMOS_DATABASE": "DocumentDB",
    "COSMOS_CONTAINER": "Extractions"
  }
}
EOF
```
> Replace with your actual values from `.env`

### Step 5.5 — Run Function App locally
```bash
cd function_app
func start
```

Test:
```bash
curl -X POST http://localhost:7071/api/process \
  -H "Content-Type: application/json" \
  -d '{
    "url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf",
    "type": "invoice"
  }'
```

---

## Phase 6: Local Integration Test

### Step 6.1 — Create test_local.py

```python
# test_local.py
"""Run this to test all components locally without Azure Functions."""
from document_processor import process_document
from cosmos_client import save_extraction, get_extractions_by_type

SAMPLE_INVOICE = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf"
SAMPLE_RECEIPT = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-receipt.png"


def test_invoice():
    print("\n--- Testing Invoice Extraction ---")
    result = process_document(SAMPLE_INVOICE, "invoice")
    print(f"Fields extracted: {list(result['fields'].keys())}")
    print(f"Average confidence: {result.get('averageConfidence', 'N/A')}")
    assert result.get("averageConfidence", 0) > 0.85, \
        f"Confidence {result.get('averageConfidence')} < 0.85 threshold"
    print("PASS: Invoice confidence meets threshold")

    cosmos_id = save_extraction(result)
    print(f"PASS: Saved to Cosmos DB with ID: {cosmos_id}")
    return result


def test_receipt():
    print("\n--- Testing Receipt Extraction ---")
    result = process_document(SAMPLE_RECEIPT, "receipt")
    print(f"Fields: {list(result['fields'].keys())}")
    print(f"Items: {result.get('items', [])[:2]}")
    cosmos_id = save_extraction(result)
    print(f"PASS: Saved receipt to Cosmos DB: {cosmos_id}")


def test_query():
    print("\n--- Testing Cosmos DB Query ---")
    invoices = get_extractions_by_type("invoice")
    print(f"Found {len(invoices)} invoice records in Cosmos DB")
    assert len(invoices) > 0, "No invoices found in Cosmos DB"
    print("PASS: Query returned results")


if __name__ == "__main__":
    test_invoice()
    test_receipt()
    test_query()
    print("\n=== All tests passed ===")
```

### Step 6.2 — Run tests
```bash
python test_local.py
```

---

## Phase 7: Architecture Decision Record

Create `docs/adr/ADR-001-storage-selection.md`:

```markdown
# ADR-001: Storage Selection for Extraction Results

**Date:** 2026-05-27
**Status:** Accepted

## Context
We need to store document extraction results (JSON, variable schema) from Azure AI Document Intelligence.
Requirements: flexible schema (different models return different fields), queryable by document type,
low operational overhead, serverless-friendly.

## Decision
Use **Azure Cosmos DB (NoSQL API)** partitioned by `documentType`.

## Alternatives Considered
| Option | Pros | Cons |
|--------|------|------|
| **Cosmos DB NoSQL** | Schema-flexible JSON, partition by doc type, serverless mode, SDK support | Cost at scale |
| Azure SQL Database | Familiar SQL, structured | Schema must be predefined; not flexible for varying Document Intelligence fields |
| Azure Blob Storage | Cheapest | Not queryable; requires separate index |
| Azure Table Storage | Cheap, simple | Limited query capabilities |

## Rationale
- Invoice, Receipt, and ID Document models return completely different field sets — Cosmos DB's schemaless storage handles this naturally
- Partition key `/documentType` enables efficient queries like "all invoices"
- Free tier (1000 RU/s, 25 GB) covers the project's volume
- Cosmos DB SDK integrates cleanly with Azure Functions Python runtime

## Consequences
- Free tier sufficient for <400 documents/month
- No joins across document types (acceptable for this use case)
- Future: add TTL on old records to manage storage costs
```

---

## Verification Checklist
- [ ] Azure Resource Group created with all 4 services (Doc Intelligence, Storage, Cosmos DB, optionally Functions)
- [ ] `python test_local.py` runs without errors
- [ ] Invoice extraction returns `averageConfidence > 0.85`
- [ ] Receipt extraction returns items list
- [ ] Both results appear in Cosmos DB (verify in Azure Portal → Cosmos DB → Data Explorer)
- [ ] `python batch_processor.py` processes 3+ documents concurrently
- [ ] `func start` runs the Function App locally
- [ ] `POST /api/process` returns valid JSON with extracted fields
- [ ] ADR-001 written
- [ ] Total Azure cost < $5

---

## Troubleshooting

**`ResourceNotFoundError` from Document Intelligence**
Verify the endpoint URL ends with `/` and the key is correct. Check `az cognitiveservices account show`.

**Cosmos DB `Forbidden` error**
Ensure the Cosmos DB key in `.env` is the primary master key (not read-only key).

**F0 tier rate limit (20 requests/minute)**
Add a `time.sleep(3)` between calls in batch_processor.py when testing with many documents.

**Function App `ImportError`**
Ensure `sys.path.insert` points correctly to the parent directory containing `document_processor.py`.

---

## Next Steps → Project 04: AWS Bedrock Chat
```bash
mkdir -p ~/Documents/ai-journey/projects/04-aws-bedrock
cd ~/Documents/ai-journey/projects/04-aws-bedrock
```
