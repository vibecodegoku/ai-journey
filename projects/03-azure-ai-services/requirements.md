# Project 03: Azure AI Document Intelligence App

**XP:** 250 | **Cost:** Free tier | **Duration:** Weeks 5–6

## Objective
Build a document processing pipeline using Azure AI Document Intelligence (formerly Form Recognizer) that extracts structured data from PDFs and images, processes invoices and receipts, and stores results in Azure Cosmos DB. Deploy the pipeline as an Azure Function.

## Azure Services Used
- **Azure AI Document Intelligence** — OCR + layout extraction
- **Azure Blob Storage** — document uploads
- **Azure Cosmos DB** — structured extraction results
- **Azure Functions** — serverless processing pipeline
- **Azure Key Vault** — API key management
- **Azure Monitor** — logs and alerts

## Project Structure
```
03-azure-ai-services/
├── function_app/
│   ├── function_app.py           ← Azure Functions entry point
│   ├── document_processor.py     ← Core extraction logic
│   └── cosmos_client.py          ← Cosmos DB write logic
├── notebooks/
│   ├── 01_explore_document_intelligence.ipynb
│   └── 02_batch_processing.ipynb
├── sample_docs/
│   └── .gitkeep                  ← Add sample PDFs here
├── docs/adr/
│   └── 001-storage-selection.md
├── requirements.txt
└── local.settings.json           ← Local dev config (git-ignored)
```

## Phase 1: Azure Setup (Day 1)

### Create resources via Azure CLI
```bash
# Login
az login

# Create resource group
az group create --name rg-doc-intelligence --location eastus

# Create Document Intelligence resource (free tier: 500 pages/month)
az cognitiveservices account create \
  --name doc-intel-demo \
  --resource-group rg-doc-intelligence \
  --kind FormRecognizer \
  --sku F0 \
  --location eastus \
  --yes

# Get endpoint and key
az cognitiveservices account show \
  --name doc-intel-demo \
  --resource-group rg-doc-intelligence \
  --query "properties.endpoint" -o tsv

az cognitiveservices account keys list \
  --name doc-intel-demo \
  --resource-group rg-doc-intelligence \
  --query "key1" -o tsv

# Create storage account
az storage account create \
  --name stdocsintel$(date +%s) \
  --resource-group rg-doc-intelligence \
  --location eastus \
  --sku Standard_LRS

# Create Cosmos DB (serverless = lowest cost)
az cosmosdb create \
  --name cosmos-doc-results \
  --resource-group rg-doc-intelligence \
  --capabilities EnableServerless

az cosmosdb sql database create \
  --account-name cosmos-doc-results \
  --resource-group rg-doc-intelligence \
  --name DocumentResults

az cosmosdb sql container create \
  --account-name cosmos-doc-results \
  --resource-group rg-doc-intelligence \
  --database-name DocumentResults \
  --name Extractions \
  --partition-key-path "/documentType"
```

## Phase 2: Explore Document Intelligence SDK (Days 2–3)

Install dependencies:
```bash
pip install azure-ai-formrecognizer azure-storage-blob azure-cosmos azure-identity azure-functions
```

In `notebooks/01_explore_document_intelligence.ipynb`:

### Prebuilt Invoice Model
```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
import os

endpoint = os.environ["AZURE_DOC_INTEL_ENDPOINT"]
key = os.environ["AZURE_DOC_INTEL_KEY"]

client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key))

# Analyze a PDF invoice from URL (or use a local file)
invoice_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf"

with client:
    poller = client.begin_analyze_document_from_url("prebuilt-invoice", invoice_url)
    result = poller.result()

for invoice in result.documents:
    print(f"Vendor: {invoice.fields.get('VendorName', {}).get('value', 'N/A')}")
    print(f"Invoice Date: {invoice.fields.get('InvoiceDate', {}).get('value', 'N/A')}")
    print(f"Total: {invoice.fields.get('InvoiceTotal', {}).get('value', 'N/A')}")
    
    items = invoice.fields.get("Items", {}).get("value", [])
    for item in items:
        desc = item.value.get("Description", {}).get("value", "")
        amount = item.value.get("Amount", {}).get("value", "")
        print(f"  - {desc}: {amount}")
```

### Prebuilt Receipt Model
```python
receipt_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/contoso-allinone.jpg"

with client:
    poller = client.begin_analyze_document_from_url("prebuilt-receipt", receipt_url)
    result = poller.result()

for receipt in result.documents:
    print(f"Merchant: {receipt.fields.get('MerchantName', {}).get('value', 'N/A')}")
    print(f"Transaction Date: {receipt.fields.get('TransactionDate', {}).get('value', 'N/A')}")
    total = receipt.fields.get("Total", {}).get("value", 0)
    print(f"Total: ${total:.2f}" if total else "Total: N/A")
```

### Layout Analysis (any document)
```python
with client:
    poller = client.begin_analyze_document_from_url("prebuilt-layout", invoice_url)
    result = poller.result()

for page in result.pages:
    print(f"Page {page.page_number}: {len(page.lines)} lines, {len(page.tables or [])} tables")

for table in result.tables:
    print(f"\nTable: {table.row_count} rows x {table.column_count} cols")
    for cell in table.cells:
        print(f"  [{cell.row_index},{cell.column_index}]: {cell.content}")
```

## Phase 3: Core Processing Logic (Days 4–5)

`function_app/document_processor.py`:
```python
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from datetime import datetime, timezone
import uuid

SUPPORTED_MODELS = {
    "invoice": "prebuilt-invoice",
    "receipt": "prebuilt-receipt",
    "id_document": "prebuilt-idDocument",
    "business_card": "prebuilt-businessCard",
    "layout": "prebuilt-layout",
}

class DocumentProcessor:
    def __init__(self, endpoint: str, key: str):
        self.client = DocumentAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(key)
        )

    def process_document(self, document_url: str, doc_type: str = "invoice") -> dict:
        model_id = SUPPORTED_MODELS.get(doc_type, "prebuilt-layout")
        
        with self.client:
            poller = self.client.begin_analyze_document_from_url(model_id, document_url)
            result = poller.result()

        return {
            "id": str(uuid.uuid4()),
            "documentType": doc_type,
            "modelUsed": model_id,
            "processedAt": datetime.now(timezone.utc).isoformat(),
            "sourceUrl": document_url,
            "pageCount": len(result.pages),
            "confidence": self._avg_confidence(result),
            "fields": self._extract_fields(result, doc_type),
        }

    def _avg_confidence(self, result) -> float:
        confidences = [
            field.confidence
            for doc in (result.documents or [])
            for field in doc.fields.values()
            if hasattr(field, "confidence") and field.confidence is not None
        ]
        return round(sum(confidences) / len(confidences), 3) if confidences else 0.0

    def _extract_fields(self, result, doc_type: str) -> dict:
        if not result.documents:
            return {"raw_text": " ".join(
                line.content for page in result.pages for line in page.lines
            )}
        
        fields = {}
        for doc in result.documents:
            for name, field in doc.fields.items():
                fields[name] = {
                    "value": str(field.value) if field.value is not None else None,
                    "confidence": field.confidence,
                }
        return fields
```

## Phase 4: Azure Function (Days 6–8)

`function_app/function_app.py`:
```python
import azure.functions as func
import json
import logging
import os
from document_processor import DocumentProcessor
from cosmos_client import CosmosWriter

app = func.FunctionApp(http_auth_level=func.AuthLevel.FUNCTION)

processor = DocumentProcessor(
    endpoint=os.environ["AZURE_DOC_INTEL_ENDPOINT"],
    key=os.environ["AZURE_DOC_INTEL_KEY"],
)
writer = CosmosWriter(
    url=os.environ["COSMOS_ENDPOINT"],
    key=os.environ["COSMOS_KEY"],
    database="DocumentResults",
    container="Extractions",
)

@app.route(route="process-document", methods=["POST"])
def process_document(req: func.HttpRequest) -> func.HttpResponse:
    try:
        body = req.get_json()
        doc_url = body.get("url")
        doc_type = body.get("type", "invoice")
        
        if not doc_url:
            return func.HttpResponse(
                json.dumps({"error": "Missing 'url' field"}),
                status_code=400,
                mimetype="application/json",
            )
        
        result = processor.process_document(doc_url, doc_type)
        writer.save(result)
        
        logging.info(f"Processed {doc_type} with {result['pageCount']} pages")
        return func.HttpResponse(
            json.dumps(result),
            status_code=200,
            mimetype="application/json",
        )
    except Exception as e:
        logging.error(f"Processing failed: {e}")
        return func.HttpResponse(
            json.dumps({"error": str(e)}),
            status_code=500,
            mimetype="application/json",
        )
```

`function_app/cosmos_client.py`:
```python
from azure.cosmos import CosmosClient

class CosmosWriter:
    def __init__(self, url: str, key: str, database: str, container: str):
        client = CosmosClient(url, credential=key)
        db = client.get_database_client(database)
        self.container = db.get_container_client(container)

    def save(self, document: dict) -> dict:
        return self.container.upsert_item(document)

    def query(self, doc_type: str, limit: int = 10) -> list:
        query = f"SELECT TOP {limit} * FROM c WHERE c.documentType = '{doc_type}' ORDER BY c._ts DESC"
        return list(self.container.query_items(query, enable_cross_partition_query=True))
```

## Phase 5: Batch Processing Notebook (Days 9–10)

In `notebooks/02_batch_processing.ipynb`:
```python
import concurrent.futures
from document_processor import DocumentProcessor

# Sample batch of document URLs
documents = [
    {"url": "https://...invoice1.pdf", "type": "invoice"},
    {"url": "https://...receipt1.jpg", "type": "receipt"},
    {"url": "https://...invoice2.pdf", "type": "invoice"},
]

processor = DocumentProcessor(
    endpoint=os.environ["AZURE_DOC_INTEL_ENDPOINT"],
    key=os.environ["AZURE_DOC_INTEL_KEY"],
)

def process_one(doc):
    return processor.process_document(doc["url"], doc["type"])

# Process up to 5 documents concurrently (respect rate limits)
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
    results = list(ex.map(process_one, documents))

# Analyze results
import pandas as pd
df = pd.DataFrame([
    {
        "type": r["documentType"],
        "pages": r["pageCount"],
        "confidence": r["confidence"],
        "processed_at": r["processedAt"],
    }
    for r in results
])
print(df.describe())
```

## Run Locally

```bash
# Install Azure Functions Core Tools
brew tap azure/functions
brew install azure-functions-core-tools@4

# Set up local.settings.json (git-ignored)
cat > local.settings.json << 'EOF'
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "AZURE_DOC_INTEL_ENDPOINT": "<your-endpoint>",
    "AZURE_DOC_INTEL_KEY": "<your-key>",
    "COSMOS_ENDPOINT": "<your-cosmos-endpoint>",
    "COSMOS_KEY": "<your-cosmos-key>"
  }
}
EOF

# Start function locally
cd function_app
func start
```

Test:
```bash
curl -X POST http://localhost:7071/api/process-document \
  -H "Content-Type: application/json" \
  -d '{"url": "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-invoice.pdf", "type": "invoice"}'
```

## Acceptance Criteria
- [ ] Invoice extraction returns vendor, date, line items, total with confidence > 0.85
- [ ] Receipt extraction returns merchant name, total, date
- [ ] Results saved to Cosmos DB and queryable
- [ ] Azure Function deployed and invocable via HTTP
- [ ] Batch processing notebook handles 5+ documents concurrently
- [ ] ADR-001 written: why Azure AI Document Intelligence vs AWS Textract vs Google Document AI
- [ ] Total Azure cost under $5 for the project

## ADR Template
Create `docs/adr/001-storage-selection.md`:
```markdown
# ADR-001: Storage Selection for Extraction Results

## Status: Accepted

## Context
Need to store semi-structured extraction results with varying schemas per document type.
Query patterns: by document type, date range, and confidence threshold.

## Decision
Azure Cosmos DB with NoSQL API, serverless capacity mode, partitioned by documentType.

## Consequences
+ Schema-flexible — invoices and receipts have different fields
+ Serverless = zero cost when idle (important for dev/test)
+ Native Azure integration with managed identity
- More expensive than Blob Storage for large result sets
- Cosmos query language differs from standard SQL
```
