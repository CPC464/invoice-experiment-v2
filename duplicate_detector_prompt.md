# Task: Build an Invoice Duplication Detector with Embeddings

Create a Python system that detects duplicate or nearly-identical invoices using vector embeddings. The system should prevent companies from accidentally paying the same invoice twice.

## Core Requirements:

1. Use an open-source embedding model with strong performance (like SBERT/MPNet, Sentence-BERT, E5, or BGE)
2. Keep the embedding model size reasonable (<1GB) to allow deployment without excessive resources
3. Generate embeddings for entire invoices to capture comprehensive semantic information
4. Implement efficient similarity search for invoice comparison
5. Provide a standardized similarity scoring system from 0-100

## Implementation Details:

### Embedding Model

- Use a production-ready open-source embedding model like SentenceTransformers' `all-mpnet-base-v2`, `all-MiniLM-L6-v2` or BAAI's `bge-small-en`
- Process the entire invoice content to generate holistic embeddings
- Balance accuracy vs performance (model size) for practical deployment

### Invoice Processing

- Extract full text content from invoices, including all fields and contextual information
- Create normalized text representations that preserve document structure
- Generate embedding vectors for complete invoice documents
- Implement a vector database or efficient similarity search mechanism

### Similarity Scoring System

- Implement a standardized 0-100 similarity score where:
  - 100 = identical copy
  - 75-99 = likely duplicate with minor variations
  - 50-74 = related documents (e.g., invoice and receipt for same service)
  - 0-49 = distinct documents
- Calculate scores using cosine similarity with appropriate scaling
- Flag invoices that exceed configurable thresholds

### Related Document Detection

- Specifically design the system to recognize when an invoice and receipt relate to the same transaction
- Implement semantic matching that understands document types and their relationships
- Use contextual information (dates, amounts, services) to link related but non-duplicate documents
- Provide relationship classifications (e.g., "invoice-receipt pair", "partial payment")

### System Architecture

- Create modular components for preprocessing, embedding, and matching
- Implement an efficient storage solution for document vectors
- Create a simple API for integration with existing systems
- Provide clear documentation of the approach

## Technical Implementation Guide:

1. Set up efficient vector storage using FAISS, Hnswlib, or similar
2. Develop preprocessing to extract complete text from various document formats
3. Create a scoring system that scales similarity to the 0-100 range
4. Build evaluation metrics to measure accuracy for both duplicates and related documents
5. Implement a simple interface for testing and demonstration

The system should process documents in common formats (PDF, images) and identify both exact duplicates and semantically related documents like invoice-receipt pairs.
