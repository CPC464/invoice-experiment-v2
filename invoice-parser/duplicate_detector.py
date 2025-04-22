import os
import numpy as np
import json
import logging
import subprocess
import sys
from typing import Dict, Any, List, Tuple, Optional, Union
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from preprocess_utils import pdf_to_images, encode_image
import fitz  # PyMuPDF
import base64
from PIL import Image, ImageEnhance, ImageFilter
import io
import re
import pytesseract  # Add import for OCR
from datetime import datetime

# Setup logging
logger = logging.getLogger("duplicate_detector")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Constants for similarity scoring
IDENTICAL_THRESHOLD = 95  # Score above this is considered identical/duplicate
RELATED_THRESHOLD = 70  # Score above this is considered related document
VECTOR_DIMENSION = 384  # Default dimension for small models


class InvoiceDuplicateDetector:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        index_path: str = "invoice-parser/results/vector_index",
        metadata_path: str = "invoice-parser/results/vector_metadata.json",
    ):
        """
        Initialize the duplicate detector with an embedding model

        Args:
            model_name: Name of the SentenceTransformers model to use
            index_path: Path to save/load the FAISS index
            metadata_path: Path to save/load document metadata
        """
        # Create directories if they don't exist
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)

        self.model_name = model_name
        self.index_path = index_path
        self.metadata_path = metadata_path

        # Load the embedding model
        self.model = SentenceTransformer(model_name)

        # Get the embedding dimension
        self.vector_dim = self.model.get_sentence_embedding_dimension()

        # Initialize or load the index
        self.index, self.document_metadata = self._initialize_or_load_index()

    def _initialize_or_load_index(
        self,
    ) -> Tuple[faiss.IndexFlatIP, List[Dict[str, Any]]]:
        """Initialize a new FAISS index or load an existing one"""
        if os.path.exists(self.index_path) and os.path.exists(self.metadata_path):
            # Load existing index
            index = faiss.read_index(self.index_path)

            with open(self.metadata_path, "r") as f:
                document_metadata = json.load(f)

            return index, document_metadata
        else:
            # Create new index - using Inner Product (cosine similarity when vectors are normalized)
            index = faiss.IndexFlatIP(self.vector_dim)
            document_metadata = []

            return index, document_metadata

    def _save_index(self):
        """Save the current index and metadata to disk"""
        faiss.write_index(self.index, self.index_path)

        with open(self.metadata_path, "w") as f:
            json.dump(self.document_metadata, f)

    def _extract_text_from_pdf(self, file_path: str) -> str:
        """
        Extract text content from a PDF file

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text content
        """
        pdf_document = fitz.open(file_path)
        text = ""

        # First try to extract text directly
        for page_num in range(pdf_document.page_count):
            page = pdf_document.load_page(page_num)
            page_text = page.get_text()
            text += page_text

        # If we got minimal text, i.e. below the threshold, the PDF might be scanned/image-based
        # In that case, use OCR to extract text from rendered page images
        if len(text.strip()) < os.getenv("PDF_CHAR_THRESHOLD"):
            text = self._run_ocr_on_pdf(file_path)

        return text

    def _run_ocr_on_pdf(self, file_path: str) -> str:
        """
        Extract text from a PDF using OCR on rendered page images

        Args:
            file_path: Path to the PDF file

        Returns:
            Extracted text from OCR
        """
        try:
            # Convert PDF pages to images
            images_data = pdf_to_images(file_path)

            # Process each page image with OCR
            all_text = []
            for i, img_data in enumerate(images_data):
                try:
                    # Convert image bytes to PIL Image
                    img = Image.open(io.BytesIO(img_data))

                    # Preprocess the image
                    img = self._preprocess_image_for_ocr(img)

                    # Apply OCR
                    custom_config = (
                        r"--oem 3 --psm 6"  # Primary Tesseract config option
                    )
                    page_text = pytesseract.image_to_string(img, config=custom_config)

                    if page_text.strip():
                        all_text.append(page_text)
                        logger.info(
                            f"OCR extracted {len(page_text.split())} words from PDF page {i+1}"
                        )
                    else:
                        # Try with different PSM mode
                        custom_config = (
                            r"--oem 3 --psm 3"  # Alternative Tesseract config option
                        )
                        page_text = pytesseract.image_to_string(
                            img, config=custom_config
                        )
                        all_text.append(page_text)
                        logger.info(
                            f"OCR with PSM 3 extracted {len(page_text.split())} words from PDF page {i+1}"
                        )

                except Exception as e:
                    logger.error(f"Error extracting text from PDF page {i+1}: {str(e)}")

            # Combine all page texts
            combined_text = "\n\n".join(all_text)
            return combined_text

        except Exception as e:
            logger.error(f"Error extracting text from PDF with OCR: {str(e)}")
            return os.path.basename(file_path)

    def _preprocess_image_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess an image to improve OCR results

        Args:
            image: PIL Image object

        Returns:
            Preprocessed PIL Image object
        """
        # Convert to grayscale if not already
        if image.mode != "L":
            image = image.convert("L")

        # Increase size for better OCR if the image is small
        width, height = image.size
        if width < 1000 or height < 1000:
            ratio = 1500 / min(width, height)
            new_width = int(width * ratio)
            new_height = int(height * ratio)

            # Use a resampling filter compatible with different Pillow versions
            try:
                # For newer Pillow versions
                resample_filter = Image.LANCZOS
            except AttributeError:
                try:
                    # For older Pillow versions (pre-9.1.0)
                    resample_filter = Image.ANTIALIAS
                except AttributeError:
                    # Fallback to bicubic for very old versions
                    resample_filter = Image.BICUBIC

            image = image.resize((new_width, new_height), resample=resample_filter)

        # Increase contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        # Apply slight blur to reduce noise
        image = image.filter(ImageFilter.GaussianBlur(radius=0.5))

        # Apply sharpening to enhance text
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(2.0)

        return image

    def _extract_text_from_image(self, file_path: str) -> str:
        """
        Extract text from an image using OCR

        Args:
            file_path: Path to the image file

        Returns:
            Extracted text from the image
        """

        try:
            # Open the image using PIL
            img = Image.open(file_path)

            # Preprocess the image
            img = self._preprocess_image_for_ocr(img)

            # Try multiple OCR configurations to get the best results
            # Order from most specific (for structured invoices) to most general
            psm_modes = [6, 4, 3, 1]  # Different page segmentation modes

            text_results = []
            for psm in psm_modes:
                try:
                    custom_config = f"--oem 3 --psm {psm}"
                    text = pytesseract.image_to_string(img, config=custom_config)
                    if text.strip():
                        text_results.append(text)
                except Exception as e:
                    logger.error(f"OCR with PSM {psm} failed: {str(e)}")
                    continue

            # If we got results, choose the one with the most content
            if text_results:
                # Select the text with the most words as it's likely the most complete
                final_text = max(text_results, key=lambda x: len(x.split()))
                word_count = len(final_text.split())
                logger.info(f"Extracted {word_count} words from image using OCR")
                return final_text
            else:
                logger.warning(f"OCR couldn't extract text from {file_path}")
                # Fall back to filename as a last resort
                return os.path.basename(file_path)

        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            # Fall back to filename as a last resort
            return os.path.basename(file_path)

    def _prepare_document_text(
        self, file_path: str, extracted_data: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a normalized text representation of the document

        Args:
            file_path: Path to the document file
            extracted_data: Optional dictionary of already extracted invoice data

        Returns:
            Normalized text representation of the document
        """
        file_extension = file_path.split(".")[-1].lower()

        # Extract text based on file type
        if file_extension == "pdf":
            raw_text = self._extract_text_from_pdf(file_path)
        else:  # Image files
            raw_text = self._extract_text_from_image(file_path)

        # If we have extracted data, create a structured representation
        if extracted_data:
            return self._create_invoice_representation(raw_text, extracted_data)

        return raw_text

    def _create_invoice_representation(
        self, raw_text: str, extracted_data: Dict[str, Any]
    ) -> str:
        """
        Create a structured representation of an invoice by combining extracted data with raw text

        Args:
            raw_text: Raw text extracted from the document
            extracted_data: Dictionary of structured data extracted from the invoice

        Returns:
            A combined representation optimized for similarity matching because it places the most important fields first, which gives them higher importance in the vector space.
        """
        # Start with high-value fields that are most important for duplicate detection
        representation_parts = []

        # Key invoice identifiers - these help identify exact duplicates
        key_identifiers = {
            "document_number": "Invoice Number",
            "vendor_name": "Vendor",
            "issue_date": "Issue Date",
            "due_date": "Due Date",
            "gross_amount": "Total Amount",
            "currency": "Currency",
            "document_type": "Document Type",
        }

        # Add key identifiers with their labels to maintain context
        identifiers_text = ""
        for field, label in key_identifiers.items():
            if field in extracted_data and extracted_data[field]:
                value = extracted_data[field]
                # Handle lists (like document_type might be)
                if isinstance(value, list):
                    value = " ".join(str(v) for v in value)
                identifiers_text += f"{label}: {value} "

        # Start with the key identifiers (highest weight for duplicate detection)
        representation_parts.append(identifiers_text)

        # Add secondary fields that help with context and related document detection
        secondary_fields = {
            "vendor_address": "Vendor Address",
            "vendor_vat": "Vendor VAT",
            "net_amount": "Net Amount",
            "vat_amount": "VAT Amount",
            "service_from": "Service From",
            "service_to": "Service To",
            "paid_date": "Paid Date",
            "payment_method": "Payment Method",
        }

        secondary_text = ""
        for field, label in secondary_fields.items():
            if field in extracted_data and extracted_data[field]:
                secondary_text += f"{label}: {extracted_data[field]} "

        representation_parts.append(secondary_text)

        # Add line items if available - important for catching similar invoices with different line items
        line_items_text = ""
        if "line_items" in extracted_data and extracted_data["line_items"]:
            for item in extracted_data["line_items"]:
                if "description" in item and item["description"]:
                    line_items_text += f"Item: {item['description']} "
                if "total" in item and item["total"]:
                    line_items_text += f"Price: {item['total']} "

        if line_items_text:
            representation_parts.append(line_items_text)

        # Finally add the raw text but with lower weight (by adding it last)
        # This captures any information not in the structured data
        representation_parts.append(raw_text)

        # Join all parts with extra spaces to ensure separation
        combined_text = "  ".join(representation_parts)

        return combined_text

    def _normalize_text(self, text: str) -> str:
        """Normalize text by removing excess whitespace and converting to lowercase"""
        # Replace multiple whitespace with single space
        normalized = re.sub(r"\s+", " ", text)
        # Convert to lowercase
        normalized = normalized.lower().strip()
        return normalized

    def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate embedding vector for text"""
        # Normalize the text
        normalized_text = self._normalize_text(text)

        # Generate embedding
        embedding = self.model.encode(normalized_text, normalize_embeddings=True)
        return embedding

    def _calculate_similarity_score(self, similarity: float) -> int:
        """Convert raw similarity to 0-100 scale"""
        # Cosine similarity is between -1 and 1, but with normalized vectors it's 0 to 1
        # Scale to 0-100 range
        score = int(max(0, min(100, similarity * 100)))
        return score

    def add_document(
        self,
        file_path: str,
        document_id: str,
        metadata: Optional[Dict[str, Any]] = None,
        extracted_data: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Add a document to the index

        Args:
            file_path: Path to the document file
            document_id: Unique identifier for the document
            metadata: Additional metadata to store with the document
            extracted_data: Optional dictionary of already extracted invoice data
        """
        try:
            # Prepare document text
            document_text = self._prepare_document_text(file_path, extracted_data)

            # Generate embedding
            embedding = self._generate_embedding(document_text)

            # Add to index
            self.index.add(np.array([embedding], dtype=np.float32))

            # Store metadata
            doc_metadata = {
                "id": document_id,
                "file_path": file_path,
                "added_at": str(Path(file_path).stat().st_mtime),
                "vector_id": self.index.ntotal - 1,  # Index is 0-based
            }

            # Add any additional metadata
            if metadata:
                doc_metadata.update(metadata)

            # Add extracted data summary if available
            if extracted_data:
                # Store compact version of the data
                doc_metadata["extracted_summary"] = {
                    k: v
                    for k, v in extracted_data.items()
                    if k
                    in [
                        "vendor_name",
                        "document_type",
                        "document_number",
                        "issue_date",
                        "due_date",
                        "gross_amount",
                    ]
                    and v
                }

            self.document_metadata.append(doc_metadata)

            # Save updated index and metadata
            self._save_index()

            logger.info(f"Added document to index: {document_id}")

        except Exception as e:
            logger.error(f"Error adding document to index: {str(e)}")
            raise

    def find_similar_documents(
        self,
        file_path: str,
        extracted_data: Optional[Dict[str, Any]] = None,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        Find similar documents to the given file

        Args:
            file_path: Path to the document file to check
            extracted_data: Optional dictionary of already extracted invoice data
            top_k: Number of similar documents to return

        Returns:
            List of dictionaries with similarity information
        """
        if self.index.ntotal == 0:
            logger.info("Index is empty, no similar documents to find")
            return []

        try:
            # Prepare document text
            document_text = self._prepare_document_text(file_path, extracted_data)

            # Generate embedding
            embedding = self._generate_embedding(document_text)

            # Search the index - get distances and indices of similar documents
            k = min(top_k, self.index.ntotal)  # Cannot retrieve more than exist
            similarities, indices = self.index.search(
                np.array([embedding], dtype=np.float32), k
            )

            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < 0:  # FAISS returns -1 when fewer than k results are found
                    continue

                # Get document metadata
                doc_metadata = self.document_metadata[idx]

                # Calculate similarity score (0-100)
                score = self._calculate_similarity_score(similarity)

                # Determine match type
                match_type = "distinct"
                if score >= IDENTICAL_THRESHOLD:
                    match_type = "duplicate"
                elif score >= RELATED_THRESHOLD:
                    match_type = "related"

                result = {
                    "document_id": doc_metadata["id"],
                    "similarity_score": score,
                    "match_type": match_type,
                    "metadata": doc_metadata,
                }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Error finding similar documents: {str(e)}")
            raise

    def check_for_duplicates(
        self, file_path: str, extracted_data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Check if a document is a duplicate or related to existing documents

        Args:
            file_path: Path to the document file to check
            extracted_data: Optional dictionary of already extracted invoice data

        Returns:
            Dictionary with duplicate check results
        """
        if self.index.ntotal == 0:
            return {
                "is_duplicate": False,
                "is_related": False,
                "similar_documents": [],
                "highest_score": 0,
            }

        # Find similar documents
        similar_docs = self.find_similar_documents(file_path, extracted_data)

        if not similar_docs:
            return {
                "is_duplicate": False,
                "is_related": False,
                "similar_documents": [],
                "highest_score": 0,
            }

        # Get highest similarity score
        highest_score = max(doc["similarity_score"] for doc in similar_docs)

        # Check if it's a duplicate
        is_duplicate = any(doc["match_type"] == "duplicate" for doc in similar_docs)

        # Check if it's related to any document
        is_related = any(doc["match_type"] == "related" for doc in similar_docs)

        return {
            "is_duplicate": is_duplicate,
            "is_related": is_related,
            "similar_documents": similar_docs,
            "highest_score": highest_score,
        }


# Singleton instance for application-wide use
_duplicate_detector = None


def get_duplicate_detector() -> InvoiceDuplicateDetector:
    """Get or create the duplicate detector singleton instance"""
    global _duplicate_detector
    if _duplicate_detector is None:
        _duplicate_detector = InvoiceDuplicateDetector()
    return _duplicate_detector
