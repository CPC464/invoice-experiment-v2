import os
import base64
import io
from typing import List
import fitz  # PyMuPDF
from PIL import Image


def encode_image(image_path: str) -> str:
    """
    Encode an image file to base64 string

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string of the image
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def pdf_to_images(pdf_path: str) -> List[bytes]:
    """
    Convert PDF pages to images

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of image byte data for each page
    """
    # Open the PDF document
    pdf_document = fitz.open(pdf_path)
    images = []

    # Iterate through pages
    for page_num in range(pdf_document.page_count):
        # Get page and render at higher resolution
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x resolution
        img_bytes = pix.pil_tobytes(format="JPEG")
        images.append(img_bytes)

    return images


def generate_thumbnail(
    file_path: str, thumbnail_path: str, max_size: tuple = (1200, 1600)
) -> str:
    """
    Generate a thumbnail for an invoice file (PDF or image)

    Args:
        file_path: Path to the invoice file
        thumbnail_path: Path where to save the thumbnail
        max_size: Maximum dimensions for the thumbnail (width, height)

    Returns:
        Path to the generated thumbnail
    """
    try:
        # Get the file extension
        file_extension = file_path.split(".")[-1].lower()

        if file_extension == "pdf":
            # Handle PDF by converting first page to image
            pdf_document = fitz.open(file_path)

            # Get first page
            if pdf_document.page_count > 0:
                page = pdf_document.load_page(0)
                pix = page.get_pixmap(
                    matrix=fitz.Matrix(
                        2.0, 2.0
                    )  # Higher resolution for better readability
                )
                img_data = pix.pil_tobytes(format="JPEG")

                # Open as PIL Image
                img = Image.open(io.BytesIO(img_data))
            else:
                # Empty PDF, create blank image
                img = Image.new("RGB", (100, 100), color="white")

        else:
            # Handle image formats directly
            img = Image.open(file_path)

        # Resize while maintaining aspect ratio
        img.thumbnail(max_size)

        # Save thumbnail with higher quality
        img.save(thumbnail_path, "JPEG", quality=95)

        return thumbnail_path

    except Exception as e:
        print(f"Error generating thumbnail: {str(e)}")
        # Create a blank thumbnail in case of error
        try:
            blank = Image.new("RGB", (100, 100), color="#eeeeee")
            blank.save(thumbnail_path, "JPEG")
            return thumbnail_path
        except:
            return ""
