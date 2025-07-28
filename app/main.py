import os
import json
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import re
import sys
import nltk

# Download NLTK resources at build time to avoid network calls during execution
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
except Exception as e:
    print(f"Warning: NLTK download failed: {e}")
    print("Continuing with fallback methods...")

def extract_text_blocks_from_pdf(pdf_path):
    """
    Extract all text blocks and their properties from a PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        List of dictionaries containing text block properties
    """
    all_blocks = []
    
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Get the filename without extension
        filename = os.path.basename(pdf_path)
        
        # Process each page
        for page_num, page in enumerate(doc):
            # Extract text blocks with their properties
            blocks = page.get_text("dict")["blocks"]
            
            # Process each block
            for block_num, block in enumerate(blocks):
                block_info = {
                    "filename": filename,
                    "page_num": page_num,  # 0-indexed page number
                    "block_num": block_num + 1,
                    "block_type": block.get("type", ""),
                    "bbox_x0": block.get("bbox", [0, 0, 0, 0])[0],
                    "bbox_y0": block.get("bbox", [0, 0, 0, 0])[1],
                    "bbox_x1": block.get("bbox", [0, 0, 0, 0])[2],
                    "bbox_y1": block.get("bbox", [0, 0, 0, 0])[3],
                }
                
                # Process lines within blocks (if it's a text block)
                if "lines" in block:
                    block_text = []
                    
                    for line_num, line in enumerate(block["lines"]):
                        line_text = []
                        
                        # Process spans within lines
                        for span in line.get("spans", []):
                            span_text = span.get("text", "")
                            line_text.append(span_text)
                            
                            # Add span-level properties
                            span_info = {
                                **block_info,
                                "line_num": line_num + 1,
                                "text": span_text,
                                "font": span.get("font", ""),
                                "font_size": span.get("size", 0),
                                "color": span.get("color", 0),
                                "flags": span.get("flags", 0),
                                "ascender": span.get("ascender", 0),
                                "descender": span.get("descender", 0),
                                "origin_x": span.get("origin", [0, 0])[0],
                                "origin_y": span.get("origin", [0, 0])[1],
                                "is_bold": bool(span.get("flags", 0) & 2**0),  # 1
                                "is_italic": bool(span.get("flags", 0) & 2**1),  # 2
                                "is_superscript": bool(span.get("flags", 0) & 2**2),  # 4
                                "is_subscript": bool(span.get("flags", 0) & 2**3),  # 8
                                "is_monospace": bool(span.get("flags", 0) & 2**17),  # 131072
                            }
                            all_blocks.append(span_info)
                        
                        block_text.append(" ".join(line_text))
                    
                    # Add block text to the last span info
                    if all_blocks:
                        all_blocks[-1]["block_text"] = "\n".join(block_text)
                
                # Process image blocks
                elif block.get("type") == 1:  # Image block
                    image_info = {
                        **block_info,
                        "text": "[IMAGE]",
                        "line_num": 0,
                        "font": "",
                        "font_size": 0,
                        "color": 0,
                        "flags": 0,
                        "is_image": True,
                        "block_text": "[IMAGE]",
                    }
                    all_blocks.append(image_info)
        
        doc.close()
        return all_blocks
    
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return []

def detect_base_text_properties(blocks_data):
    """
    Detect the most common text properties which likely represent the base text.
    
    Args:
        blocks_data: List of dictionaries containing text block properties
        
    Returns:
        Dictionary with the most common text properties
    """
    # Filter out empty text blocks and very short text (likely not base text)
    valid_blocks = [block for block in blocks_data if len(block.get("text", "").strip()) > 10]
    
    if not valid_blocks:
        return {}
    
    # Count occurrences of font properties
    font_counter = Counter([block["font"] for block in valid_blocks])
    font_size_counter = Counter([round(block["font_size"], 1) for block in valid_blocks])
    is_bold_counter = Counter([block["is_bold"] for block in valid_blocks])
    is_italic_counter = Counter([block["is_italic"] for block in valid_blocks])
    
    # Get the most common properties
    base_properties = {
        "font": font_counter.most_common(1)[0][0],
        "font_size": font_size_counter.most_common(1)[0][0],
        "is_bold": is_bold_counter.most_common(1)[0][0],
        "is_italic": is_italic_counter.most_common(1)[0][0],
    }
    
    # Get min and max font sizes for normalization
    font_sizes = [block["font_size"] for block in blocks_data if block["font_size"] > 0]
    if font_sizes:
        base_properties["min_font_size"] = min(font_sizes)
        base_properties["max_font_size"] = max(font_sizes)
    else:
        base_properties["min_font_size"] = base_properties["font_size"]
        base_properties["max_font_size"] = base_properties["font_size"]
    
    return base_properties

def contains_alphabetic_characters(text):
    """
    Check if the text contains at least one alphabetic character.
    
    Args:
        text: String to check
        
    Returns:
        Boolean indicating if the text contains alphabetic characters
    """
    return bool(re.search('[a-zA-Z]', text))

def identify_potential_headings(blocks_data, base_properties):
    """
    Identify potential headings based on differences from base text properties.
    
    Args:
        blocks_data: List of dictionaries containing text block properties
        base_properties: Dictionary with the base text properties
        
    Returns:
        Updated blocks_data with heading classification
    """
    for block in blocks_data:
        # Skip image blocks
        if block.get("is_image", False):
            block["is_potential_heading"] = False
            block["heading_confidence"] = 0
            continue
        
        # Get the text and perform initial validation
        text = block.get("text", "").strip()
        
        # Skip if text is too short (less than 3 characters) or doesn't contain any alphabetic characters
        if len(text) < 3 or not contains_alphabetic_characters(text):
            block["is_potential_heading"] = False
            block["heading_confidence"] = 0
            block["heading_level"] = "not_heading"
            block["rejected_reason"] = "Too short or no alphabetic characters"
            continue
        
        # Calculate relative font size
        base_font_size = base_properties["font_size"]
        if base_font_size > 0:
            block["relative_font_size"] = block["font_size"] / base_font_size
        else:
            block["relative_font_size"] = 1.0
        
        # Normalize font size between 0 and 1 based on min and max in document
        min_font_size = base_properties["min_font_size"]
        max_font_size = base_properties["max_font_size"]
        font_size_range = max_font_size - min_font_size
        
        if font_size_range > 0:
            block["normalized_font_size"] = (block["font_size"] - min_font_size) / font_size_range
        else:
            block["normalized_font_size"] = 0.5  # Default if all fonts are the same size
        
        # Initialize confidence score
        confidence = 0
        
        # Check for font size difference (larger than base text)
        if block["relative_font_size"] > 1.1:
            confidence += block["relative_font_size"] - 1  # More weight for larger differences
        
        # Check for bold when base is not bold
        if block["is_bold"] and not base_properties["is_bold"]:
            confidence += 1
        
        # Check for different font
        if block["font"] != base_properties["font"]:
            confidence += 0.5
        
        # Check for italic when base is not italic
        if block["is_italic"] and not base_properties["is_italic"]:
            confidence += 0.5
        
        # Check for text length (headings are typically shorter)
        if len(text) < 100 and text.count(" ") < 10:
            confidence += 0.5
        
        # Normalize confidence score (0 to 1)
        block["heading_confidence"] = min(confidence / 4, 1.0)  # Cap at 1.0
        
        # Classify as potential heading if confidence is high enough
        block["is_potential_heading"] = block["heading_confidence"] >= 0.25
        
        # Attempt to determine heading level
        if block["is_potential_heading"]:
            # H1 is typically the largest and boldest
            if block["relative_font_size"] >= 1.5:
                block["heading_level"] = "H1"
            # H2 is typically larger than base but smaller than H1
            elif block["relative_font_size"] >= 1.25:
                block["heading_level"] = "H2"
            # H3 is typically slightly larger than base or bold
            elif block["relative_font_size"] > 1.0 or block["is_bold"]:
                block["heading_level"] = "H3"
            else:
                block["heading_level"] = "unknown"
            block["rejected_reason"] = ""
        else:
            block["heading_level"] = "not_heading"
            block["rejected_reason"] = "Insufficient style differences from base text"
    
    return blocks_data

def merge_multiline_headings(blocks_data):
    """
    Merge consecutive text spans that are likely parts of the same heading.
    Merges spans with similar formatting and vertical proximity.

    Args:
        blocks_data: List of dictionaries containing text block properties

    Returns:
        Updated list of blocks with merged multiline headings
    """
    if not blocks_data:
        return blocks_data

    # Sort blocks by page and position
    sorted_blocks = sorted(
        blocks_data,
        key=lambda x: (x["page_num"], x["bbox_y0"], x["bbox_x0"])
    )

    merged_blocks = []
    current_heading = None

    for block in sorted_blocks:
        # Skip images or non-text
        if block.get("is_image") or not block.get("text", "").strip():
            if current_heading:
                merged_blocks.append(current_heading)
                current_heading = None
            merged_blocks.append(block)
            continue

        # Use normalized properties for comparison
        font = block["font"]
        font_size = round(block["font_size"], 1)
        is_bold = block["is_bold"]
        is_italic = block["is_italic"]
        page_num = block["page_num"]

        # If no current heading, start a new one
        if current_heading is None:
            current_heading = block.copy()
            continue

        # Check if this block is similar
        same_page = (current_heading["page_num"] == page_num)
        similar_font = (current_heading["font"] == font)
        similar_size = (abs(current_heading["font_size"] - font_size) < 0.5)
        same_bold = (current_heading["is_bold"] == is_bold)
        same_italic = (current_heading["is_italic"] == is_italic)

        # If all conditions match, merge
        if all([same_page, similar_font, similar_size, same_bold, same_italic]):
            # Append this block's text to current heading
            current_heading["text"] += " " + block["text"].strip()
            # Optionally update bbox to encompass both
            current_heading["bbox_x1"] = max(current_heading["bbox_x1"], block["bbox_x1"])
            current_heading["bbox_y1"] = block["bbox_y1"]
            # Update block_text if present
            if "block_text" in current_heading:
                current_heading["block_text"] += "\n" + block["text"].strip()
            else:
                current_heading["block_text"] = current_heading["text"]
        else:
            # Finalize current heading and start new
            merged_blocks.append(current_heading)
            current_heading = block.copy()

    # Don't forget the last heading
    if current_heading:
        merged_blocks.append(current_heading)

    return merged_blocks

def normalize_features_for_ml(blocks_data):
    """
    Normalize features for machine learning model training.
    
    Args:
        blocks_data: List of dictionaries containing text block properties
        
    Returns:
        Updated blocks_data with normalized features
    """
    # Create binary features for ML
    for block in blocks_data:
        # Binary features (already normalized)
        block["feature_is_bold"] = 1 if block.get("is_bold", False) else 0
        block["feature_is_italic"] = 1 if block.get("is_italic", False) else 0
        block["feature_is_different_font"] = 1 if block.get("font", "") != block.get("base_font", "") else 0
        
        # Text length features
        text = block.get("text", "").strip()
        block["feature_text_length"] = len(text)
        block["feature_word_count"] = len(text.split()) if text else 0
        
        # Normalize text length (0-1 scale)
        if block["feature_text_length"] > 0:
            # Log transform to handle varying text lengths better
            block["feature_normalized_text_length"] = min(np.log1p(block["feature_text_length"]) / 10, 1.0)
        else:
            block["feature_normalized_text_length"] = 0
            
        # Position features
        block["feature_normalized_x_position"] = block.get("bbox_x0", 0) / 1000  # Arbitrary normalization
        block["feature_normalized_y_position"] = block.get("bbox_y0", 0) / 1000  # Arbitrary normalization
        
        # Create a composite feature for ML
        block["ml_feature_vector"] = [
            block.get("normalized_font_size", 0.5),
            block.get("feature_is_bold", 0),
            block.get("feature_is_italic", 0),
            block.get("feature_is_different_font", 0),
            block.get("feature_normalized_text_length", 0),
            block.get("feature_normalized_x_position", 0),
        ]
    
    return blocks_data

def extract_title_and_headings(blocks_data):
    """
    Extract the document title and headings from the processed blocks.
    
    Args:
        blocks_data: List of dictionaries containing text block properties
        
    Returns:
        Dictionary with title and outline
    """
    # Group blocks by page number
    blocks_by_page = defaultdict(list)
    for block in blocks_data:
        if block.get("is_potential_heading", False) or block.get("normalized_font_size", 0) > 0.8:
            blocks_by_page[block["page_num"]].append(block)
    
    # Sort blocks within each page by y-position (top to bottom)
    for page_num in blocks_by_page:
        blocks_by_page[page_num] = sorted(blocks_by_page[page_num], key=lambda x: x["bbox_y0"])
    
    # Extract title (typically the first large text on the first page)
    title = ""
    if 0 in blocks_by_page and blocks_by_page[0]:  # Check if first page has any blocks
        # Find the largest font on the first page
        first_page_blocks = blocks_by_page[0]
        largest_font_block = max(first_page_blocks, key=lambda x: x.get("font_size", 0))
        
        # If it's significantly larger than others, it's likely the title
        if largest_font_block.get("normalized_font_size", 0) > 0.7:
            title = largest_font_block.get("text", "").strip()
            
            # If the title is too short, try to combine with adjacent blocks
            if len(title) < 10 and len(first_page_blocks) > 1:
                # Get blocks near the title block (vertically)
                title_y = largest_font_block.get("bbox_y0", 0)
                nearby_blocks = [b for b in first_page_blocks if abs(b.get("bbox_y0", 0) - title_y) < 50 and b != largest_font_block]
                
                # Sort by x-position for horizontal reading order
                nearby_blocks = sorted(nearby_blocks, key=lambda x: x.get("bbox_x0", 0))
                
                # Combine with nearby blocks
                for block in nearby_blocks:
                    title += " " + block.get("text", "").strip()
    
    # If no title found, use the first block of text on the first page
    if not title and 0 in blocks_by_page and blocks_by_page[0]:
        title = blocks_by_page[0][0].get("text", "").strip()
    
    # Extract headings
    headings = []
    for page_num in sorted(blocks_by_page.keys()):
        for block in blocks_by_page[page_num]:
            if block.get("is_potential_heading", False) and block.get("heading_level", "") in ["H1", "H2", "H3"]:
                # Skip if this is likely the title (on first page with very large font)
                if page_num == 0 and block.get("text", "").strip() == title:
                    continue
                
                heading = {
                    "level": block.get("heading_level", ""),
                    "text": block.get("text", "").strip(),
                    "page": page_num + 1  # Convert to 1-indexed page number for output
                }
                headings.append(heading)
    
    # Remove duplicate headings (same text and level on the same page)
    unique_headings = []
    seen_headings = set()
    for heading in headings:
        heading_key = (heading["level"], heading["text"], heading["page"])
        if heading_key not in seen_headings:
            unique_headings.append(heading)
            seen_headings.add(heading_key)
    
    return {
        "title": title,
        "outline": unique_headings
    }

def process_pdf_for_challenge1a(pdf_path, output_path):
    """
    Process a PDF file and extract title and headings for Challenge 1A.
    
    Args:
        pdf_path: Path to the PDF file
        output_path: Path to save the output JSON
    """
    print(f"Processing {os.path.basename(pdf_path)}...")
    
    # Extract text blocks and their properties
    blocks_data = extract_text_blocks_from_pdf(pdf_path)
    
    if blocks_data:
        # Detect base text properties
        base_properties = detect_base_text_properties(blocks_data)
        
        # Identify potential headings
        blocks_data = identify_potential_headings(blocks_data, base_properties)
        
        # Merge multiline headings
        blocks_data = merge_multiline_headings(blocks_data)
        
        # Normalize features for ML
        blocks_data = normalize_features_for_ml(blocks_data)
        
        # Extract title and headings
        result = extract_title_and_headings(blocks_data)
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"Output saved to {output_path}")
    else:
        print(f"No data extracted from {pdf_path}")
        
        # Create an empty result with just a title based on filename
        filename = os.path.basename(pdf_path)
        title = os.path.splitext(filename)[0]
        
        result = {
            "title": title,
            "outline": []
        }
        
        # Save to JSON file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=4, ensure_ascii=False)
        
        print(f"Empty output saved to {output_path}")

def main():
    """
    Main function to process PDFs based on the challenge requirements.
    """
    # Default paths for Docker execution
    input_dir = "/app/input"
    output_dir = "/app/output"
    
    # Check if running in development environment
    if not os.path.exists(input_dir):
        # Use local paths for development
        input_dir = "input"
        output_dir = "output"
        
        # Create directories if they don't exist
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all PDF files in the input directory
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return
    
    print(f"Found {len(pdf_files)} PDF files to process")
    
    # Process each PDF file
    for pdf_file in pdf_files:
        pdf_path = os.path.join(input_dir, pdf_file)
        output_path = os.path.join(output_dir, pdf_file.replace('.pdf', '.json'))
        process_pdf_for_challenge1a(pdf_path, output_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main() 