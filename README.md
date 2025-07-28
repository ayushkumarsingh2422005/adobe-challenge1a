# Challenge 1A: PDF Structure Extraction

## Overview
This solution extracts structured outlines from PDF documents, identifying the document title and headings (H1, H2, H3) with their corresponding page numbers. The output is provided in a clean, hierarchical JSON format.

## Approach

Our approach uses a combination of visual formatting cues and content analysis to accurately identify document structure:

1. **Text Block Extraction**: We use PyMuPDF (fitz) to extract text blocks along with their properties (font, size, position, etc.)

2. **Base Text Detection**: The algorithm identifies the most common text properties in the document to establish a baseline for comparison.

3. **Heading Detection**: We analyze multiple features to identify headings:
   - Relative font size (compared to base text)
   - Text formatting (bold, italic)
   - Font differences
   - Text length and position

4. **Heading Classification**: Headings are classified into H1, H2, or H3 based on their relative properties.

5. **Title Extraction**: The document title is identified based on its position (typically at the beginning) and distinctive formatting.

6. **Multiline Heading Merging**: The solution handles cases where headings span multiple lines by merging text blocks with similar properties.

## Libraries Used

- **PyMuPDF (fitz)**: For PDF parsing and text extraction with formatting information
- **pandas/numpy**: For data manipulation and analysis
- **scikit-learn**: For feature normalization and processing
- **NLTK**: For text processing

## How to Build and Run

### Using Docker

1. Build the Docker image:
```bash
docker build --platform linux/amd64 -t challenge1a-solution .
```

2. Run the container:
```bash
docker run --rm -v "$(pwd)/input:/app/input" -v "$(pwd)/output:/app/output" --network none challenge1a-solution
```

For Windows PowerShell:
```powershell
docker run --rm -v "${PWD}/input:/app/input" -v "${PWD}/output:/app/output" --network none challenge1a-solution
```

For Windows Command Prompt:
```cmd
docker run --rm -v "%cd%/input:/app/input" -v "%cd%/output:/app/output" --network none challenge1a-solution
```

### Output Format

The solution generates a JSON file for each input PDF with the following structure:

```json
{
  "title": "Document Title",
  "outline": [
    { "level": "H1", "text": "Section Heading", "page": 1 },
    { "level": "H2", "text": "Subsection Heading", "page": 2 },
    { "level": "H3", "text": "Sub-subsection Heading", "page": 3 }
  ]
}
```

## Performance

- Processes a 50-page PDF in under 10 seconds
- Works offline with no network calls
- Runs efficiently on CPU (amd64) with 8 CPUs and 16GB RAM

## Key Features

- **Format-Agnostic**: Works with various PDF layouts and styles
- **Robust Heading Detection**: Doesn't rely solely on font size
- **Multiline Heading Support**: Properly handles headings that span multiple lines
- **Hierarchical Structure**: Correctly identifies heading levels
- **Efficient Processing**: Optimized for speed and memory usage 