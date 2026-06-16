# Business Card OCR Scanner

A sophisticated web-based business card scanner that uses AI to extract contact information from business card images. This application combines OpenCV for basic image loading/normalization with OpenAI's GPT-4.1-mini vision model for intelligent text extraction, featuring specialized prompt engineering for high-precision Indian name detection.

## Features

- **AI-Powered OCR**: Utilizes OpenAI's gpt-4.1-mini model for vision-based text extraction.
- **Indian Name Detection**: Specialized prompts and techniques for identifying Indian names regardless of font styling.
- **Image Enhancement Primitives**: Built-in OpenCV utilities (CLAHE, deskewing, unsharp mask, Richardson-Lucy deconvolution, bilateral filter, gamma correction) available for advanced processing.
- **Web Interface**: Clean, responsive web UI for easy business card scanning.
- **Knowledge Base**: Stores confirmed extractions in a JSON file for future reference.

## Project Structure

```
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables (API keys)
├── templates/
│   └── index.html       # Main web interface
├── static/
│   ├── style.css        # CSS styling
│   └── app.js           # Frontend JavaScript
└── knowledgebase.json   # Storage for confirmed extractions
```

## Libraries Used

### Core Dependencies
- **Flask** (2.x): Web framework for the application server
- **python-dotenv**: Environment variable management
- **opencv-python**: Computer vision library for image processing
- **numpy**: Numerical computing for image operations
- **openai**: Official OpenAI Python SDK for vision API calls

### System Dependencies
- **Python 3.8+**: Required runtime environment

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)
- OpenAI API key

### Step 1: Open the Project Directory
Navigate to the project directory.

### Step 2: Create Virtual Environment
```bash
python -m venv .venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
source .venv/bin/activate
```

### Step 4: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 5: Configure Environment Variables
Create a `.env` file in the project root:
```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4.1-mini
```

## How to Use

### Starting the Application
```bash
python app.py
```

The application will start on `http://localhost:5000`

### Using the Web Interface
1. Open your browser and navigate to `http://localhost:5000`
2. Click "Upload Image" to select a business card image from your device
3. Or click "Take Picture" to use your device's camera (mobile devices)
4. Click "Scan Card" to process the image
5. Review the extracted information in the form fields
6. Edit any fields if necessary
7. Click "Confirm Entries" to save the data to the knowledge base

### API Usage
You can also use the API directly:

**POST /scan**
- Upload a business card image
- Returns extracted contact information in JSON format

Example:
```bash
curl -X POST -F "image=@card.jpg" http://localhost:5000/scan
```

## Technical Details

### Image Processing Pipeline
The application implements a clean, efficient pipeline:

1. **Image Loading and Normalization**
   - Decodes image bytes and validates structure.
   - Optimizes resolution: Minimum 900px and Maximum 2400px using Lanczos and Area interpolation to balance detail versus payload size for the OpenAI Vision API.
2. **OpenAI Vision API Call**
   - Encodes the normalized image as a base64 JPEG (88% quality).
   - Sends the image directly to the GPT-4.1-mini vision model with deterministic parameters (`temperature=0`).

### Image Enhancement Primitives (Utility Library)
The code retains 6 modular image enhancement functions for potential pre-processing pipelines or custom utilities:
- `_clahe`: Contrast Limited Adaptive Histogram Equalization.
- `deskew`: Hough line median angle tilt correction (±10°).
- `_unsharp`: Gaussian blur-based unsharp masking to enhance edges.
- `_richardson_lucy`: Richardson-Lucy deconvolution for deblurring text.
- `deblur_image`: Combination of Richardson-Lucy deconvolution, multi-scale unsharp masking, and bilateral filtering.
- `gamma_correct`: Look-up table based gamma correction for low-contrast/dark images.

### OCR and Text Extraction
1. **OpenAI Vision Integration**
   - Sends normalized images to OpenAI's gpt-4.1-mini model.
   - Temperature set to 0.0 for consistent results.
2. **Indian Name Detection**
   - Specialized prompts for Indian name recognition.
   - Visual hierarchy analysis (font size, prominence, position).
   - OCR artifact correction (e.g., "JO HN" → "JOHN").
3. **Field Extraction**
   The system extracts the following fields:
   - **Name**: Person's full name (with special Indian name handling)
   - **Number**: Phone/WhatsApp/mobile numbers
   - **Email**: Email addresses with validation
   - **Address**: Full postal address with Indian format support
   - **Website**: Company website URLs
   - **Company Name**: Organization/business name
   - **Designation**: Job title or role

## Configuration

### Environment Variables
- `OPENAI_API_KEY`: Your OpenAI API key (required)
- `OPENAI_MODEL`: OpenAI model to use (default: gpt-4.1-mini)

### Image Processing Parameters
- Minimum image dimension: 900px
- Maximum image dimension: 2400px

### API Settings
- OpenAI model: gpt-4.1-mini (configurable via `OPENAI_MODEL`)
- Request timeout: 60 seconds (SDK default)
- Temperature: 0.0 (for deterministic outputs)
- Max tokens: 1024

---

**Built with ❤️ for efficient business card scanning and Indian name detection**
