import base64
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, UploadFile, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════════
# SETUP
# ═══════════════════════════════════════════════════════════════════════════════

BASE_DIR            = Path(__file__).resolve().parent
KNOWLEDGE_BASE_FILE = BASE_DIR / "knowledgebase.json"

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

if not OPENAI_API_KEY:
    raise RuntimeError("Missing OPENAI_API_KEY in .env file.")

openai_client = OpenAI(api_key=OPENAI_API_KEY)

app = FastAPI(title="Business Card Scanner", version="2.0.0")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

TARGET_FIELDS = ["name", "number", "email", "address", "website", "company_name", "designation"]


def count_non_null(data: Dict[str, Any]) -> int:
    return sum(1 for v in data.values() if v)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def decode_image(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image data.")

    h, w = img.shape[:2]
    # 900px minimum — good balance of detail vs payload size for OpenAI vision
    if min(h, w) < 900:
        s = 900 / min(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_LANCZOS4)

    h, w = img.shape[:2]
    # Cap at 2400px — vision models do not benefit beyond this
    if max(h, w) > 2400:
        s = 2400 / max(h, w)
        img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)

    return img


def _clahe(gray: np.ndarray, clip: float = 3.5, tile: int = 8) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile)).apply(gray)


def deskew(gray: np.ndarray) -> np.ndarray:
    """Correct tilt up to ±10° via Hough line median angle."""
    try:
        edges = cv2.Canny(gray, 40, 120, apertureSize=3)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=60,
            minLineLength=gray.shape[1] // 5, maxLineGap=15
        )
        if lines is None:
            return gray
        angles = [
            np.degrees(np.arctan2(y2 - y1, x2 - x1))
            for x1, y1, x2, y2 in (l[0] for l in lines)
            if x2 != x1 and abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) < 10
        ]
        if not angles:
            return gray
        angle = float(np.median(angles))
        if abs(angle) < 0.4:
            return gray
        h, w = gray.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                               flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    except Exception:
        return gray


def _unsharp(gray: np.ndarray, sigma: float, strength: float) -> np.ndarray:
    blur = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
    return cv2.addWeighted(gray, 1 + strength, blur, -strength, 0)


def _richardson_lucy(gray: np.ndarray, iterations: int = 4) -> np.ndarray:
    """Fast Richardson-Lucy deconvolution (4 iterations — sufficient for card text)."""
    psf = cv2.getGaussianKernel(5, 1.5)
    psf = psf @ psf.T
    est = np.clip(np.float64(gray) / 255.0, 1e-6, 1.0)
    obs = est.copy()
    for _ in range(iterations):
        conv = np.clip(cv2.filter2D(est, -1, psf), 1e-6, None)
        est *= cv2.filter2D(obs / conv, -1, np.flip(psf))
        est = np.clip(est, 0.0, 1.0)
    return cv2.normalize(est, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def deblur_image(gray: np.ndarray) -> np.ndarray:
    """RL deconvolution → multi-scale unsharp → bilateral cleanup."""
    rl   = _richardson_lucy(gray, iterations=4)
    fine = _unsharp(rl, sigma=0.8, strength=0.8)
    crs  = _unsharp(rl, sigma=2.5, strength=0.5)
    comb = cv2.addWeighted(fine, 0.65, crs, 0.35, 0)
    return cv2.bilateralFilter(comb, d=5, sigmaColor=25, sigmaSpace=25)


def gamma_correct(gray: np.ndarray, gamma: float) -> np.ndarray:
    lut = np.array(
        [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(gray, lut)


# ═══════════════════════════════════════════════════════════════════════════════
# IMAGE → BASE64
# ═══════════════════════════════════════════════════════════════════════════════

def image_to_base64_jpeg(image: np.ndarray, quality: int = 88) -> str:
    """Encode a BGR or grayscale image as JPEG base64 for the OpenAI vision API."""
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("Failed to encode image as JPEG.")
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# NORMALISE RESPONSE
# ═══════════════════════════════════════════════════════════════════════════════

def normalize_response(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized = {}
    for field in TARGET_FIELDS:
        value = data.get(field)
        if value in ("", "N/A", "n/a", "null", "None", None):
            value = None
        elif isinstance(value, str):
            value = value.strip() or None
        normalized[field] = value
    return normalized


# ═══════════════════════════════════════════════════════════════════════════════
# OPENAI VISION — structured extraction
# ═══════════════════════════════════════════════════════════════════════════════

_OPENAI_SYSTEM = (
    "You are a business card OCR and data extraction engine. "
    "You receive a photo of a business card and extract all visible contact information into structured JSON. "
    "Return ONLY valid JSON — no markdown fences, no backticks, no explanation. "
    "If a field is not visible or illegible, use null. Never invent or hallucinate data."
)

_OPENAI_PROMPT = """\
Extract all contact information from this business card image.

Return a JSON object with exactly these keys (no extras):
  name, number, email, address, website, company_name, designation

JSON KEYS - return exactly these, no extras
name, number, email, address, website, company_name, designation

FIELD RULES
name (REQUIRED)
  Full person name or partial name (first name only, surname only, or first+surname). There can be cases when first name is only present like arun or mamta, remember the name is not supposed to be first name + surname/second name
  Never include titles or designations. there can be cases when company name is ahead human name. 
  Reassemble visually split names: "JO HN SM ITH" -> "JOHN SMITH"
  Name is typically in large text regions, often at the top of card
  Common Indian names: Amit, Suresh, Priya, Rahul, Anjali, Vikram, Sneha, Ravi, Neha, Sunil
  If card shows "Company Name | Person Name" or "Company Name - Person Name", extract the person's name part
  Extract even single names or partial names (e.g., "firstname" from "company name | firstname")
  Exclude: company names (unless preceded by separator like | or -), job titles, descriptive text

number (OPTIONAL)
  All phone, tel, mobile, WhatsApp numbers joined with " / "
  Preserve country codes exactly as printed (e.g., +91, +1, +44)
  Do NOT add country codes if not visible on card
  Accept only: digits, spaces, hyphens, dots, parentheses, plus sign
  Multiple numbers separated by " / ": "+91 98765 43210 / +91 011 1234 5678"

email (OPTIONAL)
  Must contain "@" and valid domain
  Common domains: @gmail.com, @yahoo.com, @company domain
  Remove stray spaces inside address
  Example: "john.doe@company.com" 

address (OPTIONAL)
  Full street address: building, street, city, state/province, ZIP/postcode, country (if visible)
  Indian postcode format: 6 digits total (e.g., 400001 or 400 001)
  Concatenate multi-line addresses into single string with commas
  Keywords: Street, Lane, Road, Ave, Building, Flat, Apt, Suite, Floor
  Example: "123 Business Park, Mumbai, Maharashtra 400001, India"

website (OPTIONAL)
  Domain or full URL as printed on card
  Fix visual artifacts: "vww" -> "www", missing dots before extension
  Strip trailing punctuation
  Look for www, http://, https://, or domain patterns
  Example: "www.company.com" or "company.com"

company_name (OPTIONAL)
  Organization name - typically largest/most prominent text
  May be all caps: "INFOSYS", "TCS", "WIPRO"
  May be branded phrase: "Ameya Innovex", "Vinayak Solutions"
  Indicators: Corp, Inc, Ltd, LLC, Solutions, Technologies, Consulting, Group, Co., LLP
  Usually in large text regions

designation (OPTIONAL)
  Job title/role only - NOT company name or statements
  Common titles: CEO, Founder, Director, Manager, Engineer, Senior Developer, VP, CTO, CFO
  Exclude: descriptive statements like "Empowering", "Innovating", "Transforming"
  Do NOT include: "CEO at CompanyName" - extract only "CEO"
  Do NOT extract: "Higher Education - Empowering Girls"
  Often near name or company

EXTRACTION RULES (CRITICAL - FOLLOW STRICTLY)
1. Use null ONLY if field is truly not visible - never guess or invent
2. Fix visual artifacts: 0↔O, 1↔l↔I, rn↔m, vv↔w, |↔l
3. Return ONLY valid JSON - NO markdown fences, NO backticks, NO commentary whatsoever
4. Format example: {"name": "John Doe", "number": "+91 98765 43210", "email": "john@company.com", "address": null, "website": "www.company.com", "company_name": "Company Inc", "designation": "Manager"}
"""


def get_structured_data(image: np.ndarray) -> Dict[str, Any]:
    """
    Send the image to OpenAI vision (gpt-4.1-mini).
    Returns structured contact data as a normalised dict.
    """
    img_b64 = image_to_base64_jpeg(image, quality=88)

    response = openai_client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.0,
        max_completion_tokens=2048,
        messages=[
            {"role": "system", "content": _OPENAI_SYSTEM},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}",
                        },
                    },
                    {"type": "text", "text": _OPENAI_PROMPT},
                ],
            },
        ],
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown fences if the model wraps output despite instructions
    if raw.startswith("```"):
        raw = re.sub(r"^```[a-z]*\n?", "", raw).rstrip("` \n")

    return normalize_response(json.loads(raw))


# ═══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE BASE
# ═══════════════════════════════════════════════════════════════════════════════

def append_to_knowledge_base(entry: Dict[str, Any]) -> None:
    records: list = []
    if KNOWLEDGE_BASE_FILE.exists():
        try:
            records = json.loads(KNOWLEDGE_BASE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    records.append({
        "confirmed_at": datetime.utcnow().isoformat() + "Z",
        "data": normalize_response(entry),
    })
    KNOWLEDGE_BASE_FILE.write_text(json.dumps(records, indent=2), encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# FASTAPI ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
def index():
    return FileResponse(str(BASE_DIR / "templates" / "index.html"))


@app.post("/scan")
async def scan_card(image: UploadFile = File(...)):
    """Upload a business card image and extract structured contact information."""
    try:
        raw_bytes = await image.read()

        # 1. Load & normalise resolution
        img = decode_image(raw_bytes)

        # 2. Send to OpenAI vision for extraction
        structured = get_structured_data(img)

    except json.JSONDecodeError as exc:
        return JSONResponse(status_code=500, content={"error": f"OpenAI returned invalid JSON: {exc}"})
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"error": str(exc)})
    except Exception as exc:
        return JSONResponse(status_code=500, content={"error": str(exc)})

    return {"raw_text": "", "data": structured}


@app.post("/confirm")
async def confirm_data(payload: Dict[str, Any] = Body(...)):
    """Confirm and save extracted card data to the knowledge base."""
    confirmed = normalize_response(payload.get("data", {}))
    append_to_knowledge_base(confirmed)
    return {"message": "Saved to knowledge base.", "data": confirmed}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=5000, reload=True)
