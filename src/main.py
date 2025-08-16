import cv2
import torch
import numpy as np
import re
import pytesseract
import easyocr
from cvzone.FaceMeshModule import FaceMeshDetector
from openlocationcode import openlocationcode as olc

ALLOWED = "23456789CFGHJMPQRVWX+"
import difflib

# Allowed Plus Code characters (excluding vowels/ambiguous chars)
VALID_CHARS = "23456789CFGHJMPQRVWX"

# Common OCR mistake mapping
OCR_FIXES = {
    "O": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "B": "8",
    "G": "6",  # sometimes mixed
}


def clean_code(raw_code: str) -> str:
    """Apply OCR fixes and keep only allowed chars + '+' sign."""
    code = raw_code.upper().replace(" ", "")
    fixed = []
    for ch in code:
        if ch in VALID_CHARS or ch == "+":
            fixed.append(ch)
        elif ch in OCR_FIXES:
            fixed.append(OCR_FIXES[ch])
        else:
            # Fuzzy match single character
            match = difflib.get_close_matches(ch, VALID_CHARS, n=1, cutoff=0.6)
            fixed.append(match[0] if match else "")
    return "".join(fixed)


def recover_code(raw_code: str, ref_lat: float, ref_lng: float) -> str:
    code = clean_code(raw_code)
    print(f'cleaned code: {code}')
    if olc.isValid(code):
        return code

    if olc.isShort(code):
        try:
            return olc.recoverNearest(code, ref_lat, ref_lng)
        except:
            return None

    return None


def nothing(x):
    pass


def pickers():
    cv2.namedWindow("HSV Picker")

    # Create trackbars for lower/upper HSV
    cv2.createTrackbar("LH", "HSV Picker", 0, 180, nothing)
    cv2.createTrackbar("LS", "HSV Picker", 0, 255, nothing)
    cv2.createTrackbar("LV", "HSV Picker", 200, 255, nothing)

    cv2.createTrackbar("UH", "HSV Picker", 180, 180, nothing)
    cv2.createTrackbar("US", "HSV Picker", 40, 255, nothing)
    cv2.createTrackbar("UV", "HSV Picker", 255, 255, nothing)


def preprocess2(img):
    gray = preprocess(img)

    # Step 3: CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    # Step 4: Light blur for noise reduction
    blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)

    # Step 5: Adaptive thresholding (invert for white text)
    binary = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # Step 6: Morphological ops (optional: erode to thin, dilate to connect)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    return cleaned


def clean_plus_code(plus_code):
    reference_latitude = 34.8673
    reference_longitude = 32.6832

    # ex_code = '8G6JVMG8+746'
    if (plus_code.startswith('8G6') or plus_code.startswith('BG')) and olc.isValid(plus_code):
        print(f"OCR extracted: {plus_code} is a valid Plus Code.")


def preprocess(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Get current positions of trackbars
    lh = cv2.getTrackbarPos("LH", "HSV Picker")
    ls = cv2.getTrackbarPos("LS", "HSV Picker")
    lv = cv2.getTrackbarPos("LV", "HSV Picker")

    uh = cv2.getTrackbarPos("UH", "HSV Picker")
    us = cv2.getTrackbarPos("US", "HSV Picker")
    uv = cv2.getTrackbarPos("UV", "HSV Picker")

    # Apply the HSV filter
    lower_white = np.array([lh, ls, lv], dtype=np.uint8)
    upper_white = np.array([uh, us, uv], dtype=np.uint8)

    # Create mask
    mask = cv2.inRange(hsv, lower_white, upper_white)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Apply mask to keep only white letters
    white_only = cv2.bitwise_and(img, img, mask=mask)

    # Convert to grayscale for OCR
    gray = cv2.cvtColor(white_only, cv2.COLOR_BGR2GRAY)

    # Optional: small dilation to connect thin strokes
    kernel = np.ones((1, 1), np.uint8)
    # gray = cv2.dilate(gray, kernel, iterations=1)
    return gray


cap = cv2.VideoCapture('/Users/apolyakov/Downloads/last plane fligth.mp4')
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Compute 3/4 position
start_frame = int(frame_count * 0.75)

# Set position
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

faceMeshDetector = FaceMeshDetector()
pickers()
if torch.backends.mps.is_available():
    device = 'mps'
elif torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print("Device: ", device)

while True:
    start_frame +=1
    print(f"Starting frame: {start_frame} out of {frame_count}")
    success, img = cap.read();
    if not success:  # end of video
        break
    # img = cv2.rectangle(img, (0, 0), (250, 30), (255, 255, 255), 2)
    img = img[0:30, 0:250]
    img = preprocess(img)

    # Perform OCR on the cropped image

    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(img, detail=0)
    texts = [res[1] for res in result]

    # Join into one string (with spaces if multiple words)
    full_text = "".join(result).replace(" ", "")
    if len(result) > 1 and not result[1].startswith("+"):
        full_text = "+".join(result).replace(" ", "")

    # config = r'--oem 3 --psm 10 -c tessedit_char_whitelist=23456789CFGHJMPQRVWX+ --dpi 300'
    # plus_code = pytesseract.image_to_string(img, config=config).strip()
    # clean_plus_code(plus_code)
    print(f'Raw: {full_text}')
    reference_lat, reference_lng = 35.1667, 33.3667
    code = recover_code(full_text, reference_lat, reference_lng)
    codes = []
    if code is not None:
        codes.append(code)

    #cv2.imshow('Video', img)
    #cv2.waitKey(1)
with open("codes.txt", "a", encoding="utf-8") as f:
    for code in codes:
        f.write(code.strip() + "\n")

print("Appended", len(codes), "codes to codes.txt")
