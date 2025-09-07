# bank_statement_modules/ai_core.py

import logging
import json
import pandas as pd
from PIL import Image
import base64
from io import BytesIO
import google.generativeai as genai
from groq import Groq
import re

from bank_statement_modules.config import (
    GROQ_API_KEY,
    GEMINI_API_KEY,
    LLAMA_MODEL,
    GEMINI_MODEL,
    DEFAULT_SCHEMA,
    MAX_COMPLETION_TOKENS_LLAMA,
)
from bank_statement_modules.prompts import prompt1, prompt2


# ---------------------------
# API Clients
# ---------------------------
client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel(GEMINI_MODEL)


# ---------------------------
# Utility
# ---------------------------
def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for API consumption."""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"


# ---------------------------
# Llama-based functions
# ---------------------------
def is_transaction_table(image: Image.Image) -> bool:
    """Check if the table contains transactions by calling Llama."""
    base64_img = encode_image(image)
    try:
        completion = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt1},
                    {"type": "image_url", "image_url": {"url": base64_img}},
                ],
            }],
            temperature=0.0,
            max_completion_tokens=10,
        )
        response = completion.choices[0].message.content.strip().upper()
        return response == "YES"
    except Exception as e:
        logging.warning(f"Error checking if transaction table: {e}")
        return True


def detect_schema_from_first_table(image: Image.Image) -> str:
    """Detect column order from first transactional table and return reordered schema"""
    base64_img = encode_image(image)
    
    try:
        completion = client.chat.completions.create(
            model=LLAMA_MODEL,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt2},
                    {"type": "image_url", "image_url": {"url": base64_img}},
                ],
            }],
            temperature=0.0,
            max_completion_tokens=MAX_COMPLETION_TOKENS_LLAMA,
        )
        raw = completion.choices[0].message.content.strip()
        # Extract the JSON array portion (if any)
        m = re.search(r'(\[.*\])', raw, flags=re.DOTALL)
        if m:
            schema = m.group(1).strip()
            # validate
            try:
                json.loads(schema)
                return schema
            except Exception:
                logging.warning("Detected schema not valid JSON; falling back to default schema")
                return DEFAULT_SCHEMA
        else:
            return DEFAULT_SCHEMA
    except Exception as e:
        logging.error(f"Error detecting schema: {e}")
        return DEFAULT_SCHEMA

# ---------------------------
# Gemini-based functions
# ---------------------------
def extract_table_with_schema(image: Image.Image, schema_template: str) -> str:
    """Extract table content using the reordered schema template - Using Gemini Vision"""
    try:
        prompt = f"""You are a bank statement data extractor. Extract ALL transactions as JSON array using this schema:

{schema_template}

🔍 TABLE ANALYSIS:
- Identify columns: Date, Description, Debit, Credit, Balance
- Count transaction rows (ignore headers/footers)
- Determine date order: ASCENDING (oldest→newest) or DESCENDING (newest→oldest)

💰 AMOUNT MAPPING (Follow schema order exactly):
- Schema "dr" field → Table's DEBIT column value
- Schema "cr" field → Table's CREDIT column value
- Withdrawal/Payment → amount in "dr", "cr"=0.00
- Deposit/Credit → amount in "cr", "dr"=0.00

📝 DESCRIPTION: Extract COMPLETE text (no truncation)

⚖️ VALIDATION (VERY CRITICAL - Check EVERY row):

FOR ASCENDING DATES (oldest→newest):
Row N: balance_previous_row + credit - debit = balance_current_row
Example: 1000 + 500 - 0 = 1500 ✓

FOR DESCENDING DATES (newest→oldest):
Row N: balance_current_row + debit - credit = balance_previous_row
Example: 1300 + 200 - 0 = 1500 ✓

Please check If validation fails, you've swapped debit/credit - FIX immediately by swapping credit and debit!

📋 SCHEMA MAPPING:
- dt: DD-MM-YYYY format
- desc: COMPLETE description text
- ref: Reference ID (null if none)
- dr: Debit amount (0.00 if none)
- cr: Credit amount (0.00 if none)
- bal: Account balance
- type: "W" for withdrawal, "D" for deposit

📝 OUTPUT FORMAT: If it is a non-transactional table, return an empty JSON array: []

** Check again for validation and if all balance and all rows are there is fine you can return the json !! **

🚀 OUTPUT: JSON array only, no markdown. Must Do : Validate EACH row with previous row before proceeding to next row with respect to {schema_template}!
"""

        response = gemini_model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error extracting table with Gemini: {e}")
        return f"Error extracting table: {str(e)}"

def clean_and_fix_json(raw_text: str) -> str:
    """
    Try to sanitize model output into a valid JSON array string.
    - Strip code fences
    - Extract first [...] JSON array if present
    - Fallback: collect {...} objects and wrap into [...]
    - Try a few heuristic fixes (trailing commas, single->double quotes)
    Returns a string (always at least "[]")
    """
    if not raw_text or not raw_text.strip():
        return "[]"
    try:
        txt = raw_text.strip()
        # remove fenced code blocks
        txt = re.sub(r'```(?:json)?\s*', '', txt, flags=re.IGNORECASE)
        txt = txt.replace('```', '')

        # try to extract the first JSON array
        m = re.search(r'(\[.*\])', txt, flags=re.DOTALL)
        if m:
            candidate = m.group(1)
        else:
            # fallback: find all { ... } blocks and combine
            objs = re.findall(r'\{[^{}]*\}', txt, flags=re.DOTALL)
            if objs:
                candidate = "[" + ",".join(objs) + "]"
            else:
                candidate = txt

        # common fixes
        candidate = re.sub(r',\s*]', ']', candidate)   # trailing comma before closing ]
        candidate = re.sub(r',\s*}', '}', candidate)   # trailing comma before closing }
        candidate = candidate.replace('\\n', ' ')
        candidate = candidate.replace('\\', '')

        # Try parse; if fails, replace single quotes with double quotes and try again
        try:
            json.loads(candidate)
            return candidate
        except Exception:
            alt = candidate.replace("'", '"')
            alt = re.sub(r',\s*]', ']', alt)
            try:
                json.loads(alt)
                return alt
            except Exception:
                # final fallback: return empty array
                return "[]"
    except Exception:
        return "[]"

def refine_with_camelot_reference(llm_transactions, camelot_df, detected_schema: str = DEFAULT_SCHEMA):
    """Refine extracted transactions by comparing LLM output with Camelot reference table."""

    if not llm_transactions or camelot_df.empty:
        logging.warning("No transactions or empty Camelot reference - skipping refinement")
        return llm_transactions

    try:
        llm_transactions_json = json.dumps(llm_transactions, indent=2)
        camelot_raw_data = [
            [str(val) if not pd.isna(val) else "" for val in row.values]
            for _, row in camelot_df.iterrows()
        ]
        camelot_raw_json = json.dumps(camelot_raw_data, indent=2)

        refinement_prompt = f"""
You are a **bank statement transaction validator and corrector**.  
Your job is to refine the extracted JSON transactions by comparing them with Camelot’s raw extracted table.

📌 **Detected Schema** (strictly follow this):
{detected_schema}

📌 **Source 1: LLM Extracted JSON**
{llm_transactions_json}

📌 **Source 2: Camelot Raw Table**
{camelot_raw_json}

🔍 **Rules for Validation & Correction**:
1. Preserve schema: fields = dt, desc, ref, dr, cr, bal, type
2. Map amounts correctly:
   - dr → Debit (money deducted) → txn_type = "W"
   - cr → Credit (money added) → txn_type = "D"
3. Validate balances row-by-row:
   - Ascending dates (oldest→newest): balance[n] = balance[n-1] + cr - dr
   - Descending dates (newest→oldest): balance[n-1] = balance[n] + dr - cr
4. If validation fails, fix by:
   - Swapping debit/credit values
   - Correcting amounts using Camelot table as reference
5. Ensure **full description text** is preserved (no truncation).
6. Keep reference IDs (ref) where available, otherwise null.
7. If any row cannot be validated, make the **closest correction** but do not drop rows.

⚠️ Output strictly the **corrected JSON array only** (no markdown, no extra text).
"""

        response = gemini_model.generate_content(refinement_prompt)
        corrected_raw = response.text.strip()
        
        cleaned_json = clean_and_fix_json(corrected_raw)

        try:
            corrected_transactions = json.loads(cleaned_json)
            if isinstance(corrected_transactions, list):
                logging.info("✅ Refinement successful with Camelot reference")
                return corrected_transactions
        except Exception:
            logging.warning("❌ Refinement produced invalid JSON, falling back to LLM output")

        return llm_transactions
    except Exception as e:
        logging.error(f"Error refining with Camelot reference: {e}")
        return llm_transactions
