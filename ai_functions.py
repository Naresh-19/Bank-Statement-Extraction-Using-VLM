import os
import logging
import base64
import json
import streamlit as st
import pandas as pd
from io import BytesIO
from PIL import Image
from groq import Groq
import google.generativeai as genai
from prompts import prompt1, prompt2

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = Groq(api_key=GROQ_API_KEY)
genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel("gemini-2.5-flash")


def encode_image(image: Image.Image) -> str:
    """Convert PIL Image to base64 string for Groq API"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    base64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{base64_image}"


def is_transaction_table(image: Image.Image) -> bool:
    """Check if the table contains transactions by looking for transaction indicators"""
    base64_img = encode_image(image)
    
    try:
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt1},
                        {"type": "image_url", "image_url": {"url": base64_img}},
                    ],
                }
            ],
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
            model="meta-llama/llama-4-maverick-17b-128e-instruct",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt2},
                        {"type": "image_url", "image_url": {"url": base64_img}},
                    ],
                }
            ],
            temperature=0.0,
            max_completion_tokens=300,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error detecting schema: {e}")
        return '[{"dt":"DD-MM-YYYY","desc":"COMPLETE_EXACT_DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]'


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


def refine_with_camelot_reference_simple(llm_transactions, camelot_df):
    """
    Simple approach: Send raw Camelot data to LLM and let it figure everything out
    No complex preprocessing - just raw data + schema context
    """
    if not llm_transactions or camelot_df.empty:
        logging.warning(
            "No transactions or empty Camelot reference - skipping refinement"
        )
        return llm_transactions
    
    try:
        detected_schema = st.session_state.get(
            "detected_schema",
            '[{"dt":"DD-MM-YYYY","desc":"DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]',
        )
        
        llm_transactions_json = json.dumps(llm_transactions, indent=2)
        
        camelot_raw_data = []
        for idx, row in camelot_df.iterrows():
            row_values = [str(val) if not pd.isna(val) else "" for val in row.values]
            camelot_raw_data.append(row_values)
        
        camelot_raw_json = json.dumps(camelot_raw_data, indent=2)
        
        logging.info(
            f"✅ Sending {len(camelot_raw_data)} raw Camelot rows to LLM for analysis"
        )
        
        refinement_prompt = f"""You are a bank transaction validator with expertise in data analysis.

**DETECTED SCHEMA** (Your column order from primary extraction): {detected_schema}

**SOURCE 1** - Our Perfect Extraction (may have wrong dr/cr swaps): {llm_transactions_json}

**SOURCE 2** - Raw Camelot Data (no headers, just row values in array format): {camelot_raw_json}

**YOUR TASK**: Fix SOURCE 1 debit/credit errors using SOURCE 2 as reference for validation.

**ANALYSIS INSTRUCTIONS**:
1. **Understand Schema Order**: Look at the detected schema to understand our column sequence
2. **Analyze Raw Camelot**: Each row in SOURCE 2 is [value1, value2, value3, ...]
   - Identify which values are dates (patterns like DD-MM-YYYY)
   - Identify which values are descriptions (text content)
   - Identify which values are amounts (numeric values)
   - Determine if amounts represent debits or credits based on:
     * Position in the row (left amounts often debits, right amounts often credits)
     * Negative values (usually debits)
     * Context from descriptions (ATM/Withdrawal = debit, Deposit/Credit = credit)

3. **Match Transactions**:
   - Match SOURCE 1 and SOURCE 2 transactions by:
     * Date similarity (exact or close dates)
     * Description keyword overlap
     * Amount value similarity

4. **Correct Errors**:
   - If Camelot suggests transaction is DEBIT but our transaction has dr=0, cr>0 → SWAP dr and cr
   - If Camelot suggests transaction is CREDIT but our transaction has cr=0, dr>0 → SWAP dr and cr
   - Keep all other fields (dt, desc, ref, bal, type) exactly the same
   - Only make corrections when you're confident about the match

⚖️ VALIDATION (VERY CRITICAL - Check EVERY row):

FOR ASCENDING DATES (oldest→newest):
Row N: balance_previous_row + credit - debit = balance_current_row
Example: 1000 + 500 - 0 = 1500 ✓

FOR DESCENDED DATES (newest→oldest):
Row N: balance_current_row + debit - credit = balance_previous_row
Example: 1300 + 200 - 0 = 1500 ✓

Please check If validation fails, you've swapped debit/credit - FIX immediately by swapping credit and debit!

**EXAMPLE ANALYSIS**:
Schema: [{{"dt":"DD-MM-YYYY","desc":"DESC","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}}]
Our data: {{"dt":"01-01-2024","desc":"ATM WITHDRAWAL","dr":0.00,"cr":500.00,"bal":1000.00}}
Camelot row: ["01-01-2024", "ATM", "WITHDRAWAL", "500.00", "0.00", "1000.00"]

Analysis: Date matches, description "ATM WITHDRAWAL" matches "ATM" + "WITHDRAWAL", amount 500 appears in position suggesting debit
Correction: Swap dr/cr → {{"dt":"01-01-2024","desc":"ATM WITHDRAWAL","dr":500.00,"cr":0.00,"bal":1000.00}}

**OUTPUT**: Return corrected JSON array in exact same format as SOURCE 1. No explanations, just the corrected JSON."""
        
        gemini_pro_model = genai.GenerativeModel("gemini-2.5-flash")
        response = gemini_pro_model.generate_content(refinement_prompt)
        corrected_json = response.text.strip()
        
        cleaned_json = clean_and_fix_json(corrected_json)
        corrected_transactions = json.loads(cleaned_json)
        
        if isinstance(corrected_transactions, list) and len(
            corrected_transactions
        ) == len(llm_transactions):
            corrections_made = 0
            for orig, corrected in zip(llm_transactions, corrected_transactions):
                if orig.get("dr") != corrected.get("dr") or orig.get(
                    "cr"
                ) != corrected.get("cr"):
                    corrections_made += 1
                    desc = corrected.get("desc", "Unknown")[:40]
                    logging.info(
                        f"🔄 Fixed: {desc} | Dr: {orig.get('dr')}→{corrected.get('dr')} | Cr: {orig.get('cr')}→{corrected.get('cr')}"
                    )
            
            logging.info(
                f"✅ Simple refinement completed. Made {corrections_made} corrections from {len(camelot_raw_data)} raw Camelot rows"
            )
            return corrected_transactions
        else:
            logging.warning(
                f"⚠️ Response format issue - returning original transactions"
            )
            return llm_transactions
    
    except Exception as e:
        logging.warning(f"❌ Simple Camelot refinement failed: {e}")
        logging.info("📝 Returning original transactions")
        return llm_transactions


def clean_and_fix_json(json_text):
    """Clean and fix common JSON formatting issues"""
    import re
    
    json_text = re.sub(r"```\s*json", "", json_text)
    json_text = re.sub(r"```", "", json_text)
    
    start_idx = json_text.find("[")
    end_idx = json_text.rfind("]")
    
    if start_idx != -1 and end_idx != -1:
        json_text = json_text[start_idx : end_idx + 1]
    
    json_text = re.sub(r",\s*}", "}", json_text)
    json_text = re.sub(r",\s*]", "]", json_text)
    
    def fix_string_content(match):
        content = match.group(1)
        return '"' + re.sub(r"\s+", " ", content.strip()) + '"'
    
    json_text = re.sub(r'"([^"]*(?:\n[^"]*)*)"', fix_string_content, json_text)
    return json_text