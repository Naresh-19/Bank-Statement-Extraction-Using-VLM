import os
import logging
from pathlib import Path
import streamlit as st
from table_cropper import crop_tables_from_pdf
from prompts import prompt1,prompt2
from PIL import Image
import warnings
import base64
from io import BytesIO, StringIO
from dotenv import load_dotenv
from groq import Groq
import google.generativeai as genai
import pandas as pd
import json
import re

from bank_extractor import extract_bank_statement

load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    raise ValueError(
        "GROQ_API_KEY not found in environment variables. Please set it in .env file"
    )

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables. Please set it in .env file"
    )

warnings.filterwarnings("ignore", category=UserWarning, message=".*meta parameter.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing keys.*")

# Logging 
logging.basicConfig(level=logging.INFO, format="%(message)s")

client = Groq(api_key=GROQ_API_KEY)

genai.configure(api_key=GEMINI_API_KEY)
gemini_model = genai.GenerativeModel('gemini-2.5-flash')

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
                        {
                            "type": "text",
                            "text": prompt1
                        },
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
                        {
                            "type": "text",
                            "text": prompt2
                        },
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

def clean_and_fix_json(json_text):
    """Clean and fix common JSON formatting issues"""
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

def refine_with_camelot_reference_simple(llm_transactions, camelot_df):
    """
    Simple approach: Send raw Camelot data to LLM and let it figure everything out
    No complex preprocessing - just raw data + schema context
    """
    if not llm_transactions or camelot_df.empty:
        logging.warning("No transactions or empty Camelot reference - skipping refinement")
        return llm_transactions
    
    try:
        # Get detected schema from session state
        detected_schema = st.session_state.get('detected_schema', 
            '[{"dt":"DD-MM-YYYY","desc":"DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]')
        
        # Convert our transactions to JSON for Gemini
        llm_transactions_json = json.dumps(llm_transactions, indent=2)
        
        # Convert Camelot DataFrame to simple raw data (no headers, just values)
        camelot_raw_data = []
        for idx, row in camelot_df.iterrows():
            # Just convert row to list of values, removing NaN
            row_values = [str(val) if not pd.isna(val) else '' for val in row.values]
            camelot_raw_data.append(row_values)
        
        # Convert to JSON for the prompt
        camelot_raw_json = json.dumps(camelot_raw_data, indent=2)
        
        logging.info(f"✅ Sending {len(camelot_raw_data)} raw Camelot rows to LLM for analysis")

        # Simple but powerful prompt - let LLM do all the work
        refinement_prompt = f"""You are a bank transaction validator with expertise in data analysis.

**DETECTED SCHEMA** (Your column order from primary extraction):
{detected_schema}

**SOURCE 1** - Our Perfect Extraction (may have wrong dr/cr swaps):
{llm_transactions_json}

**SOURCE 2** - Raw Camelot Data (no headers, just row values in array format):
{camelot_raw_json}

**YOUR TASK**: 
Fix SOURCE 1 debit/credit errors using SOURCE 2 as reference for validation.

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

FOR DESCENDING DATES (newest→oldest):
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

        # Use Gemini for refinement
        gemini_pro_model = genai.GenerativeModel('gemini-2.5-flash')
        response = gemini_pro_model.generate_content(refinement_prompt)
        corrected_json = response.text.strip()
        
        # Clean and parse response
        cleaned_json = clean_and_fix_json(corrected_json)
        corrected_transactions = json.loads(cleaned_json)
        
        # Validation and logging
        if isinstance(corrected_transactions, list) and len(corrected_transactions) == len(llm_transactions):
            corrections_made = 0
            for orig, corrected in zip(llm_transactions, corrected_transactions):
                if orig.get('dr') != corrected.get('dr') or orig.get('cr') != corrected.get('cr'):
                    corrections_made += 1
                    desc = corrected.get('desc', 'Unknown')[:40]
                    logging.info(f"🔄 Fixed: {desc} | Dr: {orig.get('dr')}→{corrected.get('dr')} | Cr: {orig.get('cr')}→{corrected.get('cr')}")
            
            logging.info(f"✅ Simple refinement completed. Made {corrections_made} corrections from {len(camelot_raw_data)} raw Camelot rows")
            return corrected_transactions
        else:
            logging.warning(f"⚠️ Response format issue - returning original transactions")
            return llm_transactions
            
    except Exception as e:
        logging.warning(f"❌ Simple Camelot refinement failed: {e}")
        logging.info("📝 Returning original transactions")
        return llm_transactions

def expand_compact_json(compact_transactions):
    """Convert compact JSON format to full schema"""
    expanded_transactions = []

    for transaction in compact_transactions:
        expanded = {
            "date": transaction.get("dt"),
            "narration": transaction.get("desc"),
            "reference_number": transaction.get("ref"),
            "withdrawal_dr": float(transaction.get("dr", 0.0)),
            "deposit_cr": float(transaction.get("cr", 0.0)),
            "balance": float(transaction.get("bal", 0.0)),
            "transaction_type": "Withdrawal"
            if transaction.get("type") == "W"
            else "Deposit",
        }
        expanded_transactions.append(expanded)

    return expanded_transactions

# ========== MODIFY combine_json_texts_to_dataframe FUNCTION ==========
def combine_json_texts_to_dataframe(json_texts, image_paths, temp_pdf_path=None):
    """Combine multiple JSON texts with Camelot refinement and enhanced error handling"""
    all_transactions = []

    # First, collect all raw transactions
    for idx, (json_text, img_path) in enumerate(zip(json_texts, image_paths), start=1):
        try:
            if json_text.startswith("Error extracting table:"):
                continue

            clean_json = clean_and_fix_json(json_text)

            try:
                transactions = json.loads(clean_json)
            except json.JSONDecodeError as e:
                logging.warning(f"Table {idx}: JSON parse failed, attempting recovery: {e}")

                pattern = r'\{[^{}]*"dt"[^{}]*?\}'
                matches = re.finditer(pattern, clean_json, re.DOTALL)
                transactions = []

                for match in matches:
                    try:
                        obj_text = match.group(0)
                        obj_text = re.sub(r",\s*}", "}", obj_text)
                        obj_text = re.sub(r"\\+", "\\", obj_text)
                        transaction = json.loads(obj_text)
                        transactions.append(transaction)
                    except Exception as inner_e:
                        logging.warning(f"Failed to parse individual transaction: {inner_e}")
                        continue

                if not transactions:
                    st.error(f"Table {idx}: Could not parse JSON. Raw: {json_text[:300]}...")
                    continue

            if not isinstance(transactions, list):
                logging.warning(f"Table {idx}: Expected array, got {type(transactions)}")
                continue

            # Add raw transactions to combined list
            all_transactions.extend(transactions)
            logging.info(f"Added {len(transactions)} raw transactions from Table {idx}")

        except Exception as e:
            logging.warning(f"Failed to process table {idx}: {e}")
            continue

    # CAMELOT REFINEMENT - Use Camelot as reference for debit/credit validation
    if all_transactions and temp_pdf_path:
        try:
            logging.info("🤖 Running Camelot extraction for debit/credit reference...")
            
            # Extract using Camelot
            def camelot_progress(msg):
                logging.info(f"Camelot: {msg}")
            
            camelot_df, camelot_summary = extract_bank_statement(
                temp_pdf_path, 
                progress_callback=camelot_progress
            )
            
            if not camelot_df.empty:
                logging.info(f"✅ Camelot extracted {len(camelot_df)} transactions for reference")
                
                # Refine our transactions using Camelot reference
                logging.info("🔍 Refining debit/credit classification using Camelot reference...")
                all_transactions = refine_with_camelot_reference_simple(all_transactions, camelot_df)
            else:
                logging.warning("⚠️ Camelot extraction returned empty results - skipping refinement")
                
        except Exception as e:
            logging.warning(f"❌ Camelot extraction failed: {e}")
            logging.info("📝 Continuing without Camelot refinement")

    # Expand transactions and add metadata
    if all_transactions:
        expanded_transactions = []
        transaction_idx = 0
        
        for idx, (json_text, img_path) in enumerate(zip(json_texts, image_paths), start=1):
            if json_text.startswith("Error extracting table:"):
                continue
                
            clean_json = clean_and_fix_json(json_text)
            try:
                original_transactions = json.loads(clean_json)
                if isinstance(original_transactions, list):
                    table_transaction_count = len(original_transactions)
                    
                    # Get the refined transactions for this table
                    table_refined_transactions = all_transactions[transaction_idx:transaction_idx + table_transaction_count]
                    
                    # Expand and add metadata
                    table_expanded = expand_compact_json(table_refined_transactions)
                    filename = Path(img_path).name.replace(".png", "")
                    
                    for transaction in table_expanded:
                        transaction["source_table"] = f"Table_{idx}"
                        transaction["source_file"] = filename
                        expanded_transactions.append(transaction)
                    
                    transaction_idx += table_transaction_count
                    logging.info(f"Processed {len(table_expanded)} refined transactions from Table {idx}")
            except:
                continue
        
        if expanded_transactions:
            df = pd.DataFrame(expanded_transactions)
            logging.info(f"✅ Final result: {len(expanded_transactions)} validated transactions")
            return df
        else:
            return pd.DataFrame()
    else:
        return pd.DataFrame()

def process_pdf_extraction(temp_pdf_path, uploaded_filename):
    """Main extraction processing function"""
    logging.info(f"Starting extraction process for: {uploaded_filename}")

    cropped_image_paths = crop_tables_from_pdf(
        temp_pdf_path,
        confidence_threshold=0.5,
        padding=10,
    )

    if not cropped_image_paths:
        st.warning("No tables detected in the uploaded PDF.")
        return None, None

    extracted_json_texts = []
    reordered_schema = None
    schema_detected_from_table = None
    first_transaction_table_found = False

    for idx, img_path in enumerate(cropped_image_paths, start=1):
        filename = Path(img_path).name
        page_table_info = filename.replace(".png", "")
        logging.info(f"Processing Table : {page_table_info.replace('_', ' ')}")

        img = Image.open(img_path)
        st.image(img, caption=f"Table {idx}", use_container_width=True)

        if not first_transaction_table_found:
            with st.spinner(f"Checking if Table {idx} contains transactions..."):
                is_transaction = is_transaction_table(img)
                
            if is_transaction:
                first_transaction_table_found = True
                schema_detected_from_table = idx
                
                with st.spinner(f"Analyzing Table {idx} (first transaction table) to detect column order..."):
                    reordered_schema = detect_schema_from_first_table(img)
                    # Make the schema publicly available
                    st.session_state.detected_schema = reordered_schema
                    st.success(f"✅ Schema detected from Table {idx}: {reordered_schema}")
                    logging.info(f"Detected reordered schema from Table {idx}: {reordered_schema}")
            else:
                st.info(f"⏭️ Table {idx} is not a transaction table - skipping schema detection")
                logging.info(f"Table {idx} is not a transaction table")

        if reordered_schema:
            with st.spinner(f"Extracting transaction data for Table {idx} using detected schema..."):
                json_text = extract_table_with_schema(img, reordered_schema)
        else:
            default_schema = '[{"dt":"DD-MM-YYYY","desc":"COMPLETE_EXACT_DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]'
            with st.spinner(f"Extracting Table {idx} with default schema..."):
                json_text = extract_table_with_schema(img, default_schema)

        with st.expander(f"View Raw JSON for Table {idx}"):
            st.text_area(f"JSON Response:", json_text, height=150, key=f"json_{idx}")

        extracted_json_texts.append(json_text)

    if first_transaction_table_found:
        st.success(f"🎯 Schema successfully detected from Table {schema_detected_from_table} (first transaction table)")
    else:
        st.warning("⚠️ No transaction tables found - used default schema for all tables")

    if extracted_json_texts:
        combined_df = combine_json_texts_to_dataframe(extracted_json_texts, cropped_image_paths, temp_pdf_path)
        return combined_df, first_transaction_table_found
    else:
        return None, False

def cleanup_temp_files():
    """Clean up all temporary files"""
    temp_files = [f for f in os.listdir('.') if f.startswith('temp_')]
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            logging.info(f"Cleaned up temporary file: {temp_file}")
        except Exception as e:
            logging.warning(f"Failed to clean up {temp_file}: {e}")
    
    # Also clean up any cropped table images
    table_files = [f for f in os.listdir('.') if f.startswith('page') and f.endswith('.png')]
    for table_file in table_files:
        try:
            os.remove(table_file)
            logging.info(f"Cleaned up table image: {table_file}")
        except Exception as e:
            logging.warning(f"Failed to clean up {table_file}: {e}")

def main():
    st.title("PDF Bank Statement Extraction - Hybrid AI Models")
    st.write("Upload a PDF file and then click 'Extract' to process transactions with smart schema detection using Llama for analysis and Gemini Vision for extraction.")

    st.info("🎯 **Hybrid AI Approach**: Llama analyzes table structure & schema, Gemini Vision extracts transaction data for optimal accuracy")

    # Step 1: File Upload
    uploaded_pdf = st.file_uploader("Choose a PDF file", type="pdf", help="Upload your bank statement PDF file")

    if uploaded_pdf is not None:
        # Store uploaded file info in session state for later use
        st.session_state.uploaded_filename = uploaded_pdf.name
        
        # Show file details
        file_details = {
            "Filename": uploaded_pdf.name,
            "File size": f"{uploaded_pdf.size / 1024:.2f} KB"
        }
        st.success("✅ PDF uploaded successfully!")
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(file_details)
        with col2:
            st.info("📋 Click 'Extract to CSV/JSON' below to start processing")

        # Step 2: Extract Button
        if st.button("🚀 Extract to CSV/JSON", type="primary", help="Start the extraction process"):
            # Save uploaded file temporarily
            temp_pdf_path = f"temp_{uploaded_pdf.name}"
            with open(temp_pdf_path, "wb") as f:
                f.write(uploaded_pdf.getbuffer())

            # Store temp file path in session state for later cleanup
            if 'temp_files' not in st.session_state:
                st.session_state.temp_files = []
            st.session_state.temp_files.append(temp_pdf_path)

            # Process the extraction
            combined_df, schema_found = process_pdf_extraction(temp_pdf_path, uploaded_pdf.name)

            if combined_df is not None and not combined_df.empty:
                # Store results in session state
                st.session_state.extraction_results = combined_df
                st.session_state.extraction_complete = True
                
                st.subheader("📊 Extraction Results")
                
                # Display summary statistics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Transactions", len(combined_df))
                with col2:
                    total_withdrawals = combined_df["withdrawal_dr"].sum()
                    st.metric("Total Withdrawals", f"₹{total_withdrawals:,.2f}")
                with col3:
                    total_deposits = combined_df["deposit_cr"].sum()
                    st.metric("Total Deposits", f"₹{total_deposits:,.2f}")
                with col4:
                    withdrawal_count = len(combined_df[combined_df["withdrawal_dr"] > 0])
                    deposit_count = len(combined_df[combined_df["deposit_cr"] > 0])
                    st.metric("W/D Ratio", f"{withdrawal_count}/{deposit_count}")

                # Show sample descriptions
                st.subheader("📝 Sample Transaction Descriptions")
                if len(combined_df) > 0:
                    sample_descriptions = combined_df["narration"].head(3).tolist()
                    for i, desc in enumerate(sample_descriptions, 1):
                        st.text(f"{i}. {desc}")

                # Show full data
                st.subheader("📋 All Extracted Transactions")
                st.dataframe(combined_df, use_container_width=True)

                st.success(f"✅ Successfully extracted {len(combined_df)} transactions!")
                logging.info(f"Extraction complete: {len(combined_df)} transactions ready for download")
                
                st.info(f"""
                🎯 **COMPLETE FINANCIAL OVERVIEW**
                - **Total Credits (Money In):** ₹{total_withdrawals:,.2f} across {deposit_count} transactions
                - **Total Debits (Money Out):** ₹{total_deposits:,.2f} across {withdrawal_count} transactions  
                """)

            else:
                st.error("❌ No valid transaction data could be extracted from the PDF.")

    # Step 3: Download and Clear sections (show only if extraction is complete)
    if 'extraction_complete' in st.session_state and st.session_state.extraction_complete:
        combined_df = st.session_state.extraction_results
        uploaded_filename = st.session_state.get('uploaded_filename', 'bank_statement')
        
        st.subheader("💾 Download Options")
        col1, col2, col3 = st.columns(3)

        with col1:
            csv_buffer = StringIO()
            combined_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()

            pdf_name = Path(uploaded_filename).stem
            csv_filename = f"{pdf_name}_hybrid_transactions.csv"

            st.download_button(
                label="📥 Download CSV",
                data=csv_data,
                file_name=csv_filename,
                mime="text/csv",
                help="Download transactions as CSV file",
            )

        with col2:
            json_data = combined_df.to_json(orient="records", indent=2)
            json_filename = f"{pdf_name}_hybrid_transactions.json"

            st.download_button(
                label="📥 Download JSON",
                data=json_data,
                file_name=json_filename,
                mime="application/json",
                help="Download transactions as JSON file",
            )

        with col3:
            if st.button("🗑️ Clear & Reset", help="Clean up temporary files and reset the session", type="secondary"):
                cleanup_temp_files()
                
                for key in ['extraction_complete', 'extraction_results', 'temp_files', 'uploaded_filename']:
                    if key in st.session_state:
                        del st.session_state[key]
                
                st.success("🧹 All temporary files cleaned up and session reset!")
                st.rerun()  

    if uploaded_pdf is None and ('extraction_complete' not in st.session_state or not st.session_state.extraction_complete):
        st.info("👆 Please upload a PDF file to get started")

if __name__ == "__main__":
    main()