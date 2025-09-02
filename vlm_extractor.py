import os
import logging
from pathlib import Path
import streamlit as st
from PIL import Image
import warnings
import base64
from io import BytesIO, StringIO
from dotenv import load_dotenv
import pandas as pd
import json
import re
import PyPDF2
import time
import gc
import shutil

from camelot_cropper import crop_tables_from_pdf
from camelot_extractor import extract_bank_statement
from css import streamlit_css
from ai_functions import (
    is_transaction_table,
    detect_schema_from_first_table,
    extract_table_with_schema,
    refine_with_camelot_reference_simple,
    clean_and_fix_json
)

load_dotenv(override=True)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY or not GEMINI_API_KEY:
    raise ValueError(
        "GROQ_API_KEY or GEMINI_API_KEY not found in environment variables. Please set them in .env file"
    )

warnings.filterwarnings("ignore", category=UserWarning, message=".*meta parameter.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*missing keys.*")

logging.basicConfig(level=logging.INFO, format="%(message)s")


def handle_password_protected_pdf(uploaded_file, filename):
    """Handle password-protected PDFs and return temp file path (same name always)"""
    temp_pdf_path = f"temp_{filename}"
    
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    try:
        with open(temp_pdf_path, "rb") as file_handle:
            pdf_reader = PyPDF2.PdfReader(file_handle)
            
            if pdf_reader.is_encrypted:
                st.warning("🔐 This PDF is password protected")
                
                password = st.text_input(
                    "Enter PDF password:",
                    type="password",
                    key="pdf_password",
                    help="Enter the password to unlock this PDF",
                )
                
                if password:
                    with open(temp_pdf_path, "rb") as pdf_file:
                        encrypted_data = pdf_file.read()
                    
                    from io import BytesIO
                    
                    pdf_stream = BytesIO(encrypted_data)
                    pdf_reader = PyPDF2.PdfReader(pdf_stream)
                    
                    if pdf_reader.decrypt(password):
                        pdf_writer = PyPDF2.PdfWriter()
                        for page in pdf_reader.pages:
                            pdf_writer.add_page(page)
                        
                        with open(temp_pdf_path, "wb") as output_file:
                            pdf_writer.write(output_file)
                        
                        pdf_stream.close()
                        del pdf_reader, pdf_writer, pdf_stream
                        
                        st.success("✅ PDF unlocked successfully!")
                        return temp_pdf_path
                    else:
                        pdf_stream.close()
                        st.error("❌ Incorrect password. Please try again.")
                        return None
                else:
                    st.info("👆 Please enter the password to continue")
                    return None
            else:
                return temp_pdf_path
    
    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None


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


def process_pdf_extraction(temp_pdf_path, uploaded_filename):
    """Main extraction processing function"""
    logging.info(f"Starting extraction process for: {uploaded_filename}")
    
    try:
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
                    
                    with st.spinner(
                        f"Analyzing Table {idx} (first transaction table) to detect column order..."
                    ):
                        reordered_schema = detect_schema_from_first_table(img)
                        st.session_state.detected_schema = reordered_schema
                        with st.expander("View Detected Schema"):
                            st.success(f"✅ Schema detected from Table {idx}: {reordered_schema}")
                        
                        logging.info(
                            f"Detected reordered schema from Table {idx}: {reordered_schema}"
                        )
                else:
                    st.info(
                        f"⏭️ Table {idx} is not a transaction table - skipping schema detection"
                    )
                    logging.info(f"Table {idx} is not a transaction table")
            
            if reordered_schema:
                with st.spinner(
                    f"Extracting transaction data for Table {idx} using detected schema..."
                ):
                    json_text = extract_table_with_schema(img, reordered_schema)
            else:
                with st.expander("View Schema Template"):
                    default_schema = '[{"dt":"DD-MM-YYYY","desc":"COMPLETE_EXACT_DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]'
                with st.spinner(f"Extracting Table {idx} with default schema..."):
                    json_text = extract_table_with_schema(img, default_schema)
            
            with st.expander(f"View Raw JSON for Table {idx}"):
                st.text_area(
                    "JSON Response:", json_text, height=150, key=f"json_{idx}"
                )
            
            extracted_json_texts.append(json_text)
        
        if first_transaction_table_found:
            st.success(
                f"Schema successfully detected from Table {schema_detected_from_table} (first transaction table)"
            )
        else:
            st.warning(
                "⚠️ No transaction tables found - used default schema for all tables"
            )
        
        if extracted_json_texts:
            combined_df = combine_json_texts_to_dataframe(
                extracted_json_texts, cropped_image_paths, temp_pdf_path
            )
            return combined_df, first_transaction_table_found
        else:
            return None, False
    
    except Exception as e:
        logging.error(f"Error in process_pdf_extraction: {e}")
        cleanup_temp_files(temp_pdf_path)
        raise


def cleanup_temp_files(temp_pdf_path, cropped_image_paths=None):
    """Centralized cleanup function for temporary files including cropped images"""
    gc.collect()
    time.sleep(0.5)
    
    if cropped_image_paths and len(cropped_image_paths) > 0:
        first_image_path = Path(cropped_image_paths[0])
        table_folder = first_image_path.parent
        
        if table_folder.exists():
            try:
                shutil.rmtree(table_folder)
                logging.info(f"✅ Auto-cleaned entire table folder: {table_folder}")
            except Exception as e:
                logging.warning(f"Failed to cleanup table folder {table_folder}: {e}")
                for img_path in cropped_image_paths:
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                            logging.info(
                                f"✅ Auto-cleaned cropped image: {Path(img_path).name}"
                            )
                        except Exception as e:
                            logging.warning(
                                f"Failed to cleanup cropped image {img_path}: {e}"
                            )
    
    if temp_pdf_path and os.path.exists(temp_pdf_path):
        try:
            for attempt in range(3):
                try:
                    os.remove(temp_pdf_path)
                    logging.info(f"✅ Auto-cleaned temporary PDF: {temp_pdf_path}")
                    break
                except PermissionError:
                    if attempt < 2:
                        time.sleep(1.0)
                    continue
            else:
                logging.warning(
                    f"⚠️ Could not delete {temp_pdf_path} - file may be in use. Manual cleanup needed."
                )
        
        except Exception as e:
            logging.warning(f"Failed to auto-cleanup PDF {temp_pdf_path}: {e}")
    
    try:
        table_files = [
            f for f in os.listdir(".") if f.startswith("page") and f.endswith(".png")
        ]
        for table_file in table_files:
            try:
                os.remove(table_file)
                logging.info(f"✅ Auto-cleaned remaining table image: {table_file}")
            except Exception as e:
                logging.warning(f"Failed to cleanup table image {table_file}: {e}")
    except Exception as e:
        logging.warning(f"Failed to cleanup table images: {e}")


def combine_json_texts_to_dataframe(json_texts, image_paths, temp_pdf_path=None):
    """Combine multiple JSON texts with Camelot refinement and enhanced error handling"""
    all_transactions = []
    
    try:
        for idx, (json_text, img_path) in enumerate(
            zip(json_texts, image_paths), start=1
        ):
            try:
                if json_text.startswith("Error extracting table:"):
                    continue
                
                clean_json = clean_and_fix_json(json_text)
                
                try:
                    transactions = json.loads(clean_json)
                except json.JSONDecodeError as e:
                    logging.warning(
                        f"Table {idx}: JSON parse failed, attempting recovery: {e}"
                    )
                    
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
                            logging.warning(
                                f"Failed to parse individual transaction: {inner_e}"
                            )
                            continue
                    
                    if not transactions:
                        st.error(
                            f"Table {idx}: Could not parse JSON. Raw: {json_text[:300]}..."
                        )
                        continue
                
                if not isinstance(transactions, list):
                    logging.warning(
                        f"Table {idx}: Expected array, got {type(transactions)}"
                    )
                    continue
                
                all_transactions.extend(transactions)
                logging.info(
                    f"Added {len(transactions)} raw transactions from Table {idx}"
                )
            
            except Exception as e:
                logging.warning(f"Failed to process table {idx}: {e}")
                continue
        
        if all_transactions and temp_pdf_path:
            try:
                if not os.path.exists(temp_pdf_path):
                    logging.warning(
                        f"❌ Temp PDF file not found: {temp_pdf_path} - skipping Camelot refinement"
                    )
                else:
                    logging.info(
                        "🤖 Running Camelot extraction for debit/credit reference..."
                    )
                    
                    def camelot_progress(msg):
                        logging.info(f"Camelot: {msg}")
                    
                    camelot_df, camelot_summary = extract_bank_statement(
                        temp_pdf_path, progress_callback=camelot_progress
                    )
                    
                    if not camelot_df.empty:
                        logging.info(
                            f"✅ Camelot extracted {len(camelot_df)} transactions for reference"
                        )
                        
                        logging.info(
                            "🔍 Refining debit/credit classification using Camelot reference..."
                        )
                        all_transactions = refine_with_camelot_reference_simple(
                            all_transactions, camelot_df
                        )
                    else:
                        logging.warning(
                            "⚠️ Camelot extraction returned empty results - skipping refinement"
                        )
            
            except Exception as e:
                logging.warning(f"❌ Camelot extraction failed: {e}")
                logging.info("📝 Continuing without Camelot refinement")
        
        if all_transactions:
            expanded_transactions = []
            transaction_idx = 0
            
            for idx, (json_text, img_path) in enumerate(
                zip(json_texts, image_paths), start=1
            ):
                if json_text.startswith("Error extracting table:"):
                    continue
                
                clean_json = clean_and_fix_json(json_text)
                try:
                    original_transactions = json.loads(clean_json)
                    if isinstance(original_transactions, list):
                        table_transaction_count = len(original_transactions)
                        
                        table_refined_transactions = all_transactions[
                            transaction_idx : transaction_idx + table_transaction_count
                        ]
                        
                        table_expanded = expand_compact_json(table_refined_transactions)
                        # filename = Path(img_path).name.replace(".png", "")  # COMMENTED: Source file tracking
                        
                        for transaction in table_expanded:
                            # transaction["source_table"] = f"Table_{idx}"  # COMMENTED: Source table tracking
                            # transaction["source_file"] = filename  # COMMENTED: Source file tracking
                            expanded_transactions.append(transaction)
                        
                        transaction_idx += table_transaction_count
                        logging.info(
                            f"Processed {len(table_expanded)} refined transactions from Table {idx}"
                        )
                except:
                    continue
            
            if expanded_transactions:
                df = pd.DataFrame(expanded_transactions)
                logging.info(
                    f"✅ Final result: {len(expanded_transactions)} validated transactions"
                )
                return df
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    
    finally:
        if temp_pdf_path:
            cleanup_temp_files(temp_pdf_path, image_paths)


def main():
    st.markdown(streamlit_css, unsafe_allow_html=True)
    
    st.title("Bank Statement Transaction Extraction")
    st.write(
        "Upload a PDF file and then click 'Extract to CSV/JSON' to process transactions with smart schema detection using Llama for analysis and Gemini Vision for extraction."
    )
    
    st.info(
        "🎯 **Hybrid Approach**: Llama analyzes table structure & schema, Gemini Vision extracts transaction data for optimal accuracy"
    )
    
    st.success(
        "🔒 **Privacy Protected**: Only the table images displayed below are sent to AI models for processing. No personal account details, passwords, or other sensitive information from your PDF are transmitted to any external AI service."
    )
    
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file", type="pdf", help="Upload your bank statement PDF file"
    )
    
    if uploaded_pdf is not None:
        st.session_state.uploaded_filename = uploaded_pdf.name
        
        file_details = {
            "Filename": uploaded_pdf.name,
            "File size": f"{uploaded_pdf.size / 1024:.2f} KB",
        }
        st.success("✅ PDF uploaded successfully!")
        
        temp_pdf_path = handle_password_protected_pdf(uploaded_pdf, uploaded_pdf.name)
        
        if temp_pdf_path is None:
            st.stop()
        
        st.session_state.temp_pdf_path = temp_pdf_path
        
        col1, col2 = st.columns(2)
        with col1:
            st.json(file_details)
        with col2:
            st.info("📋 Click 'Extract to CSV/JSON' below to start processing")
        if st.button(
            "🚀 Extract to CSV/JSON",
            type="primary",
            help="Start the extraction process",
        ):
            temp_pdf_path = st.session_state.temp_pdf_path
            
            combined_df, schema_found = process_pdf_extraction(
                temp_pdf_path, uploaded_pdf.name
            )
            
            if combined_df is not None and not combined_df.empty:
                st.session_state.extraction_results = combined_df
                st.session_state.extraction_complete = True
                
                st.subheader("📊 Extraction Results")
                
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
                    withdrawal_count = len(
                        combined_df[combined_df["withdrawal_dr"] > 0]
                    )
                    deposit_count = len(combined_df[combined_df["deposit_cr"] > 0])
                    st.metric("W/D Ratio", f"{withdrawal_count}/{deposit_count}")
                
                st.subheader("📋 All Extracted Transactions")
                st.dataframe(combined_df, use_container_width=True)
                
                st.success(
                    f"✅ Successfully extracted {len(combined_df)} transactions!"
                )
                logging.info(
                    f"Extraction complete: {len(combined_df)} transactions ready for download"
                )
                
                st.info(f"""
                🎯 **COMPLETE FINANCIAL OVERVIEW**
                - **Total Credits (Money In):** ₹{total_deposits:,.2f} across {deposit_count} transactions
                - **Total Debits (Money Out):** ₹{total_withdrawals:,.2f} across {withdrawal_count} transactions
                """)
            
            else:
                st.error(
                    "❌ No valid transaction data could be extracted from the PDF."
                )
    
    if (
        "extraction_complete" in st.session_state
        and st.session_state.extraction_complete
    ):
        combined_df = st.session_state.extraction_results
        uploaded_filename = st.session_state.get("uploaded_filename", "bank_statement")
        
        st.subheader("💾 Download Options")
        col1, col2 = st.columns(2)
        
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
    
    if uploaded_pdf is None and (
        "extraction_complete" not in st.session_state
        or not st.session_state.extraction_complete
    ):
        st.info("👆 Please upload a PDF file to get started")


if __name__ == "__main__":
    main()