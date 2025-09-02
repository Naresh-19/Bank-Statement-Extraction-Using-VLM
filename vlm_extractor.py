import os
import logging
from pathlib import Path
import streamlit as st
from PIL import Image
import warnings
from io import BytesIO, StringIO
from dotenv import load_dotenv
import PyPDF2

from camelot_cropper import crop_tables_from_pdf
from css import streamlit_css
from ai_functions import (
    is_transaction_table,
    detect_schema_from_first_table,
    extract_table_with_schema,
)
from utils import (
    cleanup_temp_files,
    combine_json_texts_to_dataframe,
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