import os
import json
import logging
import warnings
import streamlit as st
from pathlib import Path
from PIL import Image
import fitz
import PyPDF2
import pandas as pd

from bank_statement_modules.ui import streamlit_css, render_metrics_block
from bank_statement_modules.table_utils import PDFProcessor, SimplifiedTableExtractor
from bank_statement_modules.ai_core import (
    is_transaction_table,
    detect_schema_from_first_table,
    extract_table_with_schema,
    refine_with_camelot_reference,
)
from bank_statement_modules.file_utils import (
    cleanup_temp_dir,
    combine_json_texts_to_dataframe,
    ensure_temp_dir,
)
from bank_statement_modules.config import DEFAULT_SCHEMA

warnings.filterwarnings("ignore", category=UserWarning, message=".*meta parameter.*")
logging.basicConfig(level=logging.INFO, format="%(message)s")

def handle_password_protected_pdf(uploaded_file, filename):
    """Save uploaded PDF and handle password protection."""
    temp_folder = ensure_temp_dir(filename)
    temp_pdf_path = os.path.join(temp_folder, filename)

    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:
        needs_password = False
        try:
            needs_password = not PDFProcessor.authenticate_pdf(temp_pdf_path)
        except Exception:
            needs_password = False

        try:
            doc = fitz.open(temp_pdf_path)
            if doc.needs_pass:
                needs_password = True
            doc.close()
        except Exception:
            needs_password = True

        if not needs_password:
            return temp_pdf_path

        st.warning("🔐 This PDF appears to be password protected.")
        widget_key = f"pdf_password_{abs(hash(filename)) % (10 ** 8)}"
        password = st.text_input("Enter PDF password:", type="password", key=widget_key)

        if not password:
            st.info("👆 Please enter the password to continue")
            return None

        try:
            doc = fitz.open(temp_pdf_path)
            auth_ok = False
            try:
                auth_ok = doc.authenticate(password)
            except Exception:
                try:
                    doc.close()
                    doc = fitz.open(temp_pdf_path, filetype="pdf", password=password)
                    auth_ok = True
                except Exception:
                    auth_ok = False

            if not auth_ok:
                doc.close()
                raise RuntimeError("fitz authentication failed")

            decrypted_path = os.path.join(temp_folder, f"{Path(filename).stem}_decrypted.pdf")
            try:
                doc.save(decrypted_path)
                doc.close()
                st.success("🔓 PDF unlocked successfully.")
                return decrypted_path
            except Exception:
                doc.close()
                raise RuntimeError("fitz save failed, falling back to PyPDF2")
        except Exception:
            try:
                with open(temp_pdf_path, "rb") as f:
                    reader = PyPDF2.PdfReader(f)
                    if reader.is_encrypted:
                        try:
                            decrypt_result = reader.decrypt(password)
                        except Exception:
                            decrypt_result = 0

                        if decrypt_result:
                            writer = PyPDF2.PdfWriter()
                            for p in reader.pages:
                                writer.add_page(p)
                            decrypted_path = os.path.join(temp_folder, f"{Path(filename).stem}_decrypted.pdf")
                            with open(decrypted_path, "wb") as out_f:
                                writer.write(out_f)
                            st.success("✅ PDF unlocked successfully (saved decrypted copy).")
                            return decrypted_path
                        else:
                            st.error("❌ Incorrect password. Please try again.")
                            return None
                    else:
                        return temp_pdf_path
            except Exception as e:
                st.error(f"Error trying to decrypt PDF: {e}")
                return None

    except Exception as e:
        st.error(f"Error processing PDF: {e}")
        return None


def process_pdf_extraction(temp_pdf_path, uploaded_filename, use_camelot_refiner=False):
    """Process PDF extraction with progress tracking."""
    logging.info(f"Starting extraction process for: {uploaded_filename}")

    try:
        progress = st.progress(0)
        status_text = st.empty()

        status_text.text("Extracting tables from PDF...")
        progress.progress(0.05)

        extractor = SimplifiedTableExtractor()
        temp_dir = ensure_temp_dir(uploaded_filename)

        cropped_image_paths = extractor.extract_all_tables(
            temp_pdf_path,
            output_dir=Path(temp_dir),
            padding=10,
            password=None,
        )

        if not cropped_image_paths:
            st.warning("No tables detected in the uploaded PDF.")
            return None, None

        extracted_json_texts = []
        reordered_schema = None
        first_transaction_table_found = False

        total_tables = len(cropped_image_paths)
        for idx, img_path in enumerate(cropped_image_paths, start=1):
            current_progress = 0.15 + 0.6 * (idx / max(1, total_tables))
            progress.progress(min(current_progress, 0.85))
            status_text.text(f"Processing Table {idx} of {total_tables}...")

            page_table_info = Path(img_path).name
            logging.info(f"🔃 Processing Table: {page_table_info.replace('_', ' ')}")

            img = Image.open(img_path)
            st.image(img, caption=f"Table {idx}", use_container_width=True)

            if not first_transaction_table_found:
                with st.spinner(f"Checking if Table {idx} contains transactions..."):
                    if is_transaction_table(img):
                        first_transaction_table_found = True
                        with st.spinner(f"Analyzing Table {idx} to detect column order..."):
                            reordered_schema = detect_schema_from_first_table(img)
                            with st.expander("Detected Schema (raw)"):
                                st.text_area("Schema", str(reordered_schema), height=120)
                        logging.info(f"Detected schema from Table {idx}: {reordered_schema}")
                    else:
                        st.info(f"⏭️ Table {idx} is not a transaction table - skipping schema detection")

            schema_used = reordered_schema or DEFAULT_SCHEMA
            with st.spinner(f"Extracting Table {idx} with schema..."):
                json_text = extract_table_with_schema(img, schema_used)

            with st.expander(f"Raw Extracted JSON — Table {idx}"):
                st.text_area("Output", json_text, height=150)

            extracted_json_texts.append(json_text)

        combined_df = None
        if extracted_json_texts:
            combined_df = combine_json_texts_to_dataframe(extracted_json_texts, cropped_image_paths, temp_pdf_path)

            if use_camelot_refiner and combined_df is not None and not combined_df.empty:
                progress.progress(0.90)
                status_text.text("🔍 Final refinement with Camelot (optional)...")
                try:
                    from bank_statement_modules.camelot_refiner import extract_bank_statement

                    camelot_df, summary = extract_bank_statement(temp_pdf_path)
                    if camelot_df is not None and not camelot_df.empty:
                        refined_transactions = refine_with_camelot_reference(
                            combined_df.to_dict(orient="records"),
                            camelot_df,
                            reordered_schema or DEFAULT_SCHEMA,
                        )
                        combined_df = combine_json_texts_to_dataframe(
                            [json.dumps(refined_transactions)], cropped_image_paths, temp_pdf_path
                        )
                        st.success("✨ Final Camelot refinement applied to all transactions")
                    else:
                        logging.info("No valid Camelot transactions found, skipping refinement")
                except Exception as e:
                    logging.warning(f"Camelot refinement skipped: {e}")

        progress.progress(1.0)
        status_text.text("✅ Extraction completed!")
        return combined_df, first_transaction_table_found

    except Exception as e:
        logging.error(f"Error in process_pdf_extraction: {e}")
        cleanup_temp_dir(uploaded_filename)
        raise


def main():
    st.set_page_config(page_title="Bank Statement Extractor", page_icon="🏦", layout="wide")
    st.markdown(streamlit_css, unsafe_allow_html=True)
    
    st.title("🏦 Bank Statement Extractor")
    st.info("🔒 Privacy Protected: Confidential information and personal account details are automatically masked for security.")

    # Sidebar
    st.sidebar.header("⚙️ Options")
    debug_mode = st.sidebar.checkbox("Enable Debug Mode", value=False)
    keep_temp = st.sidebar.checkbox("Keep Temp Files", value=False)
    use_camelot_refiner = st.sidebar.checkbox("Use Advanced Camelot Refinement", value=True)

    left_col, right_col = st.columns([1, 2], gap="large")

    # LEFT COLUMN
    with left_col:
        st.subheader("Upload & Controls")
        uploaded = st.file_uploader("Upload Bank Statement (PDF)", type="pdf", key="uploader")
        
        if 'last_uploaded_filename' in st.session_state:
            if uploaded is None and st.session_state.last_uploaded_filename is not None:
                cleanup_temp_dir(st.session_state.last_uploaded_filename)
                st.session_state.last_uploaded_filename = None
                if 'extraction_results' in st.session_state:
                    del st.session_state.extraction_results
                st.info("🗑️ File removed. Temporary files cleaned up.")
        
        st.session_state.last_uploaded_filename = uploaded.name if uploaded else None
        
        if uploaded:
            temp_pdf_path = handle_password_protected_pdf(uploaded, uploaded.name)
            
            if temp_pdf_path and st.button("🚀 Start Extraction", type="primary"):
                with st.spinner("Processing PDF... please wait ⏳"):
                    try:
                        combined_df, schema_found = process_pdf_extraction(
                            temp_pdf_path, uploaded.name, use_camelot_refiner=use_camelot_refiner
                        )

                        if combined_df is not None and not combined_df.empty:
                            st.session_state.extraction_results = combined_df
                            st.success(f"✅ Extraction completed! Found {len(combined_df)} transactions.")
                        else:
                            st.error("❌ No transactions extracted.")
                            
                        if not keep_temp:
                            cleanup_temp_dir(uploaded.name)
                            
                    except Exception as e:
                        st.error(f"⚠️ Extraction failed: {e}")

    # RIGHT COLUMN
    with right_col:
        st.subheader("Results")
        
        if hasattr(st.session_state, 'extraction_results') and st.session_state.extraction_results is not None:
            df = st.session_state.extraction_results
            
            render_metrics_block(df)
            
            st.markdown("### Extracted Transactions")
            st.dataframe(df, use_container_width=True)

            col1, col2 = st.columns(2)
            
            csv_data = df.to_csv(index=False).encode()
            col1.download_button(
                "⬇️ Download CSV",
                csv_data,
                file_name="transactions.csv",
                mime="text/csv"
            )
            
            json_data = df.to_json(orient="records", indent=2).encode()
            col2.download_button(
                "⬇️ Download JSON", 
                json_data,
                file_name="transactions.json",
                mime="application/json"
            )
        else:
            st.markdown(
                "<div style='color:#6b7280; text-align:center; padding:2rem;'>"
                "No results yet. Upload a PDF and click <b>Start Extraction</b>."
                "</div>", 
                unsafe_allow_html=True
            )

    if debug_mode:
        with st.sidebar.expander("Debug Info", expanded=True):
            debug_info = {}
            for k, v in st.session_state.items():
                if isinstance(v, pd.DataFrame):
                    debug_info[k] = f"DataFrame({v.shape})"
                else:
                    debug_info[k] = str(v)[:100]
            st.json(debug_info)

if __name__ == "__main__":
    main()