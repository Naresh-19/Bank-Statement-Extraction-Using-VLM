import os
import time
import gc
import shutil
import logging
import json
import re
from pathlib import Path
import pandas as pd
import streamlit as st
from camelot_extractor import extract_bank_statement
from ai_functions import refine_with_camelot_reference_simple, clean_and_fix_json


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