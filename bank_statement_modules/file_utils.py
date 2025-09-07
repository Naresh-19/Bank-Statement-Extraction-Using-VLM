# bank_statement_modules/file_utils.py
import re
import os
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from bank_statement_modules.ai_core import clean_and_fix_json

TEMP_ROOT = "temp"

# --------------------------------------------------------
# File Handling Utilities
# --------------------------------------------------------

def cleanup_temp_dir(pdf_name: str):
    """
    Delete the entire temp folder for a given PDF.
    """
    try:
        base_stem = os.path.splitext(os.path.basename(pdf_name))[0]
        folder_path = os.path.join(TEMP_ROOT, base_stem)
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            logging.info(f"🧹 Deleted temp folder: {folder_path}")
    except Exception as e:
        logging.warning(f"Cleanup failed: {e}")
        
def ensure_temp_dir(pdf_name: str) -> str:
    """
    Create a dedicated temp folder for each PDF.
    Example: temp/<pdf_stem>/
    Returns the folder path.
    """
    base_stem = os.path.splitext(os.path.basename(pdf_name))[0]
    folder_path = os.path.join(TEMP_ROOT, base_stem)
    os.makedirs(folder_path, exist_ok=True)
    return folder_path


# --------------------------------------------------------
# DataFrame Helpers
# --------------------------------------------------------
def combine_json_texts_to_dataframe(json_texts, cropped_image_paths, source_pdf):
    all_records = []
    for idx, json_text in enumerate(json_texts):
        try:
            if not json_text or json_text.strip().startswith("Error extracting table:"):
                logging.warning(f"Skipping table {idx+1} due to error or empty response")
                continue

            clean_json = clean_and_fix_json(json_text)

            try:
                transactions = json.loads(clean_json)
            except json.JSONDecodeError as e:
                logging.warning(f"Table {idx+1}: JSON parse failed, attempting recovery: {e}")
                # Attempt to extract individual objects as fallback
                pattern = r'\{[^{}]*"dt"[^{}]*?\}'
                matches = re.finditer(pattern, clean_json, re.DOTALL)
                transactions = []
                for match in matches:
                    try:
                        obj_text = match.group(0)
                        obj_text = re.sub(r",\s*}", "}", obj_text)
                        obj_text = obj_text.replace('\\\\', '\\')
                        transaction = json.loads(obj_text)
                        transactions.append(transaction)
                    except Exception as inner_e:
                        logging.warning(f"Failed to parse individual transaction: {inner_e}")
                        continue

                if not transactions:
                    logging.error(f"Table {idx+1}: Could not parse JSON. Raw start: {clean_json[:200]}")
                    continue

            if not isinstance(transactions, list):
                logging.warning(f"Table {idx+1}: Expected array, got {type(transactions)}")
                continue

            for r in transactions:
                r["_source_table"] = Path(cropped_image_paths[idx]).name if idx < len(cropped_image_paths) else ""
                r["_source_pdf"] = Path(source_pdf).name if source_pdf else ""
            all_records.extend(transactions)

        except Exception as e:
            logging.warning(f"Failed to process table {idx+1}: {e}")
            continue

    if not all_records:
        logging.info("No valid JSON records extracted.")
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    expected_cols = ["dt", "desc", "ref", "dr", "cr", "bal", "type"]
    for col in expected_cols:
        if col not in df.columns:
            df[col] = None

    # Rename to standard schema
    df = df.rename(
        columns={
            "dt": "date",
            "desc": "description",
            "ref": "reference",
            "dr": "withdrawal_dr",
            "cr": "deposit_cr",
            "bal": "balance",
            "type": "txn_type",
        }
    )

    # Normalize transaction types
    if "txn_type" in df.columns:
        df["txn_type"] = df["txn_type"].replace({
            "d": "Debit",
            "D": "Debit",
            "w": "Withdrawal",
            "W": "Withdrawal",
            "c": "Credit",
            "C": "Credit",
            "dp": "Deposit",
            "DP": "Deposit"
        })

    # Keep metadata, just comment out removal for now
    if "_source_table" in df.columns:
        df.drop(columns=["_source_table"], inplace=True)
    if "_source_pdf" in df.columns:
        df.drop(columns=["_source_pdf"], inplace=True)

    return df
