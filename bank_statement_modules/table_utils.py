# bank_statement_modules/table_utils.py

import re
import logging
from pathlib import Path
from collections import defaultdict
from PIL import Image
import fitz
import camelot
import warnings
import pandas as pd

from bank_statement_modules.config import (
    DATE_REGEX,
    AMOUNT_REGEX,
    SUMMARY_BLACKLIST,
    COLUMN_KEYWORDS,
    OFFSET,
)

warnings.filterwarnings("ignore", category=UserWarning, message=".*meta parameter.*")


# --------------------------------------------------------
# PDF Preprocessing
# --------------------------------------------------------
class PDFProcessor:
    """Handles PDF validation, authentication, and header redaction."""

    @staticmethod
    def authenticate_pdf(pdf_path, password=None) -> bool:
        """Check if PDF is password-protected and attempt authentication."""
        doc = fitz.open(pdf_path)
        try:
            if doc.needs_pass:
                if password and doc.authenticate(password):
                    return True
                return False
            return True
        finally:
            doc.close()

    @staticmethod
    def merge_blocks_by_line(blocks, tolerance=6):
        """Group blocks of text into merged lines based on y-position."""
        lines = defaultdict(list)
        for b in blocks:
            y0 = round(b[1] / tolerance) * tolerance
            lines[y0].append(b)

        merged = []
        for y, blks in sorted(lines.items()):
            blks_sorted = sorted(blks, key=lambda x: x[0])
            text = " ".join(b[4].strip() for b in blks_sorted if b[4].strip())
            merged.append((y, text))
        return merged

    @staticmethod
    def is_transaction_line(idx, merged_lines):
        """Check if a line looks like a transaction (date + amount)."""
        text = merged_lines[idx][1]
        if DATE_REGEX.search(text) and AMOUNT_REGEX.search(text):
            return True
        if AMOUNT_REGEX.search(text):
            for back in range(1, 3):
                if idx - back >= 0 and DATE_REGEX.search(merged_lines[idx - back][1]):
                    return True
        return False

    @staticmethod
    def is_header_line(text: str) -> bool:
        """Check if a line is likely a table header."""
        text_low = text.lower()
        if any(b in text_low for b in SUMMARY_BLACKLIST):
            return False
        if "date" not in text_low:
            return False
        matches = sum(1 for k in COLUMN_KEYWORDS if k in text_low)
        return matches >= 1

    @staticmethod
    def detect_header_y(merged_lines):
        """Detect header position by scanning lines with 'date' + keywords."""
        for i, (y, text) in enumerate(merged_lines):
            if PDFProcessor.is_header_line(text):
                tx_count = sum(
                    PDFProcessor.is_transaction_line(j, merged_lines)
                    for j in range(i + 1, min(i + 10, len(merged_lines)))
                )
                if tx_count >= 2:
                    return y
            # Handle 3-line headers like "Date / Transaction / Amount"
            header_chunk = merged_lines[i : i + 3]
            combined = " ".join(txt.lower() for _, txt in header_chunk)
            if "date" in combined and "transaction" in combined and "amount" in combined:
                tx_count = sum(
                    PDFProcessor.is_transaction_line(j, merged_lines)
                    for j in range(i + 3, min(i + 13, len(merged_lines)))
                )
                if tx_count >= 2:
                    return max(y for y, _ in header_chunk)
        return None

    @staticmethod
    def redact_pdf(pdf_path, output_path, password=None):
        """Redact header section of PDF (above detected table)."""
        doc = fitz.open(pdf_path)
        if doc.needs_pass and password:
            doc.authenticate(password)

        modified = False
        for page in doc:
            blocks = page.get_text("blocks")
            merged_lines = PDFProcessor.merge_blocks_by_line(blocks)
            header_y = PDFProcessor.detect_header_y(merged_lines)
            if header_y:
                rect = fitz.Rect(0, 0, page.rect.width, max(0, header_y - OFFSET))
                page.draw_rect(rect, color=(1, 1, 1), fill=(1, 1, 1))
                modified = True

        if modified:
            doc.save(output_path)
        doc.close()
        return modified


# --------------------------------------------------------
# Simplified Table Extraction (Camelot + heuristics)
# --------------------------------------------------------
class SimplifiedTableExtractor:
    """Camelot wrapper with heuristics for identifying transaction tables."""

    def __init__(self):
        self.date_patterns = [
            r"\d{1,2}[-/]\d{1,2}[-/]\d{2,4}",
            r"\d{1,2}\s+\w{3}\s*,?\s+\d{4}",
            r"\d{1,2}-\w{3}-\d{2,4}",
            r"\d{4}[-/]\d{1,2}[-/]\d{1,2}",
        ]
        self.header_patterns = [
            r"date|dt|txn.*date|transaction.*date",
            r"balance|bal|amount|amt",
            r"debit|credit|withdrawal|deposit",
            r"description|particulars|narration|details|reference",
        ]
        self.transaction_keywords = [
            "debit", "credit", "balance", "withdrawal", "deposit",
            "transfer", "payment", "upi", "imps", "neft", "rtgs",
        ]

    def is_date_like(self, value):
        if not value or str(value).strip().lower() in ["", "nan"]:
            return False
        return any(re.search(pattern, str(value)) for pattern in self.date_patterns)

    def is_header_row(self, row) -> bool:
        row_text = " ".join([str(cell).lower().strip() for cell in row if str(cell).strip() != "nan"]).strip()
        if not row_text:
            return False
        matches = sum(1 for pattern in self.header_patterns if re.search(pattern, row_text))
        return matches >= 2

    def is_transaction_table(self, df: pd.DataFrame) -> bool:
        """Check if dataframe contains transactions."""
        if df.empty or len(df) < 2:
            return False

        has_headers = any(self.is_header_row(df.iloc[i]) for i in range(min(3, len(df))))

        # Look for date column
        date_found = False
        for col in range(min(4, len(df.columns))):
            sample_vals = [str(df.iloc[i, col]).strip() for i in range(min(10, len(df))) if col < len(df.iloc[i])]
            if len(sample_vals) >= 2:
                date_matches = sum(1 for val in sample_vals if self.is_date_like(val))
                if date_matches >= 1:
                    date_found = True
                    break

        all_text = df.to_string().lower()
        keyword_matches = sum(1 for keyword in self.transaction_keywords if keyword in all_text)

        return (has_headers and keyword_matches >= 1) or (date_found and keyword_matches >= 1)

    def merge_boxes(self, bbox1, bbox2):
        """Merge two bounding boxes."""
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return (x1, y1, x2, y2)

    def boxes_overlap(self, bbox1, bbox2, threshold=0.3):
        """Check overlap ratio of two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2

        inter_x1 = max(x1_1, x1_2)
        inter_y1 = max(y1_1, y1_2)
        inter_x2 = min(x2_1, x2_2)
        inter_y2 = min(y2_1, y2_2)

        if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
            return False

        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)

        overlap_ratio = inter_area / min(area1, area2)
        return overlap_ratio > threshold

    def merge_overlapping_tables(self, tables):
        """Merge overlapping Camelot tables page-wise."""
        page_tables = {}
        for table in tables:
            if self.is_transaction_table(table.df):
                page_num = table.page
                page_tables.setdefault(page_num, []).append(table)

        merged_tables = []
        for page_num, page_table_list in page_tables.items():
            used = [False] * len(page_table_list)
            for i in range(len(page_table_list)):
                if used[i]:
                    continue
                current_table = page_table_list[i]
                current_bbox = current_table._bbox
                overlapping = [i]
                for j in range(i + 1, len(page_table_list)):
                    if not used[j] and self.boxes_overlap(current_bbox, page_table_list[j]._bbox):
                        overlapping.append(j)
                        current_bbox = self.merge_boxes(current_bbox, page_table_list[j]._bbox)
                for idx in overlapping:
                    used[idx] = True
                merged_table = page_table_list[i]
                merged_table._bbox = current_bbox
                merged_tables.append(merged_table)
        return merged_tables

    def extract_all_tables(self, pdf_path, output_dir, padding=20, password=None, redact=True):
        """
        Extract all transaction tables as cropped images.
        If redact=True, the area above the detected header will be masked.
        """
        redacted_path = Path(output_dir) / f"{Path(pdf_path).stem}_redacted.pdf"
        pdf_to_process = pdf_path
        if redact:
            try:
                modified = PDFProcessor.redact_pdf(pdf_path, str(redacted_path), password=password)
                if modified:
                    pdf_to_process = str(redacted_path)
                    logging.info(f"Redacted sensitive areas → using {pdf_to_process}")
            except Exception as e:
                logging.warning(f"Redaction failed, using original PDF: {e}")

        pdf_doc = fitz.open(pdf_to_process)
        cropped_paths = []

        try:
            logging.info("Processing entire PDF with Camelot...")

            try:
                tables = camelot.read_pdf(
                    pdf_to_process, pages="all", flavor="stream",
                    edge_tol=75, row_tol=10, password=password
                )
                if not tables:
                    tables = camelot.read_pdf(pdf_to_process, pages="all",
                                              flavor="stream", password=password)
            except Exception as e:
                logging.warning(f"Stream flavor failed ({e}), trying lattice flavor...")
                tables = camelot.read_pdf(pdf_to_process, pages="all",
                                          flavor="lattice", password=password)

            merged_tables = self.merge_overlapping_tables(tables)

            for idx, table in enumerate(merged_tables, start=1):
                page_num = table.page - 1
                bbox = table._bbox
                page = pdf_doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)

                scale_x = pix.width / page.rect.width
                scale_y = pix.height / page.rect.height

                img_x1 = max(0, int(bbox[0] * scale_x) - padding)
                img_y1 = max(0, int((page.rect.height - bbox[3]) * scale_y) - padding)
                img_x2 = min(pix.width, int(bbox[2] * scale_x) + padding)
                img_y2 = min(pix.height, int((page.rect.height - bbox[1]) * scale_y) + padding)

                cropped_img = page_img.crop((img_x1, img_y1, img_x2, img_y2))
                output_path = Path(output_dir) / f"{Path(pdf_path).stem}_table{idx}.png"
                cropped_img.save(output_path)
                cropped_paths.append(str(output_path))
                logging.info(f"Extracted table → {output_path}")

        finally:
            pdf_doc.close()
            # Cleanup redacted copy if created
            if redact and Path(redacted_path).exists():
                Path(redacted_path).unlink()
                logging.info(f"🧹 Cleaned up redacted temp file: {redacted_path}")

        return cropped_paths

# --------------------------------------------------------
# Camelot Schema-based Extraction
# --------------------------------------------------------

def extract_table_with_camelot_schema(pdf_path: str, page_num: int, bbox: tuple | None, schema_columns: list):
    """
    Extract table from a specific PDF region using Camelot and enforce schema columns.

    Args:
        pdf_path (str): Path to PDF
        page_num (int): Page number (1-indexed for Camelot)
        bbox (tuple | None): Bounding box (x1, y1, x2, y2) in PDF coordinates. 
                             If None → whole page is used.
        schema_columns (list): Expected column names in schema

    Returns:
        pd.DataFrame: Cleaned table dataframe with schema columns
    """
    import camelot
    import pandas as pd

    try:
        table_kwargs = {
            "flavor": "stream",
            "pages": str(page_num),
            "strip_text": "\n",
        }

        if bbox is not None:
            x1, y1, x2, y2 = bbox
            table_kwargs["table_areas"] = [f"{x1},{y2},{x2},{y1}"]

        tables = camelot.read_pdf(pdf_path, **table_kwargs)

        if not tables:
            logging.warning(f"No tables detected in page {page_num} (bbox={bbox})")
            return pd.DataFrame(columns=schema_columns)

        df = tables[0].df

        # Normalize columns
        df.columns = [str(i) for i in range(len(df.columns))]
        df = df.rename(columns=dict(zip(df.columns, schema_columns[: len(df.columns)])))

        return df

    except Exception as e:
        logging.error(f"Error in extract_table_with_camelot_schema: {e}")
        return pd.DataFrame(columns=schema_columns)

