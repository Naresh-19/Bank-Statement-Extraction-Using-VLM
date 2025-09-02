import os
import re
from pathlib import Path
from PIL import Image
import fitz  # PyMuPDF
import camelot
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*meta parameter.*")

class SimplifiedTableExtractor:
    def __init__(self):
        self.date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{1,2}\s+\w{3}\s*,?\s+\d{4}',
            r'\d{1,2}-\w{3}-\d{2,4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
        ]
        
        self.header_patterns = [
            r'date|dt|txn.*date|transaction.*date',
            r'balance|bal|amount|amt',
            r'debit|credit|withdrawal|deposit',
            r'description|particulars|narration|details|reference',
        ]
        
        self.transaction_keywords = [
            'debit', 'credit', 'balance', 'withdrawal', 'deposit', 
            'transfer', 'payment', 'upi', 'imps', 'neft', 'rtgs'
        ]
    
    def is_date_like(self, value):
        """Check if value looks like a date"""
        if not value or str(value).strip() in ['', 'nan']:
            return False
        return any(re.search(pattern, str(value)) for pattern in self.date_patterns)
    
    def is_header_row(self, row):
        """Check if a row contains banking column headers"""
        row_text = ' '.join([str(cell).lower().strip() for cell in row 
                            if str(cell).strip() != 'nan']).strip()
        if not row_text:
            return False
        
        matches = sum(1 for pattern in self.header_patterns 
                     if re.search(pattern, row_text))
        return matches >= 2
    
    def is_transaction_table(self, df):
        """Check if dataframe contains transaction data"""
        if df.empty or len(df) < 2:
            return False
        
        # Check for headers
        has_headers = any(self.is_header_row(df.iloc[i]) for i in range(min(3, len(df))))
        
        # Check for dates in first few columns
        date_found = False
        for col in range(min(4, len(df.columns))):
            sample_vals = [str(df.iloc[i, col]).strip() for i in range(min(10, len(df))) 
                          if col < len(df.iloc[i]) and str(df.iloc[i, col]).strip() not in ['', 'nan']]
            
            if len(sample_vals) >= 2:
                date_matches = sum(1 for val in sample_vals if self.is_date_like(val))
                if date_matches >= 2:
                    date_found = True
                    break
        
        # Check for transaction keywords
        all_text = df.to_string().lower()
        keyword_matches = sum(1 for keyword in self.transaction_keywords if keyword in all_text)
        
        return (has_headers and keyword_matches >= 1) or (date_found and keyword_matches >= 1)
    
    def boxes_overlap(self, bbox1, bbox2, threshold=0.3):
        """Check if two bounding boxes overlap significantly"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
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
    
    def merge_boxes(self, bbox1, bbox2):
        """Merge two overlapping bounding boxes"""
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return (x1, y1, x2, y2)
    
    def merge_overlapping_tables(self, tables):
        """Merge tables with overlapping bounding boxes on the same page"""
        page_tables = {}
        
        # Group tables by page
        for table in tables:
            if self.is_transaction_table(table.df):
                page_num = table.page
                if page_num not in page_tables:
                    page_tables[page_num] = []
                page_tables[page_num].append(table)
        
        merged_tables = []
        
        for page_num, page_table_list in page_tables.items():
            if len(page_table_list) == 1:
                merged_tables.extend(page_table_list)
                continue
            
            # Find overlapping tables and merge them
            used = [False] * len(page_table_list)
            
            for i in range(len(page_table_list)):
                if used[i]:
                    continue
                
                current_table = page_table_list[i]
                current_bbox = current_table._bbox
                
                # Find all overlapping tables
                overlapping = [i]
                for j in range(i + 1, len(page_table_list)):
                    if used[j]:
                        continue
                    
                    if self.boxes_overlap(current_bbox, page_table_list[j]._bbox):
                        overlapping.append(j)
                        current_bbox = self.merge_boxes(current_bbox, page_table_list[j]._bbox)
                
                # Mark as used
                for idx in overlapping:
                    used[idx] = True
                
                # Create merged table (use the first table's data but update bbox)
                merged_table = page_table_list[i]
                merged_table._bbox = current_bbox
                merged_tables.append(merged_table)
        
        return merged_tables
    
    def extract_all_tables(self, pdf_path, output_dir, padding=20):
        """Extract all transaction tables from entire PDF at once"""
        pdf_doc = fitz.open(pdf_path)
        cropped_paths = []
        
        try:
            # Process entire PDF with Camelot (all pages at once)
            print("Processing entire PDF with Camelot...")
            
            try:
                print("Trying stream flavor...")
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream',
                                          edge_tol=75, row_tol=10)
                if not tables:
                    print("Stream flavor failed, trying parameterized stream flavor...")
                    tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
            except Exception as e:
                print(f"Stream flavor failed with error: {e}")
                print("Trying lattice flavor as fallback...")
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            # Merge overlapping tables
            merged_tables = self.merge_overlapping_tables(tables)
            
            table_count = 0
            for table in merged_tables:
                page_num = table.page - 1  # Camelot uses 1-based page numbers
                bbox = table._bbox  # (x1, y1, x2, y2) in PDF coordinates
                
                # Get page and convert to image
                page = pdf_doc.load_page(page_num)
                pix = page.get_pixmap(dpi=300)
                page_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert PDF bbox to image coordinates
                scale_x = pix.width / page.rect.width
                scale_y = pix.height / page.rect.height
                
                img_x1 = max(0, int(bbox[0] * scale_x) - padding)
                img_y1 = max(0, int((page.rect.height - bbox[3]) * scale_y) - padding)
                img_x2 = min(page_img.width, int(bbox[2] * scale_x) + padding)
                img_y2 = min(page_img.height, int((page.rect.height - bbox[1]) * scale_y) + padding)
                
                # Crop and save
                cropped_table = page_img.crop((img_x1, img_y1, img_x2, img_y2))
                table_count += 1
                save_path = output_dir / f"page{table.page}_table{table_count}.png"
                cropped_table.save(save_path)
                cropped_paths.append(str(save_path))
                
                print(f"Extracted table: {save_path}")
            
        except Exception as e:
            print(f"Error processing PDF: {e}")
            raise
        finally:
            pdf_doc.close()
            
        return cropped_paths


def crop_tables_from_pdf(pdf_path, output_folder=None, **kwargs):
    """
    Simplified table extraction - processes entire PDF at once
    
    Args:
        pdf_path (str): Path to PDF file
        output_folder (str, optional): Output directory
        **kwargs: Ignored for compatibility
    
    Returns:
        list: Paths to cropped table images
    """
    pdf_name = Path(pdf_path).stem
    output_dir = Path(output_folder) if output_folder else Path(f"table/{pdf_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    extractor = SimplifiedTableExtractor()
    cropped_paths = extractor.extract_all_tables(pdf_path, output_dir)
    
    print(f"\nSummary: Extracted {len(cropped_paths)} transaction tables")
    print(f"Output directory: {output_dir}")
    
    return cropped_paths
    

# Backward compatibility
def crop_tables_from_pdf_legacy(pdf_path, output_folder=None, **kwargs):
    """Legacy function name - calls simplified version"""
    return crop_tables_from_pdf(pdf_path, output_folder, **kwargs)