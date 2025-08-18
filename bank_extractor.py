import camelot
import pandas as pd
import re
from datetime import datetime
import warnings
import PyPDF2
import os
warnings.filterwarnings('ignore')

class BankStatementExtractor:
    def __init__(self):
        self.all_transactions = []
        self.extracted_headers = None
        self.first_table_processed = False
        self.header_row_index = None
        
    def validate_pdf(self, pdf_path):
        """Validate PDF and get basic information"""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                print(f"PDF validation successful - {num_pages} pages detected")
                return num_pages
        except Exception as e:
            print(f"PDF validation failed: {e}")
            raise Exception("Cannot read PDF file")
    
    def is_date_like(self, value):
        """Check if value looks like a date"""
        if not value or str(value).strip() == '' or str(value) == 'nan':
            return False
            
        date_patterns = [
            r'\d{1,2}[-/]\d{1,2}[-/]\d{2,4}',
            r'\d{1,2}\s+\w{3}\s*,?\s+\d{4}',  # Fixed: Added optional comma and flexible spacing
            r'\d{1,2}-\w{3}-\d{2,4}',
            r'\d{4}[-/]\d{1,2}[-/]\d{1,2}',
            r'\d{1,2}\s+\w{3}\s*,?\s+\d{2}',  # Fixed: Added optional comma and flexible spacing
        ]
        return any(re.search(pattern, str(value)) for pattern in date_patterns)
    
    def standardize_date(self, date_str):
        """Convert various date formats to standard format"""
        if not date_str:
            return None
            
        date_str = str(date_str).strip()
        
        patterns = [
            ('%d-%m-%Y', r'\d{1,2}-\d{1,2}-\d{4}'),
            ('%d/%m/%Y', r'\d{1,2}/\d{1,2}/\d{4}'),
            ('%d %b, %Y', r'\d{1,2}\s+\w{3},\s+\d{4}'),  # Added: For "01 Jun, 2025"
            ('%d %b %Y', r'\d{1,2}\s+\w{3}\s+\d{4}'),
            ('%d-%b-%Y', r'\d{1,2}-\w{3}-\d{4}'),
            ('%d-%b-%y', r'\d{1,2}-\w{3}-\d{2}'),
            ('%d %b, %y', r'\d{1,2}\s+\w{3},\s+\d{2}'),  # Added: For "01 Jun, 25"
            ('%d %b %y', r'\d{1,2}\s+\w{3}\s+\d{2}'),
        ]
        
        for date_format, pattern in patterns:
            if re.search(pattern, date_str):
                try:
                    match = re.search(pattern, date_str)
                    if match:
                        date_part = match.group()
                        parsed_date = datetime.strptime(date_part, date_format)
                        return parsed_date.strftime('%d %b %Y')
                except:
                    continue
        
        return date_str
    
    def is_header_row(self, row):
        """Check if a row contains typical banking column headers"""
        header_patterns = [
            r'date|dt|txn.*date|transaction.*date',
            r'balance|bal|amount|amt',
            r'debit|credit|withdrawal|deposit',
            r'description|particulars|narration|details|reference|remark',
            r'cheque|chq|ref.*no|reference.*no'
        ]
        
        row_text = ' '.join([str(cell).lower().strip() for cell in row if str(cell).strip() != 'nan']).strip()
        
        if not row_text:
            return False
        
        matches = sum(1 for pattern in header_patterns 
                      if re.search(pattern, row_text))
        
        return matches >= 2
    
    def extract_headers_from_table(self, df):
        """Extract headers from first few rows of the table"""
        for i in range(min(5, len(df))):
            row = df.iloc[i]
            if self.is_header_row(row):
                headers = []
                for cell in row:
                    header = str(cell).strip().replace('\n', ' ').replace('\r', ' ')
                    header = re.sub(r'\s+', ' ', header)
                    if header and header != 'nan':
                        headers.append(header)
                    else:
                        headers.append(f'Column_{len(headers)}')
                
                return headers, i
        
        return None, None
    
    def is_transaction_table(self, df):
        """Check if dataframe contains transaction data"""
        if df.empty or len(df) < 2:
            return False
        
        date_found = False
        for col in range(min(4, len(df.columns))):
            sample_values = []
            for i in range(min(20, len(df))):
                if col < len(df.iloc[i]):
                    val = str(df.iloc[i, col]).strip()
                    if val and val != 'nan' and val != '':
                        sample_values.append(val)
            
            if len(sample_values) >= 3:
                date_matches = sum(1 for val in sample_values if self.is_date_like(val))
                if date_matches >= 2 or (date_matches >= 1 and date_matches / len(sample_values) >= 0.15):
                    date_found = True
                    break
        
        return date_found    
    
    def find_date_column(self, df, start_row=0):
        """Find which column contains dates, starting from a specific row"""
        for col in range(len(df.columns)):
            sample_values = []
            for i in range(start_row, min(start_row + 10, len(df))):
                if col < len(df.iloc[i]):
                    val = str(df.iloc[i, col]).strip()
                    if val and val != 'nan':
                        sample_values.append(val)
            
            if sample_values:
                date_matches = sum(1 for val in sample_values if self.is_date_like(val))
                if len(sample_values) > 0 and date_matches / len(sample_values) > 0.5:
                    return col
        return None
    
    def merge_multiline_transactions(self, df, date_col, start_row=0):
        """Merge multi-line transactions into single rows, starting from a specific row"""
        if df.empty or date_col is None:
            return df
        
        merged_rows = []
        current_transaction = None
        
        for i in range(start_row, len(df)):
            row = df.iloc[i]
            
            date_value = str(row.iloc[date_col]).strip()
            has_date = self.is_date_like(date_value)
            
            if has_date:
                if current_transaction is not None:
                    merged_rows.append(current_transaction)
                
                current_transaction = row.copy()
            else:
                if current_transaction is not None:
                    for col_idx in range(len(row)):
                        continuation_value = str(row.iloc[col_idx]).strip()
                        if continuation_value and continuation_value != 'nan' and continuation_value != '':
                            current_value = str(current_transaction.iloc[col_idx]).strip()
                            if not current_value or current_value == 'nan':
                                current_transaction.iloc[col_idx] = continuation_value
                            else:
                                if col_idx != date_col:
                                    current_transaction.iloc[col_idx] = current_value + ' ' + continuation_value
        
        if current_transaction is not None:
            merged_rows.append(current_transaction)
        
        if merged_rows:
            return pd.DataFrame(merged_rows).reset_index(drop=True)
        else:
            return pd.DataFrame()
    
    def clean_dataframe(self, df):
        """Clean the dataframe by removing empty rows and normalizing headers"""
        if df.empty:
            return df
        
        df = df.dropna(how='all').reset_index(drop=True)
        return df
    
    def process_table(self, table, table_num, progress_callback=None):
        """Process a single table and extract transactions"""
        df = table.df.copy()
        
        if progress_callback:
            progress_callback(f"Processing table {table_num}: Original shape {df.shape}")
        
        df = self.clean_dataframe(df)
        
        if df.empty:
            if progress_callback:
                progress_callback(f"Table {table_num}: Empty after cleaning")
            return
        
        if not self.is_transaction_table(df):
            if progress_callback:
                progress_callback(f"Table {table_num}: Doesn't look like transaction table")
            return
        
        header_start_row = 0
        if not self.first_table_processed:
            headers, header_row_idx = self.extract_headers_from_table(df)
            if headers:
                self.extracted_headers = headers
                self.header_row_index = header_row_idx
                header_start_row = header_row_idx + 1
                if progress_callback:
                    progress_callback(f"Table {table_num}: Headers extracted from row {header_row_idx}: {headers}")
            else:
                self.extracted_headers = [f'Column_{i}' for i in range(len(df.columns))]
                if progress_callback:
                    progress_callback(f"Table {table_num}: No headers found, using generic headers")
            
            self.first_table_processed = True
        else:
            if progress_callback:
                progress_callback(f"Table {table_num}: Using headers from first table, skipping potential header row")
            if len(df) > 0 and self.is_header_row(df.iloc[0]):
                header_start_row = 1
        
        date_col = self.find_date_column(df, header_start_row)
        if date_col is None:
            if progress_callback:
                progress_callback(f"Table {table_num}: No date column found")
            return
        
        if progress_callback:
            progress_callback(f"Table {table_num}: Date column found at index {date_col}")
        
        df_transactions = self.merge_multiline_transactions(df, date_col, header_start_row)
        if df_transactions.empty:
            if progress_callback:
                progress_callback(f"Table {table_num}: No transactions after merging")
            return
            
        if progress_callback:
            progress_callback(f"Table {table_num}: After merging multi-line: {df_transactions.shape}")
        
        for i in range(len(df_transactions)):
            row = df_transactions.iloc[i]
            
            date_value = str(row.iloc[date_col]).strip()
            if not self.is_date_like(date_value):
                continue
            
            transaction = {}
            
            for col_idx in range(len(df_transactions.columns)):
                cell_value = str(row.iloc[col_idx]).strip()
                
                if col_idx < len(self.extracted_headers):
                    header_name = self.extracted_headers[col_idx]
                else:
                    header_name = f'Column_{col_idx}'
                
                if cell_value and cell_value != 'nan':
                    transaction[header_name] = cell_value
                else:
                    transaction[header_name] = ''
            
            transaction['standardized_date'] = self.standardize_date(date_value)
            # transaction['table_source'] = f"Table_{table_num}"
            
            self.all_transactions.append(transaction)
    
    def extract_transactions(self, pdf_path, progress_callback=None):
        """Main extraction method"""
        try:
            num_pages = self.validate_pdf(pdf_path)
            
            if progress_callback:
                progress_callback("Attempting Camelot extraction...")
            
            tables = camelot.read_pdf(pdf_path, pages='all', flavor='lattice')
            
            if not tables:
                if progress_callback:
                    progress_callback("Camelot lattice failed, trying stream flavor...")
                tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream', edge_tol=75, row_tol=10)
            
            if tables:
                if progress_callback:
                    progress_callback(f"Found {len(tables)} tables")
                
                for i, table in enumerate(tables):
                    self.process_table(table, i + 1, progress_callback)
                    
                if progress_callback and self.extracted_headers:
                    progress_callback(f"Using headers: {self.extracted_headers}")
            else:
                if progress_callback:
                    progress_callback("No tables found")
                
        except Exception as e:
            error_msg = f"Error in extraction: {e}"
            if progress_callback:
                progress_callback(error_msg)
            raise Exception(error_msg)
    
    def get_dataframe(self):
        """Return all transactions as DataFrame with extracted headers"""
        if not self.all_transactions:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.all_transactions)
        return df
    
    def save_to_csv(self, output_path):
        """Save transactions to CSV file"""
        df = self.get_dataframe()
        if not df.empty:
            df.to_csv(output_path, index=False)
            return len(df)
        return 0
    
    def get_summary(self):
        """Get extraction summary as dictionary"""
        summary = {
            'total_transactions': len(self.all_transactions),
            'columns': [],
            'date_range': None,
            'extracted_headers': self.extracted_headers,
            'header_detection_success': self.extracted_headers is not None
        }
        
        if self.all_transactions:
            df = self.get_dataframe()
            summary['columns'] = list(df.columns)
            
            if 'standardized_date' in df.columns:
                dates = df['standardized_date'].dropna()
                if not dates.empty:
                    summary['date_range'] = f"{dates.min()} to {dates.max()}"
        
        return summary


def extract_bank_statement(pdf_path, output_csv=None, progress_callback=None):
    """
    Enhanced bank statement extraction with automatic header detection
    
    Args:
        pdf_path (str): Path to PDF bank statement
        output_csv (str, optional): Path to save CSV output
        progress_callback (function, optional): Callback function for progress updates
    
    Returns:
        pandas.DataFrame: Extracted transactions with detected headers
        dict: Summary with header detection info
    """
    extractor = BankStatementExtractor()
    
    if progress_callback:
        progress_callback(f"Extracting from: {pdf_path}")
    
    extractor.extract_transactions(pdf_path, progress_callback)
    
    df = extractor.get_dataframe()
    
    if output_csv:
        num_saved = extractor.save_to_csv(output_csv)
        if progress_callback:
            progress_callback(f"Saved {num_saved} transactions to {output_csv}")
    
    return df, extractor.get_summary()