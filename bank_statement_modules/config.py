import re
import os
from dotenv import load_dotenv

"""
Configuration file for Bank Statement Transaction Extraction
"""


# Load environment variables
load_dotenv(override=True)

# API Keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

OFFSET = 15

HEADER_KEYWORDS = [
    "date", "transaction", "txn", "description", "details", "narration", 
    "amount", "debit", "credit", "balance", "particulars", "withdrawal",
    "deposit", "transaction type", "type"
]

COLUMN_KEYWORDS = [
    "transaction", "txn", "description", "details", "narration", 
    "amount", "debit", "credit", "balance", "withdrawal", "deposit", "particulars"
]

SUMMARY_BLACKLIST = [
    "summary", "minimum amount", "payment due", "credit limit", 
    "available credit", "statement date", "account details", "transaction period"
]

DATE_REGEX = re.compile(
    r"""
    (
        \b(?:\d{1,2}[\/\-.]\d{1,2}[\/\-.]\d{2,4}|\d{4}[\/\-.]\d{1,2}[\/\-.]\d{1,2})\b
        |
        \b\d{1,2}(st|nd|rd|th)?[\s\-]?(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*[\s\-]?\d{2,4}\b
        |
        \b\d{1,2}(st|nd|rd|th)?[\s\-]?(January|February|March|April|May|June|July|August|September|October|November|December)[\s\-]?\d{2,4}\b
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

AMOUNT_REGEX = re.compile(
    r"""
    (?:INR|Rs\.?|₹)?      
    \s*                  
    [-+]?                
    (?:\d{1,3}(?:,\d{2,3})*|\d+)  
    (?:\.\d+)?           
    \s*(Cr|Dr|CR|DR)?    
    """,
    re.VERBOSE,
)

GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GEMINI_MODEL = "gemini-2.0-flash-exp"
DEFAULT_BATCH_SIZE = 1
DEFAULT_DPI = 400
MAX_RETRIES = 3
TEMPERATURE = 0.01
MAX_COMPLETION_TOKENS = 8192

# Model Names
LLAMA_MODEL = "meta-llama/llama-4-maverick-17b-128e-instruct"
GEMINI_MODEL = "gemini-2.5-flash"

# Processing Parameters
CONFIDENCE_THRESHOLD = 0.5
TABLE_PADDING = 10
TEMPERATURE = 0.0
MAX_COMPLETION_TOKENS_LLAMA = 300
MAX_COMPLETION_TOKENS_SCHEMA = 10

# Default Schema Template
DEFAULT_SCHEMA = '[{"dt":"DD-MM-YYYY","desc":"COMPLETE_EXACT_DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]'

# File Extensions
ALLOWED_PDF_EXTENSIONS = ['.pdf']
TEMP_FILE_PREFIX = "temp_"
TEMP_DECRYPTED_PREFIX = "temp_decrypted_"

# Output File Suffixes
CSV_SUFFIX = "_hybrid_transactions.csv"
JSON_SUFFIX = "_hybrid_transactions.json"

# UI Messages
APP_TITLE = "Bank Statement Transaction Extraction"
APP_DESCRIPTION = "Upload a PDF file and then click 'Extract to CSV/JSON' to process transactions with smart schema detection using Llama for analysis and Gemini Vision for extraction."

HYBRID_INFO = "🎯 **Hybrid Approach**: Llama analyzes table structure & schema, Gemini Vision extracts transaction data for optimal accuracy"

PRIVACY_INFO = "🔒 **Privacy Protected**: Only the table images displayed below are sent to AI models for processing. No personal account details, passwords, or other sensitive information from your PDF are transmitted to any external AI service."

# Validation function
def validate_api_keys():
    """Validate that required API keys are present"""
    if not GROQ_API_KEY or not GEMINI_API_KEY:
        raise ValueError(
            "GROQ_API_KEY or GEMINI_API_KEY not found in environment variables. Please set them in .env file"
        )