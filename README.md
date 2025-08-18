# VLM-based | Bank Transaction Extraction

This project leverages Vision Language Models (VLMs) to extract and process transaction data from bank statements in PDF format. It combines advanced table detection with AI-powered data extraction to generate structured CSV outputs.

## Features
- Automated table detection and extraction from bank statement PDFs
- Export cropped table images and structured CSV files
- Multi-model support for enhanced accuracy

## Setup & Usage
1. Install dependencies:
    ```powershell
    pip install -r requirements.txt
    ```
2. Configure API keys in `.env` file:
    ```env
    GOOGLE_API_KEY=your_google_api_key
    GROQ_API_KEY=your_groq_api_key
    ```
3. Launch the application:
    ```powershell
    streamlit run vlm_extractor.py
    ```
4. Processed table images are saved to `table/` directory
5. Extracted transaction data exports to `outputs/` directory as CSV files

## Core Components
- `vlm_extractor.py`: Main Streamlit application interface
- `bank_extractor.py`, `table_cropper.py`: Core extraction scripts
- `prompts.py`, `css.py`: Configuration and styling modules

## AI Models & Technologies
- **Table Detection**: Table Transformer (IFRS-adapted) - [HuggingFace Model](https://huggingface.co/apkonsta/table-transformer-detection-ifrs)
- **Vision Language Models**: 
    - Gemini-2.5-flash
    - Llama-4-Maverick-17B-128E-Instruct
- **Complementary Extraction**: `CAMELOT` for additional data validation