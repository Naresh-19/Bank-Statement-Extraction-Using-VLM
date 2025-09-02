base64_img = []

prompt1 = """You are a strict transaction table investigatorAnalyze this table and determine if it contains bank transactions.

Transactional Table have these features . Look for:
- Multiple rows of transaction data (not just headers)
- Date column with transaction dates
- Amount columns (debit/credit or withdrawal/deposit)
- Transaction descriptions/particulars
- Running balance column

Return ONLY:
- "YES" if this is a transaction table with actual transaction rows
- "NO" if this is a header, summary, account info, or non-transaction table

Be strict - only return YES if you see actual transaction rows with dates, amounts, and descriptions."""

prompt2 = """Analyze this bank statement table and identify the column order. Look for transaction tables with headers like Date, Description/Particulars, Debit, Credit, Balance.

Based on the column order you observe, reorder this JSON schema to match:

Original: [{"dt":"DD-MM-YYYY","desc":"DESCRIPTION","ref":null,"dr":0.00,"cr":0.00,"bal":0.00,"type":"W"}]

- If Credit comes before Debit in table, put "cr" before "dr" in JSON
- If Debit comes before Credit in table, put "dr" before "cr" in JSON  
- Detect Date is in ASCENDING / DESCENDING order

Change it into : 

Re-Ordered: [{"dt":"DD-MM-YYYY","desc":"DESCRIPTION","ref":null,"cr":0.00,"dr":0.00,"bal":0.00,"type":"W"}]
Date_Order: ASCENDING / DESCENDING

Rules:
- Keep all fields but reorder them to match the visual column sequence
- If Credit comes before Debit in table, put "cr" before "dr" in JSON
- If Debit comes before Credit in table, put "dr" before "cr" in JSON  
- Always keep dt first and type last
- Strictly Return ONLY the reordered JSON schema array and Date_Order :(ASCENDING/DESCENDING) alone, nothing else

Avoid including any additional text or explanations.Just return the JSON array and Date_Order.
"""