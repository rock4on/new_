import re

import os

from azure.ai.formrecognizer import DocumentAnalysisClient

from azure.core.credentials import AzureKeyCredential

from azure.core.pipeline.transport import RequestsTransport

 

# --- CONFIG ---

endpoint =""

key = ""

 

# If your PDF is in a 'data' folder inside your project directory:

base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory where your script is located

file_name = "CCG - 7115 Standard Drive - Office - Administrative or Professional - 11300 (1) (1).pdf"

data_folder = os.path.join(base_dir)  # Adjust path as needed

file_path = os.path.join(data_folder, file_name)

 

print(f"Looking for file at: {file_path}")  # This will help debug path issues

 

if not os.path.exists(file_path):

    raise FileNotFoundError(f"File does not exist: {file_path}")

 

 

# --- Helper: Extract ZIP ---

def extract_zip(text):

    match = re.search(r'\b\d{5}(?:-\d{4})?\b', text)

    return match.group(0) if match else ""

 

# Set up a custom transport with SSL verification disabled

transport = RequestsTransport(connection_verify=False)

 

# --- Connect to Azure ADI ---

client = DocumentAnalysisClient(endpoint=endpoint, credential=AzureKeyCredential(key),transport=transport)

 

with open(file_path, "rb") as f:

    poller = client.begin_analyze_document("prebuilt-document", document=f,)

    result = poller.result()

 

# ---- Gather all text for flexible regex matching ----

full_text = ""

for page in result.pages:

    for line in page.lines:

        full_text += line.content + "\n"

 

# --- Extraction Logic ---

# 1. Vendor/Supplier Name

vendor_match = re.search(r"(Communications Construction Group LLC|BGE|[A-Z][A-Za-z &]+LLC|Inc\.|Corporation)", full_text)

vendor = vendor_match.group(1) if vendor_match else ""

 

# 2. Invoice/Account/Doc ref Number

invoice_match = re.search(r"Account #\s*(\d+)", full_text)

account_number = invoice_match.group(1) if invoice_match else ""

invoice_number = account_number  # BGE invoice uses "Account #" as main reference

 

# 3. Supporting documentation date (Invoice date)

date_match = re.search(r"Issued Date:?\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", full_text)

invoice_date = date_match.group(1) if date_match else ""

 

# 4. Location per supporting documentation (Service address)

loc_match = re.search(r"7115 Standard Dr\s*,?\s*Hanover, MD \d{5}", full_text)

location = loc_match.group(0) if loc_match else ""

 

# 5. Measurement period start date/end date (Billing period)

period_match = re.search(r"Billing Period:\s*([A-Za-z]{3}\s*\d{1,2},?\s*\d{4})?\s*-?\s*([A-Za-z]{3}\s*\d{1,2},?\s*\d{4})", full_text)

if not period_match:

    # Try the BGE format

    period_match = re.search(r"Billing Period: (\w{3}) (\d{1,2}), (\d{4}) - (\w{3}) (\d{1,2}), (\d{4})", full_text)

if not period_match:

    period_match = re.search(r"Billing Period: ([A-Za-z]+\s+\d{1,2},\s*\d{4})\s*-\s*([A-Za-z]+\s+\d{1,2},\s*\d{4})", full_text)

if not period_match:

    # Try simple DD/MM/YYYY - DD/MM/YYYY

    period_match = re.search(r"Billing Period: ([0-9]{2}/[0-9]{2}/[0-9]{4}) - ([0-9]{2}/[0-9]{2}/[0-9]{4})", full_text)

 

if period_match and period_match.lastindex >= 2:

    period_start = period_match.group(1)

    period_end = period_match.group(2)

else:

    # For BGE, try explicit dates from your sample

    period_alt = re.search(r"Billing Period:\s*([A-Za-z]{3} \d{1,2}, \d{4}) - ([A-Za-z]{3} \d{1,2}, \d{4})", full_text)

    if period_alt:

        period_start, period_end = period_alt.group(1), period_alt.group(2)

    else:

        # Fallback: look for any two dates

        dates = re.findall(r"([A-Za-z]{3,9} \d{1,2}, \d{4})", full_text)

        if len(dates) >= 2:

            period_start, period_end = dates[0], dates[1]

        else:

            period_start = period_end = ""

 

# 6. Consumption/activity amount (kWh usage: sum of peak, intermed, off-peak)

# Try to extract usage line (see "Electric details")

usage_match = re.search(r"Peak\s*([\d,]+)\s*kWh\s*([\d,]+)", full_text)

intermed_match = re.search(r"Intermed\s*([\d,]+)\s*kWh", full_text)

offpeak_match = re.search(r"Off Peak\s*([\d,]+)\s*kWh", full_text)

if usage_match and intermed_match and offpeak_match:

    consumption = int(usage_match.group(1).replace(',', '')) + int(intermed_match.group(1).replace(',', '')) + int(offpeak_match.group(1).replace(',', ''))

else:

    # Fallback: use summary kWh if found

    summary_match = re.search(r"Adj Annual Usage Ele ([\d,]+) kWh", full_text)

    if summary_match:

        consumption = summary_match.group(1).replace(',', '')

    else:

        consumption = ""

 

# 7. Unit of measure

unit_match = re.search(r"([\d,]+)\s*kWh", full_text)

unit = "kWh" if unit_match else ""

 

# ---- Result as dictionary ----

extracted = {

    "vendor_name": vendor,

    "account_or_invoice_number": account_number,

    "invoice_date": invoice_date,

    "location": location,

    "measurement_period_start": period_start,

    "measurement_period_end": period_end,

    "consumption_amount": consumption,

    "unit_of_measure": unit,

    "emission_source": "Electricity"

}

 

print("\n=== Extracted Data ===")

for k, v in extracted.items():

    print(f"{k}: {v}")

 

# Optional: Save to JSON

import json

with open("extracted_bge_invoice_fields.json", "w", encoding="utf-8") as f:

    json.dump(extracted, f, indent=2)