"""
Document categories configuration
"""

# All supported document categories based on ground-truth.json
DOCUMENT_CATEGORIES = [
    # Identity Documents
    "passport",
    "id_card",
    "driver_license",
    "passport_card",
    # Financial/Business Documents
    "bank_statement",
    "utility_bill",
    "paystub",
    "employment_card",
    "green_card",
    "tax_document",
    # Fallback category
    "other",
]

# Category groups for easier management
IDENTITY_DOCUMENTS = ["passport", "id_card", "driver_license", "passport_card"]

FINANCIAL_DOCUMENTS = ["bank_statement", "utility_bill", "paystub", "tax_document"]

BUSINESS_DOCUMENTS = ["employment_card", "green_card"]

# Category descriptions for prompts
CATEGORY_DESCRIPTIONS = {
    "passport": "International passport document",
    "id_card": "National identity card or state ID",
    "driver_license": "Driver's license or driving permit",
    "passport_card": "US passport card for land/sea travel",
    "bank_statement": "Financial account statement from bank",
    "utility_bill": "Electricity, water, gas, or internet bill",
    "paystub": "Salary or wage payment statement",
    "employment_card": "Employment Authorization Card",
    "green_card": "Green card or Permanent Resident Card",
    "tax_document": "Tax document number or similar document",
    "other": "Document type not matching any specific category",
}
