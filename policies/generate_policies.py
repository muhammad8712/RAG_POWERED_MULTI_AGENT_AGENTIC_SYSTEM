from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import cm

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "policies"
OUT_DIR.mkdir(exist_ok=True)

POLICIES = {
    "01_payment_terms_policy.pdf": [
        "Payment Terms Policy",
        "",
        "1. Standard payment terms are Net 30 days from invoice issue date.",
        "2. For approved vendors with credit rating A, Net 45 may be granted.",
        "3. Vendors with rating C or D must use Net 15 unless CFO approval exists.",
        "4. Early payment discount: 1% if paid within 10 days (if stated in contract).",
    ],
    "02_late_payment_fees.pdf": [
        "Late Payment Fees and Interest",
        "",
        "1. A grace period of 5 calendar days applies after the due date.",
        "2. After the grace period, late fee is 2% of the outstanding amount.",
        "3. Additional interest of 0.05% per day may be applied starting day 10 overdue.",
        "4. Fees must be documented with a reference to the invoice_id and vendor_id.",
    ],
    "03_vendor_onboarding_compliance.pdf": [
        "Vendor Onboarding & Compliance",
        "",
        "1. Required documents: tax certificate, bank details, signed contract.",
        "2. High-risk vendors (rating D) require compliance review and director approval.",
        "3. Vendor master data must include country and onboarding_date.",
    ],
    "04_procurement_approval_matrix.pdf": [
        "Procurement Approval Matrix",
        "",
        "1. PO amount <= 5,000 EUR: Department Manager approval.",
        "2. 5,001 to 25,000 EUR: Finance Manager approval.",
        "3. > 25,000 EUR: CFO approval.",
        "4. Late deliveries > 14 days require supplier performance review.",
    ],
    "05_invoice_dispute_and_validation.pdf": [
        "Invoice Validation & Dispute Process",
        "",
        "1. DISPUTED invoices must not be paid until dispute resolved.",
        "2. Invoice must match PO amount within 3% tolerance.",
        "3. Missing PO reference triggers manual review.",
    ],
}

def write_pdf(path: Path, lines: list[str]):
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    x = 2 * cm
    y = height - 2 * cm
    line_height = 14

    for line in lines:
        if y < 2 * cm:
            c.showPage()
            y = height - 2 * cm
        c.drawString(x, y, line)
        y -= line_height

    c.save()

def main():
    for fname, lines in POLICIES.items():
        write_pdf(OUT_DIR / fname, lines)
    print(f"✅ Generated {len(POLICIES)} policy PDFs in {OUT_DIR}")

if __name__ == "__main__":
    main()