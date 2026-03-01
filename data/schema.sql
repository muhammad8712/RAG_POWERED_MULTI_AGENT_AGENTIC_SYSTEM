PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS invoices;
DROP TABLE IF EXISTS purchase_orders;
DROP TABLE IF EXISTS vendors;

CREATE TABLE vendors (
  vendor_id TEXT PRIMARY KEY,
  vendor_name TEXT NOT NULL,
  country TEXT NOT NULL,
  credit_rating TEXT NOT NULL,
  onboarding_date TEXT NOT NULL
);

CREATE TABLE invoices (
  invoice_id TEXT PRIMARY KEY,
  vendor_id TEXT NOT NULL,
  amount REAL NOT NULL,
  currency TEXT NOT NULL,
  issue_date TEXT NOT NULL,
  due_date TEXT NOT NULL,
  status TEXT NOT NULL,        -- OPEN, PAID, OVERDUE, PARTIALLY_PAID, DISPUTED
  payment_date TEXT,           -- nullable
  FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
);

CREATE TABLE purchase_orders (
  po_id TEXT PRIMARY KEY,
  vendor_id TEXT NOT NULL,
  order_date TEXT NOT NULL,
  amount REAL NOT NULL,
  status TEXT NOT NULL,        -- OPEN, APPROVED, DELIVERED, CANCELLED, LATE
  expected_delivery_date TEXT,
  actual_delivery_date TEXT,
  FOREIGN KEY (vendor_id) REFERENCES vendors(vendor_id)
);

CREATE TABLE payments (
  payment_id TEXT PRIMARY KEY,
  invoice_id TEXT NOT NULL,
  amount REAL NOT NULL,
  payment_date TEXT NOT NULL,
  method TEXT NOT NULL,        -- BANK_TRANSFER, CARD, CASH, CHECK
  FOREIGN KEY (invoice_id) REFERENCES invoices(invoice_id)
);

CREATE INDEX idx_invoices_vendor ON invoices(vendor_id);
CREATE INDEX idx_invoices_status ON invoices(status);
CREATE INDEX idx_po_vendor ON purchase_orders(vendor_id);
CREATE INDEX idx_payments_invoice ON payments(invoice_id);