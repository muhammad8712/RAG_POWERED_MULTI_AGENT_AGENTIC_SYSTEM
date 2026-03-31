PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS payments;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS customers;
DROP TABLE IF EXISTS sales_orders_odoo;
DROP TABLE IF EXISTS purchase_orders_odoo;

CREATE TABLE customers (
  customer_id INTEGER PRIMARY KEY,
  name TEXT NOT NULL,
  email TEXT UNIQUE,
  country TEXT,
  age INTEGER,
  signup_date TEXT,
  marketing_opt_in BOOLEAN
);

CREATE TABLE products (
  product_id INTEGER PRIMARY KEY,
  category TEXT,
  name TEXT NOT NULL,
  price_usd REAL,
  cost_usd REAL,
  margin_usd REAL
);

CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL,
  order_time TEXT,
  payment_method TEXT,
  discount_pct REAL,
  subtotal_usd REAL,
  total_usd REAL,
  country TEXT,
  device TEXT,
  source TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE order_items (
  order_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
  order_id INTEGER NOT NULL,
  product_id INTEGER NOT NULL,
  unit_price_usd REAL,
  quantity INTEGER,
  line_total_usd REAL,
  FOREIGN KEY (order_id) REFERENCES orders(order_id),
  FOREIGN KEY (product_id) REFERENCES products(product_id)
);

CREATE TABLE payments (
  transaction_id INTEGER PRIMARY KEY,
  payment_date TEXT,
  customer_id INTEGER NOT NULL,
  amount REAL,
  type TEXT,
  description TEXT,
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);

CREATE TABLE sales_orders_odoo (
  sales_order_ref TEXT PRIMARY KEY,
  creation_date TEXT,
  customer_name TEXT,
  salesperson TEXT,
  company TEXT,
  total REAL,
  status TEXT
);

CREATE TABLE purchase_orders_odoo (
  purchase_order_ref TEXT PRIMARY KEY,
  priority TEXT,
  vendor_name TEXT,
  company TEXT,
  buyer TEXT,
  order_deadline TEXT,
  total REAL,
  status TEXT
);

CREATE INDEX idx_orders_customer_id ON orders(customer_id);
CREATE INDEX idx_order_items_order_id ON order_items(order_id);
CREATE INDEX idx_order_items_product_id ON order_items(product_id);
CREATE INDEX idx_payments_customer_id ON payments(customer_id);
CREATE INDEX idx_sales_orders_status ON sales_orders_odoo(status);
CREATE INDEX idx_purchase_orders_status ON purchase_orders_odoo(status);