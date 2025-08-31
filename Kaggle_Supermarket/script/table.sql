CREATE DATABASE IF NOT EXISTS supermarket;
USE supermarket;

CREATE TABLE sales (
  invoice_id VARCHAR(30),
  branch VARCHAR(5),
  city VARCHAR(50),
  customer_type VARCHAR(20),
  gender VARCHAR(10),
  product_line VARCHAR(100),
  unit_price DECIMAL(10,2),
  quantity INT,
  tax_5 DECIMAL(10,2),
  total DECIMAL(10,2),
  date VARCHAR(20),  -- 改成 VARCHAR，date/time格式出问题
  payment VARCHAR(20),
  cogs DECIMAL(10,2),
  gross_margin DECIMAL(10,6),  
  gross_income DECIMAL(10,2),
  rating DECIMAL(3,1)
);

DROP TABLE IF EXISTS sales;