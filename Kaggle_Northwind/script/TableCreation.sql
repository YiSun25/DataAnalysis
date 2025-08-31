-- 创建数据库
CREATE DATABASE IF NOT EXISTS northwind;
USE northwind;

CREATE TABLE categories (
  categoryID INT,  -- 类别 ID
  categoryName VARCHAR(26),  -- 类别名称
  description VARCHAR(68)  -- 类别描述
);

CREATE TABLE customers (
  customerID VARCHAR(15),  -- 客户 ID
  companyName VARCHAR(46),  -- 公司名称
  contactName VARCHAR(33),  -- 联系人姓名
  contactTitle VARCHAR(40),  -- 联系人职位
  city VARCHAR(25),  -- 城市
  country VARCHAR(21)  -- 国家
);

CREATE TABLE employees (
  employeeID INT,  -- 员工 ID
  employeeName VARCHAR(26),  -- 员工姓名
  title VARCHAR(30),  -- 职位
  city VARCHAR(18),  -- 城市
  country VARCHAR(13),  -- 国家
  reportsTo DECIMAL(10,2)  -- 上级员工 ID
);

CREATE TABLE order_details (
  orderID INT,  -- 订单 ID
  productID INT,  -- 产品 ID
  unitPrice DECIMAL(10,2),  -- 单价
  quantity INT,  -- 数量
  discount DECIMAL(10,2)  -- 折扣
);

CREATE TABLE orders (
  orderID INT,  -- 订单 ID
  customerID VARCHAR(15),  -- 客户 ID
  employeeID INT,  -- 员工 ID
  orderDate VARCHAR(20),  -- 订单日期, here use VARCHAR
  requiredDate VARCHAR(20),  -- 要求日期
  shippedDate VARCHAR(20),  -- 发货日期
  shipperID INT,  -- 物流商 ID
  freight DECIMAL(10,2)  -- 运费
);

-- customers 表：客户信息
CREATE TABLE products (
  productID INT,  -- 产品 ID
  productName VARCHAR(42),  -- 产品名称
  quantityPerUnit VARCHAR(30),  -- 包装规格
  unitPrice DECIMAL(10,2),  -- 单价
  discontinued INT,  -- 是否停产（0/1）
  categoryID INT  -- 类别 ID
);



-- shippers 表：物流公司信息
CREATE TABLE shippers (
  shipper_id INT PRIMARY KEY,  -- 物流公司 ID
  company_name VARCHAR(100),  -- 公司名称
  phone VARCHAR(50)  -- 联系电话
);

-- products 表：产品信息
CREATE TABLE products (
  product_id INT PRIMARY KEY,  -- 产品 ID
  product_name VARCHAR(100),  -- 产品名称
  supplier_id INT,  -- 供应商 ID
  category_id INT,  -- 外键：类别 ID
  quantity_per_unit VARCHAR(50),  -- 包装规格
  unit_price DECIMAL(10,2),  -- 单价
  units_in_stock INT,  -- 库存数量
  units_on_order INT,  -- 订购中数量
  reorder_level INT,  -- 再订货临界点
  discontinued BOOLEAN,  -- 是否停产
  FOREIGN KEY (category_id) REFERENCES categories(category_id)  -- 外键约束
);

-- orders 表：订单信息
CREATE TABLE orders (
  order_id INT PRIMARY KEY,  -- 订单 ID
  customer_id VARCHAR(10),  -- 外键：客户 ID
  employee_id INT,  -- 外键：员工 ID
  order_date DATE,  -- 下单日期
  required_date DATE,  -- 要求送达日期
  shipped_date DATE,  -- 实际发货日期
  ship_via INT,  -- 外键：物流公司 ID
  freight DECIMAL(10,2),  -- 运费
  FOREIGN KEY (customer_id) REFERENCES customers(customer_id),
  FOREIGN KEY (employee_id) REFERENCES employees(employee_id),
  FOREIGN KEY (ship_via) REFERENCES shippers(shipper_id)
);

-- order_details 表：订单明细
CREATE TABLE order_details (
  order_id INT,  -- 外键：订单 ID
  product_id INT,  -- 外键：产品 ID
  unit_price DECIMAL(10,2),  -- 下单时单价
  quantity INT,  -- 数量
  discount DECIMAL(4,2),  -- 折扣
  PRIMARY KEY (order_id, product_id),  -- 联合主键（每个订单中某个产品唯一）
  FOREIGN KEY (order_id) REFERENCES orders(order_id),  -- 订单外键
  FOREIGN KEY (product_id) REFERENCES products(product_id)  -- 产品外键
);

CREATE TABLE shippers (
  `shipperID` INT,  -- 物流 ID
  `companyName` VARCHAR(26)  -- 公司名称
);

-- DROP database northwind;
