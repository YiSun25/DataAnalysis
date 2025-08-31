USE northwind;
-- 统计各个表中有多少行
-- SELECT COUNT(*) FROM orders;
-- SELECT COUNT(*) FROM employees;
-- SELECT COUNT(*) FROM order_details;
-- SELECT COUNT(*) FROM products;


-- 分析1：每年订单数量趋势
SELECT 
  LEFT(orderDate, 4) AS order_year,  -- 从字符串中提取年份
  COUNT(*) AS total_orders
FROM orders
WHERE orderDate IS NOT NULL
GROUP BY order_year
ORDER BY order_year;


-- 分析2：每位员工完成的订单总量
SELECT 
  e.employeeID,  -- 员工 ID
  e.employeeName,  -- 员工姓名（Kaggle 版合并为一列）
  COUNT(o.orderID) AS total_orders
FROM employees e
LEFT JOIN orders o ON e.employeeID = o.employeeID
GROUP BY e.employeeID, e.employeeName
ORDER BY total_orders DESC;

-- 分析3：产品销售排行（基于订单明细）
SELECT 
  p.productID,
  p.productName,
  SUM(od.quantity) AS total_quantity_sold,
  SUM(od.unitPrice * od.quantity * (1 - od.discount)) AS total_revenue
FROM order_details od
JOIN products p ON od.productID = p.productID
GROUP BY p.productID, p.productName
ORDER BY total_revenue DESC
LIMIT 10;

-- 分析4：不同国家的客户数量
SELECT 
  country,
  COUNT(*) AS customer_count
FROM customers
GROUP BY country
ORDER BY customer_count DESC;

-- 分析5：每年收入
SELECT 
  LEFT(o.orderDate, 4) AS order_year,
  ROUND(SUM(od.unitPrice * od.quantity * (1 - od.discount)), 2) AS total_revenue
FROM orders o
JOIN order_details od ON o.orderID = od.orderID
WHERE o.orderDate IS NOT NULL
GROUP BY order_year
ORDER BY order_year;
