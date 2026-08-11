-- =========================================================
-- Q1: Orders with customer information
-- =========================================================

-- Return:
-- external_order_id
-- customer_name
-- status
-- total_amount
SELECT external_order_id, customer_name, status, total_amount
FROM orders AS o 
JOIN customers AS c 
    ON o.customer_id = c.customer_id;


-- =========================================================
-- Q2: Orders with shipment record
-- =========================================================
SELECT
    external_order_id, status, shipment_status, warehouse_name
FROM orders AS o
INNER JOIN shipments AS s
    ON o.order_id = s.order_id;


-- =========================================================
-- Q3: Orders with shipment record (left join)
-- =========================================================
SELECT
    external_order_id, status, shipment_status, warehouse_name
FROM orders AS o
LEFT JOIN shipments AS s
    ON o.order_id = s.order_id;



-- =========================================================
-- Q4: Find orders without shipment record
-- =========================================================
SELECT
    o.external_order_id, o.status
FROM orders AS o
LEFT JOIN shipments AS s
    ON o.order_id = s.order_id
WHERE s.shipment_status IS NULL;


-- =========================================================
-- Q5: Find different status between order status and shipment status
-- =========================================================
SELECT
    o.external_order_id, o.status AS order_status, s.shipment_status, s.warehouse_name
FROM orders AS o
INNER JOIN shipments AS s
    ON o.order_id = s.order_id
WHERE o.status <> s.shipment_status;


-- =========================================================
-- Q6: JOIN + GROUP BY + 聚合
-- =========================================================
SELECT
    c.customer_name,
    COUNT(o.order_id) AS order_count,
    SUM(o.total_amount) AS total_order_amount
FROM customers AS c 
JOIN orders AS o 
    ON c.customer_id = o.customer_id
GROUP BY c.customer_name;


-- =========================================================
-- Q7: JOIN Shipment SUM Error
-- =========================================================
SELECT
    c.customer_name,
    SUM(o.total_amount) AS total_order_amount
FROM customers AS c
JOIN orders AS o
    ON c.customer_id = o.customer_id
JOIN shipments AS s
    ON o.order_id = s.order_id
GROUP BY c.customer_name;