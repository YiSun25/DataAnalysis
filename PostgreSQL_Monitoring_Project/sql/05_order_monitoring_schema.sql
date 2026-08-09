-- =========================================================
-- Order Monitoring Project: Database Schema
-- Database: PostgreSQL
-- =========================================================

-- Run once when initializing the database.
CREATE TABLE customers(
    customer_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    customer_name VARCHAR(100) NOT NULL,
    email VARCHAR(150) UNIQUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP  -- 表示带时区语义的时间戳。对于日志、订单和系统事件，使用带时区的时间通常更安全
);                                                             -- DEFAULT CURRENT_TIMESTAMP 插入数据时，如果没有手动提供 created_at，数据库自动使用当前时间。

-- 也就是说，后面只需要写
-- INSERT INTO customers (customer_name, email)
-- VALUES ('ABC Logistics', 'contact@abc.com');
-- 数据库会自动生成：customer_id created_at


SELECT *
FROM customers; 

-- =========================================================
-- Test data
-- =========================================================
INSERT INTO customers (customer_name, email)  
VALUES
    ('Alpha Trading', 'contact@alpha.example'),
    ('Beta Logistics', NULL);

-- =========================================================
-- Verify inserted data
-- =========================================================
SELECT
    customer_id,
    customer_name,
    email,
    created_at
FROM customers
ORDER BY customer_id;


BEGIN;

INSERT INTO customers (customer_name, email)
VALUES ('Duplicate Email Customer', 'contact@alpha.example');

ROLLBACK;  -- BEGIN + ROLLBACK 很适合做安全实验。

-- BEGIN 开始; COMMIT 保存; ROLLBACK 撤销；

CREATE TABLE orders(
    order_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    external_order_id VARCHAR(30) NOT NULL UNIQUE,
    customer_id INTEGER NOT NULL
        REFERENCES customers(customer_id),
    order_date DATE NOT NULL,
    status VARCHAR(20) NOT NULL,
    total_amount NUMERIC(10, 2) NOT NULL
        CHECK (total_amount >= 0),
    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);


SELECT
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'orders'
ORDER BY ordinal_position;



INSERT INTO orders (
    external_order_id,
    customer_id,
    order_date,
    status,
    total_amount
)
VALUES
    ('ERP-2026-001', 1, '2026-08-01', 'SHIPPED', 1250.50),
    ('ERP-2026-002', 1, '2026-08-02', 'PROCESSING', 399.99),
    ('ERP-2026-003', 2, '2026-08-02', 'CREATED', 780.00)
RETURNING
    order_id,
    external_order_id,
    customer_id,
    created_at;

BEGIN;


-- 测试 external_order_id 的 UNIQUE 约束
INSERT INTO orders (
    external_order_id,
    customer_id,
    order_date,
    status,
    total_amount
)
VALUES (
    'ERP-2026-001',   -- 故意使用已经存在的订单号
    1,
    '2026-08-03',
    'CREATED',
    100.00
);

ROLLBACK;

--测试 Foreign Key
BEGIN;

INSERT INTO orders (
    external_order_id,
    customer_id,
    order_date,
    status,
    total_amount
)
VALUES (
    'ERP-2026-999',
    999,              -- customers 中不存在
    '2026-08-03',
    'CREATED',
    100.00
);

ROLLBACK;


-- 测试CHECK，插入负数金额BEGIN;
INSERT INTO orders (
    external_order_id,
    customer_id,
    order_date,
    status,
    total_amount
)
VALUES (
    'ERP-2026-998',
    1,
    '2026-08-03',
    'CREATED',
    -100.00           -- 故意错误
);

ROLLBACK;

-- Creat shipment tables
CREATE TABLE shipments(
    shipment_id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    order_id INTEGER NOT NULL
        REFERENCES orders(order_id),
    warehouse_name VARCHAR(100) NOT NULL,
    shipment_status VARCHAR(20) NOT NULL,
    shipped_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL
        DEFAULT CURRENT_TIMESTAMP
);




-- Practice Data
INSERT INTO shipments (
    order_id,
    warehouse_name,
    shipment_status,
    shipped_at
)
VALUES
    (
        1,
        'Berlin Warehouse',
        'SHIPPED',
        '2026-08-02 10:30:00+02'
    ),
    (
        2,
        'Berlin Warehouse',
        'FAILED',
        NULL
    ),
    (
        2,
        'Berlin Warehouse',
        'CREATED',
        NULL
    )
RETURNING *;
-- =========================================================
-- Verify shipment data
-- =========================================================
SELECT *
FROM shipments
ORDER BY shipment_id;
