SELECT current_database(), current_user, version(); 
--作用是测试：当前连接的是哪个数据库;当前用户是谁; PostgreSQL 版本是什么

--Creat table
CREATE TABLE test_students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    age INT
);

-- Insert values
INSERT INTO test_students (name, age)
VALUES
('Alice', 24),
('Bob', 26),
('Charlie', 23);

-- Search for this table
SELECT * FROM test_students;

