-- =========================================================
-- 01. Environment check
-- Safe to run repeatedly
-- =========================================================

SELECT
    current_database(),
    current_user,
    version();


-- =========================================================
-- 02. Table creation
-- Run once only
-- =========================================================

-- CREATE TABLE test_students (
--     id SERIAL PRIMARY KEY,
--     name VARCHAR(50),
--     age INT
-- );


-- =========================================================
-- 03. Seed data
-- Run once only
-- =========================================================

-- INSERT INTO test_students (name, age)
-- VALUES
--     ('Alice', 24),
--     ('Bob', 26),
--     ('Charlie', 23);


-- =========================================================
-- 04. Basic SELECT
-- =========================================================

SELECT *
FROM test_students;


-- =========================================================
-- 05. Exercises
-- Write your SQL below each question
-- =========================================================

-- Q1.
-- Return only the name and age of all students.
-- Expected number of rows: 3
SELECT name, age
FROM test_students;

-- Q2.
-- Return all columns for students whose age is at least 24.
-- Before running, predict which students will appear.
SELECT * 
FROM test_students
WHERE age >= 24;

-- Q3.
-- Return all students ordered by age from highest to lowest.
-- If ages are equal, order names alphabetically.
SELECT *
FROM test_students
ORDER BY age DESC, name ASC;

-- Q4.
-- Return name, age and a calculated column called age_next_year.
SELECT name, age, (age + 1) AS age_next_year
FROM test_students;

-- Q5.
-- Find records with any of the following data-quality problems:
-- 1. name is NULL
-- 2. age is NULL
-- 3. age is below 0
-- 4. age is above 120
-- Expected number of rows with current data: 0
SELECT *
FROM test_students
WHERE name IS NULL
  OR  age IS NULL
  OR  age < 0
  OR  age > 120;