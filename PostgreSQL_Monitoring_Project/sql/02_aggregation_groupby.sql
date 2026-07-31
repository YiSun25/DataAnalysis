SELECT *
FROM test_students;

-- Q1
-- Return total student count, average age,
-- minimum age and maximum age in one row.
SELECT COUNT(*), AVG(age), MIN(age), MAX(age)
FROM test_students;

-- Q2
-- Count students for each exact age.
-- Sort by age ascending.
SELECT age, COUNT(*) AS student_count
FROM test_students
GROUP BY age
ORDER BY age ASC;

-- Q3
-- Divide students into:
-- age >= 25: 25_or_older
-- age < 25: under_25
-- Count students in each group.
SELECT 
    CASE 
        WHEN age IS NULL THEN 'unknown'
        WHEN age >= 25 THEN '25+'
        ELSE 'under_25'
    END AS age_group,
    COUNT(*) AS student_count
FROM test_students
GROUP BY age_group;


-- Q4
-- Find duplicated names.
-- Only return groups whose count is greater than 1.
SELECT name, count(*) AS duplicated_name
FROM test_students
GROUP BY name 
HAVING count(*) > 1;  -- Having 后面不能用 duplicated_name, 因为是在select中才定义的输出别名，而having判断分组时，这个别名还不能被引用