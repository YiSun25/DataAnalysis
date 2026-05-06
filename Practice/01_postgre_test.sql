SELECT current_database(), current_user, version();

CREATE TABLE test_students (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    age INT
);

INSERT INTO test_students (name, age)
VALUES
('Alice', 24),
('Bob', 26),
('Charlie', 23);

SELECT * FROM test_students;