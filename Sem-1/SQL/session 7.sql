USE hr;

-- Simple View
CREATE OR REPLACE VIEW department90 AS
    SELECT 
        *
    FROM
        employees
    WHERE
        department_id = 90;
SELECT 
    *
FROM
    department90;

-- Complex View

CREATE OR REPLACE VIEW employee_dept AS
    SELECT 
        first_name, last_name, department_name
    FROM
        employees e
            JOIN
        departments d ON e.department_id = d.department_id;

SELECT 
    *
FROM
    employee_dept;

-- Horizontal Views are nothing but rowise selection, it includes filter expression with where clause.
-- Vertical View - is fetching selective columns
-- 