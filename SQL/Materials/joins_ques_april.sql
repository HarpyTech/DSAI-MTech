use hr;

/*  TOPIC - 1
Introduction to Joins using the ER Diagram
Types of joins
	Inner Join
    Left Join
    Right Join
    Outer Join
    Full Outer Join
    Self Join
*/
#Q1. WAQ to display the details of employees like id, names , salary ,
# and the names of the departments they work in.  
SELECT 
    employee_id,
    first_name,
    last_name,
    salary,
    e.department_id,
    department_name
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id;
-- this gives the all the records of employees when there is matching record of departments.
-- in the above select query alias used since we have department_id column in both tables to diffrenciate the query, define which column data needs to be fetched   can be done with alias or the table name


#Q2. WAQ to display the details of employees like id, names , salary ,
# and the names of the departments they work in. Also include such
#employees who are not assigned to any departments yet.  
SELECT 
    e.employee_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.department_name,
    e.department_id
FROM
    employees e
        LEFT JOIN
    departments d ON e.department_id = d.department_id;
-- this gives the all the records of employees even when there is no matching record of departments.


# Q3. WAQ to display the details of employees like id, names , salary ,
# and the names of the departments they work in. Include the list of departments
#where no employees are working. 
SELECT 
    e.employee_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.department_name,
    e.department_id
FROM
    employees e
        RIGHT JOIN
    departments d ON e.department_id = d.department_id;



#Q4. WAQ to display the details of employees like id, names , salary ,
# and the names of the departments they work in. Also include such
#employees who are not assigned to any departments yet and the list of 
#departments where no employees are working.
 SELECT 
    e.employee_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.department_name,
    e.department_id
FROM
    employees e
        LEFT JOIN
    departments d ON e.department_id = d.department_id 
UNION SELECT 
    e.employee_id,
    e.first_name,
    e.last_name,
    e.salary,
    d.department_name,
    e.department_id
FROM
    employees e
        RIGHT JOIN
    departments d ON e.department_id = d.department_id;


#Q5. WAQ to display the details of employees along with the departments, cities and the country
#they work in.
SELECT 
    e.*, d.department_name, l.city, c.country_name
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    locations l ON l.location_id = d.location_id
        JOIN
    countries c ON c.country_id = l.country_id;


#Q6. WAQ to get the count of employees working in different cities.
#display such cities which has more than 20 employees working in them.
SELECT 
    COUNT(*) AS employee_count, l.city
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    locations l ON l.location_id = d.location_id
GROUP BY l.city
HAVING employee_count > 20;

#Q7. Display the list of employees who are based out of America Region.
SELECT 
    e.*, r.region_name
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    locations l ON l.location_id = d.location_id
        JOIN
    countries c ON c.country_id = l.country_id
        JOIN
    regions r ON r.region_id = c.region_id
WHERE
    r.region_name LIKE 'America%';
    -- always use the LIKE operator if the value is string and value is incomplete.

#Q8. WAQ to list the employees working in 'Seattle'.
SELECT 
    e.*, l.city
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    locations l ON l.location_id = d.location_id
WHERE
    l.city = 'Seattle';

#Q9. WAQ to list the details of employees, their department names and the job titles.
SELECT 
    e.*, d.department_name, j.job_title
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    jobs j ON j.job_id = e.job_id;
-- --------------------------------------
#Natural join - primitive join - uses any columns with same name and datatype to perform the join.

#Write a query to find the addresses (location_id, street_address, city, state_province, country_name) of all the departments.
SELECT 
    l.location_id,
    street_address,
    city,
    state_province,
    country_name
FROM
    departments
        NATURAL JOIN
    locations l
        NATURAL JOIN
    countries;

#write a query to display job title, firstname , difference between max salary and salary of all employees using natural join 
SELECT 
    job_title, first_name, max_salary - salary as salary_difference
FROM
    employees
        NATURAL JOIN
    jobs;

------------------------------------------------------------------
 -----------------------------------------------------------------------------------------

 #Self join 
-- Write a query to find the name (first_name, last_name) and hire date of the employees who was hired after 'Jones'.
SELECT 
    CONCAT_WS(' ', e1.first_name, e1.last_name) AS full_name,
    e1.hire_date
FROM
    employees e1
        JOIN
    employees e2 ON e2.last_name = 'Jones'
        AND e1.hire_date > e2.hire_date;
# Write a query to display first and last name ,salary of employees who earn less than the employee whose number is 182 using self join  
SELECT 
    e1.first_name, e1.last_name,
    e1.salary
FROM
    employees e1
        JOIN
    employees e2 ON e2.employee_id = 182
        AND e1.salary < e2.salary;
-- -----------------------------------------------------------------

/*
#Cross Join - cartersian product between tables
			- every row of 1 table mapped to every row of other table.
            - m*n rows
            - cross join   ,   join 

*/
select * from departments, employees;
select first_name, last_name, department_name from departments, employees;
select first_name, last_name, department_name from departments cross join employees;
---------------------------------------------------------------------------



-- extra practices 
 
 
 
 -- 1.Display the first name, last name, department id and department name, for all employees for departments 80 or 40.
 SELECT 
    first_name, last_name, d.department_id, department_name
FROM
    employees e
        JOIN
    departments d ON d.department_id = e.department_id
WHERE
    d.department_id IN (80 , 40);
 
-- 2. Write a query in SQL to display the full name (first and last name), and salary of those employees who working in
--  any department located in London. */
SELECT 
  concat_ws(" ", first_name, last_name) as name, l.city
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    locations l ON l.location_id = d.location_id
WHERE
    l.city = 'London';

 
 
 
 -- 3.	Write a query in SQL to display those employees who contain a letter z to their first name and also display their last name,
 -- department, city, and state province. (3 rows)
SELECT 
  first_name, last_name, city, department_name, state_province
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        JOIN
    locations l ON l.location_id = d.location_id
    where first_name like "%z%";

-- 4.	Write a query in SQL to display the job title, department id, full name (first and last name) of employee, starting date 
-- and end date for all the jobs which started on or after 1st January, 1993 and ending with on or before 31 August, 2000. (use employee,job_history)
SELECT 
    CONCAT_WS(' ', first_name, last_name) AS name,
    job_title,
    e.department_id,
    start_date,
    end_date
FROM
    employees e
        JOIN
    job_history jh ON jh.employee_id = e.employee_id
        JOIN
    jobs j ON j.job_id = jh.job_id
WHERE
    start_date >= '1993-01-01'
        AND end_date <= '2000-08-31';
	

-- 5.	.Display employee name if the employee joined before his manager.
SELECT 
    CONCAT_WS(' ', e.first_name, e.last_name) AS name
FROM
    employees e
        JOIN
    employees m ON m.employee_id = e.manager_id
        AND e.hire_date < m.hire_date;

-- 6 •Write a query in SQL to display the name of the department, average salary and number of employees working in that department who 
-- got commission. */
SELECT 
    department_name,
    AVG(salary) AS avg_salary,
    COUNT(*) AS employee_count
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
WHERE
    e.commission_pct IS NOT NULL
GROUP BY e.department_id;



-- 7. Write a query in SQL to display the details of jobs which was done by any of the employees who is  earning a salary on and above 12000( use job_history,employee) */
SELECT 
    j.*
FROM
    employees e
        JOIN
    job_history jh ON jh.employee_id = e.employee_id
        JOIN
    jobs j ON jh.job_id = j.job_id
WHERE
    e.salary > 1200;


-- 8. Write a query in SQL to display the employee ID, job name, number of days worked in for all those jobs in department 80.(use job, job_history)
SELECT 
    e.employee_id,
    j.job_title,
    DATEDIFF(jh.end_date, jh.start_date) AS no_of_days
FROM
    employees e
        JOIN
    job_history jh ON jh.employee_id = e.employee_id
        JOIN
    departments d ON d.department_id = jh.department_id
        AND d.department_id = 80
        JOIN
    jobs j ON jh.job_id = j.job_id;
    
/* TOPIC - 2
Data Integrity 
	Column Level Intergrity
		Check Constraint, data type, size
    Row Level Integrity
		Candidate Keys, Primary Keys, Unique Key, Foreign Key, Composite Key, Alternative Key
	Referential Integrity
		Establishing Parent - Child relationship between tables
	
*/

