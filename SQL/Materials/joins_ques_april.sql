use hr;

/*
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


#write a query to display job title, firstname , difference between max salary and salary of all employees using natural join 
------------------------------------------------------------------
 -----------------------------------------------------------------------------------------

 #Self join 
-- Write a query to find the name (first_name, last_name) and hire date of the employees who was hired after 'Jones'.
 
# Write a query to display first and last name ,salary of employees who earn less than the employee whose number is 182 using self join  
-------------------------------------------------------------------

/*
#Cross Join - cartersian product between tables
			- every row of 1 table mapped to every row of other table.
            - m*n rows
            - cross join   ,   join 

*/


---------------------------------------------------------------------------



-- extra practices 
 
 
 
 -- 1.Display the first name, last name, department id and department name, for all employees for departments 80 or 40.
 
 
-- 2. Write a query in SQL to display the full name (first and last name), and salary of those employees who working in
--  any department located in London. */


 
 
 
 -- 3.	Write a query in SQL to display those employees who contain a letter z to their first name and also display their last name,
 -- department, city, and state province. (3 rows)


-- 4.	Write a query in SQL to display the job title, department id, full name (first and last name) of employee, starting date 
-- and end date for all the jobs which started on or after 1st January, 1993 and ending with on or before 31 August, 2000. (use employee,job_history)

	

-- 5.	.Display employee name if the employee joined before his manager.




-- 6 •Write a query in SQL to display the name of the department, average salary and number of employees working in that department who 
-- got commission. */




-- 7. Write a query in SQL to display the details of jobs which was done by any of the employees who is  earning a salary on and above 12000( use job_history,employee) */



-- 8. Write a query in SQL to display the employee ID, job name, number of days worked in for all those jobs in department 80.(use job, job_history)






