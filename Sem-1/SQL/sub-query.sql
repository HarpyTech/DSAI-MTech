/* TOPIC - 3 SUB QUERIES
    Sub Query
        1. Single Row 
            Arithmetic Operators like less than, greater than
        2. Multiple Row
            ANY, ALL, IN, NOT IN, EXISTS
        3. Multiple Column
        4. Nested
        5. Correlated

*/


 use hr;
 # 1. write a query to list the employees who work for sales dept. 
 -- using join 
SELECT 
    e.*
FROM
    employees e
        JOIN
    departments d ON e.department_id = d.department_id
        AND department_name = 'Sales'; 
 
 # rewrite query using subquery 
 SELECT 
    *
FROM
    employees
WHERE
    department_id = (SELECT 
            department_id
        FROM
            departments
        WHERE
            department_name = 'Sales'); 
 
-- Explain will give the details of Indexes, Primary keys used or utilized in the Query
explain  select * from employees where department_id = (select department_id from departments where department_name = "Sales"); 
 
 
 /* Single row subquery :   A subquery that returns a single value and feeds to main query.
Multiple row subquery : Subquery returns multiple values (more rows) to the main query
Multiple column subqueries : Returns one or more columns.
Correlated subqueries : Reference one or more columns in the outer SQL statement. 
The subquery is known as a correlated subquery because the subquery is related to the outer SQL statement.
Nested subqueries : Subqueries are placed within another subquery.*/
-- --------------------------------------------------------------------
/*positions where sq can be placed 
insert,update,delete,select,from,having */
-- ------------------------------------------------------------------
 # relational operators  < ,>,<=>= ==<>- single row sq
 # multiple row sq- in,any ,all,exists
  -- ---------------------------------------------------------
 # single row sq

 # 1. what is the average salary of sales dept?
  SELECT 
    AVG(salary) AS avg_salary
FROM
    employees
WHERE
    department_id = (SELECT 
            department_id
        FROM
            departments
        WHERE
            department_name = 'Sales'); 
  
 # 2.how many employees have salary greater than that of susan
  SELECT 
    COUNT(*) AS emp_count
FROM
    employees
WHERE
    salary > (SELECT 
            salary
        FROM
            employees
        WHERE
            first_name = 'susan'); 

 #3. list of employees who work other than department of Den
 SELECT 
    *
FROM
    employees
WHERE
    department_id != (SELECT 
            department_id
        FROM
            employees
        WHERE
            first_name = 'Den');
 

 #4. list the employees who earn salary less than the employee gerald
   SELECT 
    *
FROM
    employees
WHERE
    salary < (SELECT 
            salary
        FROM
            employees
        WHERE
            first_name = 'gerald');
-- -------------------------------------------------------------------
 # multiple row sq - in , any, all ,exists
 # in - list of values 
 # any -or all -and  > < 
 # x > any(2,3,4)-- > smallest 
  #x < any(2,3,4)-- < greatest
  #x > all(2,3,4)-- > greatest
  #x < all(2,3,4)-- <smallest
#  =any  same as IN

-- --------------------------------------------------------------
# multiple row queries 
#1. list the employees who work for sales,finance department 
  SELECT 
    *
FROM
    employees
WHERE
    department_id IN (SELECT 
            department_id
        FROM
            departments
        WHERE
            department_name IN ('Sales', "finance"));
 #2. list all the employees  who have more salary than everybody who joined in 2000
select * from employees where salary > all ( -- the all operator uses the greatest values returnd in the subquery for this operation
select salary from employees where year(hire_date) = 2000); -- sub query returns values between 2200 to 10500
-- the outer query will fetch the salary details of 11000 - 24000
 
 select * from employees where salary > any ( -- the all operator uses the least value returned in the subquery for this operation
 select salary from employees where year(hire_date) = 2000); -- sub query returns values between 2200 to 10500
 -- the outer query will fetch the salary details of 2400 - 24000
 #3.list the employees where the sales representatives are earning more than any of the 
 #sales manager 
select * from employees where job_id='sa_rep' and salary > any (select salary from employees where job_id='sa_man');


# 4.display the names of the employees whose salary is less than the lowest salary of  any sh_clerk
select first_name, last_name from employees where salary < all (select salary from employees where job_id='sh_clerk');
 -- ----------------------------------------------
 
 # nested sq 
 #1. get the details of employees who are working in city seattle
 select * from employees where department_id in (
	select department_id from departments where location_id in (
		select location_id from locations where city = 'seattle'));

 
 # 2.list the locations in asia region 
 select * from locations where country_id in (
	select country_id from countries where region_id in (
		select region_id from regions where region_name='asia'));
 

 -- --------------------------------------------------------------
 # multiple column queries 
 #1. list all the employees who earn salary equal to that of employee gerald and
 # work in the same department as gerald. 
-- Method 1
SELECT 
    *
FROM
    employees
WHERE
    salary = (SELECT 
            salary
        FROM
            employees
        WHERE
            first_name = 'gerald')
        AND department_id = (SELECT 
            department_id
        FROM
            employees
        WHERE
            first_name = 'gerald')
        AND first_name != 'gerald';
-- Method 2
SELECT 
    *
FROM
    employees
WHERE
    (salary , department_id) = (SELECT
            salary, department_id
        FROM
            employees
        WHERE
            first_name = 'gerald')
        AND first_name != 'gerald';
        -- This Method can be useful only in the quality conditions
    
  -- -----------------------------------------------------------------  
    # sq in from clause  - derived tables - alias 
    #1. find out the  5th highest salary of the employee
     select min(salary) from (select distinct salary from employees order by salary desc limit 5) a;

    #2.display the count of employees whose name starts with 'a'
    select count(*) as emp_count from (select * from employees where first_name like 'a%') a;
   
  -- ----------------------------------------------------------------  
    # sq in having clause 
    #1. find the departments  with average salary greater than the salary of lex.
  SELECT 
    AVG(salary), department_id
FROM
    employees
GROUP BY department_id
HAVING AVG(salary) > (SELECT 
        salary
    FROM
        employees
    WHERE
        first_name = 'lex');
    
    
    -- --------------------------------------------------------------
  
    # sq in update, delete and insert statements
    -- dont execute
    # 1.update the comm_pct of the employees as 0.05 for those who belong to accounts
    update employees set commission_pct = 0.05 where department_id = 
    (select department_id from departments where department_name like 'account%');
     
    #2.give 5 % hike to all the employees of sales dept.
    update employees set salary = salary *1.05 where department_id = 
		(select department_id from departments where department_name like 'sales%');
    
    # sq in delete 
    #1. delete the records of the employees who belongs to accounts dept
	delete from employees where department_id =
		( select department_id from departments where department_name like 'account%');
    # sq in insert 
    -- create a new table - insert - using sq
   create table emp (empid int, ename varchar(20), salary int);
   insert into emp select employee_id, first_name, salary from employees where department_id = 90;
   select * from emp;
 
    -- ----------------------------------------------------
    /*Correlated subquery -inner query will depend on outer query
    inner query gets executed , once for every row that is selected by the outer query
    SELECT j FROM t2 WHERE j IN (SELECT i FROM t1); - uncorrelated 
    SELECT j FROM t2 WHERE (SELECT i FROM t1 WHERE i = j); - corrlated query
     time consuming

    
    */
	select last_name, salary, department_id from employees o where salary > 
		(select avg(salary) from employees where department_id = o.department_id);

    
    
  # exists and not exists
  # exists -- true if your sq returns atleast one row 
  
  #Find employees who have atleast one person reporting to them .
  select first_name from employees e1 where exists (
  select * from employees e2 where e1.employee_id = e2.manager_id);
   # 1 find all the dept which have atleast one employee with salary >4000
    select department_name from departments d where exists (
		select * from employees e where d.department_id = e.department_id and salary > 4000);
  
  
  -- Self Learning
  explain select department_name from departments d where exists (
		select * from employees e where d.department_id = e.department_id and salary > 4000);
  #2 list the employees who changed their jobs atleast once. 
  select first_name from employees e where exists (
   select * from job_history where  employee_id = e.employee_id );
  #3  Display only the department which has no employees
  select department_name from departments d where not exists (
		select * from employees e where d.department_id = e.department_id);
-- ================
-- Extra Practice questions
 -- 1.Find all the employees who have the  highest salary
 select * from employees where salary = ( select max(salary) from employees);
-- 2.find employees with second maximum salary 
select * from employees where salary = ( select salary from employees order by salary desc limit 1,1);
-- 3.list the employees whose salary is in the range of 10000 and 20000 and working for dept id 10 0r 20 
select * from employees where (department_id , salary) = ( 
	select department_id, salary from employees where salary between 10000 and 20000 and department_id in (10,20));
-- 4.Write a query to display the employee name (first name and last name), employee id and salary of all employees who report to Payam.
SELECT 
    CONCAT_WS(' ', first_name, last_name) AS name,
    employee_id,
    salary
FROM
    employees e1
WHERE
    EXISTS( SELECT 
            *
        FROM
            employees e2
        WHERE
            e2.employee_id = e1.manager_id
                AND first_name = 'Payam')
        AND first_name != 'Payam'; 
-- 5.Write a query to display all the information of the employees whose salary is within the range of smallest salary and 2500.
   
-- 6.Write a query to display the employee number, name (first name and last name), and salary for all employees 
-- who earn more than the average salary and who work in a department with any employee with a J in their name. 

-- 7.Display the employee name (first name and last name), employee id, and job_id for all employees whose department location is Toronto. 

-- 8.Write a query to display the employee id, name (first name and last name), salary 
-- and the SalaryStatus column with a title HIGH and LOW respectively for those employees 
-- whose salary is more than and less than the average salary of all employees. 
select salary, case when salary > (select avg(salary) from employees) then 'high' else 'low' end as salaryStatus from employees;
-- 9.Write a query in SQL to display all the information of those employees who did not have any job in the past. 

-- 10. Write a query in SQL to display the full name (first and last name) of manager who is supervising 4 or more employees. 

  
  
    
    
    
    
  
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        
        
    
    
    
    
    
    
    
    
    
    
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 