












-- Practice  Questions use hr
/*
1.Write a query to show the employee details whose job id is  'IT_PROG'
2.Write a query to show the employee details hired between 1990 and 1995 
3.Write a query to show the Department id  & hire_date for Neena  & bruce
4.Write a query to show the Name and job id of the employees working as accountant or clerk.
5.Write a query to show the employee details who has joined before 1995
6.Write a query to show the employee details who has not assigned any manager
7.Write a query to show the details of all employees whose job_id is IT_PROG name and earns salary more than 5,000.
8.Write a query to show all the employees not belonging from department id 90,80,70
9.Write a query to display employee id, first name, last name, job id  of first 5 highest salaried employees.
10.Write a query to display third highest salaried employee details having 'ST_MAN' job id.
11.Write a query to display the average, highest, lowest, and sum of monthly salaries for all sales representatives
12.Write a query to show the earliest and latest join dates of employees
13. Write a query to display the no. of weeks the employee has been employed for all employees in department 90
14.Write a query to display hire date, date after 100 days of joining and date before 1 Month of joining for those belong  department id 90
15.Write a query to display salay, and sal_grade as ‘Good’ if salary >15000 other wise ‘Bad’
16.Write a query to display id, firstname,department_id salary and list 0 if commission_pct value is NULL for those employees belong to department id 80 and 90 */

-- 1.Write a query to show the employee details whose job id is  'IT_PROG'
select * from hr.employees where job_id = "IT_PROG";
-- 2.Write a query to show the employee details hired between 1990 and 1995 
select * from hr.employees where hire_date between "1990-01-01" and '1995-12-31';
-- 3.Write a query to show the Department id  & hire_date for Neena  & bruce
select department_id, hire_date from hr.employees where first_name in ("Neena", "Bruce");
-- 4.Write a query to show the Name and job id of the employees working as accountant or clerk.
select concat_ws(" ", first_name, last_name) as full_name, job_id from hr.employees 
	where instr(job_id, "CLERK") > 0 or instr(job_id, "ACCOUNT") > 0;
-- 5.Write a query to show the employee details who has joined before 1995
select * from hr.employees where hire_date < "1995-01-01";
-- 6.Write a query to show the employee details who has not assigned any manager
select * from hr.employees where manager_id is null;
-- 7.Write a query to show the details of all employees whose job_id is IT_PROG name and earns salary more than 5,000.
select * from hr.employees where job_id = "IT_PROG" and salary > 5000;
-- 8.Write a query to show all the employees not belonging from department id 90,80,70
select * from hr.employees where department_id not in (90, 80, 70);
-- 9.Write a query to display employee id, first name, last name, job id  of first 5 highest salaried employees.
select employee_id, first_name, last_name, job_id  from hr.employees order by salary desc limit 5;
-- 10.Write a query to display third highest salaried employee details having 'ST_MAN' job id.
select * from hr.employees where job_id = "ST_MAN" order by salary desc limit 2,1;
-- 11.Write a query to display the average, highest, lowest, and sum of monthly salaries for all sales representatives
select avg(salary) as average, max(salary) as highest, min(salary) as lowest, sum(salary) as total from hr.employees where job_id = "SA_REP";
-- 12.Write a query to show the earliest and latest join dates of employees
select min(hire_date) as earliest, max(hire_date) as latest from hr.employees;
-- 13. Write a query to display the no. of weeks the employee has been employed for all employees in department 90
select datediff(curdate(),hire_date) / 7 as noOfWeeks from hr.employees where department_id = 90;
-- 14.Write a query to display hire date, date after 100 days of joining and date before 1 Month of joining for those belong  department id 90
select hire_date, date_add(hire_date, interval 100 day) as after_100_days, date_sub(hire_date, interval 1 month) as before_one_month from hr.employees where department_id = 90;
-- 15.Write a query to display salay, and sal_grade as ‘Good’ if salary >15000 other wise ‘Bad’
select salary, IF(salary > 15000, "Good", "Bad") as sal_grade from hr.employees;
-- 16.Write a query to display id, firstname,department_id salary and list 0 if commission_pct value is NULL for those employees belong to department id 80 and 90 */
select employee_id, first_name, department_id, salary, ifnull(commission_pct, 0) as commission_pct from hr.employees where department_id  = 90;


select avg(salary) as avg_salary, avg(ifnull(salary, 0)) as avg_salary_with_null from hr.employees;

/*  Execution Order:
FROM 
WHERE
GROUP BY
HAVING
SELECT
ORDER BY
LIMIT

*/
-- 
# Synatx 
# select columns from tabname where condition group by columns having condition order by column.


# use employees table 
# Question 1: display the number of employees in each department. 
select count(employee_id) as no_of_employees, department_id from hr.employees group by department_id;
# Question 2:  display total salary paid to employees  in each department
select sum(salary) as total_salary, department_id from hr.employees group by department_id;

# Question 3: display number of employees, avg salary paid to employees in each department.
select count(employee_id) as no_0f_employees, avg(salary) as avg_salary, department_id from hr.employees group by department_id;
# Question 4: display the department id, job id, min salary paid to employees group by department_id, job_id.
select department_id, job_id,min(salary) as min_salary from hr.employees group by department_id, job_id;

# Question 5: find the sum of salary, count of employees who belong to the department id 80 and 90  
select sum(salary) as total_salary, count(employee_id) as total_employees from hr.employees where department_id in (80,90);

# Question 6: display the count of the employees based on year( hiredate )
select count(*) as total_employees, date_format(hire_date, "%Y") as in_year from hr.employees group by date_format(hire_date, "%Y");
# Question 7: sort and group  the employees based on year and month wise with count of employees 
select count(*) as total_employees, date_format(hire_date, "%Y") as in_year, date_format(hire_date, "%M") as in_month from hr.employees group by date_format(hire_date, "%Y"), date_format(hire_date, "%M") order by total_employees;
# Question 8: display the department id, number of employees of those groups that have more than 2 employees -- having clause 
select count(employee_id) as no_of_employees, department_id from hr.employees group by department_id having no_of_employees > 2;
# Question 9:  display the departments which has sum of salary greater than 35000
select sum(salary) as total_salary, department_id from hr.employees group by department_id having total_salary > 35000;
  
   /*
CASE <input>
    WHEN <eval_expression_1> THEN <expression if true>
    WHEN <eval_expression_2> THEN <expression if true>
    …
    WHEN <eval_expression_N> THEN <expression if true>
    ELSE <default expression> END 
*/


-- categorize employees based on their year of service <365 as less than 1 yr, <730 as 1-2 yrs <1095 as 2-3 yrs else more than 3 yrs 
-- consider todays date as 2000-12-31.
select concat_ws(" ", first_name, last_name) as full_name, job_id, hire_date,  CASE 
	WHEN datediff(date("2000-12-31"), hire_date) < 365 THEN "less than 1 year"
    WHEN datediff(date("2000-12-31"), hire_date) < 730 THEN "1-2 years"
    WHEN datediff(date("2000-12-31"), hire_date) < 1095 THEN "2-3 years"
    ELSE "more than 3 years"
    END as service_history
 from hr.employees;
