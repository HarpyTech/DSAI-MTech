












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

# Question 2:  display total salary paid to employees  in each department


# Question 3: display number of employees, avg salary paid to employees in each department.

# Question 4: display the department id, job id, min salary paid to employees group by department_id, job_id.


# Question 5: find the sum of salary, count of employees who belong to the department id 80 and 90  


# Question 6: display the count of the employees based on year( hiredate )

# Question 7: sort and group  the employees based on year and month wise with count of employees 

# Question 8: display the department id, number of employees of those groups that have more than 2 employees -- having clause 

# Question 9:  display the departments which has sum of salary greater than 35000

  
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

