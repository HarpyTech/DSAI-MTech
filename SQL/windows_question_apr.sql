use hr;

# total salary of all the employees in the company
select sum(salary) as total_salary from employees; -- single row of info
 
select department_id, sum(salary) as total_salary from employees group by department_id;
-- aggregated total salry for each department, no of rows is equal to no of unique departments
 /*
  window /analytic function
 - performs calculations on set of rows  , do not collapase the result of the rows
 into a single rows. 
 - used for analysis - next ,previous ,n th value,running total , cumaltive 
 syntax - window function
 window_function_name(expression) over(
 [partition by]--data grouping
 [order by ] determine the order asc or desc
 [frame definition] -- limit the rows /range/window size
 )
 */
 
-- -------------------------------------------------------------
# row_number() --serial no, rowid 
-- ------------------------------------------------------------
#1. order the rows based on salary 
select first_name, salary, row_number() over() from employees;

# 2. rowid based on salary dept wise 
select department_id, salary, row_number() over(partition by department_id order by salary) from employees;

-- --------------------------------------------------------------------
# rank()
-- ---------------------------------------------------------------------  
# 1.ranking based on  salary
 select salary, rank() over(order by salary desc) from employees;
#2. ranking in order of salary  dept wise
 select department_id, salary, rank() over(partition by department_id order by salary desc) from employees;
  -- --------------------------------------------------
# dense rank 
-- --------------------------------------------------------------------
#1. dense rank dept wise 
select department_id, salary, dense_rank() over(partition by department_id order by salary desc) as `dense_rank`, rank() over(partition by department_id order by salary desc) as `rank`   from employees;
#2.display row number, rank and dense rank in the same query dept 60 and 90
select row_number() over(), department_id, salary, dense_rank() over(partition by department_id order by salary desc) as `dense_rank`, rank() over(partition by department_id order by salary desc) as `rank`   from employees;
-- --------------------------------------------------------------------
# ntile -  create n bins based on formula --  total no of rows /no of   bins
-- --------------------------------------------------------------------
#1. create ten bins 
select *, ntile(10) over() from employees;

select *, ntile(2) over() from employees where department_id = 100;
#2. same query dept wise ,no of bins 10
select *, ntile(10) over(partition by department_id) from employees;
   
select first_name, ntile(4) over(partition by department_id) from employees where department_id=100;
-----------------------------------
-- LEAD and LAG
-- LEAD computes an expression based on the next rows
-- i.e. rows coming after the current row) and return value to current row
-- LEAD (expr, offset, default)
-- expr = expression to compute from leading row
-- offset = index of the leading row relative to the current row
-- default = value to return if the <offset> points to a row beyond partition range
-----------------------------------

#1. display previous salary 
select department_id, salary, lag(salary) over(partition by department_id order by salary) as prev_salary from employees;
 
#2. display previous salary with offset 2 and default value 200
select department_id, salary, lag(salary, 2, 200) over(order by salary) as prev_salary from employees;

#3. lag - display the previous salary and difference in salary dept wise ,order by salary
select department_id, salary,lag(salary) over(partition by department_id order by salary desc) as prev_sal, (salary - lag(salary) over(partition by department_id order by salary desc)) as diff_sal from employees;


#4. lead- next value
select department_id, salary, lead(salary) over(order by salary) as next_salary from employees;

#5.find the next salary milestone dept wise order by salary
select department_id, salary, lead(salary) over( partition by department_id  order by salary) as next_salary from employees;
  
-- --------------------------------------------------------------------
# first_value last_value  range should be unbouded following and unbounded preceding
   -- --------------------------------------------------------------------
 
#1.display the first value of salary deptwise 
select salary, department_id,  first_value(first_name) over(partition by department_id order by salary desc) from employees;

select *, salary, department_id,  first_value(first_name) over(partition by department_id order by salary desc) as `first_value` from employees where department_id=100;
#2. find the name of all employees with their  start day of the first job.use job_history table 
select distinct e.employee_id, e.first_name , salary, e.department_id,  first_value(start_date) over(partition by j.employee_id order by start_date) as early from employees e join job_history j on e.employee_id = j.employee_id;

#3.last_value-- find recent job of the employee- frame partition - from where to where range of rows  
select distinct e.employee_id, e.first_name , salary, e.department_id, last_value(start_date) over(partition by j.employee_id order by start_date range between unbounded preceding and unbounded following) as latest from employees e join job_history j on e.employee_id = j.employee_id;

# 4. display the employee first name who gets highest pay dept wise - last_value
select distinct e.employee_id, e.first_name , salary, e.department_id, last_value(first_name) over(partition by employee_id order by salary range between unbounded preceding and unbounded following) as latest from employees e;

-- --------------------------------------------------------------------
-- nth_value 
select department_id. first_name, salary, nth_value(first_name, 2) over(partition by department_id order by salary desc) as `2nd_highest` from employees;
-- --------------------------------------------------------------------
 #1.- 2nd highest paid employee dept wise 
select department_id, first_name, salary, nth_value(first_name, 2) over(partition by department_id order by salary desc) as `2nd_highest` from employees;
  
 -- -------------------------------------------------------------------- 
  -- percent_rank - (rank-1)/(totalrows-1)--70th percentile is the value below which 70% of the values fall.
  -- cume_dist- cumaltive distribution of a record occupied in the total set 
  -- no of rows whose values  are less than or equal to rows value/total rows  0 to 1
  -- --------------------------------------------------------------------
 # 1. percent rank and cume dist for dept id =50
select salary, percent_rank() over(order by salary) as `percent_rank`, cume_dist() over(order by salary) * 100 as `cume_dist` from  employees where department_id = 50;

select salary, percent_rank() over(order by salary) as `percent_rank`, cume_dist() over(order by salary) * 100 as `cume_dist` from  employees where department_id = 100;
 -- --------------------------------------------------------------------
# Window function with aggregate functions      
-- --------------------------------------------------------------------     
# 1. display the count of employees in each dept 
select department_id, count(*) over(partition by department_id) from employees;
#2.print the sum of salary dept wise 
select department_id, first_name, sum(salary) over(partition by department_id) as t_salary, salary from employees;

#3 print the sum of salary dept 30 and 40
 select department_id, first_name, sum(salary) over(partition by department_id) as t_salary, salary from employees where department_id in (30, 40);

# variations with rows and range 
select department_id, first_name, 
sum(salary) over(partition by department_id) as t_salary, 
sum(salary) over(partition by department_id rows between unbounded preceding and current row) as t11_salary, 
sum(salary) over(partition by department_id rows between 1 preceding and 1 following) as `11_salary`,
sum(salary) over(partition by department_id rows between unbounded preceding and unbounded following) as `uu_salary`,
sum(salary) over(partition by department_id rows between 2 preceding and 3 following) as `23_salary`,
salary 
from employees where department_id=50;

select department_id, first_name, 
sum(salary) over(partition by department_id) as t_salary, 
sum(salary) over(partition by department_id rows between unbounded preceding and current row) as t11_salary, 
sum(salary) over(partition by department_id rows between 1 preceding and 1 following) as `11_salary`,
sum(salary) over(partition by department_id rows between unbounded preceding and unbounded following) as `uu_salary`,
sum(salary) over(partition by department_id rows between 2 preceding and 3 following) as `23_salary`,
salary 
from employees where department_id=50;

select department_id, first_name, 
sum(salary) over(partition by department_id) as t_salary, 
sum(salary) over(partition by department_id range between unbounded preceding and current row) as t11_salary, 
sum(salary) over(partition by department_id order by salary range  between 200 preceding and 200 following) as `11_salary`,
salary 
from employees where department_id=100;


  # =============

 
     
                  
#4.-- running total
select salary, department_id, first_name, 
sum(salary) over (partition by department_id rows between unbounded preceding and current row) as running_total
 from employees; 
       
#5. moving average 
 select salary, department_id, first_name, 
 avg(salary) over (partition by department_id rows between unbounded preceding and current row) as moving_avg
 from employees; 
  
  -- CTE - Common Table Expression
   # Find the minimum and maximum of the avg salary among all the depts
    with minmax as (select department_id, avg(salary) as avg_sal from employees group by department_id)
    select min(avg_sal) , max(avg_sal) from minmax;
   
   # Find the big and small departments wrt to the count of employees 
  with count1 as (select department_id, count(*) as c_emp from employees group by department_id)
    select min(c_emp) , max(c_emp) from count1;
    -- Find percentile rank  of every dept by total salary.
    with perc_rank as (select department_id, sum(salary) as t_sal from employees group by department_id)
    select t_sal, percent_rank() over(order by t_sal) from perc_rank;
 
   
  -- find employee details who earns 2nd highest salary dept wise 
 with rank2 as (select *, dense_rank() over(partition by department_id order by salary desc) as rank_sal from employees)
 select * from rank2 where rank_sal=2;
  

 
     
    
    
    
    
   
   
   
   
   
   
-- Refer Performance metrics on misc.js file   
explain format=JSON select department_id, first_name, 
sum(salary) over(partition by department_id) as t_salary, 
sum(salary) over(partition by department_id rows between unbounded preceding and current row) as t11_salary, 
sum(salary) over(partition by department_id rows between 1 preceding and 1 following) as `11_salary`,
sum(salary) over(partition by department_id rows between unbounded preceding and unbounded following) as `uu_salary`,
sum(salary) over(partition by department_id rows between 2 preceding and 3 following) as `23_salary`,
salary 
from employees where department_id=50;   

explain format=JSON select * from employees where department_id=50;   

explain format=JSON select count(*) as emp_count, sum(salary) as total_sal from employees where department_id=50;    
   
   
   
                  
                  
                  
                  
  
           
          
          
          
          
          
          
          




