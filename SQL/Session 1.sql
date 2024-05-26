 create database if not exists apr;
 use apr;
 create table if not exists student(sid int, name varchar(20), marks int);
 insert into student(sid, name, marks) values(1,"Sam", 40);
 -- comment line
 # commented line 
 
/*
multi lined comments
*/

desc student;

insert into student values (12, "Joe 1", 40), (13,"John 1", 50), (14, "Arun 1", 70), (15, "Bhanu 1", 80);
insert into student values(8, "Chitra", null);
insert into student(sid, marks) values(9,90);

select * from student;

select * from student  where name like '%a_';
-- it gives the records matching student name where name contains char a as second last.
select * from student where marks like '4%';
-- retrive the records whose marks start with 4

select * from student order by 2 limit 4,4;  
-- here limit 4,4 describes from where and how many records have to be retrived,
-- first 4 defines the offset from which index has to read, and second 4 defines the how many records to get
-- order by second column to define it its given as order by 2.

select distinct marks from student;
-- select distinct (marks, sid) from student;
select distinct marks, name from student;
select distinct (marks), sid from student;

# set sql_safe_updates = 0;  this will disable the safe execution during the delete and update operations

delete from student where marks < 45;

update student set marks = marks + 10 where sid > 4;


alter table student add column phone int, add column address varchar(20) after sid, add column sex int first;
alter table student drop column phone;
alter table student rename column sid to student_id;
alter table student change name student_name varchar(30);
alter table student modify student_name varchar(50);

alter table student rename column student_name to name, add column phone bigint after name;

desc student;
 

 