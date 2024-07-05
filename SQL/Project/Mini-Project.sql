use hr;

/*
Part - A
Cricket Players Scrores Data
*/
# 1.	Import the csv file to a table in the database.

-- Import CSV file to table with the help of Table Data Import Wizard in Workbench
-- Store the data into table name called test_batting

# 2.	Remove the column 'Player Profile' from the table.
alter table test_batting drop column `Player Profile`;

desc test_batting;
-- to verify the column is removed and current stucture of the table
# 3.	Extract the country name and player names from the given data and store it in separate columns for further usage.
select trim(
        substring_index(Player, '(', 1)
    ) AS name, substring_index(
        substring_index(Player, '(', - 1), ')', 1
    ) AS country
from test_batting;
-- 2913 Records Returned
alter table test_batting
add column player_name varchar(50) generated always as (
    trim(
        substring_index(Player, '(', 1)
    )
) stored,
add column country varchar(15) generated always as (
    substring_index(
        substring_index(Player, '(', - 1),
        ')',
        1
    )
) stored;
-- 2913 Rows affected and player_name and country columns are added.

# 4.	From the column 'Span' extract the start_year and end_year and store them in separate columns for further usage.
select span, substring_index(span, '-', 1) AS start_year, substring_index(span, '-', - 1) AS end_year
from test_batting;
-- 2913 Records Returned
alter table test_batting
add column start_year int generated always as (substring_index(span, '-', 1)) stored,
add column end_year int generated always as (
    substring_index(span, '-', - 1)
) stored;
-- 2913 Rows affected and start_year and end_year columns are added.

# 5.	The column 'HS' has the highest score scored by the player so far in any given match.
#		The column also has details if the player had completed the match in a NOT OUT status.
#		Extract the data and store the highest runs and the NOT OUT status in different columns.
select
    no,
    hs,
    substring_index(hs, "*", 1) as highest_runs,
    case hs
        when hs like "%*" then "Not Out"
        else "Out"
    end as not_out_status
from test_batting;
-- 2913 Records Returned
alter table test_batting
add column highest_runs int generated always as (substring_index(hs, "*", 1)) stored,
add column not_out_status varchar(10) generated always as (
    case
        when hs like "%*" then "Not Out"
        else "Out"
    end
) stored;
-- 2913 Rows affected and highest_runs and not_out_status columns are added.

# 6.	Using the data given, considering the players who were active in the year of 2019,
#		create a set of batting order of best 6 players using the selection criteria of those
#  		who have a good average score across all matches for India.
select *
from test_batting
where
    2019 between start_year and end_year
order by avg desc
limit 6;
-- OUT PUT: Player Names - Abis Ali, KR Patterson, DJ Mitchell, TA Blundell, MA Agarwal, SPD Smith

#7.	Using the data given, considering the players who were active in the year of 2019,
#	create a set of batting order of best 6 players using the selection criteria of those
#	who have the highest number of 100s across all matches for India.
select *
from test_batting
where
    2019 between start_year and end_year
order by `100` desc
limit 6;
-- output : Player names - AM Amla, V Kohli, SPD Smith, DA Warner, KS Williamson, LRPL Taylor

# 8.	Using the data given, considering the players who were active in the year of 2019,
# 		create a set of batting order of best 6 players using 2 selection criteria of your own for India.
# 		considering - highest runs and innings played by the player should more than 100 as selection criteria
select *
from test_batting
where (
        2019 between start_year and end_year
    )
    and inn > 100
order by highest_runs desc
limit 6;
-- OUT PUT: Player Names - DA Warner, HM Amla, Azhar. Ali, LRPL Taylor, BA Strokes, JE Root

# 9.	Create a View named ‘Batting_Order_GoodAvgScorers_SA’ using the data given, considering the players who
#		were active in the year of 2019, create a set of batting order of best 6 players using the selection
#		criteria of those who have a good average score across all matches for South Africa.
create view Batting_Order_GoodAvgScorers_SA as
select *
from test_batting
where
    2019 between start_year and end_year
order by `100` desc
limit 6;

# 10.	Create a View named ‘Batting_Order_HighestCenturyScorers_SA’ Using the data given, considering the players
#		who were active in the year of 2019, create a set of batting order of best 6 players using the selection
#		criteria of those who have highest number of 100s across all matches for South Africa.
create view Batting_Order_HighestCenturyScorers_SA as
select *
from test_batting
where
    2019 between start_year and end_year
order by `100` desc
limit 6;

# 11.	Using the data given, Give the number of player_played for each country.
select count(distinct player_name) as players_played, country
from test_batting
group by
    country;
-- Rows count: 27

# 12.	Using the data given, Give the number of player_played for Asian and Non-Asian continent
-- Asian Contenent coutries available - 'INDIA', 'AFG', 'PAK', 'SL', 'BDESH'
with
    continent_of_country as (
        select
            country,
            player_name,
            case
                when country in (
                    'INDIA',
                    'AFG',
                    'PAK',
                    'SL',
                    'BDESH'
                ) then 'Asian'
                else 'Non-Asian'
            end as continent
        from test_batting
    )
select continent, count(distinct player_name)
from continent_of_country
group by
    continent;
-- OUTPUT: Asian continent - 751, and Non-Asian - 2148

/*  PART - B   */