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
use supply_chain;

# 1.	Company sells the product at different discounted rates. Refer actual product price in product table and selling price in the order item table. Write a query to find out total amount saved in each order then display the orders from highest to lowest amount saved.
SELECT
    `Id`,
    `OrderNumber`,
    `TotalAmount`,
    actual_price,
    amount_saved
from orders as o
    join (
        select
            `OrderId`, sum(`Quantity` * p.`UnitPrice`) as actual_price, sum(`Quantity` * oi.`UnitPrice`) as selling_price, sum(
                `Quantity` * p.`UnitPrice` - `Quantity` * oi.`UnitPrice`
            ) as amount_saved
        from orderitem as oi
            join product as p on oi.`ProductId` = p.`Id`
        GROUP BY
            oi.`OrderId`
    ) as order_sales on o.`Id` = order_sales.`OrderId`
ORDER BY order_sales.amount_saved desc;

# 2.	Mr. Kavin want to become a supplier. He got the database of "Richard's Supply" for reference. Help him to pick:
#       a. List few products that he should choose based on demand.
#       b. Who will be the competitors for him for the products suggested in above questions.
SELECT
    `ProductName`,
    `Package`,
    `CompanyName` as Compitetor,
    `Country`,
    `ContactName`
from
    product
    join (
        select `ProductId`, SUM(`Quantity`) as quantity_sold
        from orderitem
        GROUP BY
            `ProductId`
        ORDER BY quantity_sold desc
        limit 15
    ) as ondemand_products on product.`Id` = ondemand_products.`ProductId`
    JOIN supplier on supplier.`Id` = product.`SupplierId`;

# 3.	Create a combined list to display customers and suppliers details considering the following criteria
#       ●	Both customer and supplier belong to the same country
#       ●	Customer who does not have supplier in their country
#       ●	Supplier who does not have customer in their country
with
    customer_with_order as (
        SELECT
            orders.`Id` as OrderId,
            CONCAT(`FirstName`, ' ', `LastName`) AS CustomerName,
            `City`,
            `Country`
        FROM customer
            JOIN orders on customer.`Id` = orders.`CustomerId`
    ),
    product_with_supplier as (
        SELECT
            `ProductName`,
            `SupplierId`,
            `CompanyName`,
            `City`,
            `Country`,
            product.`Id`
        FROM product
            JOIN supplier on product.`SupplierId` = supplier.`Id`
    ) (
        SELECT cwo.*, pws.*, IF(
                cwo.`Country` = pws.`Country`, "Both are from Same Country", 'from Different Country'
            ) AS `Country_Of_CS_Relation`
        FROM
            orderitem as oi
            JOIN customer_with_order as cwo on oi.`OrderId` = cwo.`OrderId`
            JOIN product_with_supplier as pws on pws.`Id` = oi.`ProductId`
    );

# 4.	Every supplier supplies specific products to the customers. Create a view of suppliers and total sales
#       made by their products and write a query on this view to find out top 2 suppliers (using windows function)
#       in each country by total sales done by the products.

# 5.	Find out for which products, UK is dependent on other countries for the supply.
#       List the countries which are supplying these products in the same list.

--  End