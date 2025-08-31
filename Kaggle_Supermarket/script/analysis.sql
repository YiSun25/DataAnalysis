USE supermarket;

-- 每天的销售趋势
SELECT 
  date, 
  ROUND(SUM(total), 2) AS daily_sales
FROM sales
GROUP BY date
ORDER BY date;

-- 产品线销售额排行（TOP N）
SELECT 
  product_line, 
  ROUND(SUM(total), 2) AS total_sales
FROM sales
GROUP BY product_line
ORDER BY total_sales DESC
LIMIT 5; -- TOP 5

-- 支付方式使用比例
SELECT 
  payment, 
  COUNT(*) AS payment_count
FROM sales
GROUP BY payment
ORDER BY payment_count DESC;


-- 各城市销售表现
SELECT 
  city, 
  ROUND(SUM(total), 2) AS city_sales
FROM sales
GROUP BY city
ORDER BY city_sales DESC;


-- 销售评分分布（rating）
SELECT 
  FLOOR(rating) AS rating_group,
  COUNT(*) AS num_sales
FROM sales
GROUP BY rating_group
ORDER BY rating_group;





