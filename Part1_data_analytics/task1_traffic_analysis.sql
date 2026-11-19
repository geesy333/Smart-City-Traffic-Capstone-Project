-- ==========================================
-- TASK 1.1: Data Verification
-- ==========================================
SELECT COUNT(*) AS total_rows FROM traffic;

-- ==========================================
-- TASK 1.2: Annual Traffic Trends (2012–2017)
-- ==========================================
WITH yearly_stats AS (
    SELECT 
        SUBSTR(date_time, 1, 4) AS year,
        COUNT(*) AS hours_recorded,
        SUM(traffic_volume) AS total_traffic_volume,
        ROUND(AVG(traffic_volume), 2) AS avg_hourly_volume
    FROM traffic
    WHERE SUBSTR(date_time, 1, 4) IN ('2012', '2013', '2014', '2015', '2016', '2017')
    GROUP BY year
)
SELECT 
    year,
    hours_recorded,
    total_traffic_volume,
    avg_hourly_volume,
    total_traffic_volume - LAG(total_traffic_volume) OVER (ORDER BY year) AS yoy_total_change,
    ROUND(
        (total_traffic_volume - LAG(total_traffic_volume) OVER (ORDER BY year)) * 100.0 / 
        LAG(total_traffic_volume) OVER (ORDER BY year), 
    2) AS yoy_pct_change
FROM yearly_stats
ORDER BY year;

-- ==========================================
-- TASK 1.3: Holiday Temperature Patterns (2015–2017)
-- ==========================================
SELECT 
    SUBSTR(date_time, 1, 4) AS year,
    holiday,
    COUNT(*) AS record_count,
    ROUND(AVG(temp), 2) AS avg_temp_kelvin,
    ROUND(AVG(temp) - 273.15, 2) AS avg_temp_celsius,
    ROUND(AVG(traffic_volume), 2) AS avg_traffic_volume
FROM traffic
WHERE SUBSTR(date_time, 1, 4) IN ('2015', '2016', '2017')
  AND holiday IN ('New Years Day', 'Labor Day')
GROUP BY year, holiday
ORDER BY holiday, year;