SELECT
    -- Selecting PostalCode from Customer table
    C.PostalCode AS zip_code,  
    -- Selecting latitude and longitude from Zips table
    Z.Lat AS latitude,
    Z.Lng AS longitude,
    -- Counting the number of orders for each zip code
    COUNT(O.OrderId) AS order_count,
    -- Haversine formula, to find the distance between two points on a sphere, in this case, thats Earth
    -- dist = 2 radius (earth's radius = 6371) × asin ((hav(Δlat) + cos(lat_1) × cos(lat_2) × hav(Δlng)^(1/2))
    -- Δlat and Δlng are the differences in latitude and longitude between the two points
    -- hav(θ)=sin^2(θ/2)
    ( 6371 * acos( cos( radians(42.3736) ) * cos( radians(Z.Lat) ) 
      * cos( radians(Z.Lng) - radians(-71.1097) ) + sin( radians(42.3736) ) 
      * sin( radians(Z.Lat) ) ) ) AS distance_km
FROM
    -- Starting with the Orders table
    Orders O 
-- Joining Customer table using CustomerId, and and Zips table using PostalCode 
JOIN Customer C ON O.CustomerId = C.CustomerId 
JOIN Zips Z ON C.PostalCode = Z.Zip
-- Grouping by zip, latitude, and longitude
GROUP BY
    C.PostalCode, Z.Lat, Z.Lng 
-- Only including zip codes within 100 km
HAVING
    distance_km <= 100
-- Sorting by highest order count 
ORDER BY
    order_count DESC 
 -- Limit result to top 3 zip codes
LIMIT 3;