-- Identify conversion IDs with a coherent user and conversion time.
CREATE TEMP VIEW reported_conversions AS
SELECT conversion_id,
       min(user_id) AS user_id,
       min(conversion_time) AS conversion_time,
       count(DISTINCT user_id) AS user_count,
       count(DISTINCT conversion_time) AS time_count
FROM impressions
WHERE has_conversion = 1
GROUP BY conversion_id;

-- The complete chosen lookback and its endpoint must both lie within observed time.
CREATE TEMP VIEW conversion_windows AS
SELECT r.conversion_id, r.user_id, r.conversion_time,
       CASE WHEN r.user_count <> 1 OR r.time_count <> 1 THEN 'ambiguous'
            WHEN r.conversion_time < b.first_touch + s.lookback_seconds THEN 'left_boundary'
            WHEN r.conversion_time > b.last_touch THEN 'right_boundary'
            ELSE 'within_bounds' END AS status
FROM reported_conversions r
CROSS JOIN observation_bounds b
CROSS JOIN settings s;

-- Restrict credit to strictly pre-conversion impressions in the source-linked path.
CREATE TEMP TABLE linked_touches AS
SELECT i.conversion_id, i.campaign, i.touch_time
FROM impressions i
JOIN conversion_windows w USING (conversion_id)
CROSS JOIN settings s
WHERE i.has_conversion = 1 AND w.status = 'within_bounds'
  AND i.touch_time >= w.conversion_time - s.lookback_seconds
  AND i.touch_time < w.conversion_time;

-- Anchor sensitivity to the same IDs and clip the path at any preceding coherent
-- conversion for the user. Equal conversion times cannot be ordered reliably.
CREATE TEMP VIEW conversion_moments AS
SELECT user_id, conversion_time,
       count(*) AS ids_at_moment,
       lag(conversion_time) OVER (PARTITION BY user_id ORDER BY conversion_time) AS previous_time
FROM reported_conversions
WHERE user_count = 1 AND time_count = 1
GROUP BY user_id, conversion_time;

CREATE TEMP VIEW same_user_anchors AS
SELECT w.conversion_id, w.user_id, w.conversion_time, m.previous_time, m.ids_at_moment
FROM conversion_windows w
JOIN conversion_moments m USING (user_id, conversion_time)
WHERE w.conversion_id IN (SELECT conversion_id FROM linked_touches);

CREATE TEMP VIEW nearby_touches AS
SELECT a.conversion_id, i.campaign, i.touch_time
FROM same_user_anchors a
JOIN impressions i ON i.user_id = a.user_id
CROSS JOIN settings s
WHERE i.touch_time >= a.conversion_time - s.lookback_seconds
  AND i.touch_time < a.conversion_time
  AND i.touch_time > coalesce(a.previous_time, -1);

CREATE TEMP TABLE qualifying_touches AS
SELECT scope, conversion_id, campaign, touch_time,
       min(touch_time) OVER path AS first_time,
       max(touch_time) OVER path AS last_time,
       count(*) OVER path AS touch_count,
       count(*) OVER timestamp_tie AS tie_count
FROM (
    SELECT 'source_linked' AS scope, * FROM linked_touches
    UNION ALL
    SELECT 'same_user' AS scope, * FROM nearby_touches
)
WINDOW path AS (PARTITION BY scope, conversion_id),
       timestamp_tie AS (PARTITION BY scope, conversion_id, touch_time);

-- Each method assigns one credit per eligible conversion, splitting timestamp ties.
CREATE TEMP VIEW campaign_credits AS
SELECT scope, campaign,
       count(*) AS touches,
       count(DISTINCT conversion_id) AS paths,
       sum(CASE WHEN touch_time = first_time THEN 1.0 / tie_count ELSE 0 END) AS first_touch,
       sum(CASE WHEN touch_time = last_time THEN 1.0 / tie_count ELSE 0 END) AS last_touch,
       sum(1.0 / touch_count) AS linear_touch
FROM qualifying_touches
GROUP BY scope, campaign;
