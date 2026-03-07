-- Count dynamic tools for a specific user
SELECT COUNT(*) as total_count
FROM dynamic_tools
WHERE user_id = $1;
