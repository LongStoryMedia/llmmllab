-- Count system-defined model profiles
SELECT
  COUNT(*)
FROM
  model_profiles
WHERE
  type = 1
