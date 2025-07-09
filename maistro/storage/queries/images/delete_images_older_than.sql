-- Delete images older than a certain date
DELETE FROM images
WHERE created_at < $1
