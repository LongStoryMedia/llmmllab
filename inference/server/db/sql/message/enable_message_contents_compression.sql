-- Enable compression for message_contents
ALTER TABLE message_contents SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id');

