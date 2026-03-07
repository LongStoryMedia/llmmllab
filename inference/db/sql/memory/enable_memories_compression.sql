-- Enable compression on memories hypertable
ALTER TABLE memories SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id, source');

