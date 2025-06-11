-- Enable compression on conversations hypertable
ALTER TABLE conversations SET (timescaledb.compress, timescaledb.compress_segmentby = 'user_id');

