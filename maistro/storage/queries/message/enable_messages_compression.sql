-- Enable compression on messages hypertable
ALTER TABLE messages SET (timescaledb.compress, timescaledb.compress_segmentby = 'conversation_id');

