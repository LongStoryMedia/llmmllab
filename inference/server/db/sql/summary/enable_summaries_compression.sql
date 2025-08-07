-- Enable compression on summaries hypertable
ALTER TABLE summaries SET (timescaledb.compress, timescaledb.compress_segmentby = 'conversation_id');

