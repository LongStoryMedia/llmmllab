-- Enable compression for message_embeddings table
ALTER TABLE message_embeddings SET (timescaledb.compress, timescaledb.compress_segmentby = 'message_id,chunk_index');

