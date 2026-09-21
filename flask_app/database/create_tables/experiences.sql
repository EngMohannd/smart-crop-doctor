CREATE TABLE IF NOT EXISTS experiences (
    experience_id  SERIAL PRIMARY KEY,
    position_id    INT REFERENCES positions(position_id) ON DELETE CASCADE,
    name           VARCHAR(100) NOT NULL,
    description    TEXT DEFAULT NULL,
    start_date     DATE DEFAULT NULL,
    end_date       DATE DEFAULT NULL,
    embedding      vector(1536) DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS experiences_embedding_idx ON experiences USING ivfflat (embedding vector_cosine_ops);