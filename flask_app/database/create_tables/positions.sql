CREATE TABLE IF NOT EXISTS positions (
    position_id    SERIAL PRIMARY KEY,
    inst_id        INT REFERENCES institutions(inst_id) ON DELETE CASCADE,
    title          VARCHAR(100) NOT NULL,
    start_date     DATE DEFAULT NULL,
    end_date       DATE DEFAULT NULL,
    responsibilities TEXT DEFAULT NULL,
    embedding      vector(1536) DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS positions_embedding_idx ON positions USING ivfflat (embedding vector_cosine_ops);