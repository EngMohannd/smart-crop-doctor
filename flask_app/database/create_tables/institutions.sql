CREATE TABLE IF NOT EXISTS institutions (
    inst_id        SERIAL PRIMARY KEY,
    type           VARCHAR(100) NOT NULL,
    name           VARCHAR(100) NOT NULL,
    department     VARCHAR(100) DEFAULT NULL,
    address        VARCHAR(100) DEFAULT NULL,
    city           VARCHAR(100) DEFAULT NULL,
    state          VARCHAR(100) DEFAULT NULL,
    zip            VARCHAR(10)  DEFAULT NULL,
    embedding      vector(1536) DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS institutions_embedding_idx ON institutions USING ivfflat (embedding vector_cosine_ops);