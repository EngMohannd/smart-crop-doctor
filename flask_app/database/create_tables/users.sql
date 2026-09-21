CREATE TABLE IF NOT EXISTS users (
    user_id        SERIAL PRIMARY KEY,
    email          VARCHAR(100) UNIQUE NOT NULL,
    password       VARCHAR(256) NOT NULL,
    role           VARCHAR(50) DEFAULT 'user',
    embedding      vector(1536) DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS users_embedding_idx ON users USING ivfflat (embedding vector_cosine_ops);