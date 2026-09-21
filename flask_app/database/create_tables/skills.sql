CREATE TABLE IF NOT EXISTS skills (
    skill_id       SERIAL PRIMARY KEY,
    experience_id  INT REFERENCES experiences(experience_id) ON DELETE CASCADE,
    name           VARCHAR(100) NOT NULL,
    skill_level    INT DEFAULT 1,
    embedding      vector(1536) DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS skills_embedding_idx ON skills USING ivfflat (embedding vector_cosine_ops);