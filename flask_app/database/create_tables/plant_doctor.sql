-- تفعيل إضافة المتجهات إذا لم تكن مفعلة
CREATE EXTENSION IF NOT EXISTS vector;

-- جدول توصيف أمراض النباتات
CREATE TABLE IF NOT EXISTS plant_diseases (
    disease_id SERIAL PRIMARY KEY,
    crop_name VARCHAR(100) NOT NULL,
    disease_name VARCHAR(150) NOT NULL,
    pathogen_type VARCHAR(50), -- فطر، بكتيريا، فيروس
    symptoms TEXT NOT NULL,
    preventive_measures TEXT,
    embedding VECTOR(1536)
);

-- فهرس لتسريع البحث الدلالي بالمتجهات
CREATE INDEX IF NOT EXISTS plant_diseases_embedding_idx 
ON plant_diseases USING ivfflat (embedding vector_cosine_ops);

-- جدول بروتوكولات العلاج والمبيدات وفترات الأمان
CREATE TABLE IF NOT EXISTS plant_treatments (
    treatment_id SERIAL PRIMARY KEY,
    disease_id INTEGER REFERENCES plant_diseases(disease_id) ON DELETE CASCADE,
    treatment_type VARCHAR(50) NOT NULL, -- كيميائي، عضوي، ميكانيكي
    active_ingredient VARCHAR(150),
    trade_name VARCHAR(150),
    dosage_per_liter VARCHAR(100),
    safety_interval_days INTEGER, -- فترة الأمان قبل الحصاد (PHI)
    instructions TEXT,
    embedding VECTOR(1536)
);

CREATE INDEX IF NOT EXISTS plant_treatments_embedding_idx 
ON plant_treatments USING ivfflat (embedding vector_cosine_ops);