ALTER TABLE analysis_item ADD COLUMN recorded_at TEXT;
ALTER TABLE analysis_item ADD COLUMN latitude REAL
    CHECK (latitude IS NULL OR latitude BETWEEN -90 AND 90);
ALTER TABLE analysis_item ADD COLUMN longitude REAL
    CHECK (longitude IS NULL OR longitude BETWEEN -180 AND 180);
ALTER TABLE analysis_item ADD COLUMN region_code TEXT;
ALTER TABLE analysis_item ADD COLUMN location_name TEXT;
