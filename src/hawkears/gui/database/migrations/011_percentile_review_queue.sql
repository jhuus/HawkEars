ALTER TABLE review_queue ADD COLUMN percentile_points INTEGER
    CHECK (percentile_points IS NULL OR percentile_points BETWEEN 2 AND 10);
