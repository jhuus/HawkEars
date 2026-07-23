ALTER TABLE review_queue ADD COLUMN random_sample_size INTEGER
    CHECK (random_sample_size IS NULL OR random_sample_size > 0);

ALTER TABLE review_queue ADD COLUMN random_seed INTEGER
    CHECK (random_seed IS NULL OR random_seed >= 0);
