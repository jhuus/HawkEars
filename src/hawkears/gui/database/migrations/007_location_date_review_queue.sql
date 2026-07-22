ALTER TABLE review_queue ADD COLUMN max_per_location_date INTEGER
    CHECK (max_per_location_date IS NULL OR max_per_location_date > 0);
