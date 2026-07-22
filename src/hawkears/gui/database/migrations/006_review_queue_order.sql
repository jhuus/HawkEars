ALTER TABLE review_queue ADD COLUMN review_order TEXT NOT NULL DEFAULT 'queue'
    CHECK (review_order IN ('queue', 'score', 'chronological'));
