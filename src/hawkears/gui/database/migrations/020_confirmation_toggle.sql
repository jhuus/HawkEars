ALTER TABLE review_queue ADD COLUMN confirmation_enabled INTEGER
    NOT NULL DEFAULT 0 CHECK (confirmation_enabled IN (0, 1));

UPDATE review_queue
SET confirmation_enabled = 1
WHERE confirmation_scope != 'none';
