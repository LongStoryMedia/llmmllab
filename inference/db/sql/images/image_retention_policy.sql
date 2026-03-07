-- Add or update retention policy for images data based on configuration
DO $$
BEGIN
    -- Try to add the retention policy, if it exists already, update it
    BEGIN
        PERFORM
            add_retention_policy('images', $1::interval);
    EXCEPTION
        WHEN duplicate_object THEN
            -- If policy exists, remove the old one and add the new one
            PERFORM
                remove_retention_policy('images');
                PERFORM
                    add_retention_policy('images', $1::interval);
    END;
END
$$;

