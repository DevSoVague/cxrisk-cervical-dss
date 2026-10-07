-- CxRisk clinician frontend: Postgres schema for DB mode
--
-- Reconstructed from the queries in frontend/app.R (no original DDL file was
-- kept with the project). It covers exactly the tables and columns the app
-- reads and writes; anything else in the original hosted database is out of
-- scope. Demo mode does not need a database at all.
--
-- Load with:  psql "$DATABASE_URL" -f frontend/schema.sql
-- Requires Postgres 13+ (gen_random_uuid is built in) and pgcrypto only if you
-- want to create bcrypt hashes in SQL (see the example at the bottom).

CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE app_user (
    user_id        uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email          text NOT NULL UNIQUE,
    -- Must be a bcrypt hash ($2a$/$2b$...). The app verifies logins with
    -- bcrypt::checkpw(); plaintext passwords will never match.
    password_hash  text NOT NULL,
    -- 'physician' and 'nurse' map to the app's clinician role; anything else
    -- maps to admin.
    role           text NOT NULL CHECK (role IN ('physician', 'nurse', 'admin')),
    is_active      boolean NOT NULL DEFAULT TRUE
);

CREATE TABLE patient (
    patient_id     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    first_name     text NOT NULL,
    last_name      text NOT NULL,
    dob            date,
    phone_contact  text,
    address        text,
    email          text,
    is_active      boolean NOT NULL DEFAULT TRUE
);

CREATE TABLE encounter (
    encounter_id   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id     uuid NOT NULL REFERENCES patient (patient_id),
    clinician_id   uuid NOT NULL REFERENCES app_user (user_id),
    encounter_date date NOT NULL DEFAULT CURRENT_DATE
);

CREATE TABLE risk_assessment (
    assessment_id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    encounter_id                 uuid NOT NULL REFERENCES encounter (encounter_id),
    age_at_assessment            integer CHECK (age_at_assessment BETWEEN 10 AND 120),
    first_sexual_intercourse_age integer,
    num_sexual_partners          integer CHECK (num_sexual_partners >= 0),
    num_pregnancies              integer CHECK (num_pregnancies >= 0),
    smokes                       boolean NOT NULL DEFAULT FALSE,
    smokes_years                 numeric CHECK (smokes_years >= 0),
    smokes_packs_year            numeric CHECK (smokes_packs_year >= 0),
    hormonal_contraceptives      boolean NOT NULL DEFAULT FALSE,
    hc_years                     numeric CHECK (hc_years >= 0),
    iud                          boolean NOT NULL DEFAULT FALSE,
    iud_years                    numeric CHECK (iud_years >= 0),
    std                          boolean NOT NULL DEFAULT FALSE,
    std_count                    integer CHECK (std_count >= 0),
    dx_cancer                    boolean NOT NULL DEFAULT FALSE,
    dx_cin                       boolean NOT NULL DEFAULT FALSE,
    dx_hpv                       boolean NOT NULL DEFAULT FALSE
);

CREATE TABLE dss_recommendation (
    recommendation_id       uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    assessment_id           uuid NOT NULL REFERENCES risk_assessment (assessment_id),
    predicted_probability   numeric NOT NULL
                            CHECK (predicted_probability >= 0 AND predicted_probability <= 1),
    -- stored lowercase by the app, displayed with initcap()
    risk_level              text NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    -- e.g. 'Routine recall', 'Expedited HPV/Pap testing',
    -- 'Referral for further evaluation'
    recommendation_category text NOT NULL,
    model_version           text,
    created_at              timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX ON encounter (patient_id);
CREATE INDEX ON risk_assessment (encounter_id);
CREATE INDEX ON dss_recommendation (assessment_id);

-- Example: create a login (replace the placeholders; never commit real ones).
-- INSERT INTO app_user (email, password_hash, role)
-- VALUES ('<email>', crypt('<password>', gen_salt('bf')), 'physician');
