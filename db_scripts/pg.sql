CREATE DATABASE data;


CREATE TABLE public.users (
    id SERIAL PRIMARY KEY,
    tele_id VARCHAR(200) NOT NULL
);

INSERT INTO
    public.tele_id (text)
VALUES
    ('677169336'), ('677459330'), ('688169451');