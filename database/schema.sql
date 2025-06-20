-- database/schema.sql

DROP TABLE IF EXISTS users CASCADE;
DROP TABLE IF EXISTS books CASCADE;
DROP TABLE IF EXISTS ratings CASCADE;

CREATE TABLE users (
    user_id INTEGER PRIMARY KEY,
    location TEXT,
    age INTEGER
);

CREATE TABLE books (
    isbn VARCHAR PRIMARY KEY,
    title TEXT,
    author TEXT,
    year INTEGER,
    publisher TEXT,
    image_s TEXT,
    image_m TEXT,
    image_l TEXT
);

CREATE TABLE ratings (
    user_id INTEGER REFERENCES users(user_id) ON DELETE CASCADE,
    isbn VARCHAR REFERENCES books(isbn) ON DELETE CASCADE,
    rating INTEGER,
    PRIMARY KEY (user_id, isbn)
);
