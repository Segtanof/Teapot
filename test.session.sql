--@Block

--@Block
CREATE TABLE Users(
    id INT PRIMARY KEY AUTO_INCREMENT,
    email VARCHAR(255) NOT NULL UNIQUE,
    bio TEXT,
    country VARCHAR(2)
);

--@Block
INSERT INTO Users(email, bio, country)
VALUES (
    "hello221@world.com",
    "i am happy",
    "DE"
),
("wo2r2k@more.com", "lol", "CN"),
("ia2my2ou@me.com", "lmamo", "HK");

--@Block
SELECT email, id, country FROM Users
WHERE email LIKE "hello%"
ORDER BY id DESC
LIMIT 5;

--@Block
CREATE INDEX email_index ON Users(email);

--@Block
CREATE TABLE Rooms(
    id INT AUTO_INCREMENT,
    street VARCHAR(255),
    owner_id INT NOT NULL,
    PRIMARY KEY(id),
    FOREIGN KEY(owner_id) REFERENCES Users(id)
);

--@Block
INSERT INTO rooms(owner_id, street)
VALUES
(6, "Room 20"),
(7, "Room 15"),
(7, "Room 66");


--@Block
SELECT 
    users.id AS user_id,
    rooms.id AS room_id,
    email,
    street
FROM users
LEFT JOIN rooms
ON rooms.owner_id = users.id;

--@Block
CREATE TABLE Bookings(
    id INT AUTO_INCREMENT,
    room_id INT NOT NULL,
    guest_id INT NOT NULL,
    check_in DATETIME,
    PRIMARY KEY(id),
    FOREIGN KEY(guest_id) REFERENCES Users(id),
    FOREIGN KEY(room_id) REFERENCES Rooms(id)
);

--@Block
INSERT INTO bookings(guest_id, room_id)
VALUES
(3, 10);

--@Block
SELECT
    guest_id,
    street,
    check_in
FROM bookings
INNER JOIN rooms ON Rooms.owner_id = guest_id



--@Block
SELECT * FROM bookings