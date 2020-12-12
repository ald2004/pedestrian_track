
import sqlite3

conn = sqlite3.connect('aquarium.sqlite')
c = conn.cursor()

# records or rows in a list
records = [('1', 'Glen', '8', 'c'),
           ('2', 'Elliot', '9', 'b'),
           ('3', 'Bob', '7', 'a')]

# insert multiple records in a single query
c.executemany('INSERT INTO pedestrian(null,boxes, features,detect_result,timestamp) VALUES(?,?,?,?);', records);

print('We have inserted', c.rowcount, 'records to the table.')

# commit the changes to db
conn.commit()
# close the connection
conn.close()
