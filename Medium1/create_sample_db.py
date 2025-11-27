import sqlite3

# Connect to database (will create if not exists)
conn = sqlite3.connect("data.db")
cursor = conn.cursor()

# --------------------------
# Create Tables
# --------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS Sales (
    ID INTEGER PRIMARY KEY,
    Date TEXT,
    Product TEXT,
    Quantity INTEGER,
    Sales REAL
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Customers (
    CustomerID INTEGER PRIMARY KEY,
    Name TEXT,
    Region TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS Products (
    ProductID INTEGER PRIMARY KEY,
    Product TEXT,
    Category TEXT
)
""")

# --------------------------
# Insert Sample Data
# --------------------------
sales_data = [
    ('2025-11-01', 'Product A', 10, 100.0),
    ('2025-11-02', 'Product B', 5, 50.0),
    ('2025-11-03', 'Product C', 8, 80.0),
    ('2025-11-04', 'Product A', 12, 120.0),
    ('2025-11-05', 'Product B', 7, 70.0)
]

customer_data = [
    (1, 'Alice', 'North'),
    (2, 'Bob', 'South'),
    (3, 'Charlie', 'East')
]

product_data = [
    (1, 'Product A', 'Category 1'),
    (2, 'Product B', 'Category 2'),
    (3, 'Product C', 'Category 1')
]

cursor.executemany("INSERT INTO Sales (Date, Product, Quantity, Sales) VALUES (?, ?, ?, ?)", sales_data)
cursor.executemany("INSERT INTO Customers (CustomerID, Name, Region) VALUES (?, ?, ?)", customer_data)
cursor.executemany("INSERT INTO Products (ProductID, Product, Category) VALUES (?, ?, ?)", product_data)

conn.commit()
conn.close()

print("✅ Sample database 'data.db' created with tables and sample data!")
