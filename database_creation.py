import sqlite3

# Function to create the database and tables
def create_database():
    # Connect to SQLite Database
    conn = sqlite3.connect('chemical_text.db')
    cursor = conn.cursor()


    # Create raw text table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chemical_text (
            chemical_name TEXT,
            chemical_id INTEGER,
            extracted_text TEXT,
            FOREIGN KEY (chemical_id) REFERENCES chemicals (chemical_id)
        )
    ''')

    # Commit changes and close the connection
    conn.commit()
    conn.close()

# Create the database and tables
create_database()