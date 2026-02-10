import sqlite3

conn = sqlite3.connect("chat_history.db")
cursor = conn.cursor()

cursor.execute("DROP TABLE IF EXISTS chat_history")

cursor.execute("""
CREATE TABLE IF NOT EXISTS chat_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT,
    title TEXT,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
conn.close()
