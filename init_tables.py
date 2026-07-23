"""
数据库初始化脚本 — SQLite 版本
运行方式: python init_tables.py
"""
import sqlite3
import sys

DB_PATH = "psychology_assistant.db"

TABLES = {
    "user_account": """
        CREATE TABLE IF NOT EXISTS user_account (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            phone TEXT,
            password TEXT,
            nickname TEXT,
            avatar TEXT DEFAULT '/static/7.jpeg',
            gender TEXT DEFAULT '女',
            sign TEXT DEFAULT '',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """,
    "sleep_records": """
        CREATE TABLE IF NOT EXISTS sleep_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            score INTEGER DEFAULT 0,
            duration INTEGER DEFAULT 0,
            bedtime TEXT,
            waketime TEXT,
            deep_sleep INTEGER DEFAULT 0,
            light_sleep INTEGER DEFAULT 0,
            rem_sleep INTEGER DEFAULT 0,
            awake_time INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(user_id, date),
            FOREIGN KEY (user_id) REFERENCES user_account(id)
        )
    """,
    "emotion_records": """
        CREATE TABLE IF NOT EXISTS emotion_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date TEXT NOT NULL,
            time TEXT,
            mood TEXT NOT NULL,
            note TEXT,
            tags TEXT,
            context TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_account(id)
        )
    """,
    "alarms": """
        CREATE TABLE IF NOT EXISTS alarms (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            time TEXT NOT NULL,
            label TEXT,
            repeat_type TEXT DEFAULT 'weekdays',
            smart_wake INTEGER DEFAULT 0,
            ringtone TEXT,
            enabled INTEGER DEFAULT 1,
            type TEXT DEFAULT 'wakeup',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_account(id)
        )
    """,
    "devices": """
        CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            device_type TEXT,
            name TEXT,
            brand TEXT,
            battery INTEGER DEFAULT 0,
            connection_status TEXT DEFAULT 'disconnected',
            last_sync TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_account(id)
        )
    """,
    "user_settings": """
        CREATE TABLE IF NOT EXISTS user_settings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL UNIQUE,
            theme TEXT DEFAULT 'light',
            white_noise TEXT DEFAULT 'none',
            sleep_reminder INTEGER DEFAULT 1,
            smart_wake INTEGER DEFAULT 1,
            snore_detection INTEGER DEFAULT 0,
            notification INTEGER DEFAULT 1,
            bedtime_reminder INTEGER DEFAULT 1,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_account(id)
        )
    """,
    "conversations": """
        CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            title TEXT DEFAULT '新对话',
            message_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES user_account(id)
        )
    """,
    "messages": """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT,
            emotion TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
    """
}


def init_database():
    print("=" * 60)
    print("  Lovelux Database Initialization (SQLite)")
    print("=" * 60)

    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        # Enable foreign keys
        cursor.execute("PRAGMA foreign_keys = ON")
        print(f"[OK] Connected to SQLite: {DB_PATH}")
    except Exception as e:
        print(f"[FAIL] Cannot connect to SQLite: {e}")
        sys.exit(1)

    success_count = 0
    for table_name, sql in TABLES.items():
        try:
            cursor.execute(sql)
            conn.commit()
            print(f"[OK] Table '{table_name}' ready.")
            success_count += 1
        except Exception as e:
            print(f"[FAIL] Table '{table_name}' error: {e}")
            conn.rollback()

    cursor.close()
    conn.close()

    print("-" * 60)
    print(f"  Done: {success_count}/{len(TABLES)} tables created successfully.")
    print("=" * 60)


if __name__ == "__main__":
    init_database()
