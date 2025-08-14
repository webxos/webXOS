import logging
import psycopg2
import mysql.connector
from pymongo import MongoClient
from datetime import datetime
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Manages connections to PostgreSQL, MySQL, and MongoDB for Vial MCP."""
    def __init__(self, postgres_config: dict, mysql_config: dict, mongo_config: dict):
        """Initialize DatabaseManager with database configurations.

        Args:
            postgres_config (dict): PostgreSQL connection parameters.
            mysql_config (dict): MySQL connection parameters.
            mongo_config (dict): MongoDB connection parameters.
        """
        self.postgres_config = postgres_config
        self.mysql_config = mysql_config
        self.mongo_config = mongo_config
        self.postgres_conn = None
        self.mysql_conn = None
        self.mongo_client = None
        logger.info("DatabaseManager initialized")

    def connect(self):
        """Establish connections to all databases."""
        try:
            self.postgres_conn = psycopg2.connect(**self.postgres_config)
            self.mysql_conn = mysql.connector.connect(**self.mysql_config)
            self.mongo_client = MongoClient(**self.mongo_config)
            logger.info("Connected to PostgreSQL, MySQL, and MongoDB")
        except Exception as e:
            logger.error(f"Database connection failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [DatabaseManager] Database connection failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")

    def add_note(self, wallet_id: str, content: str, resource_id: str = None, db_type: str = "postgres") -> dict:
        """Add a note to the specified database.

        Args:
            wallet_id (str): Wallet ID.
            content (str): Note content.
            resource_id (str, optional): Resource ID.
            db_type (str): Database type ('postgres', 'mysql', or 'mongo').

        Returns:
            dict: Success message with note ID.
        """
        try:
            if db_type == "postgres":
                with self.postgres_conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES (%s, %s, %s, %s) RETURNING id",
                        (content, resource_id, datetime.now(), wallet_id)
                    )
                    note_id = cursor.fetchone()[0]
                    self.postgres_conn.commit()
            elif db_type == "mysql":
                with self.mysql_conn.cursor() as cursor:
                    cursor.execute(
                        "INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES (%s, %s, %s, %s)",
                        (content, resource_id, datetime.now(), wallet_id)
                    )
                    note_id = cursor.lastrowid
                    self.mysql_conn.commit()
            elif db_type == "mongo":
                note = {
                    "content": content,
                    "resource_id": resource_id,
                    "timestamp": datetime.now(),
                    "wallet_id": wallet_id
                }
                note_id = str(self.mongo_client.vial_mcp.notes.insert_one(note).inserted_id)
            else:
                raise ValueError("Invalid database type")
            
            logger.info(f"Note {note_id} added to {db_type} for wallet {wallet_id}")
            return {"status": "success", "note_id": note_id, "wallet_id": wallet_id}
        except Exception as e:
            logger.error(f"Note add failed in {db_type}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [DatabaseManager] Note add failed in {db_type}: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Note add failed: {str(e)}")

    def get_notes(self, wallet_id: str, limit: int = 10, db_type: str = "postgres") -> dict:
        """Retrieve notes for a wallet from the specified database.

        Args:
            wallet_id (str): Wallet ID.
            limit (int): Maximum number of notes to retrieve.
            db_type (str): Database type ('postgres', 'mysql', or 'mongo').

        Returns:
            dict: List of notes.
        """
        try:
            if db_type == "postgres":
                with self.postgres_conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = %s ORDER BY timestamp DESC LIMIT %s",
                        (wallet_id, limit)
                    )
                    notes = [{"id": r[0], "content": r[1], "resource_id": r[2], "timestamp": r[3].isoformat(), "wallet_id": r[4]} for r in cursor.fetchall()]
            elif db_type == "mysql":
                with self.mysql_conn.cursor() as cursor:
                    cursor.execute(
                        "SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = %s ORDER BY timestamp DESC LIMIT %s",
                        (wallet_id, limit)
                    )
                    notes = [{"id": r[0], "content": r[1], "resource_id": r[2], "timestamp": r[3].isoformat(), "wallet_id": r[4]} for r in cursor.fetchall()]
            elif db_type == "mongo":
                notes = self.mongo_client.vial_mcp.notes.find({"wallet_id": wallet_id}).sort("timestamp", -1).limit(limit)
                notes = [{"id": str(n["_id"]), "content": n["content"], "resource_id": n["resource_id"], "timestamp": n["timestamp"].isoformat(), "wallet_id": n["wallet_id"]} for n in notes]
            else:
                raise ValueError("Invalid database type")
            
            logger.info(f"Retrieved {len(notes)} notes from {db_type} for wallet {wallet_id}")
            return {"status": "success", "notes": notes}
        except Exception as e:
            logger.error(f"Note retrieval failed in {db_type}: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [DatabaseManager] Note retrieval failed in {db_type}: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Note retrieval failed: {str(e)}")

    def close(self):
        """Close all database connections."""
        try:
            if self.postgres_conn:
                self.postgres_conn.close()
            if self.mysql_conn:
                self.mysql_conn.close()
            if self.mongo_client:
                self.mongo_client.close()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Database connection close failed: {str(e)}")
            with open("/app/errorlog.md", "a") as f:
                f.write(f"[{datetime.now().isoformat()}] [DatabaseManager] Database connection close failed: {str(e)}\n")
            raise HTTPException(status_code=500, detail=f"Database connection close failed: {str(e)}")
