const express = require('express');
const { ApolloServer, gql } = require('apollo-server-express');
const { Pool } = require('pg');
const mysql = require('mysql2/promise');
const { MongoClient } = require('mongodb');
const dotenv = require('dotenv');
const jwt = require('jsonwebtoken');

dotenv.config();

const app = express();
app.use(express.json());

const typeDefs = gql`
  type Note {
    id: ID!
    content: String!
    resource_id: String
    timestamp: String!
    wallet_id: String!
  }
  type Query {
    getNotes(wallet_id: String!, limit: Int = 10, db_type: String!): [Note!]!
  }
  type Mutation {
    addNote(wallet_id: String!, content: String!, resource_id: String, db_type: String!): Note!
  }
`;

const resolvers = {
  Query: {
    getNotes: async (_, { wallet_id, limit, db_type }, { postgresPool, mysqlPool, mongoClient }) => {
      try {
        if (db_type === 'postgres') {
          const { rows } = await postgresPool.query(
            'SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = $1 ORDER BY timestamp DESC LIMIT $2',
            [wallet_id, limit]
          );
          return rows.map(row => ({
            id: row.id.toString(),
            content: row.content,
            resource_id: row.resource_id,
            timestamp: row.timestamp.toISOString(),
            wallet_id: row.wallet_id
          }));
        } else if (db_type === 'mysql') {
          const [rows] = await mysqlPool.query(
            'SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = ? ORDER BY timestamp DESC LIMIT ?',
            [wallet_id, limit]
          );
          return rows.map(row => ({
            id: row.id.toString(),
            content: row.content,
            resource_id: row.resource_id,
            timestamp: row.timestamp.toISOString(),
            wallet_id: row.wallet_id
          }));
        } else if (db_type === 'mongo') {
          const notes = await mongoClient.db('vial_mcp').collection('notes')
            .find({ wallet_id })
            .sort({ timestamp: -1 })
            .limit(limit)
            .toArray();
          return notes.map(note => ({
            id: note._id.toString(),
            content: note.content,
            resource_id: note.resource_id,
            timestamp: note.timestamp.toISOString(),
            wallet_id: note.wallet_id
          }));
        } else {
          throw new Error('Invalid database type');
        }
      } catch (error) {
        console.error(`Note retrieval failed in ${db_type}: ${error.message}`);
        throw new Error(`Note retrieval failed: ${error.message}`);
      }
    }
  },
  Mutation: {
    addNote: async (_, { wallet_id, content, resource_id, db_type }, { postgresPool, mysqlPool, mongoClient }) => {
      try {
        let note_id;
        if (db_type === 'postgres') {
          const { rows } = await postgresPool.query(
            'INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES ($1, $2, $3, $4) RETURNING id, content, resource_id, timestamp, wallet_id',
            [content, resource_id, new Date(), wallet_id]
          );
          return {
            id: rows[0].id.toString(),
            content: rows[0].content,
            resource_id: rows[0].resource_id,
            timestamp: rows[0].timestamp.toISOString(),
            wallet_id: rows[0].wallet_id
          };
        } else if (db_type === 'mysql') {
          const [result] = await mysqlPool.query(
            'INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES (?, ?, ?, ?)',
            [content, resource_id, new Date(), wallet_id]
          );
          note_id = result.insertId;
          const [rows] = await mysqlPool.query(
            'SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE id = ?',
            [note_id]
          );
          return {
            id: rows[0].id.toString(),
            content: rows[0].content,
            resource_id: rows[0].resource_id,
            timestamp: rows[0].timestamp.toISOString(),
            wallet_id: rows[0].wallet_id
          };
        } else if (db_type === 'mongo') {
          const result = await mongoClient.db('vial_mcp').collection('notes').insertOne({
            content,
            resource_id,
            timestamp: new Date(),
            wallet_id
          });
          note_id = result.insertedId;
          const note = await mongoClient.db('vial_mcp').collection('notes').findOne({ _id: note_id });
          return {
            id: note._id.toString(),
            content: note.content,
            resource_id: note.resource_id,
            timestamp: note.timestamp.toISOString(),
            wallet_id: note.wallet_id
          };
        } else {
          throw new Error('Invalid database type');
        }
      } catch (error) {
        console.error(`Note add failed in ${db_type}: ${error.message}`);
        throw new Error(`Note add failed: ${error.message}`);
      }
    }
  }
};

async function startServer() {
  const postgresPool = new Pool({
    host: process.env.POSTGRES_HOST,
    port: process.env.POSTGRES_DOCKER_PORT,
    user: process.env.POSTGRES_USER,
    password: process.env.POSTGRES_PASSWORD,
    database: process.env.POSTGRES_DB
  });

  const mysqlPool = await mysql.createPool({
    host: process.env.MYSQL_HOST,
    port: process.env.MYSQL_DOCKER_PORT,
    user: process.env.MYSQL_USER,
    password: process.env.MYSQL_ROOT_PASSWORD,
    database: process.env.MYSQL_DB
  });

  const mongoClient = new MongoClient(`mongodb://${process.env.MONGO_USER}:${process.env.MONGO_PASSWORD}@${process.env.MONGO_HOST}:${process.env.MONGO_DOCKER_PORT}/vial_mcp?authSource=admin`);

  await mongoClient.connect();

  const server = new ApolloServer({
    typeDefs,
    resolvers,
    context: ({ req }) => {
      const token = req.headers.authorization?.replace('Bearer ', '') || '';
      try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        return { postgresPool, mysqlPool, mongoClient, user: decoded };
      } catch (error) {
        throw new Error('Invalid token');
      }
    }
  });

  await server.start();
  server.applyMiddleware({ app, path: '/graphql' });

  app.post('/api/notes/add', async (req, res) => {
    const { wallet_id, content, resource_id, db_type } = req.body;
    const token = req.headers.authorization?.replace('Bearer ', '');
    try {
      jwt.verify(token, process.env.JWT_SECRET);
      if (db_type === 'postgres') {
        const { rows } = await postgresPool.query(
          'INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES ($1, $2, $3, $4) RETURNING id',
          [content, resource_id, new Date(), wallet_id]
        );
        res.json({ status: 'success', note_id: rows[0].id });
      } else if (db_type === 'mysql') {
        const [result] = await mysqlPool.query(
          'INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES (?, ?, ?, ?)',
          [content, resource_id, new Date(), wallet_id]
        );
        res.json({ status: 'success', note_id: result.insertId });
      } else if (db_type === 'mongo') {
        const result = await mongoClient.db('vial_mcp').collection('notes').insertOne({
          content,
          resource_id,
          timestamp: new Date(),
          wallet_id
        });
        res.json({ status: 'success', note_id: result.insertedId.toString() });
      } else {
        res.status(400).json({ error: 'Invalid database type' });
      }
    } catch (error) {
      console.error(`REST note add failed: ${error.message}`);
      res.status(500).json({ error: `Note add failed: ${error.message}` });
    }
  });

  app.post('/api/notes/read', async (req, res) => {
    const { wallet_id, limit = 10, db_type } = req.body;
    const token = req.headers.authorization?.replace('Bearer ', '');
    try {
      jwt.verify(token, process.env.JWT_SECRET);
      if (db_type === 'postgres') {
        const { rows } = await postgresPool.query(
          'SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = $1 ORDER BY timestamp DESC LIMIT $2',
          [wallet_id, limit]
        );
        res.json({ status: 'success', notes: rows });
      } else if (db_type === 'mysql') {
        const [rows] = await mysqlPool.query(
          'SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = ? ORDER BY timestamp DESC LIMIT ?',
          [wallet_id, limit]
        );
        res.json({ status: 'success', notes: rows });
      } else if (db_type === 'mongo') {
        const notes = await mongoClient.db('vial_mcp').collection('notes')
          .find({ wallet_id })
          .sort({ timestamp: -1 })
          .limit(limit)
          .toArray();
        res.json({ status: 'success', notes });
      } else {
        res.status(400).json({ error: 'Invalid database type' });
      }
    } catch (error) {
      console.error(`REST note read failed: ${error.message}`);
      res.status(500).json({ error: `Note read failed: ${error.message}` });
    }
  });

  app.listen(process.env.NODE_DOCKER_PORT, () => {
    console.log(`Node.js server running on port ${process.env.NODE_DOCKER_PORT}`);
    console.log(`GraphQL endpoint at /graphql`);
  });
}

startServer().catch(error => {
  console.error(`Server startup failed: ${error.message}`);
  process.exit(1);
});
