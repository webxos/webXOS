import com.sun.net.httpserver.HttpServer;
import graphql.schema.GraphQLSchema;
import graphql.schema.idl.RuntimeWiring;
import graphql.schema.idl.SchemaGenerator;
import graphql.schema.idl.SchemaParser;
import graphql.schema.idl.TypeDefinitionRegistry;
import graphql.ExecutionResult;
import graphql.GraphQL;
import org.json.JSONObject;
import java.io.*;
import java.net.InetSocketAddress;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.util.ArrayList;
import java.util.List;
import com.mongodb.client.MongoClients;
import com.mongodb.client.MongoClient;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import org.bson.Document;

public class JavaServer {
    private static Connection postgresConn;
    private static Connection mysqlConn;
    private static MongoClient mongoClient;
    private static GraphQL graphQL;

    public static void main(String[] args) throws Exception {
        // Load environment variables
        String postgresUrl = System.getenv("POSTGRES_URL");
        String mysqlUrl = System.getenv("MYSQL_URL");
        String mongoUrl = System.getenv("MONGO_URL");
        String jwtSecret = System.getenv("JWT_SECRET");
        int port = Integer.parseInt(System.getenv("JAVA_DOCKER_PORT"));

        // Initialize database connections
        postgresConn = DriverManager.getConnection(postgresUrl);
        mysqlConn = DriverManager.getConnection(mysqlUrl);
        mongoClient = MongoClients.create(mongoUrl);

        // Define GraphQL schema
        String schema = """
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
        """;
        TypeDefinitionRegistry typeRegistry = new SchemaParser().parse(schema);
        RuntimeWiring wiring = RuntimeWiring.newRuntimeWiring()
            .type("Query", builder -> builder.dataFetcher("getNotes", env -> {
                String walletId = env.getArgument("wallet_id");
                int limit = env.getArgument("limit");
                String dbType = env.getArgument("db_type");
                return getNotes(walletId, limit, dbType);
            }))
            .type("Mutation", builder -> builder.dataFetcher("addNote", env -> {
                String walletId = env.getArgument("wallet_id");
                String content = env.getArgument("content");
                String resourceId = env.getArgument("resource_id");
                String dbType = env.getArgument("db_type");
                return addNote(walletId, content, resourceId, dbType);
            }))
            .build();
        GraphQLSchema graphQLSchema = new SchemaGenerator().makeExecutableSchema(typeRegistry, wiring);
        graphQL = GraphQL.newGraphQL(graphQLSchema).build();

        // Start HTTP server
        HttpServer server = HttpServer.create(new InetSocketAddress(port), 0);
        server.createContext("/graphql", exchange -> {
            try {
                String requestBody = new String(exchange.getRequestBody().readAllBytes());
                JSONObject json = new JSONObject(requestBody);
                String query = json.getString("query");
                ExecutionResult result = graphQL.execute(query);
                String response = new JSONObject()
                    .put("data", result.getData())
                    .put("errors", result.getErrors())
                    .toString();
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length());
                exchange.getResponseBody().write(response.getBytes());
            } catch (Exception e) {
                String error = new JSONObject().put("error", "GraphQL request failed: " + e.getMessage()).toString();
                exchange.sendResponseHeaders(500, error.length());
                exchange.getResponseBody().write(error.getBytes());
            }
            exchange.close();
        });

        server.createContext("/api/notes/add", exchange -> {
            try {
                String requestBody = new String(exchange.getRequestBody().readAllBytes());
                JSONObject json = new JSONObject(requestBody);
                String walletId = json.getString("wallet_id");
                String content = json.getString("content");
                String resourceId = json.optString("resource_id", null);
                String dbType = json.getString("db_type");
                JSONObject result = addNote(walletId, content, resourceId, dbType);
                String response = result.toString();
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length());
                exchange.getResponseBody().write(response.getBytes());
            } catch (Exception e) {
                String error = new JSONObject().put("error", "Note add failed: " + e.getMessage()).toString();
                exchange.sendResponseHeaders(500, error.length());
                exchange.getResponseBody().write(error.getBytes());
            }
            exchange.close();
        });

        server.createContext("/api/notes/read", exchange -> {
            try {
                String requestBody = new String(exchange.getRequestBody().readAllBytes());
                JSONObject json = new JSONObject(requestBody);
                String walletId = json.getString("wallet_id");
                int limit = json.optInt("limit", 10);
                String dbType = json.getString("db_type");
                List<JSONObject> notes = getNotes(walletId, limit, dbType);
                String response = new JSONObject().put("status", "success").put("notes", notes).toString();
                exchange.getResponseHeaders().set("Content-Type", "application/json");
                exchange.sendResponseHeaders(200, response.length());
                exchange.getResponseBody().write(response.getBytes());
            } catch (Exception e) {
                String error = new JSONObject().put("error", "Note read failed: " + e.getMessage()).toString();
                exchange.sendResponseHeaders(500, error.length());
                exchange.getResponseBody().write(error.getBytes());
            }
            exchange.close();
        });

        server.start();
        System.out.println("Java server running on port " + port);
    }

    private static List<JSONObject> getNotes(String walletId, int limit, String dbType) throws Exception {
        List<JSONObject> notes = new ArrayList<>();
        if (dbType.equals("postgres")) {
            PreparedStatement stmt = postgresConn.prepareStatement(
                "SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = ? ORDER BY timestamp DESC LIMIT ?"
            );
            stmt.setString(1, walletId);
            stmt.setInt(2, limit);
            ResultSet rs = stmt.executeQuery();
            while (rs.next()) {
                notes.add(new JSONObject()
                    .put("id", rs.getInt("id"))
                    .put("content", rs.getString("content"))
                    .put("resource_id", rs.getString("resource_id"))
                    .put("timestamp", rs.getTimestamp("timestamp").toString())
                    .put("wallet_id", rs.getString("wallet_id")));
            }
        } else if (dbType.equals("mysql")) {
            PreparedStatement stmt = mysqlConn.prepareStatement(
                "SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE wallet_id = ? ORDER BY timestamp DESC LIMIT ?"
            );
            stmt.setString(1, walletId);
            stmt.setInt(2, limit);
            ResultSet rs = stmt.executeQuery();
            while (rs.next()) {
                notes.add(new JSONObject()
                    .put("id", rs.getInt("id"))
                    .put("content", rs.getString("content"))
                    .put("resource_id", rs.getString("resource_id"))
                    .put("timestamp", rs.getTimestamp("timestamp").toString())
                    .put("wallet_id", rs.getString("wallet_id")));
            }
        } else if (dbType.equals("mongo")) {
            MongoDatabase db = mongoClient.getDatabase("vial_mcp");
            MongoCollection<Document> collection = db.getCollection("notes");
            for (Document doc : collection.find(new Document("wallet_id", walletId)).sort(new Document("timestamp", -1)).limit(limit)) {
                notes.add(new JSONObject()
                    .put("id", doc.getObjectId("_id").toString())
                    .put("content", doc.getString("content"))
                    .put("resource_id", doc.getString("resource_id"))
                    .put("timestamp", doc.getDate("timestamp").toInstant().toString())
                    .put("wallet_id", doc.getString("wallet_id")));
            }
        } else {
            throw new Exception("Invalid database type");
        }
        return notes;
    }

    private static JSONObject addNote(String walletId, String content, String resourceId, String dbType) throws Exception {
        if (dbType.equals("postgres")) {
            PreparedStatement stmt = postgresConn.prepareStatement(
                "INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES (?, ?, ?, ?) RETURNING id, content, resource_id, timestamp, wallet_id"
            );
            stmt.setString(1, content);
            stmt.setString(2, resourceId);
            stmt.setTimestamp(3, new java.sql.Timestamp(System.currentTimeMillis()));
            stmt.setString(4, walletId);
            ResultSet rs = stmt.executeQuery();
            rs.next();
            return new JSONObject()
                .put("id", rs.getInt("id"))
                .put("content", rs.getString("content"))
                .put("resource_id", rs.getString("resource_id"))
                .put("timestamp", rs.getTimestamp("timestamp").toString())
                .put("wallet_id", rs.getString("wallet_id"));
        } else if (dbType.equals("mysql")) {
            PreparedStatement stmt = mysqlConn.prepareStatement(
                "INSERT INTO notes (content, resource_id, timestamp, wallet_id) VALUES (?, ?, ?, ?)",
                PreparedStatement.RETURN_GENERATED_KEYS
            );
            stmt.setString(1, content);
            stmt.setString(2, resourceId);
            stmt.setTimestamp(3, new java.sql.Timestamp(System.currentTimeMillis()));
            stmt.setString(4, walletId);
            stmt.executeUpdate();
            ResultSet rs = stmt.getGeneratedKeys();
            rs.next();
            int id = rs.getInt(1);
            PreparedStatement selectStmt = mysqlConn.prepareStatement(
                "SELECT id, content, resource_id, timestamp, wallet_id FROM notes WHERE id = ?"
            );
            selectStmt.setInt(1, id);
            rs = selectStmt.executeQuery();
            rs.next();
            return new JSONObject()
                .put("id", rs.getInt("id"))
                .put("content", rs.getString("content"))
                .put("resource_id", rs.getString("resource_id"))
                .put("timestamp", rs.getTimestamp("timestamp").toString())
                .put("wallet_id", rs.getString("wallet_id"));
        } else if (dbType.equals("mongo")) {
            MongoDatabase db = mongoClient.getDatabase("vial_mcp");
            MongoCollection<Document> collection = db.getCollection("notes");
            Document doc = new Document()
                .append("content", content)
                .append("resource_id", resourceId)
                .append("timestamp", new java.util.Date())
                .append("wallet_id", walletId);
            collection.insertOne(doc);
            return new JSONObject()
                .put("id", doc.getObjectId("_id").toString())
                .put("content", content)
                .put("resource_id", resourceId)
                .put("timestamp", doc.getDate("timestamp").toInstant().toString())
                .put("wallet_id", walletId);
        } else {
            throw new Exception("Invalid database type");
        }
    }
}
