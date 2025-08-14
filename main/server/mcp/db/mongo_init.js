db.createCollection("wallets");
db.createCollection("notes");
db.createCollection("quantum_states");

db.wallets.createIndex({ "wallet_id": 1 }, { unique: true });
db.notes.createIndex({ "wallet_id": 1 });
db.quantum_states.createIndex({ "wallet_id": 1 });

db.wallets.insertOne({
    wallet_id: "wallet_123",
    user_id: "user_123",
    api_key: "api-a24cb96b-96cd-488d-a013-91cb8edbbe68",
    timestamp: new Date()
});
