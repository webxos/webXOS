const express = require('express');
const mongoose = require('mongoose');
const { spawn } = require('child_process');
const app = express();
const port = 8080;

app.use(express.json());
app.use('/static', express.static('static'));

// MongoDB connection
mongoose.connect('mongodb://mongo:27017/vial_mcp', { useNewUrlParser: true, useUnifiedTopology: true })
    .then(() => console.log('Mongo
