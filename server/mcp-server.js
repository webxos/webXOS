const { Server } = require('socket.io');
const http = require('http');
const express = require('express');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*' },
    maxHttpBufferSize: 1e7
});

const agents = [
    { name: 'agent1', status: 'stopped', startTime: null },
    { name: 'agent2', status: 'stopped', startTime: null },
    { name: 'agent3', status: 'stopped', startTime: null },
    { name: 'agent4', status: 'stopped', startTime: null }
];

let serverRunning = false;

function logError(message, error) {
    console.error(`[MCP] ${message}: ${error.message}\nStack: ${error.stack}`);
}

io.on('connection', (socket) => {
    console.log('[MCP] Client connected:', socket.id);

    socket.on('start-server', () => {
        try {
            if (serverRunning) {
                socket.emit('server-error', { message: 'Server already running.' });
                return;
            }
            serverRunning = true;
            agents.forEach(agent => {
                agent.status = 'running';
                agent.startTime = new Date();
                socket.emit('agent-status', {
                    agent: agent.name,
                    status: 'started',
                    latency: Math.random() * 100,
                    timestamp: agent.startTime.toISOString()
                });
                console.log(`[MCP] ${agent.name} started at ${agent.startTime.toISOString()}`);
            });
            socket.emit('server-started', { message: 'MCP server started.' });
        } catch (err) {
            socket.emit('server-error', { message: err.message, analysis: 'Failed to start server. Check logs.' });
            logError('Start Server Error', err);
        }
    });

    socket.on('end-server', () => {
        try {
            if (!serverRunning) {
                socket.emit('server-error', { message: 'Server not running.' });
                return;
            }
            serverRunning = false;
            agents.forEach(agent => {
                if (agent.status === 'running') {
                    const runTime = Math.round((new Date() - agent.startTime) / 1000);
                    agent.status = 'stopped';
                    socket.emit('agent-status', {
                        agent: agent.name,
                        status: `ended, ran for ${runTime}s`,
                        latency: 0,
                        timestamp: new Date().toISOString()
                    });
                    console.log(`[MCP] ${agent.name} ended, ran for ${runTime}s`);
                    agent.startTime = null;
                }
            });
        } catch (err) {
            socket.emit('server-error', { message: err.message, analysis: 'Failed to stop server. Check logs.' });
            logError('End Server Error', err);
        }
    });

    socket.on('check-agents', () => {
        try {
            if (!serverRunning) {
                socket.emit('server-error', { message: 'Server not running.', analysis: 'Start server first.' });
                return;
            }
            agents.forEach(agent => {
                const latency = Math.random() * 100;
                socket.emit('agent-status', {
                    agent: agent.name,
                    status: agent.status,
                    latency: agent.status === 'running' ? latency : 0,
                    timestamp: new Date().toISOString()
                });
                console.log(`[MCP] ${agent.name} checked: ${agent.status}, Latency: ${latency.toFixed(2)}ms`);
            });
        } catch (err) {
            socket.emit('server-error', { message: err.message, analysis: 'Failed to check agents. Check logs.' });
            logError('Check Agents Error', err);
        }
    });

    socket.on('disconnect', () => {
        console.log('[MCP] Client disconnected:', socket.id);
    });
});

server.listen(8080, () => {
    console.log('[MCP] Server running on ws://localhost:8080');
});

process.on('uncaughtException', (err) => {
    console.error('[MCP] Uncaught Exception:', err.message, '\nStack:', err.stack);
    io.emit('server-error', { message: 'Server encountered an unexpected error.', analysis: 'Check server logs.' });
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('[MCP] Unhandled Rejection at:', promise, 'Reason:', reason);
    io.emit('server-error', { message: 'Server encountered an unexpected error.', analysis: 'Check server logs.' });
});
