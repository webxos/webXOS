import aiohttp
from aiohttp import web
import sqlite3
import json
import uuid
import os
from vial_manager import VialManager
import torch
import logging

logging.basicConfig(level=logging.INFO, filename='server.log', format='%(asctime)s %(levelname)s:%(message)s')

async def ping(request):
    logging.info('Ping request received')
    return web.json_response({'status': 'online'})

async def auth(request):
    try:
        data = await request.json()
        network_id = data.get('networkId')
        if not network_id:
            logging.error('Missing networkId in auth request')
            raise web.HTTPBadRequest(reason='Missing networkId')
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        token = str(uuid.uuid4())
        cursor.execute('INSERT OR REPLACE INTO auth (network_id, token, timestamp) VALUES (?, ?, ?)',
                      (network_id, token, int(os.times()[4])))
        cursor.execute('INSERT OR REPLACE INTO wallets (network_id, address, balance) VALUES (?, ?, ?)',
                      (network_id, str(uuid.uuid4()), 0.0))
        conn.commit()
        conn.close()
        logging.info(f'Auth successful for network_id: {network_id}')
        return web.json_response({'token': token, 'address': str(uuid.uuid4())})
    except Exception as e:
        logging.error(f'Auth error: {str(e)}')
        raise

async def void(request):
    try:
        data = await request.json()
        network_id = data.get('networkId')
        if not network_id:
            logging.error('Missing networkId in void request')
            raise web.HTTPBadRequest(reason='Missing networkId')
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('DELETE FROM auth WHERE network_id = ?', (network_id,))
        cursor.execute('DELETE FROM wallets WHERE network_id = ?', (network_id,))
        cursor.execute('DELETE FROM vials WHERE network_id = ?', (network_id,))
        conn.commit()
        conn.close()
        logging.info(f'Void successful for network_id: {network_id}')
        return web.json_response({'status': 'voided'})
    except Exception as e:
        logging.error(f'Void error: {str(e)}')
        raise

async def train(request):
    try:
        form = await request.post()
        network_id = form.get('networkId')
        code = form.get('code')
        is_python = form.get('isPython') == 'true'
        if not (network_id and code):
            logging.error('Missing networkId or code in train request')
            raise web.HTTPBadRequest(reason='Missing networkId or code')
        manager = VialManager(network_id)
        balance = 0.0004
        manager.train_vials(code, is_python)
        conn = sqlite3.connect('vial.db')
        cursor = conn.cursor()
        cursor.execute('UPDATE wallets SET balance = balance + ? WHERE network_id = ?', (balance, network_id))
        conn.commit()
        conn.close()
        logging.info(f'Train successful for network_id: {network_id}')
        return web.json_response({
            'vials': manager.get_vials(),
            'balance': balance
        })
    except Exception as e:
        logging.error(f'Train error: {str(e)}')
        raise

async def upload(request):
    try:
        form = await request.post()
        network_id = form.get('networkId')
        file = form.get('file')
        if not (network_id and file):
            logging.error('Missing networkId or file in upload request')
            raise web.HTTPBadRequest(reason='Missing networkId or file')
        file_path = f'/uploads/{file.filename}'
        with open(file_path, 'wb') as f:
            f.write(file.file.read())
        logging.info(f'File uploaded to {file_path} for network_id: {network_id}')
        return web.json_response({'filePath': file_path})
    except Exception as e:
        logging.error(f'Upload error: {str(e)}')
        raise

async def init_db():
    conn = sqlite3.connect('vial.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS auth (
            network_id TEXT PRIMARY KEY,
            token TEXT,
            timestamp INTEGER
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS wallets (
            network_id TEXT PRIMARY KEY,
            address TEXT,
            balance REAL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS vials (
            network_id TEXT,
            vial_id TEXT,
            status TEXT,
            code TEXT,
            code_length INTEGER,
            is_python BOOLEAN,
            webxos_hash TEXT,
            wallet_address TEXT,
            wallet_balance REAL,
            tasks TEXT,
            PRIMARY KEY (network_id, vial_id)
        )
    ''')
    conn.commit()
    conn.close()
    logging.info('Database initialized')

app = web.Application()
app.router.add_get('/api/mcp/ping', ping)
app.router.add_post('/api/mcp/auth', auth)
app.router.add_post('/api/mcp/void', void)
app.router.add_post('/api/mcp/train', train)
app.router.add_post('/api/mcp/upload', upload)

if __name__ == '__main__':
    init_db()
    logging.info('Starting server on port 8080')
    web.run_app(app, port=8080)
