/**
 * Next.js API route for data retrieval.
 */
import { NextResponse } from 'https://cdn.jsdelivr.net/npm/next@14.2.13/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';

export async function POST(request) {
    try {
        const { wallet_id, note_id } = await request.json();
        if (!wallet_id || !note_id) {
            return NextResponse.json({ status: 'error', message: 'Wallet ID and Note ID required' }, { status: 400 });
        }
        const response = await axios.post('https://localhost:8000/api/notes/read', 
            { note_id, wallet_id }, 
            { headers: { 'Authorization': `Bearer ${request.headers.get('Authorization')?.replace('Bearer ', '')}` } }
        );
        return NextResponse.json(response.data);
    } catch (error) {
        console.error(`Retrieve error: ${error.message}`);
        return NextResponse.json({ status: 'error', message: `Retrieve failed: ${error.message}` }, { status: 500 });
    }
}
