/**
 * Next.js API route for LLM inference (quantum processing).
 */
import { NextResponse } from 'https://cdn.jsdelivr.net/npm/next@14.2.13/+esm';
import axios from 'https://cdn.jsdelivr.net/npm/axios@1.7.7/+esm';

export async function POST(request) {
    try {
        const { vial_id, prompt, wallet_id } = await request.json();
        if (!vial_id || !prompt || !wallet_id) {
            return NextResponse.json({ status: 'error', message: 'Vial ID, prompt, and wallet ID required' }, { status: 400 });
        }
        const response = await axios.post('https://localhost:8000/api/quantum/link', 
            { vial_id, prompt, wallet_id }, 
            { headers: { 'Authorization': `Bearer ${request.headers.get('Authorization')?.replace('Bearer ', '')}` } }
        );
        return NextResponse.json(response.data);
    } catch (error) {
        console.error(`LLM inference error: ${error.message}`);
        return NextResponse.json({ status: 'error', message: `LLM inference failed: ${error.message}` }, { status: 500 });
    }
}
