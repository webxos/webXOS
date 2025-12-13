// Test script using native fetch (Node 18+)
async function testServer() {
  const baseUrl = 'http://localhost:3000';
  
  console.log('Testing WEBXOS Crawler Server...\n');
  
  try {
    // Test 1: Ping endpoint
    console.log('1. Testing /api/ping...');
    const pingRes = await fetch(`${baseUrl}/api/ping`);
    const pingData = await pingRes.json();
    console.log('✓', pingData);
    
    // Test 2: Crawl endpoint
    console.log('\n2. Testing /api/crawl with example.com...');
    const crawlRes = await fetch(`${baseUrl}/api/crawl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ startUrl: 'https://example.com' })
    });
    
    const crawlData = await crawlRes.json();
    console.log('✓ Found', crawlData.linksFound?.length || 0, 'links');
    console.log('Status:', crawlData.status);
    
    if (crawlData.linksFound && crawlData.linksFound.length > 0) {
      console.log('First 3 links:');
      crawlData.linksFound.slice(0, 3).forEach((link, i) => {
        console.log(`  ${i + 1}. ${link}`);
      });
    }
    
    // Test 3: Invalid URL
    console.log('\n3. Testing invalid URL...');
    const invalidRes = await fetch(`${baseUrl}/api/crawl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ startUrl: 'invalid-url' })
    });
    
    const invalidData = await invalidRes.json();
    console.log('✓ Expected error:', invalidData.error || 'No error');
    
    // Test 4: Stats endpoint
    console.log('\n4. Testing /api/stats...');
    const statsRes = await fetch(`${baseUrl}/api/stats`);
    const statsData = await statsRes.json();
    console.log('✓ Stats:', statsData);
    
  } catch (error) {
    console.error('✗ Test failed:', error.message);
  }
  
  console.log('\nTest complete!');
}

// Run test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  testServer();
}

export { testServer };
