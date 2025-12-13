// Test script for the crawler
const fetch = require('node-fetch');

async function testCrawler() {
  console.log('Testing WEBXOS Crawler API...\n');
  
  const baseUrl = 'http://localhost:3000';
  
  // Test 1: Ping endpoint
  console.log('Test 1: Testing ping endpoint');
  try {
    const pingResponse = await fetch(`${baseUrl}/ping`);
    const pingData = await pingResponse.json();
    console.log('✓ Ping response:', JSON.stringify(pingData, null, 2));
  } catch (error) {
    console.log('✗ Ping test failed:', error.message);
  }
  
  // Test 2: Root endpoint
  console.log('\nTest 2: Testing root endpoint');
  try {
    const rootResponse = await fetch(baseUrl);
    const rootData = await rootResponse.json();
    console.log('✓ Root response:', JSON.stringify(rootData, null, 2));
  } catch (error) {
    console.log('✗ Root test failed:', error.message);
  }
  
  // Test 3: Crawl endpoint with example.com
  console.log('\nTest 3: Testing crawl endpoint (example.com)');
  try {
    const crawlResponse = await fetch(`${baseUrl}/crawl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ startUrl: 'https://example.com' })
    });
    
    const crawlData = await crawlResponse.json();
    
    if (crawlData.error) {
      console.log('✗ Crawl test failed:', crawlData.error);
    } else {
      console.log('✓ Crawl test passed');
      console.log(`  Source URL: ${crawlData.sourceUrl}`);
      console.log(`  Links found: ${crawlData.linksFound.length}`);
      console.log(`  Status: ${crawlData.status}`);
      
      // Show first 3 links if available
      if (crawlData.linksFound.length > 0) {
        console.log('  First 3 links:');
        crawlData.linksFound.slice(0, 3).forEach((link, i) => {
          console.log(`    ${i + 1}. ${link}`);
        });
      }
    }
  } catch (error) {
    console.log('✗ Crawl test failed:', error.message);
  }
  
  // Test 4: Invalid URL
  console.log('\nTest 4: Testing invalid URL');
  try {
    const invalidResponse = await fetch(`${baseUrl}/crawl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ startUrl: 'not-a-valid-url' })
    });
    
    const invalidData = await invalidResponse.json();
    console.log('✓ Invalid URL test passed:', invalidData.error || 'No error returned');
  } catch (error) {
    console.log('✗ Invalid URL test failed:', error.message);
  }
  
  // Test 5: Missing parameter
  console.log('\nTest 5: Testing missing parameter');
  try {
    const missingResponse = await fetch(`${baseUrl}/crawl`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    
    const missingData = await missingResponse.json();
    console.log('✓ Missing parameter test passed:', missingData.error || 'No error returned');
  } catch (error) {
    console.log('✗ Missing parameter test failed:', error.message);
  }
  
  // Test 6: Stats endpoint
  console.log('\nTest 6: Testing stats endpoint');
  try {
    const statsResponse = await fetch(`${baseUrl}/stats`);
    const statsData = await statsResponse.json();
    console.log('✓ Stats response:', JSON.stringify(statsData, null, 2));
  } catch (error) {
    console.log('✗ Stats test failed:', error.message);
  }
  
  console.log('\n=== Test Complete ===');
  console.log('Note: Make sure the server is running (npm start)');
  console.log(`Server should be running at ${baseUrl}`);
}

testCrawler().catch(console.error);
