// Test script for the crawler
const fetch = require('node-fetch');

async function testCrawler() {
  console.log('Testing WEBXOS Crawler API...\n');
  
  // Test 1: Ping endpoint
  console.log('Test 1: Testing ping endpoint');
  try {
    const pingResponse = await fetch('http://localhost:3000/api/ping');
    const pingData = await pingResponse.json();
    console.log('✓ Ping response:', pingData);
  } catch (error) {
    console.log('✗ Ping test failed:', error.message);
  }
  
  // Test 2: Crawl endpoint
  console.log('\nTest 2: Testing crawl endpoint');
  try {
    const crawlResponse = await fetch('http://localhost:3000/api/crawl', {
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
  
  // Test 3: Invalid URL
  console.log('\nTest 3: Testing invalid URL');
  try {
    const invalidResponse = await fetch('http://localhost:3000/api/crawl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ startUrl: 'not-a-valid-url' })
    });
    
    const invalidData = await invalidResponse.json();
    console.log('✓ Invalid URL test passed (expected error):', invalidData.error || 'No error returned');
  } catch (error) {
    console.log('✗ Invalid URL test failed:', error.message);
  }
  
  // Test 4: Missing parameter
  console.log('\nTest 4: Testing missing parameter');
  try {
    const missingResponse = await fetch('http://localhost:3000/api/crawl', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({})
    });
    
    const missingData = await missingResponse.json();
    console.log('✓ Missing parameter test passed:', missingData.error || 'No error returned');
  } catch (error) {
    console.log('✗ Missing parameter test failed:', error.message);
  }
  
  console.log('\n=== Test Complete ===');
  console.log('Note: Make sure the server is running (npm start)');
}

testCrawler().catch(console.error);