```javascript
     const { spawn } = require('child_process');

     exports.handler = async (event, context) => {
         return new Promise((resolve, reject) => {
             const process = spawn('python3', ['backend/server.py'], {
                 env: { ...process.env, PORT: '5000' }
             });

             let output = '';
             let errorOutput = '';

             process.stdout.on('data', (data) => {
                 output += data.toString();
             });

             process.stderr.on('data', (data) => {
                 errorOutput += data.toString();
             });

             process.on('close', (code) => {
                 if (code === 0) {
                     resolve({
                         statusCode: 200,
                         body: output
                     });
                 } else {
                     reject({
                         statusCode: 500,
                         body: errorOutput
                     });
                 }
             });

             // Forward the HTTP request to the Flask server
             const body = event.body ? JSON.parse(event.body) : {};
             process.stdin.write(JSON.stringify({
                 path: event.path,
                 method: event.httpMethod,
                 body: body
             }));
             process.stdin.end();
         });
     };
     ```
