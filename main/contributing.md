Contributing to WebXOS Backend
Thank you for your interest in contributing to the WebXOS Backend! This project is open-source under the MIT license, and we welcome contributions from the community.
Getting Started

Fork the Repository: Fork https://github.com/webxos/webxos.
Clone Your Fork:git clone https://github.com/<your-username>/webxos
cd webxos-backend


Install Dependencies:npm install


Run Locally:npm run dev



Contribution Guidelines

Code Style: Follow JavaScript Standard Style.
Testing: Add unit tests for new features (use a framework like Jest).
Commits: Use clear commit messages (e.g., feat: add vial data parsing).
Pull Requests:
Create a PR against the main branch.
Include a description of changes and reference any issues.
Ensure all tests pass.


Issues: Report bugs or suggest features via GitHub Issues.

Development Tips

Use the sample_wallet.md file for testing wallet parsing.
Log errors using the winston logger in src/utils/logger.js.
Test endpoints with Postman or curl (e.g., curl http://localhost:3000/v1/wallet).

Code of Conduct
Be respectful and inclusive. Report any issues to the maintainers.
License
Contributions are licensed under the MIT license.
