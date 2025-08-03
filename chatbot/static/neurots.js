body {
    margin: 0;
    background: #000;
    font-family: 'VT323', monospace;
    color: #fff;
    overflow-x: hidden;
}

#neuralCanvas {
    position: fixed;
    top: 0;
    left: 0;
    z-index: 0;
}

.container {
    position: relative;
    z-index: 1;
    max-width: 800px;
    margin: 20px auto;
    padding: 20px;
    display: flex;
    flex-direction: column;
    align-items: center;
}

.title {
    font-family: 'Orbitron', sans-serif;
    font-size: 36px;
    color: #00ff00;
    text-shadow: 0 0 10px #00ff00;
    margin-bottom: 20px;
}

#chatbox {
    background: rgba(0, 0, 0, 0.8);
    border: 2px solid #00ff00;
    border-radius: 10px;
    padding: 20px;
    width: 100%;
    max-height: 500px;
    overflow: hidden;
    box-shadow: 0 0 20px rgba(0, 255, 0, 0.5);
}

#messages {
    max-height: 400px;
    overflow-y: auto;
    margin-bottom: 20px;
}

#messages p {
    margin: 5px 0;
    font-size: 16px;
    line-height: 1.4;
}

#messages a {
    color: inherit;
    text-decoration: underline;
}

#messages a:hover {
    color: #00ff00;
}

.agent1-color {
    color: #00FF00;
}

.agent2-color {
    color: #00CCFF;
}

.agent3-color {
    color: #FF00FF;
}

.agent4-color {
    color: #FFFF00;
}

#status {
    font-size: 14px;
    color: #fff;
    margin-bottom: 10px;
    display: none;
}

#status .success {
    color: #00ff00;
}

#status .error {
    color: #ff0000;
}

.input-group {
    display: flex;
    gap: 10px;
    width: 100%;
}

#userInput {
    flex: 1;
    padding: 10px;
    font-family: 'VT323', monospace;
    font-size: 18px;
    background: #111;
    color: #fff;
    border: 2px solid #00ff00;
    border-radius: 5px;
    outline: none;
}

#userInput:focus {
    box-shadow: 0 0 10px #00ff00;
}

#searchButton {
    padding: 10px 20px;
    font-family: 'Press Start 2P', cursive;
    font-size: 16px;
    background: #00ff00;
    color: #000;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    transition: all 0.3s;
}

#searchButton:hover {
    background: #00cc00;
    box-shadow: 0 0 10px #00ff00;
}

.pagination-group {
    display: flex;
    gap: 10px;
    justify-content: center;
    margin-top: 10px;
}

.pagination-group button {
    padding: 8px 16px;
    font-family: 'VT323', monospace;
    font-size: 16px;
    background: #111;
    color: #00ff00;
    border: 2px solid #00ff00;
    border-radius: 5px;
    cursor: pointer;
}

.pagination-group button:disabled {
    color: #555;
    border-color: #555;
    cursor: not-allowed;
}

.pagination-group span {
    font-family: 'VT323', monospace;
    font-size: 16px;
    line-height: 36px;
}

.loading-spinner {
    display: none;
    width: 30px;
    height: 30px;
    border: 4px solid #00ff00;
    border-top: 4px solid transparent;
    border-radius: 50%;
    animation: spin 1s linear infinite;
    margin: 10px auto;
}

@keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
}

.certificate-modal {
    display: none;
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    background: rgba(0, 0, 0, 0.9);
    z-index: 2;
    justify-content: center;
    align-items: center;
}

.certificate-container {
    max-width: 600px;
    width: 100%;
    background: #000;
    border: 3px solid #00ff00;
    border-radius: 15px;
    box-shadow: 0 0 20px #00ff00;
    overflow: hidden;
}

.certificate {
    padding: 20px;
    text-align: center;
}

.certificate-title {
    font-family: 'Orbitron', sans-serif;
    font-size: 24px;
    color: #00ff00;
    text-shadow: 0 0 10px #00ff00;
    margin-bottom: 20px;
}

#certificateContent p {
    font-family: 'VT323', monospace;
    font-size: 18px;
    margin: 10px 0;
}

@keyframes spin-certificate {
    0% { transform: rotateY(0deg); }
    100% { transform: rotateY(360deg); }
}

.spinning {
    animation: spin-certificate 1s ease-in-out;
}

.scramble-char {
    display: inline-block;
    transition: all 0.3s;
}

.dissolving {
    opacity: 0;
    transform: translate(calc(var(--rand-x) * 50px), calc(var(--rand-y) * 50px)) scale(0);
}

.rebuilding {
    opacity: 1;
    transform: translate(calc(var(--rand-x) * 10px), calc(var(--rand-y) * 10px)) scale(1);
}

.copyright {
    position: fixed;
    bottom: 10px;
    left: 10px;
    font-family: 'VT323', monospace;
    font-size: 14px;
    color: #555;
}

@media (max-width: 600px) {
    .container {
        margin: 10px;
        padding: 10px;
    }

    .title {
        font-size: 24px;
    }

    #chatbox {
        padding: 10px;
    }

    #userInput {
        font-size: 16px;
    }

    #searchButton {
        font-size: 14px;
        padding: 8px 16px;
    }

    .pagination-group button {
        font-size: 14px;
        padding: 6px 12px;
    }

    .pagination-group span {
        font-size: 14px;
        line-height: 28px;
    }

    .certificate-title {
        font-size: 20px;
    }

    #certificateContent p {
        font-size: 16px;
    }
}
