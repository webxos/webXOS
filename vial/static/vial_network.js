let VialNetworkModule = null;

(function () {
    let module = {};
    VialNetworkModule = function () {
        return new Promise((resolve, reject) => {
            if (module.exports) {
                resolve(module.exports);
                return;
            }
            const script = document.createElement('script');
            script.src = '/static/vial_network.wasm';
            script.async = true;
            script.onload = () => {
                module.exports = window.Module;
                resolve(module.exports);
            };
            script.onerror = () => reject(new Error('Failed to load vial_network.wasm'));
            document.head.appendChild(script);
        });
    };
})();
