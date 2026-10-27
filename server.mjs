import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = path.dirname(fileURLToPath(import.meta.url));
const files = new Map([
    ['/', ['index.html', 'text/html']],
    ['/index.html', ['index.html', 'text/html']],
    ['/styles.css', ['styles.css', 'text/css']],
    ['/app.js', ['app.js', 'text/javascript']],
    ['/classifier.js', ['classifier.js', 'text/javascript']],
]);

export function createServer() {
    return http.createServer(async (request, response) => {
        response.setHeader('X-Content-Type-Options', 'nosniff');
        response.setHeader('Referrer-Policy', 'no-referrer');
        response.setHeader('Cache-Control', 'no-store');
        response.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'");
        if (!['GET', 'HEAD'].includes(request.method)) {
            response.writeHead(405, { Allow: 'GET, HEAD' });
            response.end('Method not allowed');
            return;
        }
        const entry = files.get(new URL(request.url, 'http://localhost').pathname);
        if (!entry) {
            response.writeHead(404);
            response.end('Not found');
            return;
        }
        try {
            const body = await readFile(path.join(root, entry[0]));
            response.writeHead(200, { 'Content-Type': `${entry[1]}; charset=utf-8` });
            response.end(request.method === 'HEAD' ? undefined : body);
        } catch {
            response.writeHead(500);
            response.end('Could not load application file');
        }
    });
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
    const port = Number(process.env.PORT || 3000);
    const host = process.env.HOST || '127.0.0.1';
    const server = createServer();
    server.on('error', error => {
        console.error(`Could not start local server: ${error.message}`);
        process.exitCode = 1;
    });
    server.listen(port, host, () => console.log(`Local: http://${host}:${server.address().port}`));
}
