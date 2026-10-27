import http from 'node:http';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { spawn } from 'node:child_process';

const root = path.dirname(fileURLToPath(import.meta.url));
const imageLimit = 10 * 1024 * 1024;
const files = new Map([
    ['/', ['index.html', 'text/html']],
    ['/index.html', ['index.html', 'text/html']],
    ['/styles.css', ['styles.css', 'text/css']],
    ['/app.js', ['app.js', 'text/javascript']],
    ['/classifier.js', ['classifier.js', 'text/javascript']],
]);

function runPipeline(command, modelPath, input, python) {
    return new Promise((resolve, reject) => {
        const child = spawn(python, [path.join(root, 'ml', 'pipeline.py'), command, '--model', modelPath], { stdio: ['pipe', 'pipe', 'pipe'], windowsHide: true });
        let output = '', errors = '';
        const timer = setTimeout(() => child.kill(), 30000);
        child.stdout.on('data', chunk => { output += chunk; if (output.length > 65536) child.kill(); });
        child.stderr.on('data', chunk => { errors += chunk; if (errors.length > 65536) child.kill(); });
        child.on('error', error => { clearTimeout(timer); reject(error); });
        child.on('close', code => {
            clearTimeout(timer);
            if (code !== 0) return reject(new Error('Model process failed'));
            try { resolve(JSON.parse(output)); } catch { reject(new Error('Invalid model response')); }
        });
        child.stdin.on('error', () => {});
        child.stdin.end(input);
    });
}

function readImage(request) {
    return new Promise((resolve, reject) => {
        let size = 0, chunks = [], done = false;
        request.on('data', chunk => {
            if (done) return;
            size += chunk.length;
            if (size > imageLimit) {
                done = true;
                chunks = [];
                reject(new Error('Image is too large'));
                return;
            }
            chunks.push(chunk);
        });
        request.on('end', () => { if (!done) resolve(Buffer.concat(chunks)); });
        request.on('error', reject);
        request.on('aborted', () => reject(new Error('Upload interrupted')));
    });
}

export function createServer(options = {}) {
    const modelPath = path.resolve(options.modelPath ?? process.env.MODEL_PATH ?? path.join(root, 'artifacts', 'model.json'));
    const python = options.python ?? process.env.PYTHON ?? 'python';
    return http.createServer(async (request, response) => {
        response.setHeader('X-Content-Type-Options', 'nosniff');
        response.setHeader('Referrer-Policy', 'no-referrer');
        response.setHeader('Cache-Control', 'no-store');
        response.setHeader('Content-Security-Policy', "default-src 'self'; img-src 'self' blob:; style-src 'self'; script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'");
        const pathname = new URL(request.url, 'http://localhost').pathname;
        const json = (status, body) => {
            response.writeHead(status, { 'Content-Type': 'application/json; charset=utf-8' });
            response.end(JSON.stringify(body));
        };
        if (pathname === '/api/model' && request.method === 'GET') {
            try {
                const info = await runPipeline('inspect', modelPath, undefined, python);
                json(200, { available: true, ...info });
            } catch {
                json(200, { available: false });
            }
            return;
        }
        if (pathname === '/api/predict' && request.method === 'POST') {
            const expectedOrigin = `http://${request.headers.host}`;
            if (!request.headers.origin || request.headers.origin !== expectedOrigin ||
                !['same-origin', 'none', undefined].includes(request.headers['sec-fetch-site'])) {
                request.resume();
                json(403, { error: 'Requests must originate from this local application.' });
                return;
            }
            if (!['image/png', 'image/jpeg', 'image/webp'].includes(request.headers['content-type'])) {
                request.resume();
                json(415, { error: 'Choose a PNG, JPEG, or WebP image.' });
                return;
            }
            if (Number(request.headers['content-length']) > imageLimit) {
                request.resume();
                json(413, { error: 'Image exceeds 10 MB.' });
                return;
            }
            let body;
            try { body = await readImage(request); } catch {
                if (!response.destroyed) json(413, { error: 'Image is too large or the upload was interrupted.' });
                return;
            }
            if (!body.length) { json(400, { error: 'Choose a nonempty image.' }); return; }
            try {
                const result = await runPipeline('predict', modelPath, body, python);
                json(200, result);
            } catch {
                json(422, { error: 'Prediction failed. Check that a compatible model is available and the image is valid.' });
            }
            return;
        }
        if (!['GET', 'HEAD'].includes(request.method)) {
            response.writeHead(405, { Allow: 'GET, HEAD' });
            response.end('Method not allowed');
            return;
        }
        const entry = files.get(pathname);
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
