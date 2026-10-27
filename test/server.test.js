import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from '../server.mjs';

test('local server serves the workspace and refuses private files and writes', async t => {
    const server = createServer();
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    t.after(() => new Promise(resolve => server.close(resolve)));
    const base = `http://127.0.0.1:${server.address().port}`;
    for (const [route, mime] of [['/', 'text/html'], ['/styles.css', 'text/css'], ['/app.js', 'text/javascript'], ['/classifier.js', 'text/javascript']]) {
        const response = await fetch(base + route);
        assert.equal(response.status, 200);
        assert.ok(response.headers.get('content-type').startsWith(mime));
        assert.ok((await response.text()).length > 0);
        assert.match(response.headers.get('content-security-policy'), /default-src 'self'/);
    }
    for (const route of ['/.git/config', '/package.json', '/README.md', '/%2e%2e/README.md']) {
        assert.equal((await fetch(base + route)).status, 404);
    }
    assert.equal((await fetch(base, { method: 'POST', body: 'not accepted' })).status, 405);
    const head = await fetch(base, { method: 'HEAD' });
    assert.equal(head.status, 200);
    assert.equal(await head.text(), '');
});
