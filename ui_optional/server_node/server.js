const http = require('http');
const fs = require('fs');
const path = require('path');

const ROOT = path.join(__dirname, '..', 'web');
const RUNS = path.join(__dirname, '..', '..', 'runs');
const PORT = Number(process.env.PORT || 5173);

const MIME = {
  '.html': 'text/html; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.svg': 'image/svg+xml',
  '.ico': 'image/x-icon',
  '.map': 'application/json; charset=utf-8',
};

function safeJoin(base, target) {
  const resolved = path.normalize(path.join(base, target));
  if (!resolved.startsWith(base)) {
    return null;
  }
  return resolved;
}

function send(res, code, data, contentType) {
  res.writeHead(code, { 'Content-Type': contentType || 'text/plain; charset=utf-8' });
  res.end(data);
}

function serveFile(res, filePath) {
  fs.readFile(filePath, (err, data) => {
    if (err) {
      send(res, 404, 'Not Found');
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    send(res, 200, data, MIME[ext] || 'application/octet-stream');
  });
}

const server = http.createServer((req, res) => {
  const urlPath = decodeURIComponent((req.url || '/').split('?')[0]);
  const isRuns = urlPath.startsWith('/runs/');
  const base = isRuns ? RUNS : ROOT;
  const rel = isRuns
    ? urlPath.replace(/^\/runs\//, '')
    : urlPath === '/' ? 'index.html' : urlPath.replace(/^\//, '');

  const filePath = safeJoin(base, rel);
  if (!filePath) {
    send(res, 403, 'Forbidden');
    return;
  }

  fs.stat(filePath, (err, stat) => {
    if (!err && stat.isDirectory()) {
      const indexPath = safeJoin(filePath, 'index.html');
      if (indexPath && fs.existsSync(indexPath)) {
        serveFile(res, indexPath);
        return;
      }
      send(res, 404, 'Not Found');
      return;
    }

    if (!err && stat.isFile()) {
      serveFile(res, filePath);
      return;
    }

    if (!isRuns) {
      const fallback = safeJoin(ROOT, 'index.html');
      if (fallback && fs.existsSync(fallback)) {
        serveFile(res, fallback);
        return;
      }
    }
    send(res, 404, 'Not Found');
  });
});

server.listen(PORT, '0.0.0.0', () => {
  // eslint-disable-next-line no-console
  console.log(`UI server running at http://localhost:${PORT}`);
});
