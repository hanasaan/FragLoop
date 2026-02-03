(() => {
  const qs = (id) => document.getElementById(id);

  const codeBlock = qs('codeBlock');
  const logBody = qs('logBody');
  const historyGrid = qs('historyGrid');
  const overlayBanner = qs('overlayBanner');
  const overlayCode = qs('overlayCode');
  const wsStatus = qs('wsStatus');
  const modeStatus = qs('modeStatus');
  const renderStatus = qs('renderStatus');

  const wsInput = qs('wsInput');
  const reconnectBtn = qs('reconnectBtn');
  const nextBtn = qs('nextBtn');
  const pauseBtn = qs('pauseBtn');
  const fullscreenBtn = qs('fullscreenBtn');
  const overlayBtn = qs('overlayBtn');
  const favoriteBtn = qs('favoriteBtn');
  const rawBtn = qs('rawBtn');
  const copyBtn = qs('copyCode');

  const tempInput = qs('tempInput');
  const topPInput = qs('topPInput');
  const maxTokensInput = qs('maxTokensInput');
  const candidatesInput = qs('candidatesInput');
  const displayInput = qs('displayInput');
  const fadeInput = qs('fadeInput');
  const renderInput = qs('renderInput');
  const applyBtn = qs('applyBtn');

  const canvasA = qs('canvasA');
  const canvasB = qs('canvasB');

  const state = {
    ws: null,
    wsUrl: localStorage.getItem('infshader_ws_url') || wsInput.value,
    connected: false,
    inflight: false,
    paused: false,
    rawMode: false,
    overlay: false,
    displaySec: Number(displayInput.value) || 10,
    crossfadeSec: Number(fadeInput.value) || 2,
    renderSize: Number(renderInput.value) || 512,
    targetFps: 30,
    queue: [],
    current: null,
    lastSwapAt: performance.now(),
    history: new Map(),
    failCount: 0,
    manualSwap: false,
  };

  wsInput.value = state.wsUrl;
  renderStatus.textContent = `${state.renderSize}px`;

  const textureUrls = [
    'textures/bluenoise_256.png',
    'textures/whitenoise_256.png',
    'textures/grad.png',
    'textures/lowfreq.png',
  ];

  const shaderTemplate = (body) => `#version 300 es
precision highp float;
precision highp sampler2D;

uniform vec3 iResolution;
uniform float iTime;
uniform float iTimeDelta;
uniform int iFrame;
uniform float iChannelTime[4];
uniform vec3 iChannelResolution[4];
uniform vec4 iMouse;
uniform vec4 iDate;
uniform float iSampleRate;
uniform sampler2D iChannel0;
uniform sampler2D iChannel1;
uniform sampler2D iChannel2;
uniform sampler2D iChannel3;

out vec4 fragColor;

#define iGlobalTime iTime
#define texture2D texture
#line 1
${body}

void main() {
  mainImage(fragColor, gl_FragCoord.xy);
}
`;

  class ShaderRenderer {
    constructor(canvas) {
      this.canvas = canvas;
      this.gl = canvas.getContext('webgl2', { antialias: true, preserveDrawingBuffer: true });
      if (!this.gl) {
        throw new Error('WebGL2 not supported');
      }
      this.program = null;
      this.uniforms = {};
      this.textures = [];
      this.texRes = new Float32Array(12);
      this.startTime = performance.now();
      this.lastTime = this.startTime;
      this.frame = 0;
      this._initGeometry();
    }

    _initGeometry() {
      const gl = this.gl;
      const vertices = new Float32Array([-1, -1, 3, -1, -1, 3]);
      this.vao = gl.createVertexArray();
      this.vbo = gl.createBuffer();
      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      gl.bufferData(gl.ARRAY_BUFFER, vertices, gl.STATIC_DRAW);
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.bindVertexArray(null);
    }

    setSize(size) {
      this.canvas.width = size;
      this.canvas.height = size;
      this.gl.viewport(0, 0, size, size);
    }

    setTextures(images) {
      const gl = this.gl;
      this.textures = images.map((img, idx) => {
        const tex = gl.createTexture();
        gl.bindTexture(gl.TEXTURE_2D, tex);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.REPEAT);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR);
        gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, 0);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
        gl.generateMipmap(gl.TEXTURE_2D);
        gl.bindTexture(gl.TEXTURE_2D, null);
        this.texRes[idx * 3 + 0] = img.width;
        this.texRes[idx * 3 + 1] = img.height;
        this.texRes[idx * 3 + 2] = 1.0;
        return tex;
      });
    }

    compile(body) {
      const gl = this.gl;
      if (this.program) {
        gl.deleteProgram(this.program);
        this.program = null;
      }
      const fragSource = shaderTemplate(body);
      const vertSource = `#version 300 es
      in vec2 aPos;
      void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
      }`;

      const vert = gl.createShader(gl.VERTEX_SHADER);
      gl.shaderSource(vert, vertSource);
      gl.compileShader(vert);
      if (!gl.getShaderParameter(vert, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(vert) || 'Vertex shader compile failed';
        gl.deleteShader(vert);
        this.program = null;
        return { ok: false, log };
      }

      const frag = gl.createShader(gl.FRAGMENT_SHADER);
      gl.shaderSource(frag, fragSource);
      gl.compileShader(frag);
      if (!gl.getShaderParameter(frag, gl.COMPILE_STATUS)) {
        const log = gl.getShaderInfoLog(frag) || 'Fragment shader compile failed';
        gl.deleteShader(vert);
        gl.deleteShader(frag);
        this.program = null;
        return { ok: false, log };
      }

      const program = gl.createProgram();
      gl.attachShader(program, vert);
      gl.attachShader(program, frag);
      gl.linkProgram(program);
      gl.deleteShader(vert);
      gl.deleteShader(frag);
      if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
        const log = gl.getProgramInfoLog(program) || 'Program link failed';
        gl.deleteProgram(program);
        this.program = null;
        return { ok: false, log };
      }

      this.program = program;
      this.uniforms = {
        iResolution: gl.getUniformLocation(program, 'iResolution'),
        iTime: gl.getUniformLocation(program, 'iTime'),
        iTimeDelta: gl.getUniformLocation(program, 'iTimeDelta'),
        iFrame: gl.getUniformLocation(program, 'iFrame'),
        iChannelTime: gl.getUniformLocation(program, 'iChannelTime'),
        iChannelResolution: gl.getUniformLocation(program, 'iChannelResolution'),
        iMouse: gl.getUniformLocation(program, 'iMouse'),
        iDate: gl.getUniformLocation(program, 'iDate'),
        iSampleRate: gl.getUniformLocation(program, 'iSampleRate'),
        iChannel0: gl.getUniformLocation(program, 'iChannel0'),
        iChannel1: gl.getUniformLocation(program, 'iChannel1'),
        iChannel2: gl.getUniformLocation(program, 'iChannel2'),
        iChannel3: gl.getUniformLocation(program, 'iChannel3'),
      };

      gl.bindVertexArray(this.vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, this.vbo);
      const loc = gl.getAttribLocation(program, 'aPos');
      if (loc >= 0) {
        gl.enableVertexAttribArray(loc);
        gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
      }
      gl.bindBuffer(gl.ARRAY_BUFFER, null);
      gl.bindVertexArray(null);

      this.startTime = performance.now();
      this.lastTime = this.startTime;
      this.frame = 0;
      return { ok: true, log: '' };
    }

    render(now) {
      const gl = this.gl;
      gl.clearColor(0.02, 0.02, 0.03, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      if (!this.program) {
        return;
      }
      gl.useProgram(this.program);
      gl.bindVertexArray(this.vao);
      const delta = (now - this.lastTime) / 1000;
      const timeSec = (now - this.startTime) / 1000;
      const date = new Date();
      if (this.uniforms.iResolution) {
        gl.uniform3f(this.uniforms.iResolution, this.canvas.width, this.canvas.height, 1.0);
      }
      if (this.uniforms.iTime) {
        gl.uniform1f(this.uniforms.iTime, timeSec);
      }
      if (this.uniforms.iTimeDelta) {
        gl.uniform1f(this.uniforms.iTimeDelta, delta);
      }
      if (this.uniforms.iFrame) {
        gl.uniform1i(this.uniforms.iFrame, this.frame);
      }
      if (this.uniforms.iChannelTime) {
        gl.uniform1fv(this.uniforms.iChannelTime, new Float32Array([0, 0, 0, 0]));
      }
      if (this.uniforms.iChannelResolution) {
        gl.uniform3fv(this.uniforms.iChannelResolution, this.texRes);
      }
      if (this.uniforms.iMouse) {
        gl.uniform4f(this.uniforms.iMouse, 0, 0, 0, 0);
      }
      if (this.uniforms.iDate) {
        gl.uniform4f(
          this.uniforms.iDate,
          date.getFullYear(),
          date.getMonth() + 1,
          date.getDate(),
          date.getHours() * 3600 + date.getMinutes() * 60 + date.getSeconds()
        );
      }
      if (this.uniforms.iSampleRate) {
        gl.uniform1f(this.uniforms.iSampleRate, 44100.0);
      }

      this.textures.forEach((tex, idx) => {
        gl.activeTexture(gl.TEXTURE0 + idx);
        gl.bindTexture(gl.TEXTURE_2D, tex);
        const loc = this.uniforms[`iChannel${idx}`];
        if (loc) {
          gl.uniform1i(loc, idx);
        }
      });

      gl.drawArrays(gl.TRIANGLES, 0, 3);
      gl.bindVertexArray(null);
      this.lastTime = now;
      this.frame += 1;
    }
  }

  function sanitizeBody(body) {
    if (!body) return '';
    return body
      .replace(/^\s*#version.*$/gim, '')
      .replace(/^\s*precision\s+\w+\s+float\s*;.*$/gim, '')
      .trim();
  }

  function loadImages(urls) {
    return Promise.all(
      urls.map(
        (url) =>
          new Promise((resolve, reject) => {
            const img = new Image();
            img.crossOrigin = 'anonymous';
            img.onload = () => resolve(img);
            img.onerror = () => reject(new Error(`Failed to load ${url}`));
            img.src = url;
          })
      )
    );
  }

  let renderers;
  let front = 0;
  let back = 1;
  let lastFrame = performance.now();

  function setCanvasActive(index) {
    const canvases = [canvasA, canvasB];
    canvases.forEach((canvas, idx) => {
      canvas.style.transitionDuration = `${state.crossfadeSec}s`;
      canvas.classList.toggle('active', idx === index);
    });
  }

  function updateOverlay(shaderInfo, compileResult) {
    const metrics = shaderInfo.metrics || {};
    let banner = '';
    if (!compileResult.ok) {
      banner = 'Compile Error';
      overlayCode.textContent = compileResult.log || metrics.error || 'compile failed';
      overlayCode.classList.remove('hidden');
    } else if (metrics.copy_suspect) {
      banner = 'Copy Detected';
      overlayCode.textContent = '';
      overlayCode.classList.add('hidden');
    } else {
      overlayCode.textContent = '';
      overlayCode.classList.add('hidden');
    }

    if (banner) {
      overlayBanner.textContent = banner;
      overlayBanner.classList.remove('hidden');
    } else {
      overlayBanner.classList.add('hidden');
    }

    if (state.overlay && compileResult.ok) {
      overlayCode.textContent = shaderInfo.body || '';
      overlayCode.classList.remove('hidden');
    }
  }

  function updateLog(shaderInfo, compileResult) {
    const metrics = shaderInfo.metrics || {};
    const lines = [];
    lines.push(`shader_id: ${shaderInfo.shader_id || '-'}`);
    if (compileResult.log) {
      lines.push('--- compile log ---');
      lines.push(compileResult.log.trim());
    }
    if (metrics) {
      lines.push('--- metrics ---');
      Object.keys(metrics).forEach((key) => {
        lines.push(`${key}: ${metrics[key]}`);
      });
    }
    logBody.textContent = lines.join('\n');
  }

  function applyShader(shaderInfo) {
    if (!shaderInfo) return;
    const renderer = renderers[back];
    const cleanBody = sanitizeBody(shaderInfo.body || '');
    const compileResult = renderer.compile(cleanBody);

    if (!compileResult.ok && !state.rawMode) {
      logBody.textContent = `ui compile failed:\\n${compileResult.log}`.trim();
      if (!state.paused && state.queue.length < 2) {
        requestNext('auto');
      }
      return;
    }

    renderer.setSize(state.renderSize);
    renderer.startTime = performance.now();
    renderer.lastTime = renderer.startTime;

    setCanvasActive(back);
    updateOverlay(shaderInfo, compileResult);
    updateLog(shaderInfo, compileResult);

    const prev = front;
    front = back;
    back = prev;
    state.current = shaderInfo;
    state.failCount = 0;
    state.lastSwapAt = performance.now();
    codeBlock.textContent = shaderInfo.body || '';
    autoAdjustResolution(shaderInfo);

    setTimeout(() => {
      // ensure the hidden canvas is faded out after transition
      setCanvasActive(front);
    }, state.crossfadeSec * 1000 + 50);
  }

  function maybeSwap(now) {
    if (state.queue.length === 0) return;
    if (state.paused) {
      if (state.manualSwap) {
        state.manualSwap = false;
        applyShader(state.queue.shift());
      }
      return;
    }
    if (!state.current) {
      applyShader(state.queue.shift());
      return;
    }
    const elapsed = (now - state.lastSwapAt) / 1000;
    if (elapsed >= state.displaySec) {
      applyShader(state.queue.shift());
      if (!state.paused && state.queue.length < 2) {
        requestNext('auto');
      }
    }
  }

  function tick(now) {
    requestAnimationFrame(tick);
    if (now - lastFrame < 1000 / state.targetFps) {
      return;
    }
    lastFrame = now;
    renderers[front].render(now);
    renderers[back].render(now);
    maybeSwap(now);
  }

  function requestNext(mode) {
    if (!state.connected || state.inflight) return;
    state.inflight = true;
    send({ type: 'request_next', mode, min_display_sec: state.displaySec });
  }

  function send(payload) {
    if (!state.ws || state.ws.readyState !== WebSocket.OPEN) return;
    state.ws.send(JSON.stringify(payload));
  }

  function setWsStatus(text, ok) {
    wsStatus.textContent = text;
    wsStatus.style.borderColor = ok ? 'rgba(67, 209, 167, 0.6)' : 'rgba(255, 93, 93, 0.6)';
  }

  function connect() {
    if (state.ws) {
      state.ws.close();
    }
    state.wsUrl = wsInput.value.trim();
    localStorage.setItem('infshader_ws_url', state.wsUrl);
    const ws = new WebSocket(state.wsUrl);
    state.ws = ws;
    setWsStatus('WS: connecting', false);

    ws.onopen = () => {
      state.connected = true;
      state.inflight = false;
      setWsStatus('WS: connected', true);
      sendParams();
      requestNext('auto');
    };

    ws.onclose = () => {
      state.connected = false;
      state.inflight = false;
      setWsStatus('WS: disconnected', false);
    };

    ws.onerror = () => {
      state.connected = false;
      setWsStatus('WS: error', false);
    };

    ws.onmessage = (event) => {
      let data;
      try {
        data = JSON.parse(event.data);
      } catch (err) {
        return;
      }
      handleMessage(data);
    };
  }

  function handleMessage(data) {
    switch (data.type) {
      case 'shader_candidate':
        state.inflight = false;
        state.queue.push(data);
        state.failCount = 0;
        if (data.shader_id) {
          state.history.set(data.shader_id, data);
        }
        if (!state.paused && state.queue.length < 2) {
          requestNext('auto');
        }
        break;
      case 'shader_rejected':
        if (data.final) {
          state.inflight = false;
          state.failCount += 1;
          if (state.failCount >= 3) {
            nudgeParamsOnFailure();
          }
          if (!state.paused && state.queue.length < 2) {
            requestNext('auto');
          }
        }
        logBody.textContent = `rejected: ${data.reason}\n${data.detail || ''}`.trim();
        break;
      case 'status':
        if (typeof data.paused === 'boolean') {
          state.paused = data.paused;
          pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
        }
        modeStatus.textContent = state.paused ? 'Paused' : 'Auto';
        if (data.last_error) {
          logBody.textContent = `server error: ${data.last_error}`;
        }
        break;
      case 'history_item':
        addHistoryItem(data);
        break;
      default:
        break;
    }
  }

  function addHistoryItem(item) {
    if (!item.thumb_path) return;
    const container = document.createElement('div');
    container.className = 'history-item';
    const img = document.createElement('img');
    img.src = item.thumb_path;
    img.alt = item.shader_id || '';
    container.appendChild(img);
    container.addEventListener('click', () => {
      const stored = state.history.get(item.shader_id || '');
      if (stored) {
        applyShader(stored);
      }
    });
    historyGrid.prepend(container);
  }

  function sendParams() {
    send({
      type: 'set_params',
      temperature: Number(tempInput.value),
      top_p: Number(topPInput.value),
      max_tokens: Number(maxTokensInput.value),
      num_candidates: Number(candidatesInput.value),
      render_size: Number(renderInput.value),
      raw_mode: state.rawMode,
    });
  }

  function autoAdjustResolution(shaderInfo) {
    const renderMs = shaderInfo.metrics && shaderInfo.metrics.render_ms_mean;
    if (!renderMs) return;
    if (renderMs > 180 && state.renderSize > 256) {
      const nextSize = Math.max(256, Math.floor(state.renderSize * 0.75 / 64) * 64);
      if (nextSize !== state.renderSize) {
        state.renderSize = nextSize;
        renderInput.value = String(nextSize);
        renderStatus.textContent = `${state.renderSize}px`;
        renderers.forEach((renderer) => renderer.setSize(state.renderSize));
        sendParams();
      }
    }
  }

  function nudgeParamsOnFailure() {
    const temp = Math.min(1.6, Number(tempInput.value) + 0.1);
    const topP = Math.min(0.99, Number(topPInput.value) + 0.02);
    const maxTokens = Math.min(4096, Number(maxTokensInput.value) + 256);
    const renderSize = Math.max(256, Math.floor(Number(renderInput.value) * 0.75));
    tempInput.value = temp.toFixed(2);
    topPInput.value = topP.toFixed(2);
    maxTokensInput.value = String(maxTokens);
    renderInput.value = String(renderSize);
    state.renderSize = renderSize;
    renderStatus.textContent = `${state.renderSize}px`;
    renderers.forEach((renderer) => renderer.setSize(state.renderSize));
    sendParams();
  }

  function initEvents() {
    nextBtn.addEventListener('click', () => {
      state.manualSwap = true;
      requestNext('manual');
    });
    pauseBtn.addEventListener('click', () => {
      state.paused = !state.paused;
      pauseBtn.textContent = state.paused ? 'Resume' : 'Pause';
      modeStatus.textContent = state.paused ? 'Paused' : 'Auto';
      send({ type: 'pause', value: state.paused });
      if (!state.paused) {
        requestNext('auto');
      }
    });

    fullscreenBtn.addEventListener('click', () => {
      const stage = document.getElementById('stage');
      if (!document.fullscreenElement) {
        stage.requestFullscreen();
      } else {
        document.exitFullscreen();
      }
    });

    document.addEventListener('fullscreenchange', () => {
      document.body.classList.toggle('is-fullscreen', !!document.fullscreenElement);
    });

    overlayBtn.addEventListener('click', () => {
      state.overlay = !state.overlay;
      overlayBtn.classList.toggle('active', state.overlay);
      if (state.overlay && state.current) {
        overlayCode.textContent = state.current.body || '';
        overlayCode.classList.remove('hidden');
      } else {
        overlayCode.classList.add('hidden');
      }
      send({ type: 'toggle_overlay', value: state.overlay });
    });

    favoriteBtn.addEventListener('click', () => {
      if (state.current && state.current.shader_id) {
        send({ type: 'save_favorite', shader_id: state.current.shader_id });
      }
    });

    rawBtn.addEventListener('click', () => {
      state.rawMode = !state.rawMode;
      rawBtn.classList.toggle('active', state.rawMode);
      sendParams();
    });

    applyBtn.addEventListener('click', () => {
      state.displaySec = Number(displayInput.value) || state.displaySec;
      state.crossfadeSec = Number(fadeInput.value) || state.crossfadeSec;
      state.renderSize = Number(renderInput.value) || state.renderSize;
      renderStatus.textContent = `${state.renderSize}px`;
      renderers.forEach((renderer) => renderer.setSize(state.renderSize));
      sendParams();
    });

    reconnectBtn.addEventListener('click', connect);

    copyBtn.addEventListener('click', async () => {
      try {
        await navigator.clipboard.writeText(codeBlock.textContent || '');
        copyBtn.textContent = 'Copied';
        setTimeout(() => (copyBtn.textContent = 'Copy'), 1200);
      } catch (err) {
        copyBtn.textContent = 'Copy failed';
        setTimeout(() => (copyBtn.textContent = 'Copy'), 1200);
      }
    });
  }

  function boot() {
    loadImages(textureUrls)
      .then((images) => {
        try {
          renderers = [new ShaderRenderer(canvasA), new ShaderRenderer(canvasB)];
        } catch (err) {
          logBody.textContent = `renderer init failed: ${err.message}`;
          return;
        }
        renderers.forEach((renderer) => {
          renderer.setSize(state.renderSize);
          renderer.setTextures(images);
        });
        setCanvasActive(front);
        initEvents();
        connect();
        requestAnimationFrame(tick);
      })
      .catch((err) => {
        logBody.textContent = `texture load failed: ${err.message}`;
      });
  }

  boot();
})();
