
<script>(function(){
  // Read the page-key carrier ONCE so any helper (the no-access landing,
  // the SCALE_PICKER, future scope-hint UI) can ask "what page am I on?"
  // without re-querying the DOM.
  //   * __current_page = permission key (for can()/scope() lookups)
  //   * __current_page_path = full URL path (for display in the landing
  //     heading, so a nested page like /settings/api-keys reads correctly
  //     instead of showing just the permission key '/api-keys').
  try {
    var meta = document.querySelector('meta[name=page-key]');
    window.__current_page = meta ? (meta.getAttribute('content') || '') : '';
    window.__current_page_path = meta
      ? (meta.getAttribute('data-page-path') || '') : '';
  } catch(_) {
    window.__current_page = '';
    window.__current_page_path = '';
  }

  var BANNER_ID = 'open-mode-banner';
  var landingRenderedFor = null;   // page-load-only — once a no-access
                                   // landing is rendered, subsequent
                                   // refreshes (post-login) don't re-render.

  // CSRF token for cookie-authenticated mutations (double-submit). Prefer the
  // cached whoami payload (correct even if the cookie was renamed); fall back
  // to the readable whisper_csrf cookie (available synchronously on load).
  // The HttpOnly session cookie itself is sent automatically by the browser.
  window._csrfToken = function() {
    try {
      if (window.__whoami && window.__whoami.csrf_token)
        return window.__whoami.csrf_token;
    } catch(_) {}
    try {
      var m = document.cookie.match(/(?:^|;\s*)whisper_csrf=([^;]+)/);
      if (m) return decodeURIComponent(m[1]);
    } catch(_) {}
    return '';
  };

  // Global sign-out: revoke the server session (CSRF-protected), announce the
  // change, then reload (re-shows the login card / no-access landing). Shared
  // by the header logout button and the no-access landing's Sign-out button.
  window._signOut = function() {
    var done = function() {
      window.dispatchEvent(new Event('whisper:auth-changed'));
      location.reload();
    };
    try {
      fetch('/auth/logout', {
        method: 'POST',
        headers: { 'X-CSRF-Token': window._csrfToken() },
        cache: 'no-store',
      }).then(done, done);
    } catch(_) { done(); }
  };

  // ---- Unified login gate ----
  // One full-screen auth screen shared by every page (replaces the old
  // per-page #token-modal / #login-wrap). Lazily injected on first show; its
  // opaque cover hides the page until the user authenticates. _showLoginGate
  // is called by _refreshAuthChrome on a whoami-401 and by per-page 401
  // handlers; on success we reload so the page re-renders authenticated.
  var LG_HTML =
    '<div id="login-gate" role="dialog" aria-modal="true" aria-label="Sign in" hidden>'
  +   '<div class="lg-card">'
  +     '<div class="lg-brand">'
  +       '<svg class="lg-mark" viewBox="0 0 120 120" aria-hidden="true">'
  +         '<defs><linearGradient id="lg-fw" x1="0" y1="0" x2="1" y2="1">'
  +         '<stop offset="0" stop-color="#79c0ff"/><stop offset="1" stop-color="#7ee787"/>'
  +         '</linearGradient></defs>'
  +         '<rect x="6" y="6" width="108" height="108" rx="26" fill="#161b22" stroke="#30363d" stroke-width="2"/>'
  +         '<g transform="translate(13 2) skewX(-9)" fill="url(#lg-fw)">'
  +           '<rect class="lg-bar" x="16" y="74" width="11" height="20" rx="5.5"/>'
  +           '<rect class="lg-bar" x="35" y="52" width="11" height="42" rx="5.5"/>'
  +           '<rect class="lg-bar" x="54" y="22" width="11" height="72" rx="5.5"/>'
  +           '<rect class="lg-bar" x="73" y="44" width="11" height="50" rx="5.5"/>'
  +           '<rect class="lg-bar" x="92" y="66" width="11" height="28" rx="5.5"/>'
  +         '</g>'
  +       '</svg>'
  +       '<div class="lg-word"><span class="lg-a">faster</span><span class="lg-b">whisper</span>'
  +       '<span class="lg-sep">&rsaquo;</span><span class="lg-c">backend</span></div>'
  +     '</div>'
  +     '<p class="lg-sub">Authenticate to continue</p>'
  +     '<form class="lg-form" autocomplete="off">'
  +       '<div class="lg-field"><span class="lg-prompt" aria-hidden="true">&#9656;</span>'
  +       '<input id="lg-input" type="password" placeholder="wk_…" autocomplete="off" '
  +       'spellcheck="false" aria-label="API key"></div>'
  +       '<button id="lg-submit" type="submit" class="lg-btn">Authenticate</button>'
  +       '<p id="lg-err" class="lg-err" role="alert"></p>'
  +     '</form>'
  +     '<p class="lg-hint">Keys are issued per user in <code>/settings/api-keys</code>.</p>'
  +   '</div>'
  + '</div>';

  function _lgError(card, err, msg) {
    if (err) err.textContent = msg || '';
    if (!card) return;
    card.classList.remove('lg-shake');
    void card.offsetWidth;           // reflow → restart the shake animation
    card.classList.add('lg-shake');
  }

  function _ensureLoginGate() {
    var g = document.getElementById('login-gate');
    if (g) return g;
    var holder = document.createElement('div');
    holder.innerHTML = LG_HTML;
    g = holder.firstChild;
    (document.body || document.documentElement).appendChild(g);
    var form = g.querySelector('.lg-form');
    var input = g.querySelector('#lg-input');
    var err = g.querySelector('#lg-err');
    var btn = g.querySelector('#lg-submit');
    var card = g.querySelector('.lg-card');
    form.addEventListener('submit', function(ev) {
      ev.preventDefault();
      var key = (input.value || '').trim();
      if (!key) { _lgError(card, err, 'Enter your API key.'); input.focus(); return; }
      btn.disabled = true; btn.textContent = 'Authenticating…'; err.textContent = '';
      fetch('/auth/login', {
        method: 'POST', cache: 'no-store',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ key: key }),
      })
        .then(function(r) { if (!r.ok) throw 0; })
        .then(function() {
          try { window.dispatchEvent(new Event('whisper:auth-changed')); } catch(_) {}
          location.reload();
        })
        .catch(function() {
          btn.disabled = false; btn.textContent = 'Authenticate';
          _lgError(card, err, 'That key was rejected.');
          input.focus(); input.select();
        });
    });
    return g;
  }

  window._showLoginGate = function(msg) {
    var g = _ensureLoginGate();
    var err = g.querySelector('#lg-err');
    if (err) err.textContent = msg || '';
    g.hidden = false;
    var input = g.querySelector('#lg-input');
    setTimeout(function() { try { input.focus(); } catch(_) {} }, 30);
  };
  window._hideLoginGate = function() {
    var g = document.getElementById('login-gate');
    if (g) g.hidden = true;
  };

  // Header build chip — the page ships it empty (see _header_vtag_html); the
  // facts come from the authenticated whoami payload. Text and attributes only
  // (never innerHTML): the values are server-side strings, not markup.
  function _fillBuildChip(build) {
    var chip = document.getElementById('hdr-vtag');
    if (!chip || !build) return;
    chip.textContent = build.version_short || build.version || '';
    chip.setAttribute('data-tip',
      (build.server || '') + ' ' + (build.version || '') + '\n'
      + 'boot ' + (build.boot || '') + ' · started ' + (build.started || '')
      + '\n' + 'click to copy');
    chip.setAttribute('data-build',
      (build.server || '') + ' ' + (build.version || '')
      + ' (boot ' + (build.boot || '')
      + ', started ' + (build.started || '') + ')');
  }

  // Single source of truth for "what does the current bearer let me see?".
  // Idempotent: clears nav chrome first, then re-applies based on a fresh
  // /auth/whoami. Called once on page load AND on every `whisper:auth-changed`
  // event dispatched by login/logout sites (admin_routes, api_keys_routes,
  // quick_config_routes, reports_routes, captures_routes).
  window._refreshAuthChrome = function() {
    // Clear state before the fetch so logout (401 / removed token) leaves
    // the chrome hidden by default. The old IIFE only ADDed classes — it
    // had no way to recover from a transition admin → non-admin.
    try { document.body.classList.remove('role-admin'); } catch(_) {}
    try {
      var _chip = document.getElementById('hdr-vtag');
      if (_chip) {
        _chip.textContent = '';
        _chip.setAttribute('data-tip', '');
        _chip.setAttribute('data-build', '');
      }
    } catch(_) {}
    try {
      document.querySelectorAll('header a.page-link[data-page].allowed')
        .forEach(function(a){ a.classList.remove('allowed'); });
    } catch(_) {}

    // The HttpOnly session cookie (and any Authorization header from a
    // non-browser caller) is sent automatically — no manual header needed.
    fetch('/auth/whoami', {
      headers: { Accept: 'application/json' }, cache: 'no-store',
    })
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(j){
        if (!j) {
          // 401 (locked-down + no/invalid session). Leave chrome cleared
          // and drop the cached whoami so stale permissions don't linger,
          // then show the shared login gate so the user can authenticate.
          try { delete window.__whoami; } catch(_) {}
          _syncAuthActions();
          try { window._showLoginGate(); } catch(_) {}
          return;
        }
        // Cache the whoami payload so pages that want to consult
        // permissions later (e.g. for inline scope hints) don't re-fetch.
        try { window.__whoami = j; } catch(_) {}
        // Authenticated (or open mode): make sure the login gate is gone.
        try { window._hideLoginGate(); } catch(_) {}
        // Logout-button visibility tracks login state; the HttpOnly cookie
        // isn't JS-readable, so this is driven by whoami, not storage.
        _syncAuthActions();
        try { _fillBuildChip(j.build); } catch(_) {}

        // OPEN-mode warning banner — only when no admin key configured.
        // Idempotent: the banner gets a stable id so re-runs don't stack.
        // In normal flow (not sticky): the header below it is sticky at
        // top:0, so a sticky banner would sit on top of the brand row as
        // soon as the page scrolls. The banner is seen on every page load.
        if (j.open_mode && !document.getElementById(BANNER_ID)) {
          var b = document.createElement('div');
          b.id = BANNER_ID;
          b.setAttribute('role','alert');
          b.style.cssText = 'background:#5a2424;color:#fff;padding:0.5rem 1rem;'
            + 'text-align:center;font-weight:600;font-size:0.95rem;';
          b.innerHTML = '⚠ No admin API key set — the server is in '
            + 'OPEN mode and anyone reachable can use it. '
            + '<a href="/settings/api-keys" style="color:#ffd1d1;text-decoration:underline">'
            + 'Generate the first admin key</a>.';
          document.body.insertBefore(b, document.body.firstChild);
        }

        var isAdmin = !!j.is_admin;
        var perms = (j.permissions && j.permissions.pages) || {};

        // `body.role-admin` reveals admin-only chrome (/settings +
        // /settings/api-keys nav links, severity pills, in-page admin
        // buttons). Pages used to add it unconditionally after a successful
        // state fetch, which leaked admin chrome to non-admins on /stats,
        // /logs and /reports.
        if (isAdmin) document.body.classList.add('role-admin');

        // Per-page nav-link visibility. The nav renders every link with
        // class `.page-link` default-hidden; add `.allowed` per link the
        // caller can reach. Admins pass on every link via is_admin.
        document.querySelectorAll('header a.page-link[data-page]').forEach(
          function(a) {
            var page = a.getAttribute('data-page');
            var scope = perms[page];
            if (isAdmin || (scope && scope !== 'none')) {
              a.classList.add('allowed');
            }
          }
        );

        // No-access landing — page-load only. The landing replaces <main>
        // content; re-rendering it on every refresh would clobber a page
        // the user just successfully logged into. The per-page 403
        // handlers (admin_routes loadState, quick_config_routes loadState)
        // cover the "valid bearer, wrong scope" case after login.
        var current = window.__current_page;
        if (
          landingRenderedFor === null
          && current && current !== '__admin_only__'
          && !isAdmin
          && (!perms[current] || perms[current] === 'none')
          && typeof _renderNoAccessLanding === 'function'
        ) {
          landingRenderedFor = current;
          _renderNoAccessLanding({ page: current });
        }
      })
      .catch(function(){});
  };

  // Refresh on every login/logout. Each login/logout site dispatches
  // `whisper:auth-changed` after the server sets/clears the session cookie.
  window.addEventListener('whisper:auth-changed', window._refreshAuthChrome);

  // ---- Global sign-out ----
  // Every page renders #logout-btn (.auth-action) in the header utility
  // cluster. The HttpOnly session cookie isn't JS-readable, so visibility is
  // driven by the cached whoami: shown only when locked-down AND logged in
  // (open mode has no session to end). Clicking hits the server /auth/logout.
  function _syncAuthActions() {
    var loggedIn = false;
    try {
      loggedIn = !!(window.__whoami && window.__whoami.open_mode === false);
    } catch(_) {}
    document.querySelectorAll('.auth-action').forEach(function(el){
      el.hidden = !loggedIn;
    });
  }
  var _logoutBtn = document.getElementById('logout-btn');
  if (_logoutBtn) {
    _logoutBtn.addEventListener('click', function(){ window._signOut(); });
  }
  window.addEventListener('whisper:auth-changed', _syncAuthActions);
  _syncAuthActions();

  // ---- Global reload ----
  // Every page renders #reload-btn in the header utility cluster. Pages that
  // can refresh in place set `window._pageReload` (e.g. /settings + /settings/api-
  // keys re-fetch their data); elsewhere we do a full page reload. Read at
  // click time so the page's init can register the hook after this runs.
  var _reloadBtn = document.getElementById('reload-btn');
  if (_reloadBtn) {
    _reloadBtn.addEventListener('click', function(){
      if (typeof window._pageReload === 'function') { window._pageReload(); }
      else { location.reload(); }
    });
  }

  // Initial page-load pass — replaces the old IIFE body.
  window._refreshAuthChrome();
})();</script>
