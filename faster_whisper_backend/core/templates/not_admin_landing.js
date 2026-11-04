
// The landing's action buttons. Each label is its `href` (no "Open" verb:
// stacked verbs read like a redundant menu), so they read as links.
// `free` marks a page with no per-page permission key: /dictate is
// reachable by every signed-in identity (an ungated navlink in _NAV_SPEC,
// tier "any" on the hub), so a key with every page scope at 'none' still
// gets it.
var _PAGE_LINK_INFO = {
  quick_config: { href: '/quick-config' },
  captures:     { href: '/captures' },
  reports:      { href: '/reports' },
  stats:        { href: '/stats' },
  logs:         { href: '/logs' },
  dictate:      { href: '/dictate', free: true }
};

// Fresh whoami for the landing's page list: cached on success. Null when the
// request fails or answers non-ok; never rejects.
async function _refreshWhoami() {
  try {
    var r = await fetch('/auth/whoami');
    if (!r.ok) return null;
    var j = await r.json();
    try { window.__whoami = j; } catch (_) {}
    return j;
  } catch (_) { return null; }
}

function _renderNoAccessLanding(opts) {
  // The landing means the caller is not entitled to admin UI here.
  document.body.classList.remove('role-admin');
  var main = document.getElementsByTagName('main')[0];
  if (!main) return;

  var current = (opts && opts.page) || window.__current_page || '';
  var who = window.__whoami || {};
  var perms = (who.permissions && who.permissions.pages) || {};
  var isAdmin = !!who.is_admin;

  // Skip the current page: it is the one the caller can't reach.
  var allowed = Object.keys(_PAGE_LINK_INFO).filter(function(p) {
    if (p === current) return false;
    if (isAdmin || _PAGE_LINK_INFO[p].free) return true;
    return perms[p] && perms[p] !== 'none';
  });

  var btns = allowed.map(function(p) {
    var info = _PAGE_LINK_INFO[p];
    return '<a href="' + info.href + '" class="landing-btn">'
         + info.href + '</a>';
  }).join('');

  // Prefer OPEN_MODE_BANNER_JS's stashed URL path: a nested page's key
  // does not map back to it (/settings/api-keys). Never show the
  // __admin_only__ sentinel.
  var displayPath = window.__current_page_path || '';
  var slug;
  if (displayPath) {
    slug = ' to ' + displayPath;
  } else if (current && current !== '__admin_only__') {
    slug = ' to /' + current.replace(/_/g, '-');
  } else {
    slug = '';
  }
  var body;
  if (allowed.length) {
    body = '<p>Your API key does not grant access to this page.</p>'
         + '<p class="landing-hint">Pages you can access:</p>'
         + '<div class="landing-actions">' + btns + '</div>';
  } else {
    body = '<p>Your API key does not grant access to this page, and no '
         + 'other pages are available either. Ask an admin to grant '
         + 'access.</p>';
  }

  main.innerHTML =
    '<div class="no-access-landing">'
    + '<h2>No access' + slug + '</h2>'
    + body
    + '<p class="landing-signout">'
    + '<button onclick="window._signOut()">Sign out</button>'
    + '</p></div>';
}

// Legacy alias with no remaining in-tree caller.
function _renderNotAdminLanding() {
  _renderNoAccessLanding({ page: window.__current_page || '' });
}
