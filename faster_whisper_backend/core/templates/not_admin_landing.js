
// Per-page metadata used to render the landing's pill-style action
// buttons. `href` is what the button navigates to. The button label
// reuses `href` directly (no "Open " verb prefix — stacked verbs read
// like the redundant menu) so the buttons look + behave like links.
var _PAGE_LINK_INFO = {
  quick_config: { href: '/quick-config' },
  captures:     { href: '/captures' },
  reports:      { href: '/reports' },
  stats:        { href: '/stats' },
  logs:         { href: '/logs' }
};

function _renderNoAccessLanding(opts) {
  // Clear admin chrome — same reason it's removed: the landing shows
  // when the caller is NOT entitled to admin UI on this page.
  document.body.classList.remove('role-admin');
  var main = document.getElementsByTagName('main')[0];
  if (!main) return;

  var current = (opts && opts.page) || window.__current_page || '';
  var who = window.__whoami || {};
  var perms = (who.permissions && who.permissions.pages) || {};
  var isAdmin = !!who.is_admin;

  // Pages the caller can reach, minus the one they're on (it's the
  // page we can't reach — listing it would be confusing).
  var allowed = Object.keys(_PAGE_LINK_INFO).filter(function(p) {
    if (p === current) return false;
    if (isAdmin) return true;
    return perms[p] && perms[p] !== 'none';
  });

  var btns = allowed.map(function(p) {
    var info = _PAGE_LINK_INFO[p];
    // Button text is the URL path itself (`/logs`, `/quick-config`,
    // …) — drops the redundant "Open" verb and matches what the user
    // expects to see in the address bar.
    return '<a href="' + info.href + '" class="landing-btn">'
         + info.href + '</a>';
  }).join('');

  // Heading slug: prefer the full URL path stashed by OPEN_MODE_BANNER_JS
  // (e.g. "/settings/api-keys"), then fall back to a derived form, then to
  // an empty suffix. Admin-only pages get the URL too — never the bare
  // sentinel.
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

// Backwards-compat alias — old callers used `_renderNotAdminLanding()`.
function _renderNotAdminLanding() {
  _renderNoAccessLanding({ page: window.__current_page || '' });
}
