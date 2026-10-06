
<script>
function _pad2(n) { return (n < 10 ? '0' : '') + n; }

function absTime(ts) {
  if (!ts) return '—';
  var d = new Date(ts * 1000);
  return d.getFullYear() + '.' + _pad2(d.getMonth() + 1) + '.' + _pad2(d.getDate())
    + ' | ' + _pad2(d.getHours()) + ':' + _pad2(d.getMinutes()) + ':' + _pad2(d.getSeconds());
}

function relTime(ts) {
  if (!ts) return '';
  var sec = Math.max(0, Date.now() / 1000 - ts);
  if (sec < 5)     return 'just now';
  if (sec < 60)    return Math.floor(sec) + 's ago';
  if (sec < 3600)  return Math.floor(sec / 60) + 'm ago';
  if (sec < 86400) return Math.floor(sec / 3600) + 'h ago';
  return '';
}

function fmtWhen(ts) {
  if (!ts) return '—';
  var a = absTime(ts), r = relTime(ts);
  return r ? (a + ' | ' + r) : a;
}

function timeTick(sel, ms) {
  sel = sel || '[data-ts]';
  ms = ms || 30000;
  function paint() {
    document.querySelectorAll(sel).forEach(function(el) {
      var ts = parseFloat(el.dataset.ts);
      if (!isFinite(ts) || ts <= 0) return;
      var next = fmtWhen(ts);
      if (el.textContent !== next) el.textContent = next;
      if (!el.title) el.title = absTime(ts);
    });
  }
  paint();
  setInterval(paint, ms);
}
</script>
