
<script>(function(){
  if(!document.getElementById('sev-warn'))return;
  function setPill(id, n){
    var el=document.getElementById(id); if(!el)return;
    var numEl=el.querySelector('.n'); if(!numEl)return;
    var prev=+numEl.textContent || 0;
    numEl.textContent=n;
    el.classList.toggle('hot',  n > 0);
    el.classList.toggle('zero', n === 0);
    if(n > prev){
      el.classList.remove('flash'); void el.offsetWidth; el.classList.add('flash');
    }
  }
  function tick(){
    fetch('/sev', {cache:'no-store'})
      .then(function(r){ return r.ok ? r.json() : null; })
      .then(function(j){
        if(!j) return;
        setPill('sev-warn', j.warn|0);
        setPill('sev-err',  j.err |0);
        setPill('sev-crit', j.crit|0);
      })
      .catch(function(){});
  }
  tick();
  setInterval(tick, 5000);
})();</script>
