
<script>(function(){
  var KEY='whisper-ui-fs-base';
  var sel=document.getElementById('scale-picker');
  if(!sel)return;
  // Storage can be blocked (getItem throws) and a hand-edited value can match
  // no option (selectedIndex -1): either used to stop this IIFE before the
  // cycle button and the width toggle were wired.
  var saved=null;try{saved=localStorage.getItem(KEY);}catch(e){}
  if(saved&&[].some.call(sel.options,function(o){return o.value===saved;})){sel.value=saved;}
  sel.addEventListener('change',function(){
    document.documentElement.style.setProperty('--fs-base',sel.value+'px');
    try{localStorage.setItem(KEY,sel.value);}catch(e){}
    sync();
  });
  var cyc=document.getElementById('scale-cycle');
  function sync(){var o=sel.options[sel.selectedIndex];if(cyc&&o)cyc.title='UI scale '+o.text+' — click for next';}
  if(cyc){cyc.addEventListener('click',function(){
    sel.selectedIndex=(sel.selectedIndex+1)%sel.options.length;
    sel.dispatchEvent(new Event('change'));});}
  sync();
  // Page width preference — class on <html> (applied pre-paint by
  // SCALE_BOOTSTRAP_HEAD), persisted like the scale.
  var WKEY='whisper-ui-width', wt=document.getElementById('width-toggle');
  function wsync(){if(!wt)return;var f=document.documentElement.classList.contains('pref-fluid');
    wt.setAttribute('aria-pressed',f?'true':'false');
    wt.title='Page width: '+(f?'fluid — click for fixed':'fixed — click for fluid');}
  if(wt){wt.addEventListener('click',function(){
    var f=!document.documentElement.classList.contains('pref-fluid');
    document.documentElement.classList.toggle('pref-fluid',f);
    try{localStorage.setItem(WKEY,f?'fluid':'fixed');}catch(e){}
    wsync();});}
  wsync();
})();</script>
