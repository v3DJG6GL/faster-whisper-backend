
<script>(function(){
  var KEY='whisper-ui-fs-base';
  var sel=document.getElementById('scale-picker');
  if(!sel)return;
  var saved=localStorage.getItem(KEY);
  if(saved){sel.value=saved;}
  sel.addEventListener('change',function(){
    document.documentElement.style.setProperty('--fs-base',sel.value+'px');
    localStorage.setItem(KEY,sel.value);
    sync();
  });
  var cyc=document.getElementById('scale-cycle');
  function sync(){if(cyc)cyc.title='UI scale '+sel.options[sel.selectedIndex].text+' — click for next';}
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
