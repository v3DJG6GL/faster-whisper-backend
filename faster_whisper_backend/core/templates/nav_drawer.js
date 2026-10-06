
<script>(function(){
  var hdr=document.querySelector('header');
  var btn=document.querySelector('.nav-toggle');
  var nav=document.getElementById('navrow');
  var bd=document.querySelector('.nav-backdrop');
  if(!hdr||!btn||!nav)return;
  var lastFocus=null;
  // Mark everything OUTSIDE the header's branch inert while the drawer is open.
  // Walk header -> body and inert each path node's siblings, so this works
  // whether <header> is a direct <body> child (most pages) or wrapped in a
  // container like /settings' <div id="app-wrap"> (where inerting the wrapper
  // would otherwise cascade onto the header and kill the drawer links). Track
  // only what we set so close() never clears pre-existing inert.
  var _inerted=[];
  function inertRest(on){
    if(on){
      _inerted=[];
      var n=hdr;
      while(n&&n!==document.body){
        var p=n.parentElement; if(!p)break;
        Array.prototype.forEach.call(p.children,function(s){
          if(s!==n&&!s.hasAttribute('inert')){s.setAttribute('inert','');_inerted.push(s);}
        });
        n=p;
      }
    }else{
      _inerted.forEach(function(e){e.removeAttribute('inert');});
      _inerted=[];
    }
  }
  function onKey(e){if(e.key==='Escape'){e.preventDefault();close();}}
  function open(){
    if(hdr.classList.contains('nav-open'))return;
    lastFocus=document.activeElement;
    hdr.classList.add('nav-open');
    btn.setAttribute('aria-expanded','true');
    inertRest(true);
    var links=nav.querySelectorAll('.navlink'),first=null;
    for(var i=0;i<links.length;i++){if(links[i].offsetParent!==null){first=links[i];break;}}
    (first||btn).focus();
    document.addEventListener('keydown',onKey);
  }
  function close(){
    if(!hdr.classList.contains('nav-open'))return;
    hdr.classList.remove('nav-open');
    btn.setAttribute('aria-expanded','false');
    inertRest(false);
    document.removeEventListener('keydown',onKey);
    if(lastFocus&&lastFocus.focus)lastFocus.focus();else btn.focus();
  }
  btn.addEventListener('click',function(){
    hdr.classList.contains('nav-open')?close():open();
  });
  if(bd)bd.addEventListener('click',close);
  nav.addEventListener('click',function(e){if(e.target.closest('.navlink'))close();});
})();</script>
