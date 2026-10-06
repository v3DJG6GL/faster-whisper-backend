
<script>
(function(){
  var hdr=document.querySelector('header');
  var nav=document.getElementById('navrow');
  var more=nav&&nav.parentElement.querySelector('.nav-more');
  if(!hdr||!nav||!more||!window.ResizeObserver)return;
  var inner=nav.parentElement;
  var btn=more.querySelector('.nav-more-btn'),list=more.querySelector('.nav-more-list');
  var toggle=document.querySelector('.nav-toggle');
  var title=inner.querySelector('.title'),status=inner.querySelector('.hdr-status');
  var items=Array.prototype.slice.call(nav.children).filter(function(c){
    return c.classList.contains('navlink')||c.classList.contains('nav-gsep');});
  var STEPS=['c1a','c1','c2','c7','c5','c6','c3','c4','c8','c9','c10','c11'],SINGLE_ROW_MAX=2;
  var busy=false;
  function navFits(){return nav.scrollWidth<=nav.clientWidth+1;}
  /* .title may shrink and clip its own wordmark — that is not a fit */
  function brandFits(){return !title||title.scrollWidth<=title.clientWidth+1;}
  function rowFits(){
    if(hdr.classList.contains('nav-row2')&&status&&title){
      var st=status.getBoundingClientRect();
      if(st.width>0){
        var br=title.getBoundingClientRect();
        var same=Math.abs((br.top+br.bottom)/2-(st.top+st.bottom)/2)<br.height/2;
        return same&&st.right<=inner.getBoundingClientRect().right+1&&brandFits();
      }
    }
    return inner.scrollWidth<=inner.clientWidth+1&&brandFits();
  }
  function fits(){return navFits()&&rowFits();}
  function reset(){STEPS.forEach(function(c){hdr.classList.remove(c);});}
  function ladder(ok,max){var n=max||STEPS.length;for(var i=0;i<n&&!ok();i++)hdr.classList.add(STEPS[i]);}
  function restore(){
    items.forEach(function(el){el.hidden=false;nav.appendChild(el);});
    list.replaceChildren();
  }
  function setOpen(o){btn.setAttribute('aria-expanded',o?'true':'false');list.hidden=!o;}
  function tuck(){
    more.hidden=false;
    for(var i=items.length-1;i>=0&&!navFits();i--){
      var el=items[i];
      if(el.classList.contains('active'))continue;
      if(el.classList.contains('nav-gsep')){el.hidden=true;continue;}
      /* a gated (unrendered) link frees no room and must not surface in the list */
      if(el.offsetParent===null)continue;
      var li=document.createElement('li');li.appendChild(el);list.insertBefore(li,list.firstChild);
    }
    /* a separator left trailing in the bar says nothing */
    var vis=items.filter(function(e){return e.parentElement===nav&&!e.hidden&&e.offsetParent!==null;});
    var last=vis[vis.length-1];if(last&&last.classList.contains('nav-gsep'))last.hidden=true;
    var n=list.querySelectorAll('.navlink').length;
    btn.querySelector('.cnt').textContent=n?String(n):'';
    if(!n)more.hidden=true;
  }
  function layout(){
    if(busy)return;busy=true;
    var wasOpen=!list.hidden;
    hdr.classList.remove('nav-row2');reset();restore();more.hidden=true;
    var drawer=toggle&&getComputedStyle(toggle).display!=='none';
    if(drawer){
      ladder(rowFits);            /* phone: brand row only, nav is off-canvas */
    }else{
      ladder(fits,SINGLE_ROW_MAX);
      if(!fits()){hdr.classList.add('nav-row2');reset();ladder(rowFits);}
      if(!navFits())tuck();
    }
    /* a relayout (resize, poller tick, gate change) must not slam an open menu shut */
    setOpen(wasOpen&&!more.hidden);
    busy=false;
  }
  var ro=new ResizeObserver(function(){requestAnimationFrame(layout);});
  ro.observe(inner);ro.observe(nav);if(title)ro.observe(title);if(status)ro.observe(status);
  /* Watch ONLY the links' class lists (the whoami gate adds .allowed);
     the rail's own class churn is covered by its ResizeObserver. */
  var mo=new MutationObserver(function(){requestAnimationFrame(layout);});
  mo.observe(nav,{attributes:true,attributeFilter:['class'],subtree:true});
  mo.observe(list,{attributes:true,attributeFilter:['class'],subtree:true});
  btn.addEventListener('click',function(){setOpen(list.hidden);});
  document.addEventListener('keydown',function(e){if(e.key==='Escape'&&!list.hidden){setOpen(false);btn.focus();}});
  document.addEventListener('click',function(e){if(!more.contains(e.target)&&!list.hidden)setOpen(false);});
  layout();
})();</script>
