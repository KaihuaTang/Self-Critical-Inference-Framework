// copy BibTeX
document.querySelectorAll('.copy').forEach(function (b) {
  b.addEventListener('click', function () {
    var pre = document.getElementById(b.dataset.copy);
    var text = pre.innerText.replace(/^copy\n?/, '').replace(/^copied\n?/, '');
    navigator.clipboard.writeText(text).then(function () { b.textContent = 'copied'; setTimeout(function () { b.textContent = 'copy'; }, 1400); });
  });
});
// chart hover layer: crosshair + tooltip
document.querySelectorAll('.chart svg[data-chart]').forEach(function (svg) {
  var d = JSON.parse(svg.dataset.chart), tip = svg.parentNode.querySelector('.tip'), cross = svg.querySelector('.cross');
  svg.querySelectorAll('.hit').forEach(function (r) {
    function show() {
      var i = +r.dataset.i, cx = +r.getAttribute('x') + +r.getAttribute('width') / 2;
      cross.setAttribute('x1', cx); cross.setAttribute('x2', cx); cross.style.visibility = 'visible';
      tip.innerHTML = '<b>' + d.x[i] + '</b>' + d.series.slice().sort(function (a, b) { return b.v[i] - a.v[i]; }).map(function (s) {
        return '<div><i style="background:' + s.color + '"></i>' + s.name + ' &nbsp;<strong>' + s.v[i].toFixed(2) + '</strong></div>'; }).join('');
      var box = svg.getBoundingClientRect(), host = svg.parentNode.getBoundingClientRect();
      var px = box.left - host.left + cx / svg.viewBox.baseVal.width * box.width;
      tip.style.left = Math.min(px + 12, host.width - tip.offsetWidth - 8) + 'px';
      tip.style.top = (box.top - host.top + 8) + 'px';
      tip.style.visibility = 'visible';
    }
    function hide() { tip.style.visibility = 'hidden'; cross.style.visibility = 'hidden'; }
    r.addEventListener('mouseenter', show); r.addEventListener('mouseleave', hide);
    r.addEventListener('touchstart', show, { passive: true });
  });
});
