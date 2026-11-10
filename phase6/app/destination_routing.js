/* =====================================================================
   SafeRoute -- "Route to any place" upgrade (add-on module)
   ---------------------------------------------------------------------
   Adds ONE new way to choose a destination. Everything that exists today
   (nearest police / nearest hospital, Online / Offline start location,
   SOS, safety heatmap, login, database) is left exactly as it is.

   Two boxes, like a maps app: FROM and TO.
     FROM  empty = your current / last recorded location (Online / Offline mode),
           or type a place, or pick it on the map.
     TO    type a place, or pick it on the map.
   Place search = OpenStreetMap Nominatim (needs internet). Press Enter or Search.

   How the route is computed -- identical to the existing feature:
     start  = FROM place, or (if FROM is empty) the same start node as the
              police/hospital buttons (nearestStartNodeId)
     routes = Dijkstra on the loaded walking graph, twice:
                weight 1 = safety-weighted cost  -> safest route
                weight 2 = physical length       -> shortest route
     output = both routes drawn on the map + the existing comparison cards.

   Uses only globals that index.html already defines: map, graph, adjacency,
   nodeById, nearestStartNodeId, dijkstra, routeStats, pathToLatLngs,
   clearRoutes, haversineKm, detectCityFromCoords, cityRegistry,
   activeCityKey, noRouteStartMessage, getRouteStartLocation, loadCityData, L.
   ===================================================================== */
(function () {
  'use strict';

  var SNAP_WARN_KM = 0.30;   // place further than this from a street: warn
  var SNAP_MAX_KM  = 1.00;   // ... further than this: refuse
  var WALK_KMH     = 5;      // assumed walking speed for the time estimate
  var CANDIDATES   = 6;      // nearest graph nodes tried if one is cut off from the network
  var NOMINATIM    = 'https://nominatim.openstreetmap.org/search';

  var place = { from: null, to: null };   // {lat, lon, name}; from === null means "my location"
  var pickTarget = null;                  // 'from' | 'to' | null
  var fromMarker = null;
  var searchSeq = { from: 0, to: 0 };
  var busy = false;

  function $(id) { return document.getElementById(id); }
  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // ------------------------------------------------------------------ UI
  function field(which, label, ph) {
    return '<label class="dr-lab" for="dr-' + which + '">' + label + '</label>' +
      '<div class="dr-row"><input type="text" id="dr-' + which + '" placeholder="' + ph + '" autocomplete="off">' +
      '<button class="btn dr-ic" id="dr-' + which + '-search" type="button" title="Search">🔍</button>' +
      '<button class="btn dr-ic" id="dr-' + which + '-pick" type="button" title="Pick on map">📍</button></div>' +
      '<ul class="dr-results" id="dr-' + which + '-results"></ul>';
  }

  function injectPanel() {
    if ($('dest-panel')) return;
    var style = document.createElement('style');
    style.textContent =
      '#dest-panel .dr-lab{display:block;margin-top:10px;font-size:11px;letter-spacing:.05em;text-transform:uppercase;color:var(--muted)}' +
      '#dest-panel .dr-row{display:flex;gap:6px;margin-top:4px}' +
      '#dest-panel input[type=text]{flex:1;min-width:0;padding:10px 12px;border-radius:9px;' +
      'border:1px solid var(--panel-border);background:#0F1829;color:var(--ink);font:500 13.5px Inter,sans-serif}' +
      '#dest-panel input[type=text]:focus{outline:none;border-color:var(--accent)}' +
      '#dest-panel .dr-ic{width:40px;padding:8px 0;margin:0 !important;flex:none}' +
      '#dest-panel .dr-ic.dr-active{background:var(--accent);border-color:var(--accent)}' +
      '#dest-panel .dr-results{list-style:none;margin:0;padding:0}' +
      '#dest-panel .dr-results li{padding:8px 10px;border:1px solid var(--panel-border);border-radius:8px;margin-top:6px;' +
      'background:#101A2E;font-size:12.5px;line-height:1.35;cursor:pointer;color:var(--ink)}' +
      '#dest-panel .dr-results li:hover{border-color:var(--accent)}' +
      '#dest-panel .dr-results li.dr-none{cursor:default;color:var(--muted)}' +
      '#dr-actions{display:flex;gap:8px;margin-top:12px}#dr-actions .btn{margin:0 !important}#dr-swap{width:auto;flex:none;padding:11px 14px}' +
      '#dr-status{margin-top:8px;font-size:12.5px;line-height:1.4;color:var(--muted)}' +
      '#dr-status.dr-warn{color:var(--mid)}#dr-status.dr-bad{color:var(--risk)}#dr-status.dr-ok{color:var(--safe)}' +
      '#dr-summary{margin-top:10px;font-size:12.5px;line-height:1.5;display:none}' +
      '#dr-summary .dr-line{display:flex;justify-content:space-between;gap:8px;padding:3px 0}' +
      '#dr-summary .dr-line b{font-weight:600;text-align:right}' +
      '#dr-summary .dr-note{color:var(--muted);margin-top:4px}' +
      '.leaflet-container.dr-picking{cursor:crosshair !important}';
    document.head.appendChild(style);

    var panel = document.createElement('div');
    panel.className = 'panel-block';
    panel.id = 'dest-panel';
    panel.innerHTML =
      '<h2>Route between any two places</h2>' +
      '<p style="margin:0">Leave <b>From</b> empty to start from your current (or last recorded) location. ' +
      'Type a place and press Enter, or use 📍 and tap the map.</p>' +
      field('from', 'From', 'My location (current / last recorded)') +
      field('to', 'To', 'Where do you want to go?') +
      '<div id="dr-actions"><button class="btn btn-primary" id="dr-go" type="button">Find safest route</button>' +
      '<button class="btn" id="dr-swap" type="button" title="Swap From and To">⇅</button></div>' +
      '<div id="dr-status"></div><div id="dr-summary"></div>';

    var anchor = $('btn-police');
    var block = anchor && anchor.closest ? anchor.closest('.panel-block') : null;
    if (block && block.parentNode) block.parentNode.insertBefore(panel, block);
    else { var sb = $('sidebar') || document.body; sb.appendChild(panel); }

    ['from', 'to'].forEach(function (w) {
      $('dr-' + w + '-search').addEventListener('click', function () { doSearch(w); });
      $('dr-' + w).addEventListener('keydown', function (e) { if (e.key === 'Enter') doSearch(w); });
      $('dr-' + w).addEventListener('input', function () {
        place[w] = null;                       // text edited: the old pick no longer applies
        if (w === 'from') removeFromMarker();
      });
      $('dr-' + w + '-pick').addEventListener('click', function () { setPickTarget(pickTarget === w ? null : w); });
    });
    $('dr-go').addEventListener('click', onGo);
    $('dr-swap').addEventListener('click', onSwap);
    var reset = $('btn-reset');
    if (reset) reset.addEventListener('click', function () { setSummary(null); setStatus('', ''); removeFromMarker(); });
  }

  function setStatus(text, kind) {
    var el = $('dr-status'); if (!el) return;
    el.textContent = text;
    el.className = kind ? 'dr-' + kind : '';
  }
  function setSummary(html) {
    var el = $('dr-summary'); if (!el) return;
    el.style.display = html ? 'block' : 'none';
    el.innerHTML = html || '';
  }
  function setPickTarget(w) {
    pickTarget = w;
    ['from', 'to'].forEach(function (x) { $('dr-' + x + '-pick').classList.toggle('dr-active', x === w); });
    map.getContainer().classList.toggle('dr-picking', !!w);
    if (w) setStatus('Tap the ' + (w === 'from' ? 'starting point' : 'destination') + ' on the map.', '');
  }
  function removeFromMarker() { if (fromMarker) { map.removeLayer(fromMarker); fromMarker = null; } }

  // -------------------------------------------------------------- search
  function cityViewbox() {
    var c = cityRegistry && activeCityKey && cityRegistry[activeCityKey];
    var b = c && c.boundary;
    return b ? '&viewbox=' + [b.lonMin, b.latMax, b.lonMax, b.latMin].join(',') : '';
  }

  function doSearch(w) {
    var q = $('dr-' + w).value.trim();
    var list = $('dr-' + w + '-results');
    list.innerHTML = '';
    if (!q) { if (w === 'to') setStatus('Type a destination first.', 'warn'); return; }
    var city = cityRegistry && activeCityKey && cityRegistry[activeCityKey];
    var full = city && !/chennai|coimbatore/i.test(q) ? q + ', ' + city.name : q;
    var url = NOMINATIM + '?format=jsonv2&limit=6&countrycodes=in&addressdetails=0' +
              cityViewbox() + '&q=' + encodeURIComponent(full);
    var seq = ++searchSeq[w];
    setStatus('Searching…', '');
    fetch(url, { headers: { 'Accept': 'application/json' } })
      .then(function (r) { if (!r.ok) throw new Error('HTTP ' + r.status); return r.json(); })
      .then(function (rows) {
        if (seq !== searchSeq[w]) return;
        if (!rows.length) {
          list.innerHTML = '<li class="dr-none">No match. Try a more specific name, or use 📍 to pick it on the map.</li>';
          setStatus('', ''); return;
        }
        setStatus('Choose a result for ' + (w === 'from' ? 'From' : 'To') + ':', '');
        rows.forEach(function (row) {
          var li = document.createElement('li');
          li.textContent = row.display_name;
          li.addEventListener('click', function () {
            list.innerHTML = '';
            var short = row.display_name.split(',').slice(0, 2).join(',').trim();
            setPlace(w, parseFloat(row.lat), parseFloat(row.lon), short);
          });
          list.appendChild(li);
        });
      })
      .catch(function () {
        if (seq !== searchSeq[w]) return;
        setStatus('Place search needs an internet connection and could not be reached. ' +
                  'You can still use 📍 to pick on the map.', 'bad');
      });
  }

  function setPlace(w, lat, lon, name) {
    if (!isFinite(lat) || !isFinite(lon)) { setStatus('That place has no usable coordinates.', 'bad'); return; }
    place[w] = { lat: lat, lon: lon, name: name };
    $('dr-' + w).value = name;
    $('dr-' + w + '-results').innerHTML = '';
    if (w === 'from') {
      removeFromMarker();
      fromMarker = L.circleMarker([lat, lon], { radius: 8, color: '#0B1220', weight: 2,
        fillColor: '#22D3B0', fillOpacity: 1 }).addTo(map).bindTooltip('Start: ' + name);
    }
    if (place.to) onGo(); else setStatus(w === 'from' ? 'Start set. Now choose the destination.' : '', '');
  }

  function onGo() {
    // Resolve typed-but-not-chosen text by searching, never by guessing.
    if (!place.to) {
      if ($('dr-to').value.trim()) { doSearch('to'); return; }
      setStatus('Choose a destination first.', 'warn'); return;
    }
    if (!place.from && $('dr-from').value.trim()) { doSearch('from'); return; }
    routeBetween(place.from, place.to);
  }

  function onSwap() {
    var oldFrom = place.from, oldTo = place.to;
    var here = null;
    if (!oldFrom) {
      var s = typeof getRouteStartLocation === 'function' ? getRouteStartLocation() : null;
      if (s) here = { lat: s.lat, lon: s.lon, name: 'My location' };
    }
    var newFrom = oldTo, newTo = oldFrom || here;
    place.from = newFrom; place.to = newTo;
    $('dr-from').value = newFrom ? newFrom.name : '';
    $('dr-to').value = newTo ? newTo.name : '';
    removeFromMarker();
    if (newFrom) fromMarker = L.circleMarker([newFrom.lat, newFrom.lon], { radius: 8, color: '#0B1220', weight: 2,
      fillColor: '#22D3B0', fillOpacity: 1 }).addTo(map).bindTooltip('Start: ' + newFrom.name);
    if (place.from && place.to) onGo();
  }

  // ------------------------------------------------------------ routing
  function nearestNodes(lat, lon, k) {
    var best = [];
    for (var i = 0; i < graph.nodes.length; i++) {
      var n = graph.nodes[i];
      var dLat = n.lat - lat, dLon = (n.lon - lon) * Math.cos(lat * Math.PI / 180);
      var d2 = dLat * dLat + dLon * dLon;
      if (best.length < k || d2 < best[best.length - 1].d2) {
        best.push({ node: n, d2: d2 });
        best.sort(function (a, b) { return a.d2 - b.d2; });
        if (best.length > k) best.pop();
      }
    }
    return best.map(function (b) { return b.node; });
  }

  function walkTime(km) {
    var min = Math.round(km / WALK_KMH * 60);
    return min < 60 ? min + ' min' : Math.floor(min / 60) + ' h ' + (min % 60) + ' min';
  }

  function snapCheck(lat, lon, label) {
    var c = nearestNodes(lat, lon, CANDIDATES);
    var d = haversineKm(lat, lon, c[0].lat, c[0].lon);
    if (d > SNAP_MAX_KM) {
      return { error: label + ' is about ' + d.toFixed(1) + ' km from the nearest street in the loaded road network, so a walking route cannot be calculated.' };
    }
    return { cands: c, snapKm: d };
  }

  function routeBetween(from, to) {
    if (busy) return Promise.resolve();
    busy = true;
    return routeBetweenInner(from, to).then(function () { busy = false; }, function (e) {
      busy = false; console.error(e); setStatus('Something went wrong while routing. Please try again.', 'bad');
    });
  }

  function routeBetweenInner(from, to) {
    if (!cityRegistry) { setStatus('The map data is still loading. Try again in a moment.', 'warn'); return Promise.resolve(); }

    var startCity, startLabel;
    if (from) { startCity = detectCityFromCoords(from.lat, from.lon); startLabel = 'The starting point'; }
    else {
      var s = getRouteStartLocation();
      if (!s || nearestStartNodeId === null || nearestStartNodeId === undefined) {
        setStatus(noRouteStartMessage() + ' Or type a starting place in From.', 'warn'); return Promise.resolve();
      }
      startCity = activeCityKey; startLabel = 'Your location';
    }
    var destCity = detectCityFromCoords(to.lat, to.lon);
    if (!startCity) { setStatus(startLabel + ' is outside the supported cities (Chennai and Coimbatore).', 'bad'); return Promise.resolve(); }
    if (!destCity) { setStatus('That destination is outside the supported cities (Chennai and Coimbatore), so no route can be calculated.', 'bad'); return Promise.resolve(); }
    if (startCity !== destCity) {
      setStatus(startLabel + ' is in ' + cityRegistry[startCity].name + ' but the destination is in ' + cityRegistry[destCity].name +
                '. Routes can only be calculated inside one city’s road network.', 'bad');
      return Promise.resolve();
    }

    var ready = Promise.resolve();
    if (destCity !== activeCityKey || !graph || !adjacency) {
      setStatus('Loading the ' + cityRegistry[destCity].name + ' map data…', '');
      ready = loadCityData(destCity, { source: 'destination routing' });
    }
    return ready.then(function () { drawRoutes(from, to); });
  }

  function drawRoutes(from, to) {
    if (!graph || !adjacency) { setStatus('The map data is still loading. Try again in a moment.', 'warn'); return; }

    var dest = snapCheck(to.lat, to.lon, 'The destination');
    if (dest.error) { setStatus(dest.error, 'bad'); return; }
    var startCands, startSnap = 0;
    if (from) {
      var st = snapCheck(from.lat, from.lon, 'The starting point');
      if (st.error) { setStatus(st.error, 'bad'); return; }
      startCands = st.cands; startSnap = st.snapKm;
    } else {
      startCands = [nodeById.get(nearestStartNodeId)];
    }

    var startNode = null, destNode = null, safe = null, shortest = null;
    outer:
    for (var i = 0; i < startCands.length; i++) {
      for (var j = 0; j < dest.cands.length; j++) {
        var s1 = dijkstra(startCands[i].id, dest.cands[j].id, 1);
        if (s1) {
          startNode = startCands[i]; destNode = dest.cands[j]; safe = s1;
          shortest = dijkstra(startNode.id, destNode.id, 2);
          break outer;
        }
      }
    }

    clearRoutes();            // existing function: removes old routes, marker and cards
    setSummary(null);
    var title = to.name || (to.lat.toFixed(5) + ', ' + to.lon.toFixed(5));
    destinationMarker = L.marker([to.lat, to.lon]).addTo(map).bindPopup('Destination: ' + esc(title)).openPopup();

    if (!safe || !shortest) {
      setStatus('No street-connected route was found between these places. One of them may be cut off in the loaded road network. Try a nearby point.', 'bad');
      return;
    }

    shortRouteLayer = L.polyline(pathToLatLngs(shortest.path), {
      color: '#8792A8', weight: 3, dashArray: '6 6', opacity: 0.85 }).addTo(map);
    safeRouteLayer = L.polyline(pathToLatLngs(safe.path), {
      color: '#8B7CF6', weight: 4.5, opacity: 0.95 }).addTo(map);
    map.fitBounds(safeRouteLayer.getBounds(), { padding: [60, 60] });

    var a = routeStats(safe.path), b = routeStats(shortest.path);
    $('safe-dist').textContent = '~' + a.km + ' km';
    $('safe-score').textContent = 'avg ' + a.avg;
    $('short-dist').textContent = '~' + b.km + ' km';
    $('short-score').textContent = 'avg ' + b.avg;
    $('route-stats-block').style.display = 'block';

    var same = safe.path.length === shortest.path.length &&
               safe.path.every(function (id, k) { return id === shortest.path[k]; });
    var extraKm = parseFloat(a.km) - parseFloat(b.km);
    var gain = parseFloat(a.avg) - parseFloat(b.avg);
    var note;
    if (same) note = 'The shortest route is also the safest one here.';
    else note = 'The safest route is ' + (Math.abs(extraKm) < 0.005 ? 'about the same length' :
                (extraKm >= 0 ? extraKm.toFixed(2) + ' km longer' : Math.abs(extraKm).toFixed(2) + ' km shorter')) +
                ' and its average safety score is ' + (gain >= 0 ? gain.toFixed(1) + ' higher.' : Math.abs(gain).toFixed(1) + ' lower.');
    if (parseFloat(a.avg) < 40) note += ' Both routes pass through low-scoring areas; consider travelling with someone or by vehicle.';

    setSummary(
      '<div class="dr-line"><span>From</span><b>' + esc(from ? from.name : 'My location') + '</b></div>' +
      '<div class="dr-line"><span>To</span><b>' + esc(title) + '</b></div>' +
      '<div class="dr-line"><span>Safest route</span><b>' + a.km + ' km · ~' + walkTime(parseFloat(a.km)) + '</b></div>' +
      '<div class="dr-line"><span>Lowest score on it</span><b>' + a.min + '</b></div>' +
      '<div class="dr-line"><span>Shortest route</span><b>' + b.km + ' km · ~' + walkTime(parseFloat(b.km)) + '</b></div>' +
      '<div class="dr-note">' + esc(note) + ' Walking time assumes ' + WALK_KMH + ' km/h.</div>');

    var snapped = Math.max(haversineKm(to.lat, to.lon, destNode.lat, destNode.lon),
                           from ? haversineKm(from.lat, from.lon, startNode.lat, startNode.lon) : 0);
    if (snapped > SNAP_WARN_KM) setStatus('Route drawn to the nearest street, about ' + Math.round(snapped * 1000) + ' m from a place you chose.', 'warn');
    else setStatus('Route ready.', 'ok');
    var r = $('btn-reset'); if (r) r.disabled = false;
  }

  // For other code / tests: routeToDestination(lat, lon, name) = From "My location" to a place.
  window.routeToDestination = function (lat, lon, name) {
    place.to = { lat: lat, lon: lon, name: name || (lat.toFixed(5) + ', ' + lon.toFixed(5)) };
    $('dr-to').value = place.to.name;
    return routeBetween(place.from, place.to);
  };

  // ------------------------------------------------------- map picking
  function onMapClick(e) {
    if (!pickTarget) return;
    var w = pickTarget;
    setPickTarget(null);
    var lat = e.latlng.lat, lon = e.latlng.lng;
    setPlace(w, lat, lon, 'Dropped pin (' + lat.toFixed(4) + ', ' + lon.toFixed(4) + ')');
  }

  function init() {
    if (typeof map === 'undefined' || typeof dijkstra !== 'function') {
      console.warn('destination_routing.js: SafeRoute map code not found; module not started.');
      return;
    }
    injectPanel();
    map.on('click', onMapClick);
  }
  init();
})();
