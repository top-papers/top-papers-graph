from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def balanced_graph_palette() -> Dict[str, str]:
    return {
        'step': '#4c78a8',
        'paper': '#7f8c8d',
        'term': '#72b7b2',
        'time': '#e0ac2b',
        'assertion': '#b279a2',
        'default': '#9aa5b1',
        'edge': '#94a3b8',
        'edge_temporal': '#7c8ba1',
        'edge_support': '#6ba292',
        'edge_warning': '#c28e6a',
    }


def node_visual_style(attrs: Dict[str, Any]) -> Dict[str, Any]:
    palette = balanced_graph_palette()
    raw_type = str(attrs.get('type') or attrs.get('node_type') or '').lower()
    label = str(attrs.get('label') or attrs.get('term') or attrs.get('id') or '')
    group = 'default'
    if 'trajectory' in raw_type or 'step' in raw_type:
        group = 'step'
    elif 'paper' in raw_type or attrs.get('paper_id') or attrs.get('papers'):
        group = 'paper'
    elif 'time' in raw_type or any(k in attrs for k in ('yearly_doc_freq', 'valid_from', 'valid_to')):
        group = 'time'
    elif 'assertion' in raw_type:
        group = 'assertion'
    elif attrs.get('term') or raw_type in {'term', 'entity', 'concept'}:
        group = 'term'
    return {'group': group, 'color': palette.get(group, palette['default']), 'label': label}


def edge_visual_style(attrs: Dict[str, Any]) -> Dict[str, Any]:
    palette = balanced_graph_palette()
    predicate = str(attrs.get('predicate') or attrs.get('label') or '').lower()
    color = palette['edge']
    if 'time' in predicate or 'valid' in predicate:
        color = palette['edge_temporal']
    elif any(tok in predicate for tok in ('support', 'influence', 'relate', 'correl', 'improve', 'increase', 'reduce')):
        color = palette['edge_support']
    elif any(tok in predicate for tok in ('contrad', 'conflict', 'reject')):
        color = palette['edge_warning']
    return {'color': color}


def networkx_from_payload(payload: Dict[str, Any]):
    import networkx as nx

    directed = True
    edges = payload.get('edges') or []
    if any(not bool(e.get('directed', True)) for e in edges if isinstance(e, dict)):
        directed = False
    G = nx.DiGraph() if directed else nx.Graph()

    for node in payload.get('nodes', []) or []:
        if not isinstance(node, dict):
            continue
        node_id = node.get('id') or node.get('term') or node.get('label')
        if node_id is None:
            continue
        attrs = dict(node)
        attrs.setdefault('label', attrs.get('label') or attrs.get('term') or str(node_id))
        style = node_visual_style(attrs)
        attrs.setdefault('viz_group', style['group'])
        attrs.setdefault('viz_color', style['color'])
        G.add_node(str(node_id), **attrs)

    for edge in edges:
        if not isinstance(edge, dict):
            continue
        src = edge.get('source')
        tgt = edge.get('target') or edge.get('object')
        if src is None or tgt is None:
            continue
        if str(src) not in G:
            G.add_node(str(src), label=str(src), viz_group='default', viz_color=balanced_graph_palette()['default'])
        if str(tgt) not in G:
            G.add_node(str(tgt), label=str(tgt), viz_group='default', viz_color=balanced_graph_palette()['default'])
        attrs = dict(edge)
        attrs.setdefault('viz_color', edge_visual_style(attrs)['color'])
        G.add_edge(str(src), str(tgt), **attrs)
    return G


def _strip_self_loops_for_metrics(graph):
    import networkx as nx

    metric_graph = graph.copy()
    self_loops = list(nx.selfloop_edges(metric_graph))
    if self_loops:
        metric_graph.remove_edges_from(self_loops)
    return metric_graph, len(self_loops)


def compute_graph_analytics(payload: Dict[str, Any], *, top_k: int = 20, max_cliques: int = 30) -> Dict[str, Any]:
    import networkx as nx

    G = networkx_from_payload(payload)
    H = G.to_undirected() if hasattr(G, 'to_undirected') else G
    G_metrics, directed_self_loop_count = _strip_self_loops_for_metrics(G)
    H_metrics, undirected_self_loop_count = _strip_self_loops_for_metrics(H)
    self_loop_count = max(int(directed_self_loop_count), int(undirected_self_loop_count))

    pagerank = nx.pagerank(G_metrics) if G_metrics.number_of_nodes() else {}
    degree = nx.degree_centrality(H_metrics) if H_metrics.number_of_nodes() > 1 else {n: 0.0 for n in H_metrics.nodes()}
    betweenness = nx.betweenness_centrality(H_metrics, normalized=True) if H_metrics.number_of_nodes() > 1 else {n: 0.0 for n in H_metrics.nodes()}
    closeness = nx.closeness_centrality(H_metrics) if H_metrics.number_of_nodes() > 1 else {n: 0.0 for n in H_metrics.nodes()}
    core_numbers = nx.core_number(H_metrics) if H_metrics.number_of_nodes() and H_metrics.number_of_edges() else {n: 0 for n in H_metrics.nodes()}

    communities_raw = []
    modularity = None
    if H_metrics.number_of_nodes() and H_metrics.number_of_edges():
        try:
            communities_raw = list(nx.algorithms.community.greedy_modularity_communities(H_metrics))
            if communities_raw:
                modularity = float(nx.algorithms.community.modularity(H_metrics, communities_raw))
        except Exception:
            communities_raw = []

    community_lookup: Dict[str, int] = {}
    communities: List[Dict[str, Any]] = []
    for idx, members in enumerate(communities_raw, start=1):
        sorted_members = sorted(str(x) for x in members)
        for node in sorted_members:
            community_lookup[node] = idx
        communities.append({'community_id': idx, 'size': len(sorted_members), 'nodes': sorted_members[:80]})

    cliques: List[Dict[str, Any]] = []
    if H_metrics.number_of_nodes() and H_metrics.number_of_edges():
        try:
            found = sorted((sorted(list(c)) for c in nx.find_cliques(H_metrics)), key=lambda c: (len(c), c), reverse=True)
            for clique in found[:max_cliques]:
                cliques.append({'size': len(clique), 'nodes': clique[:80]})
        except Exception:
            cliques = []

    components_raw = list(nx.connected_components(H_metrics)) if H_metrics.number_of_nodes() else []
    components = [
        {'component_id': idx, 'size': len(comp), 'nodes': sorted(str(x) for x in list(comp)[:80])}
        for idx, comp in enumerate(sorted(components_raw, key=lambda c: len(c), reverse=True), start=1)
    ]

    def _top(metric: Dict[str, float]) -> List[Dict[str, Any]]:
        items = sorted(((str(k), float(v)) for k, v in metric.items()), key=lambda kv: kv[1], reverse=True)
        out: List[Dict[str, Any]] = []
        for node_id, score in items[:top_k]:
            out.append({
                'node_id': node_id,
                'label': str(G.nodes[node_id].get('label') or node_id),
                'score': round(score, 6),
                'community_id': community_lookup.get(node_id),
                'group': str(G.nodes[node_id].get('viz_group') or 'default'),
            })
        return out

    node_metrics: Dict[str, Dict[str, Any]] = {}
    for node_id in G.nodes():
        node_metrics[str(node_id)] = {
            'label': str(G.nodes[node_id].get('label') or node_id),
            'group': str(G.nodes[node_id].get('viz_group') or 'default'),
            'community_id': community_lookup.get(str(node_id)),
            'pagerank': round(float(pagerank.get(node_id, 0.0)), 6),
            'degree': round(float(degree.get(node_id, 0.0)), 6),
            'betweenness': round(float(betweenness.get(node_id, 0.0)), 6),
            'closeness': round(float(closeness.get(node_id, 0.0)), 6),
            'core_number': int(core_numbers.get(node_id, 0)),
        }

    summary = {
        'node_count': int(G.number_of_nodes()),
        'edge_count': int(G.number_of_edges()),
        'self_loop_count': self_loop_count,
        'density': round(float(nx.density(H_metrics)) if H_metrics.number_of_nodes() > 1 else 0.0, 6),
        'is_directed': bool(G.is_directed()),
        'component_count': len(components),
        'largest_component_size': int(max((c['size'] for c in components), default=0)),
        'community_count': len(communities),
        'modularity': round(float(modularity), 6) if modularity is not None else None,
        'largest_clique_size': int(max((c['size'] for c in cliques), default=0)),
        'average_clustering': round(float(nx.average_clustering(H_metrics)) if H_metrics.number_of_nodes() > 1 and H_metrics.number_of_edges() else 0.0, 6),
    }

    return {
        'summary': summary,
        'communities': communities,
        'centrality': {
            'pagerank': _top(pagerank),
            'degree': _top(degree),
            'betweenness': _top(betweenness),
            'closeness': _top(closeness),
        },
        'cliques': cliques,
        'components': components,
        'node_metrics': node_metrics,
    }


def _spring_positions(payload: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    import networkx as nx
    G = networkx_from_payload(payload)
    if not G.number_of_nodes():
        return {}
    try:
        pos = nx.spring_layout(G.to_undirected(), seed=7, k=0.9 / max(1, G.number_of_nodes() ** 0.5))
    except Exception:
        pos = nx.circular_layout(G)
    out: Dict[str, Dict[str, float]] = {}
    for node_id, xy in pos.items():
        out[str(node_id)] = {'x': float(xy[0]), 'y': float(xy[1])}
    return out





def _short_text(value: Any, limit: int = 42) -> str:
    text = ' '.join(str(value or '').split())
    return text if len(text) <= limit else text[: limit - 1] + '…'


def build_interactive_graph_view(
    payload: Dict[str, Any],
    *,
    analytics: Dict[str, Any] | None = None,
    title: str = 'Интерактивный граф',
    width: int = 980,
    height: int = 680,
):
    """Return a Panel-based interactive graph view backed by hvPlot/Bokeh.

    Falls back to ``None`` when optional notebook visualization dependencies are
    unavailable so callers can keep the older HTML/SVG path.
    """
    G = networkx_from_payload(payload)
    try:
        import networkx as nx
        import holoviews as hv
        import hvplot.networkx as hvnx  # noqa: F401
        import panel as pn
        import pandas as pd
    except Exception:
        return G, None

    hv.extension('bokeh')
    try:
        pn.extension(sizing_mode='stretch_width')
    except Exception:
        pass

    analytics = analytics or compute_graph_analytics(payload)
    node_metrics = analytics.get('node_metrics') or {}
    communities = analytics.get('communities') or []
    cliques = analytics.get('cliques') or []
    summary = analytics.get('summary') or {}

    if G.number_of_nodes():
        try:
            pos = nx.spring_layout(G.to_undirected(), seed=7, k=0.95 / max(1, G.number_of_nodes() ** 0.5))
        except Exception:
            pos = nx.spring_layout(G, seed=7)
    else:
        pos = {}

    pagerank_max = max((float((node_metrics.get(str(node_id)) or {}).get('pagerank') or 0.0) for node_id in G.nodes()), default=0.0)
    pagerank_max = max(pagerank_max, 1e-9)

    for node_id, attrs in G.nodes(data=True):
        metrics = node_metrics.get(str(node_id)) or {}
        full_label = str(attrs.get('label') or attrs.get('term') or node_id)
        attrs['node_id'] = str(node_id)
        attrs['full_label'] = full_label
        attrs['short_label'] = _short_text(full_label, 30)
        attrs['group'] = str(attrs.get('viz_group') or attrs.get('type') or 'default')
        attrs['community_id'] = '' if metrics.get('community_id') in (None, '') else str(metrics.get('community_id'))
        attrs['pagerank'] = float(metrics.get('pagerank') or 0.0)
        attrs['degree'] = float(metrics.get('degree') or 0.0)
        attrs['betweenness'] = float(metrics.get('betweenness') or 0.0)
        attrs['closeness'] = float(metrics.get('closeness') or 0.0)
        attrs['core_number'] = int(metrics.get('core_number') or 0)
        attrs['node_fill_color'] = str(attrs.get('viz_color') or balanced_graph_palette()['default'])
        attrs['node_size'] = float(18.0 + 42.0 * attrs['pagerank'] / pagerank_max)

    for src, tgt, attrs in G.edges(data=True):
        attrs['start_label'] = str(G.nodes[src].get('full_label') or src)
        attrs['end_label'] = str(G.nodes[tgt].get('full_label') or tgt)
        attrs['edge_label'] = str(attrs.get('predicate') or attrs.get('label') or '')
        attrs['confidence'] = float(attrs.get('confidence') or 0.0)
        attrs['edge_line_color'] = str(attrs.get('viz_color') or balanced_graph_palette()['edge'])
        attrs['edge_width'] = 1.2 + min(2.2, attrs['confidence'] * 2.0)

    graph = hvnx.draw(
        G,
        pos,
        with_labels=False,
        node_color='node_fill_color',
        edge_color='edge_line_color',
        node_size='node_size',
        edge_line_width='edge_width',
        arrows=bool(getattr(G, 'is_directed', lambda: False)()),
        arrowhead_length=0.018,
        width=width,
        height=height,
    ).opts(
        bgcolor='#ffffff',
        responsive=False,
        xaxis=None,
        yaxis=None,
        padding=0.08,
        show_frame=False,
        tools=['pan', 'wheel_zoom', 'box_zoom', 'reset', 'save', 'tap', 'hover'],
        active_tools=['wheel_zoom'],
        toolbar='above',
        hover_tooltips=[
            ('Node', '@full_label'),
            ('Type', '@group'),
            ('Community', '@community_id'),
            ('PageRank', '@pagerank{0.0000}'),
            ('Degree', '@degree{0.0000}'),
            ('Betweenness', '@betweenness{0.0000}'),
            ('Closeness', '@closeness{0.0000}'),
            ('K-core', '@core_number'),
        ],
        inspection_policy='nodes',
    )

    label_nodes = [
        (node_id, float((node_metrics.get(str(node_id)) or {}).get('pagerank') or 0.0))
        for node_id in G.nodes()
    ]
    label_nodes = [node_id for node_id, _ in sorted(label_nodes, key=lambda item: item[1], reverse=True)[: min(14, G.number_of_nodes())]]
    if label_nodes:
        label_rows = [
            {
                'x': float(pos[node_id][0]),
                'y': float(pos[node_id][1]),
                'text': str(G.nodes[node_id].get('short_label') or node_id),
            }
            for node_id in label_nodes
            if node_id in pos
        ]
        labels = hv.Labels(pd.DataFrame(label_rows), kdims=['x', 'y'], vdims=['text']).opts(
            text_font_size='9pt',
            text_color='#0f172a',
            xoffset=6,
            yoffset=6,
        )
        graph = graph * labels

    summary_md = "\n".join([
        f"### {title}",
        "",
        f"- Nodes: **{summary.get('node_count', G.number_of_nodes())}**",
        f"- Edges: **{summary.get('edge_count', G.number_of_edges())}**",
        f"- Communities: **{summary.get('community_count', 0)}**",
        f"- Largest clique: **{summary.get('largest_clique_size', 0)}**",
        f"- Density: **{summary.get('density', 0)}**",
        "",
        "Наводите курсор на узлы, используйте zoom/pan в toolbar и кнопку Save для сохранения текущего интерактивного вида.",
    ])

    def _df(rows: List[Dict[str, Any]], columns: List[str]) -> Any:
        if not rows:
            return pn.pane.Markdown('_Нет данных_')
        frame = pd.DataFrame(rows)
        keep = [col for col in columns if col in frame.columns]
        frame = frame[keep] if keep else frame
        return pn.pane.DataFrame(frame, sizing_mode='stretch_width', index=False, height=260)

    centrality_rows: List[Dict[str, Any]] = []
    for metric_name in ('pagerank', 'degree', 'betweenness', 'closeness'):
        for row in (analytics.get('centrality') or {}).get(metric_name, [])[:12]:
            centrality_rows.append({
                'metric': metric_name,
                'label': row.get('label') or row.get('node_id'),
                'score': row.get('score'),
                'community_id': row.get('community_id'),
                'group': row.get('group'),
            })

    tabs = pn.Tabs(
        ('Top nodes', _df(centrality_rows, ['metric', 'label', 'score', 'community_id', 'group'])),
        ('Communities', _df(communities, ['community_id', 'size', 'nodes'])),
        ('Cliques', _df(cliques[:20], ['size', 'nodes'])),
        dynamic=False,
    )

    sidebar = pn.Column(
        pn.pane.Markdown(summary_md, sizing_mode='stretch_width'),
        tabs,
        width=360,
    )
    return G, pn.Row(sidebar, graph, sizing_mode='stretch_width')
def write_graph_analytics_json(graph_json_path: Path, analytics_path: Path) -> Path:
    payload = json.loads(graph_json_path.read_text(encoding='utf-8'))
    analytics = compute_graph_analytics(payload)
    analytics_path.write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding='utf-8')
    return analytics_path


def _build_graph_html(payload: Dict[str, Any], analytics: Dict[str, Any], title: str) -> str:
    doc = {
        'graph': payload,
        'analytics': analytics,
        'positions': _spring_positions(payload),
        'palette': balanced_graph_palette(),
        'title': title,
    }
    app_json = json.dumps(doc, ensure_ascii=False).replace('</', '<\\/')
    return f'''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    :root {{ --bg:#f8fafc; --card:#fff; --border:#d0d7de; --text:#0f172a; --muted:#475569; --accent:#2563eb; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--text); background:var(--bg); }}
    .wrap {{ max-width: 1480px; margin:0 auto; padding:20px; }}
    .hero {{ background:linear-gradient(135deg, rgba(37,99,235,.08), rgba(37,99,235,.02)); border:1px solid rgba(37,99,235,.16); border-radius:18px; padding:18px; margin-bottom:16px; }}
    .hero h1 {{ margin:0 0 8px; font-size:26px; }}
    .hero p {{ margin:0; color:var(--muted); line-height:1.45; }}
    .grid {{ display:grid; gap:16px; grid-template-columns: 360px minmax(0, 1fr); align-items:start; }}
    .card {{ background:var(--card); border:1px solid rgba(148,163,184,.28); border-radius:16px; padding:16px; box-shadow:0 12px 28px rgba(15,23,42,.06); }}
    .toolbar {{ display:grid; gap:10px; }}
    label.field {{ display:flex; flex-direction:column; gap:6px; font-size:13px; font-weight:700; }}
    select,input[type=range],input[type=number] {{ width:100%; }}
    .row {{ display:flex; gap:8px; align-items:center; flex-wrap:wrap; }}
    .pill {{ display:inline-flex; align-items:center; gap:6px; padding:6px 10px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:12px; font-weight:700; }}
    .legend {{ display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }}
    .legend span {{ display:inline-flex; align-items:center; gap:6px; font-size:12px; color:var(--muted); }}
    .legend i {{ width:11px; height:11px; border-radius:999px; display:inline-block; }}
    .stats {{ display:grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr)); gap:10px; margin-top:10px; }}
    .stats .mini {{ border:1px solid rgba(148,163,184,.22); border-radius:12px; padding:10px; background:#fff; }}
    .muted {{ color:var(--muted); }}
    .viz {{ min-height:760px; }}
    svg {{ width:100%; height:760px; display:block; border-radius:14px; background:#fff; border:1px solid rgba(148,163,184,.24); }}
    .edge-label {{ font-size:10px; fill:#64748b; }}
    .node-label {{ font-size:11px; fill:#111827; }}
    details {{ border-top:1px solid rgba(148,163,184,.2); padding-top:10px; margin-top:10px; }}
    summary {{ cursor:pointer; font-weight:700; }}
    table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
    th, td {{ text-align:left; border-bottom:1px solid rgba(148,163,184,.16); padding:8px 6px; vertical-align:top; font-size:13px; }}
    th {{ position:sticky; top:0; background:#f8fafc; }}
    .table-wrap {{ max-height:280px; overflow:auto; border:1px solid rgba(148,163,184,.18); border-radius:12px; }}
    .hidden {{ display:none!important; }}
    @media (max-width: 1080px) {{ .grid {{ grid-template-columns: 1fr; }} svg {{ height:620px; }} .viz {{ min-height:620px; }} }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>{title}</h1>
      <p>Страница объединяет визуализацию графа и результаты базовых/продвинутых графовых алгоритмов: сообщества, клики, центральности и структурные метрики. Переключатели справа позволяют включать и выключать подсветки, менять размер top-k, порог клики и режим окраски.</p>
      <div id="summaryPills" class="legend"></div>
    </section>
    <div class="grid">
      <aside class="card toolbar">
        <label class="field">Режим окраски<select id="colorMode"><option value="type">Тип узла</option><option value="community">Сообщество</option></select></label>
        <label class="field">Размер узлов<select id="sizeMetric"><option value="none">Фиксированный</option><option value="pagerank">PageRank</option><option value="degree">Degree centrality</option><option value="betweenness">Betweenness</option><option value="closeness">Closeness</option><option value="core_number">K-core</option></select></label>
        <label class="field">Показывать top-k узлов по выбранной метрике<input id="topK" type="range" min="3" max="50" step="1" value="15"><span id="topKValue" class="muted"></span></label>
        <label class="field">Минимальный размер клики<input id="minClique" type="range" min="2" max="8" step="1" value="3"><span id="minCliqueValue" class="muted"></span></label>
        <div class="row">
          <label><input id="showLabels" type="checkbox" checked> Подписи узлов</label>
          <label><input id="showEdgeLabels" type="checkbox"> Подписи рёбер</label>
          <label><input id="dimNonTop" type="checkbox"> Затемнять вне top-k</label>
        </div>
        <div class="row">
          <label><input id="showCommunities" type="checkbox" checked> Сообщества</label>
          <label><input id="showCentrality" type="checkbox" checked> Центральности</label>
          <label><input id="showCliques" type="checkbox" checked> Клики</label>
        </div>
        <div class="legend" id="legend"></div>
      </aside>
      <section class="card viz">
        <svg id="graphSvg" viewBox="0 0 1200 760" aria-label="graph visualization"></svg>
        <div class="stats" id="statsGrid"></div>
        <details id="communitiesCard" open><summary>Сообщества</summary><div class="table-wrap"><table id="communitiesTable"></table></div></details>
        <details id="centralityCard" open><summary>Центральности</summary><div class="table-wrap"><table id="centralityTable"></table></div></details>
        <details id="cliquesCard" open><summary>Клики</summary><div class="table-wrap"><table id="cliquesTable"></table></div></details>
        <details><summary>Raw graph JSON</summary><pre id="rawJson"></pre></details>
      </section>
    </div>
  </div>
  <script>
    const APP = {app_json};
    const state = {{ colorMode:'type', sizeMetric:'none', topK:15, minClique:3, showLabels:true, showEdgeLabels:false, dimNonTop:false, showCommunities:true, showCentrality:true, showCliques:true }};
    const htmlEscape = (v) => String(v ?? '').replace(/[&<>"']/g, (ch) => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
    const trunc = (v,n=64) => {{ const t=String(v??''); return t.length<=n ? t : t.slice(0,n-1)+'…'; }};
    function legend() {{
      const host = document.getElementById('legend'); host.innerHTML = '';
      if (state.colorMode === 'type') {{
        Object.entries(APP.palette).filter(([k]) => ['step','paper','term','time','assertion','default'].includes(k)).forEach(([name, color]) => {{
          const span = document.createElement('span'); span.innerHTML = `<i style="background:${{color}}"></i>${{htmlEscape(name)}}`; host.appendChild(span);
        }});
      }} else {{
        (APP.analytics.communities || []).slice(0, 12).forEach((item, idx) => {{
          const hue = (idx * 47) % 360; const color = `hsl(${{hue}} 58% 54%)`; const span = document.createElement('span');
          span.innerHTML = `<i style="background:${{color}}"></i>community ${{item.community_id}}`; host.appendChild(span);
        }});
      }}
    }}
    function metricMap(metric) {{
      const out = {{}}; Object.entries(APP.analytics.node_metrics || {{}}).forEach(([nodeId, row]) => {{ out[nodeId] = Number(row[metric] || 0); }}); return out;
    }}
    function topMetricNodes(metric, topK) {{
      const mm = metricMap(metric); return new Set(Object.entries(mm).sort((a,b) => b[1]-a[1]).slice(0, topK).map(([k]) => k));
    }}
    function nodeColor(nodeId, attrs) {{
      if (state.colorMode === 'community') {{
        const cid = Number((APP.analytics.node_metrics[nodeId] || {{}}).community_id || 0);
        if (cid) return `hsl(${{(cid * 47) % 360}} 58% 54%)`;
      }}
      return attrs.viz_color || APP.palette.default;
    }}
    function renderSummary() {{
      const pills = document.getElementById('summaryPills'); pills.innerHTML = '';
      const s = APP.analytics.summary || {{}};
      Object.entries(s).forEach(([k, v]) => {{ const span = document.createElement('span'); span.className = 'pill'; span.textContent = `${{k}}: ${{v ?? '—'}}`; pills.appendChild(span); }});
      const grid = document.getElementById('statsGrid'); grid.innerHTML = '';
      Object.entries(s).forEach(([k, v]) => {{ const div = document.createElement('div'); div.className = 'mini'; div.innerHTML = `<b>${{htmlEscape(k)}}</b><div class="muted">${{htmlEscape(v ?? '—')}}</div>`; grid.appendChild(div); }});
    }}
    function renderGraph() {{
      document.getElementById('topKValue').textContent = String(state.topK);
      document.getElementById('minCliqueValue').textContent = String(state.minClique);
      const svg = document.getElementById('graphSvg'); svg.innerHTML = '';
      const NS = 'http://www.w3.org/2000/svg';
      const width = 1200, height = 760, cx = width/2, cy = height/2;
      const nodes = Array.isArray(APP.graph.nodes) ? APP.graph.nodes : [];
      const edges = Array.isArray(APP.graph.edges) ? APP.graph.edges : [];
      const positions = APP.positions || {{}};
      const metric = state.sizeMetric;
      const mm = metric === 'none' ? {{}} : metricMap(metric);
      const maxMetric = Math.max(0.000001, ...Object.values(mm).map(Number), 0.000001);
      const topSet = metric === 'none' ? new Set() : topMetricNodes(metric, state.topK);
      const posFor = (node, idx) => {{
        const nodeId = String(node.id || node.term || node.label || `node-${{idx}}`);
        if (positions[nodeId]) return {{ x: cx + positions[nodeId].x * 290, y: cy + positions[nodeId].y * 245 }};
        const angle = (Math.PI * 2 * idx) / Math.max(1, nodes.length);
        return {{ x: cx + 280 * Math.cos(angle), y: cy + 230 * Math.sin(angle) }};
      }};
      const coords = {{}};
      nodes.forEach((node, idx) => {{ coords[String(node.id || node.term || node.label || `node-${{idx}}`)] = posFor(node, idx); }});
      edges.forEach((edge) => {{
        const src = String(edge.source || ''); const tgt = String(edge.target || edge.object || '');
        if (!coords[src] || !coords[tgt]) return;
        const line = document.createElementNS(NS, 'line');
        line.setAttribute('x1', coords[src].x); line.setAttribute('y1', coords[src].y);
        line.setAttribute('x2', coords[tgt].x); line.setAttribute('y2', coords[tgt].y);
        line.setAttribute('stroke', String(edge.viz_color || '#94a3b8')); line.setAttribute('stroke-opacity', '0.55'); line.setAttribute('stroke-width', '1.6');
        svg.appendChild(line);
        if (state.showEdgeLabels) {{
          const text = document.createElementNS(NS, 'text'); text.setAttribute('class', 'edge-label');
          text.setAttribute('x', (coords[src].x + coords[tgt].x) / 2); text.setAttribute('y', (coords[src].y + coords[tgt].y) / 2 - 4); text.setAttribute('text-anchor', 'middle');
          text.textContent = trunc(edge.predicate || edge.label || '', 22); svg.appendChild(text);
        }}
      }});
      nodes.forEach((node, idx) => {{
        const nodeId = String(node.id || node.term || node.label || `node-${{idx}}`);
        const attrs = APP.analytics.node_metrics[nodeId] || {{}};
        const baseColor = nodeColor(nodeId, node);
        const score = metric === 'none' ? 0 : Number(mm[nodeId] || 0);
        const radius = metric === 'none' ? 10.5 : (9 + (score / maxMetric) * 16);
        const inTop = metric === 'none' ? true : topSet.has(nodeId);
        const opacity = state.dimNonTop && !inTop ? 0.2 : 0.92;
        const circle = document.createElementNS(NS, 'circle');
        circle.setAttribute('cx', coords[nodeId].x); circle.setAttribute('cy', coords[nodeId].y); circle.setAttribute('r', radius.toFixed(2));
        circle.setAttribute('fill', baseColor); circle.setAttribute('opacity', opacity.toFixed(2)); circle.setAttribute('stroke', '#fff'); circle.setAttribute('stroke-width', '1.2');
        const title = document.createElementNS(NS, 'title'); title.textContent = `${{node.label || node.term || nodeId}}\ncommunity=${{attrs.community_id || '—'}}\npagerank=${{attrs.pagerank || 0}}\ndegree=${{attrs.degree || 0}}\ncore=${{attrs.core_number || 0}}`; circle.appendChild(title);
        svg.appendChild(circle);
        if (state.showLabels) {{
          const text = document.createElementNS(NS, 'text'); text.setAttribute('class', 'node-label'); text.setAttribute('x', coords[nodeId].x); text.setAttribute('y', coords[nodeId].y + radius + 14); text.setAttribute('text-anchor', 'middle'); text.textContent = trunc(node.label || node.term || nodeId, 24); text.style.opacity = opacity; svg.appendChild(text);
        }}
      }});
      legend();
    }}
    function fillTable(id, headers, rows) {{
      const table = document.getElementById(id); table.innerHTML = '';
      const thead = document.createElement('thead'); const tr = document.createElement('tr'); headers.forEach((h) => {{ const th = document.createElement('th'); th.textContent = h; tr.appendChild(th); }}); thead.appendChild(tr); table.appendChild(thead);
      const tbody = document.createElement('tbody'); rows.forEach((row) => {{ const tr = document.createElement('tr'); row.forEach((cell) => {{ const td = document.createElement('td'); td.innerHTML = cell; tr.appendChild(td); }}); tbody.appendChild(tr); }}); table.appendChild(tbody);
    }}
    function renderTables() {{
      document.getElementById('communitiesCard').classList.toggle('hidden', !state.showCommunities);
      document.getElementById('centralityCard').classList.toggle('hidden', !state.showCentrality);
      document.getElementById('cliquesCard').classList.toggle('hidden', !state.showCliques);
      fillTable('communitiesTable', ['community_id','size','nodes'], (APP.analytics.communities || []).map((row) => [htmlEscape(row.community_id), htmlEscape(row.size), htmlEscape((row.nodes || []).slice(0,10).join(', '))]));
      const centralityRows = [];
      ['pagerank','degree','betweenness','closeness'].forEach((metric) => {{ (APP.analytics.centrality[metric] || []).slice(0, state.topK).forEach((row, idx) => centralityRows.push([htmlEscape(metric), htmlEscape(idx + 1), `<b>${{htmlEscape(row.label || row.node_id)}}</b>`, htmlEscape(row.score), htmlEscape(row.community_id || '—')])); }});
      fillTable('centralityTable', ['metric','#','node','score','community'], centralityRows);
      const cliqueRows = (APP.analytics.cliques || []).filter((row) => Number(row.size || 0) >= state.minClique).map((row, idx) => [htmlEscape(idx + 1), htmlEscape(row.size), htmlEscape((row.nodes || []).join(', '))]);
      fillTable('cliquesTable', ['#','size','nodes'], cliqueRows);
      document.getElementById('rawJson').textContent = JSON.stringify(APP.graph, null, 2);
    }}
    ['colorMode','sizeMetric','topK','minClique','showLabels','showEdgeLabels','dimNonTop','showCommunities','showCentrality','showCliques'].forEach((id) => {{
      const el = document.getElementById(id); if (!el) return;
      el.addEventListener('input', () => {{ state[id] = el.type === 'checkbox' ? !!el.checked : (el.type === 'range' ? Number(el.value) : el.value); renderGraph(); renderTables(); }});
      state[id] = el.type === 'checkbox' ? !!el.checked : (el.type === 'range' ? Number(el.value) : el.value);
    }});
    renderSummary(); renderGraph(); renderTables();
  </script>
</body>
</html>'''




def write_graph_html(graph_json_path: Path, html_path: Path, *, analytics_path: Path | None = None) -> Path:
    payload = json.loads(graph_json_path.read_text(encoding='utf-8'))
    analytics = compute_graph_analytics(payload)
    if analytics_path is not None:
        analytics_path.write_text(json.dumps(analytics, ensure_ascii=False, indent=2), encoding='utf-8')
    title = graph_json_path.stem.replace('_', ' ')
    G, interactive_view = build_interactive_graph_view(payload, analytics=analytics, title=title)
    if interactive_view is not None:
        try:
            interactive_view.save(html_path, resources='inline', embed=False, title=title)
            return html_path
        except Exception:
            pass
    html_path.write_text(_build_graph_html(payload, analytics, title), encoding='utf-8')
    return html_path
