import json
from pathlib import Path
from typing import Optional

import networkx as nx
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = Path("affinity_cache")


def load_cache(name: str) -> list:
    path = CACHE_DIR / f"{name.replace(' ', '_').lower()}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)["data"]
    return []


def build_graph() -> nx.Graph:
    G = nx.Graph()

    tracker = load_cache("Master DVC Advantage Tracker")
    us_pipeline = load_cache("DVC US Pipeline")
    india_pipeline = load_cache("DVC India Pipeline")
    interactions = load_cache("DVC VC Ecosystem Interactions")
    contacts = load_cache("DVC Contacts 4D")

    def get_name(entry: dict) -> str:
        entity = entry.get("entity", {}) or {}
        if not isinstance(entity, dict):
            entity = {}
        aff = entry.get("_affinity") or {}
        if isinstance(aff, dict) and aff.get("company_name"):
            return str(aff["company_name"]).strip()
        if isinstance(aff, dict) and aff.get("contact_name"):
            return str(aff["contact_name"]).strip()
        return (
            entity.get("name")
            or f"{entity.get('first_name', '')} {entity.get('last_name', '')}".strip()
            or f"entity_{entry.get('entity_id', 'unknown')}"
        ).strip()

    def add_company(entry: dict, list_name: str):
        name = get_name(entry)
        if not name or name == "Unknown":
            return None
        entity_id = str(entry.get("entity_id", ""))
        G.add_node(
            name,
            type="company",
            list=list_name,
            date=(entry.get("created_at") or "")[:10],
            entity_id=entity_id,
        )
        org_ids = entry.get("organization_ids", [])
        person_ids = entry.get("person_ids", [])
        return name, org_ids, person_ids

    def add_contact(entry: dict):
        entity = entry.get("entity", {}) or {}
        if not isinstance(entity, dict):
            entity = {}
        aff = entry.get("_affinity") or {}
        if isinstance(aff, dict) and aff.get("contact_name"):
            name = str(aff["contact_name"]).strip()
            email = aff.get("email", "") or ""
        else:
            first = entity.get("first_name", "") or ""
            last = entity.get("last_name", "") or ""
            email = entity.get("primary_email", "") or ""
            name = f"{first} {last}".strip()
        if not name:
            return None
        G.add_node(
            name,
            type="person",
            email=email,
            entity_id=str(entry.get("entity_id", "")),
        )
        return name

    print("Building graph nodes...")

    company_nodes: dict = {}
    for entry in tracker:
        result = add_company(entry, "Master Tracker")
        if result:
            name, org_ids, person_ids = result
            company_nodes[name] = {"org_ids": org_ids, "person_ids": person_ids}

    for entry in us_pipeline:
        result = add_company(entry, "US Pipeline")
        if result:
            name, org_ids, person_ids = result
            company_nodes[name] = {"org_ids": org_ids, "person_ids": person_ids}

    for entry in india_pipeline:
        result = add_company(entry, "India Pipeline")
        if result:
            name, org_ids, person_ids = result
            company_nodes[name] = {"org_ids": org_ids, "person_ids": person_ids}

    contact_nodes: dict = {}
    for entry in contacts:
        name = add_contact(entry)
        if name:
            eid = str(entry.get("entity_id", ""))
            contact_nodes[eid] = name

    print(f"Nodes: {G.number_of_nodes()} entities")

    print("Building edges from interactions...")
    for entry in interactions:
        entity = entry.get("entity", {}) or {}
        if not isinstance(entity, dict):
            entity = {}
        aff = entry.get("_affinity") or {}
        company_name = (
            aff.get("company_name") if isinstance(aff, dict) else None
        ) or entity.get("name", "")
        person_ids = entry.get("person_ids", [])

        if company_name and G.has_node(company_name):
            for pid in person_ids:
                contact_name = contact_nodes.get(str(pid))
                if contact_name and G.has_node(contact_name):
                    if G.has_edge(company_name, contact_name):
                        G[company_name][contact_name]["weight"] = (
                            G[company_name][contact_name].get("weight", 1) + 1
                        )
                    else:
                        G.add_edge(
                            company_name,
                            contact_name,
                            relationship="interaction",
                            weight=1,
                        )

    print("Connecting shared relationships...")
    person_to_companies: dict = {}
    for company, data in company_nodes.items():
        for pid in data.get("person_ids", []):
            pid_str = str(pid)
            person_to_companies.setdefault(pid_str, []).append(company)

    for pid, companies in person_to_companies.items():
        contact_name = contact_nodes.get(pid)
        for company in companies:
            if (
                contact_name
                and G.has_node(contact_name)
                and G.has_node(company)
            ):
                if G.has_edge(company, contact_name):
                    G[company][contact_name]["weight"] = (
                        G[company][contact_name].get("weight", 1) + 1
                    )
                else:
                    G.add_edge(
                        company,
                        contact_name,
                        relationship="contact",
                        weight=2,
                    )
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                a, b = companies[i], companies[j]
                if G.has_node(a) and G.has_node(b):
                    if G.has_edge(a, b):
                        G[a][b]["weight"] = G[a][b].get("weight", 1) + 1
                    else:
                        G.add_edge(
                            a,
                            b,
                            relationship="shared_contact",
                            weight=1,
                        )

    print(f"Graph complete: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return G


def find_connections(G: nx.Graph, entity: str, _max_hops: int = 2) -> dict:
    if not G.has_node(entity):
        matches = [n for n in G.nodes() if entity.lower() in n.lower()]
        if not matches:
            return {"found": False, "entity": entity}
        entity = matches[0]

    result = {
        "found": True,
        "entity": entity,
        "type": G.nodes[entity].get("type"),
        "direct_connections": [],
        "second_hop": [],
        "path_summary": [],
    }

    neighbors = list(G.neighbors(entity))
    for neighbor in neighbors[:20]:
        edge_data = G.get_edge_data(entity, neighbor) or {}
        result["direct_connections"].append(
            {
                "name": neighbor,
                "type": G.nodes[neighbor].get("type"),
                "relationship": edge_data.get("relationship"),
                "weight": edge_data.get("weight", 1),
            }
        )

    result["direct_connections"].sort(key=lambda x: x["weight"], reverse=True)

    second_hop: set = set()
    for neighbor in neighbors[:10]:
        for second in G.neighbors(neighbor):
            if second != entity and second not in neighbors:
                second_hop.add(second)

    for node in list(second_hop)[:15]:
        paths = list(nx.all_simple_paths(G, entity, node, cutoff=2))
        if paths:
            first_path = paths[0]
            result["second_hop"].append(
                {
                    "name": node,
                    "type": G.nodes[node].get("type"),
                    "via": first_path[1] if len(first_path) > 1 else None,
                }
            )

    return result


def find_shortest_path(G: nx.Graph, source: str, target: str) -> dict:
    if not G.has_node(source):
        matches = [n for n in G.nodes() if source.lower() in n.lower()]
        if matches:
            source = matches[0]

    if not G.has_node(target):
        matches = [n for n in G.nodes() if target.lower() in n.lower()]
        if matches:
            target = matches[0]

    try:
        path = nx.shortest_path(G, source, target)
        return {
            "found": True,
            "path": path,
            "hops": len(path) - 1,
            "description": " → ".join(path),
        }
    except nx.NetworkXNoPath:
        return {"found": False, "path": [], "hops": -1}
    except nx.NodeNotFound:
        return {"found": False, "path": [], "hops": -1}


def get_most_connected(
    G: nx.Graph, node_type: Optional[str] = None, top_n: int = 10
) -> list:
    nodes = [
        (n, G.degree(n))
        for n in G.nodes()
        if not node_type or G.nodes[n].get("type") == node_type
    ]
    nodes.sort(key=lambda x: x[1], reverse=True)
    return [
        {"name": n, "connections": d, "type": G.nodes[n].get("type")}
        for n, d in nodes[:top_n]
    ]


_graph = None


def get_graph() -> nx.Graph:
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def invalidate_graph_cache() -> None:
    global _graph
    _graph = None
