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


def _bump_company_person_edge(
    G: nx.Graph,
    company_name: str,
    person_name: str,
    relationship: str,
    weight: int,
) -> None:
    if not person_name or not G.has_node(person_name):
        return
    if G.has_edge(company_name, person_name):
        G[company_name][person_name]["weight"] = (
            G[company_name][person_name].get("weight", 1) + 1
        )
    else:
        G.add_edge(company_name, person_name, relationship=relationship, weight=weight)


def build_graph() -> nx.Graph:
    G = nx.Graph()

    tracker = load_cache("Master DVC Advantage Tracker")
    us_pipeline = load_cache("DVC US Pipeline")
    india_pipeline = load_cache("DVC India Pipeline")
    interactions = load_cache("DVC VC Ecosystem Interactions")
    contacts = load_cache("DVC Contacts 4D")

    print("Building contact index...")

    contact_by_entity_id: dict = {}
    contact_by_creator_id: dict = {}

    for entry in contacts:
        entity = entry.get("entity", {}) or {}
        if not isinstance(entity, dict):
            entity = {}
        affinity = entry.get("_affinity", {}) or {}
        if not isinstance(affinity, dict):
            affinity = {}
        name = (
            affinity.get("contact_name")
            or f"{entity.get('first_name', '')} {entity.get('last_name', '')}".strip()
        )
        if not name:
            continue
        entity_id = str(entry.get("entity_id", ""))
        creator_id = str(entry.get("creator_id", ""))
        email = affinity.get("email") or entity.get("primary_email", "") or ""

        G.add_node(name, type="person", entity_id=entity_id, email=email)

        if entity_id:
            contact_by_entity_id[entity_id] = name
        if creator_id:
            contact_by_creator_id[creator_id] = name

    print(f"Contacts indexed: {len(contact_by_entity_id)}")

    print("Building company nodes and edges from notes...")

    def process_list(entries: list, list_name: str) -> None:
        for entry in entries:
            entity = entry.get("entity", {}) or {}
            if not isinstance(entity, dict):
                entity = {}
            affinity = entry.get("_affinity", {}) or {}
            if not isinstance(affinity, dict):
                affinity = {}

            company_name = (
                affinity.get("company_name") or entity.get("name", "")
            ).strip()
            if not company_name:
                continue

            G.add_node(
                company_name,
                type="company",
                list=list_name,
                date=(entry.get("created_at") or "")[:10],
                entity_id=str(entry.get("entity_id", "")),
            )

            notes = affinity.get("notes", []) or []
            for note in notes:
                if not isinstance(note, dict):
                    continue

                creator_id = str(note.get("creator_id", "") or "")
                author = contact_by_entity_id.get(
                    creator_id
                ) or contact_by_creator_id.get(creator_id)
                if author:
                    _bump_company_person_edge(
                        G, company_name, author, "note_author", 1
                    )

                for pid in note.get("mentioned_person_ids", []) or []:
                    pid_str = str(pid)
                    if pid_str in contact_by_entity_id:
                        _bump_company_person_edge(
                            G,
                            company_name,
                            contact_by_entity_id[pid_str],
                            "mentioned",
                            1,
                        )

                for pid in note.get("interaction_person_ids", []) or []:
                    pid_str = str(pid)
                    if pid_str in contact_by_entity_id:
                        _bump_company_person_edge(
                            G,
                            company_name,
                            contact_by_entity_id[pid_str],
                            "interaction",
                            2,
                        )

                for pid in note.get("associated_person_ids", []) or []:
                    pid_str = str(pid)
                    if pid_str in contact_by_entity_id:
                        _bump_company_person_edge(
                            G,
                            company_name,
                            contact_by_entity_id[pid_str],
                            "associated",
                            1,
                        )

    process_list(tracker, "Master Tracker")
    process_list(us_pipeline, "US Pipeline")
    process_list(india_pipeline, "India Pipeline")
    process_list(interactions, "VC Ecosystem")

    print("Connecting companies via shared contacts...")
    # Skip pairwise company-company links for hub people (e.g. partners on 100s of
    # deals) — otherwise the graph becomes millions of dense shared_contact edges.
    max_companies_per_person_for_shared_clique = 48

    contact_to_companies: dict = {}
    for u, v, _data in G.edges(data=True):
        u_type = G.nodes[u].get("type")
        v_type = G.nodes[v].get("type")
        if u_type == "company" and v_type == "person":
            company, person = u, v
        elif u_type == "person" and v_type == "company":
            company, person = v, u
        else:
            continue
        contact_to_companies.setdefault(person, []).append(company)

    for person, companies in contact_to_companies.items():
        if len(companies) > max_companies_per_person_for_shared_clique:
            continue
        for i in range(len(companies)):
            for j in range(i + 1, len(companies)):
                c1, c2 = companies[i], companies[j]
                if G.has_edge(c1, c2):
                    G[c1][c2]["weight"] = G[c1][c2].get("weight", 1) + 1
                    via = G[c1][c2].get("via")
                    if via is None:
                        G[c1][c2]["via"] = [person]
                    elif person not in via:
                        via.append(person)
                else:
                    G.add_edge(
                        c1,
                        c2,
                        relationship="shared_contact",
                        weight=1,
                        via=[person],
                    )

    print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
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
