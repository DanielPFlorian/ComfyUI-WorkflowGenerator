from collections import defaultdict, deque


class WorkflowLayout:
    """
    Layout engine for ComfyUI workflows.
    Positions nodes in a logical left-to-right flow (DAG layout).
    """

    # Standard ComfyUI node dimensions and spacing
    NODE_WIDTH = 210
    NODE_HEIGHT_BASE = 60
    SLOT_HEIGHT = 20

    # Layout spacing
    RANK_SEP = 450  # Horizontal spacing between layers
    NODE_SEP = 150  # Vertical spacing between nodes

    def __init__(self, nodes: list[dict], links: list[list]):
        """
        Initialize layout engine.

        Args:
            nodes: List of node dictionaries
            links: List of link arrays [id, src_id, src_slot, dst_id, dst_slot, type]
        """
        self.nodes = nodes
        self.links = links
        self.node_map = {node["id"]: node for node in nodes}

    def apply_layout(self):
        """
        Calculate and apply positions to all nodes.
        Modifies the 'pos' attribute of nodes in-place.
        """
        if not self.nodes:
            return

        adj = defaultdict(list)
        in_degree = defaultdict(int)
        parents = defaultdict(list)

        for node in self.nodes:
            in_degree[node["id"]] = 0

        # Link format: [id, src_id, src_slot, dst_id, dst_slot, type]
        for link in self.links:
            if not link:
                continue

            if isinstance(link, (list, tuple)) and len(link) >= 5:
                src_id = link[1]
                dst_id = link[3]

                if src_id in self.node_map and dst_id in self.node_map:
                    adj[src_id].append(dst_id)
                    parents[dst_id].append(src_id)
                    in_degree[dst_id] += 1

        # Assign ranks (levels) using topological sort approach
        ranks = {}
        queue = deque()

        for node_id in self.node_map:
            if in_degree[node_id] == 0:
                queue.append((node_id, 0))
                ranks[node_id] = 0

        # We use a relaxed approach to handle cycles if any (though unlikely in valid workflows)
        processed_count = 0
        while queue:
            u, rank = queue.popleft()
            processed_count += 1

            for v in adj[u]:
                in_degree[v] -= 1
                current_rank = ranks.get(v, 0)
                new_rank = max(current_rank, rank + 1)
                ranks[v] = new_rank

                if in_degree[v] == 0:
                    queue.append((v, new_rank))

        # Handle any nodes remaining (cycles or disconnected components)
        for node_id in self.node_map:
            if node_id not in ranks:
                max_parent_rank = -1
                for p in parents[node_id]:
                    if p in ranks:
                        max_parent_rank = max(max_parent_rank, ranks[p])
                ranks[node_id] = max_parent_rank + 1

        layers = defaultdict(list)
        for node_id, rank in ranks.items():
            layers[rank].append(node_id)

        # Sort within layers to minimize crossings (heuristic) - sort by average parent Y position
        sorted_ranks = sorted(layers.keys())

        positions = {nid: [0, 0] for nid in self.node_map}

        for r in sorted_ranks:
            layer_nodes = layers[r]

            if r > 0:

                def get_avg_parent_y(nid):
                    ps = parents[nid]
                    if not ps:
                        return 0
                    known_parents = [p for p in ps if p in positions and positions[p][1] != 0]
                    if not known_parents:
                        return 0
                    return sum(positions[p][1] for p in known_parents) / len(known_parents)

                layer_nodes.sort(key=get_avg_parent_y)

            current_y = 0
            for node_id in layer_nodes:
                node = self.node_map[node_id]

                # Estimate node height based on input/output count
                input_count = len(node.get("inputs", []))
                output_count = len(node.get("outputs", []))
                height = self.NODE_HEIGHT_BASE + max(input_count, output_count) * self.SLOT_HEIGHT

                x = r * self.RANK_SEP
                y = current_y

                positions[node_id] = [x, y]
                node["pos"] = [x, y]

                current_y += height + self.NODE_SEP

            # Center the layer vertically around Y=0
            layer_height = current_y - self.NODE_SEP
            offset_y = -layer_height / 2

            for node_id in layer_nodes:
                new_y = int(self.node_map[node_id]["pos"][1] + offset_y)
                new_x = int(self.node_map[node_id]["pos"][0])
                self.node_map[node_id]["pos"] = [new_x, new_y]
                positions[node_id] = [new_x, new_y]

    @staticmethod
    def process(nodes: list[dict], links: list[list]):
        """Static convenience method"""
        layout = WorkflowLayout(nodes, links)
        layout.apply_layout()
