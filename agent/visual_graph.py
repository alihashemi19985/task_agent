from agent.graph import build_graph

graph = build_graph()

graph.get_graph().draw_mermaid_png(
    output_file_path="graph.png"
)

print("Graph saved to graph.png")