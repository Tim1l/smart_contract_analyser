import os
import json
import networkx as nx
import matplotlib.pyplot as plt

def build_detailed_graph_from_full_contract(full_contract):
    graph = nx.DiGraph()
    for contract_name, contract_data in full_contract['contracts'].items():
        for name, data in contract_data.items():
            if 'abi' in data:
                contract_node = f"contract: {contract_name}"
                graph.add_node(contract_node, label=contract_node)
                print(f"Added contract node: {contract_node}")
                for item in data['abi']:
                    if item['type'] == 'function':
                        node_label = f"function: {item['name']}"
                        graph.add_node(node_label, label=node_label)
                        graph.add_edge(contract_node, node_label)
                        print(f"Added function node: {node_label}")
                    elif item['type'] == 'event':
                        node_label = f"event: {item['name']}"
                        graph.add_node(node_label, label=node_label)
                        graph.add_edge(contract_node, node_label)
                        print(f"Added event node: {node_label}")
    return graph

def find_function_and_event_nodes(graph):
    function_nodes = []
    event_nodes = []
    for node in graph.nodes(data=True):
        if isinstance(node[1], dict) and 'label' in node[1]:
            if node[1]['label'].startswith('function:'):
                function_nodes.append(node)
            elif node[1]['label'].startswith('event:'):
                event_nodes.append(node)
    return function_nodes, event_nodes

def main(contract_directory):
    for filename in os.listdir(contract_directory):
        if filename.endswith('_full.json'):
            json_path = os.path.join(contract_directory, filename)
            print(f"Processing {json_path}")
            
            # Загрузка данных из полного JSON файла
            with open(json_path, 'r') as f:
                full_contract = json.load(f)
            
            # Построение детализированного графа
            contract_graph = build_detailed_graph_from_full_contract(full_contract)
            
            # Визуализация графа
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(contract_graph)
            nx.draw(contract_graph, pos, with_labels=True, font_size=8, node_size=500, node_color="lightblue")
            plt.show()
            
            # Поиск узлов и ребер, связанных с функциями и событиями
            function_nodes, event_nodes = find_function_and_event_nodes(contract_graph)
            print("Nodes related to functions:", function_nodes)
            print("Nodes related to events:", event_nodes)
            
            # Поиск узлов и ребер, связанных с функцией withdraw
            withdraw_function_nodes = [node for node in function_nodes if 'withdraw' in node[1]['label']]
            print("Nodes related to the withdraw function:", withdraw_function_nodes)

if __name__ == "__main__":
    main("dataset_contracts")

