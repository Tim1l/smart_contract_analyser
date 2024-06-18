import os
import json
import networkx as nx
import matplotlib.pyplot as plt

def build_graph_from_simplified_contract(simplified_contract):
    graph = nx.DiGraph()
    for contract_name, data in simplified_contract.items():
        abi = data['abi']
        for item in abi:
            if item['type'] == 'function' or item['type'] == 'event':
                node_label = item['name']
                graph.add_node(node_label, label=node_label)
    return graph

def find_withdraw_function_nodes(graph):
    withdraw_nodes = []
    for node in graph.nodes(data=True):
        if isinstance(node[1], dict) and 'label' in node[1] and 'withdraw' in node[1]['label']:
            withdraw_nodes.append(node)
    return withdraw_nodes

def main(contract_directory):
    for filename in os.listdir(contract_directory):
        if filename.endswith('_simplified.json'):
            json_path = os.path.join(contract_directory, filename)
            print(f"Processing {json_path}")
            
            # Загрузка данных из упрощенного JSON файла
            with open(json_path, 'r') as f:
                simplified_contract = json.load(f)
            
            # Построение графа
            contract_graph = build_graph_from_simplified_contract(simplified_contract)
            
            # Визуализация графа
            plt.figure(figsize=(12, 8))
            pos = nx.spring_layout(contract_graph)
            nx.draw(contract_graph, pos, with_labels=True, font_size=8, node_size=500, node_color="lightblue")
            plt.show()
            
            # Поиск узлов и ребер, связанных с функцией withdraw
            withdraw_function_nodes = find_withdraw_function_nodes(contract_graph)
            print("Nodes related to the withdraw function:", withdraw_function_nodes)

if __name__ == "__main__":
    main("vulnerable_contracts")    
