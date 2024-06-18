import os
import json
import networkx as nx
from solcx import compile_standard, install_solc, set_solc_version
from solidity_parser import parser
import pickle

def setup_solc(version='0.4.15'):
    install_solc(version)
    set_solc_version(version)

# Setting up the Solidity compiler version
setup_solc('0.4.15')

def compile_contract(contract_path):
    with open(contract_path, 'r') as file:
        source_code = file.read()
    compiled_sol = compile_standard({
        "language": "Solidity",
        "sources": {
            contract_path: {
                "content": source_code
            }
        },
        "settings": {
            "outputSelection": {
                "*": {
                    "*": [
                        "metadata", "evm.bytecode", "evm.bytecode.sourceMap", "abi"
                    ]
                }
            }
        }
    })
    print(f"Compiled Contract: {json.dumps(compiled_sol, indent=2)}")
    return compiled_sol

def simplify_compiled_contract(compiled_contract):
    simplified_contract = {}
    for contract in compiled_contract['contracts']:
        for name, data in compiled_contract['contracts'][contract].items():
            simplified_contract[name] = {
                'abi': data['abi'],
                'bytecode': data['evm']['bytecode']['object']
            }
    return simplified_contract

def save_contract_json(contract_data, output_path):
    with open(output_path, 'w') as f:
        json.dump(contract_data, f, indent=2)

def parse_ast(contract_path):
    with open(contract_path, 'r') as file:
        source_code = file.read()
    ast = parser.parse(source_code)
    print(f"AST: {json.dumps(ast, indent=2)}")
    return ast

def build_graph_from_ast(ast):
    graph = nx.DiGraph()

    def add_nodes_edges(node, parent=None, level=0):
        if isinstance(node, dict):
            for key, value in node.items():
                node_id = parent + '->' + key if parent else key
                graph.add_node(node_id)
                if parent:
                    graph.add_edge(parent, node_id)
                if isinstance(value, (dict, list)):
                    add_nodes_edges(value, node_id, level + 1)
        elif isinstance(node, list):
            for i, item in enumerate(node):
                node_id = parent + '[' + str(i) + ']'
                graph.add_node(node_id)
                graph.add_edge(parent, node_id)
                add_nodes_edges(item, node_id, level + 1)

    add_nodes_edges(ast)
    print(f"Graph Nodes: {list(graph.nodes)}")
    print(f"Graph Edges: {list(graph.edges)}")
    return graph

def extract_functions_and_events(ast):
    functions = []
    events = []

    def traverse(node):
        if isinstance(node, dict):
            if node.get('type') == 'FunctionDefinition':
                functions.append(node.get('name'))
            elif node.get('type') == 'EventDefinition':
                events.append(node.get('name'))
            for key, value in node.items():
                traverse(value)
        elif isinstance(node, list):
            for item in node:
                traverse(item)
    
    traverse(ast)
    return functions, events

def main(contract_directory):
    for filename in os.listdir(contract_directory):
        if filename.endswith('.sol'):
            contract_path = os.path.join(contract_directory, filename)
            print(f"Processing {contract_path}")
            
            # Compile the contract and print the result
            compiled_contract = compile_contract(contract_path)
            
            # Save the full JSON output
            full_json_path = contract_path.replace('.sol', '_full.json')
            save_contract_json(compiled_contract, full_json_path)
            print(f"Full contract saved to {full_json_path}")

            # Simplify the compiled contract and save to JSON
            simplified_contract = simplify_compiled_contract(compiled_contract)
            simplified_json_path = contract_path.replace('.sol', '_simplified.json')
            save_contract_json(simplified_contract, simplified_json_path)
            print(f"Simplified contract saved to {simplified_json_path}")
            
            # Parse the AST and print the result
            ast = parse_ast(contract_path)
            
            # Extract functions and events
            functions, events = extract_functions_and_events(ast)
            print(f"Functions: {functions}")
            print(f"Events: {events}")
            
            # Build a graph from the AST and print the result
            graph = build_graph_from_ast(ast)
            
            # Save the graph to a file using pickle
            output_path = contract_path.replace('.sol', '.graph')
            with open(output_path, 'wb') as f:
                pickle.dump(graph, f)
            print(f"Graph saved to {output_path}")

if __name__ == "__main__":
    main("dataset_contracts")