import os
import pickle
import networkx as nx
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Reshape
from tensorflow.keras.models import Model
from spektral.layers import GCNConv
from spektral.layers.pooling import GlobalAvgPool
from keras.layers import Flatten
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#
def load_graphs_and_labels(directory):
    graphs = []
    labels = []
    label_mapping = {
        "reentrancy": 0,
        "safe": 1
    }

    # Load graphs and their labels
    for filename in os.listdir(directory):
        if filename.endswith('.graph'):
            graph_path = os.path.join(directory, filename)
            with open(graph_path, 'rb') as f:
                graph = pickle.load(f)
            graphs.append(graph)
            if "reentrancy" in filename:
                labels.append(label_mapping["reentrancy"])
            else:
                labels.append(label_mapping["safe"])
            print(f"Loaded graph from {graph_path} with label {labels[-1]}")
    
    return graphs, labels

def normalized_adjacency(adj):
    # Normalize the adjacency matrix
    degree = np.sum(adj, axis=1) + 1e-6  # Add a small epsilon value
    adj_normalized = adj / degree[:, None]
    return adj_normalized

def graph_to_features(graph, graphs):
    # Convert graph nodes and edges to feature and adjacency matrices
    node_labels = {n: i for i, n in enumerate(graph.nodes())}
    x = to_categorical([node_labels[n] for n in graph.nodes()], num_classes=len(graph.nodes()))
    edges = list(graph.edges())
    adj = np.zeros((len(graph.nodes()), len(graph.nodes())))
    for edge in edges:
        adj[node_labels[edge[0]], node_labels[edge[1]]] = 1
    adj_normalized = normalized_adjacency(adj)
    
    # Get the maximum shape of all graphs
    max_shape = (max(len(g.nodes()) for g in graphs), max(len(g.nodes()) for g in graphs))
    
    # Pad the adjacency matrix with zeros to match the maximum shape
    adj_padded = np.pad(adj_normalized, ((0, max_shape[0] - adj_normalized.shape[0]), (0, max_shape[1] - adj_normalized.shape[1])), mode='constant')
    
    # Pad the features with zeros to match the maximum shape
    x_padded = np.pad(x, ((0, max_shape[0] - x.shape[0]), (0, 0)), mode='constant')
    
    adj_sparse = tf.sparse.from_dense(adj_padded)
    return x_padded, adj_sparse

def prepare_graph_data(graphs, labels):
    features = []
    adjacencies = []
    for graph in graphs:
        x, adj = graph_to_features(graph, graphs)
        features.append(x)
        adjacencies.append(adj)
        print(f"Graph converted to features and adjacency matrix with shapes {x.shape} and {adj.shape}")
    
    # Logging the shapes for debugging
    feature_shapes = [f.shape for f in features]
    print(f"Feature shapes: {feature_shapes}")
    
    return features, adjacencies, np.array(labels)

def create_gnn_model(max_nodes, max_features):
    input_shape = (max_nodes, max_features)
    x_in = Input(shape=input_shape, name='x_in', dtype=tf.float32)
    a_in = Input(shape=(max_nodes, max_nodes), name='a_in', dtype=tf.float32)

    # Create the GNN model
    x = GCNConv(32, activation='relu')([x_in, a_in])
    x = GCNConv(32, activation='relu')([x, a_in])
    x = GlobalAvgPool()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    output = Dense(1, activation='sigmoid')(x)

    model = Model(inputs=[x_in, a_in], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def data_generator(x, a, y, batch_size, max_features, max_nodes):
    num_samples = len(x)
    while True:
        indices = np.random.choice(num_samples, batch_size)
        x_batch = [x[i] for i in indices]
        a_batch = [tf.sparse.to_dense(a[i]) for i in indices]  # Convert SparseTensor to dense tensor
        y_batch = np.array([y[i] for i in indices]).reshape(-1, 1)
        
        # Reshape the input data to have compatible shapes
        x_batch = [tf.pad(x, [[0, 0], [0, max_features - x.shape[1]]]) for x in x_batch]
        a_batch = [tf.reshape(a, (max_nodes, max_nodes)) for a in a_batch]
        
        x_batch = np.array(x_batch)
        a_batch = np.array(a_batch)
        
        yield ([x_batch, a_batch], y_batch)

def main():
    directory = "dataset_contracts"  # Specify the path to the directory with graphs

    # Load graphs and labels
    graphs, labels = load_graphs_and_labels(directory)
    
    # Determine the maximum number of nodes in graphs
    max_nodes = max(len(graph.nodes()) for graph in graphs)
    print(f"Maximum number of nodes in graphs: {max_nodes}")

    features, adjacencies, labels = prepare_graph_data(graphs, labels)
    
    # Split data into training and testing sets
    x_train, x_test, a_train, a_test, y_train, y_test = train_test_split(features, adjacencies, labels, test_size=0.4, stratify=labels)

    print(f"Training data shapes - x_train: {len(x_train)}, a_train: {len(a_train)}, y_train: {len(y_train)}")
    print(f"Testing data shapes - x_test: {len(x_test)}, a_test: {len(a_test)}, y_test: {len(y_test)}")

    max_features = max([x.shape[1] for x in x_train])

    # Create and compile the GNN model
    gnn_model = create_gnn_model(max_nodes, max_features)
    gnn_model.summary()
    
    # Training parameters
    epochs = 100
    batch_size = 1
    
    print(f"Training parameters - epochs: {epochs}, batch size: {batch_size}")

    train_gen = data_generator(x_train, a_train, y_train, batch_size, max_features, max_nodes)
    test_gen = data_generator(x_test, a_test, y_test, batch_size, max_features, max_nodes)

    # Train the model
    history = gnn_model.fit(train_gen, steps_per_epoch=len(x_train) // batch_size, epochs=epochs, validation_data=test_gen, validation_steps=len(x_test) // batch_size)

    # Evaluate the model on the test set
    loss, accuracy = gnn_model.evaluate(test_gen, steps=len(x_test) // batch_size)
    print(f"Test loss: {loss}, Test accuracy: {accuracy}")

    # Predictions on the test set
    y_test_pred = gnn_model.predict(test_gen, steps=len(x_test) // batch_size)
    y_test_pred = (y_test_pred > 0.5).astype(int)

    # Get all training data
    x_train_full = [tf.pad(x, [[0, 0], [0, max_features - x.shape[1]]]) for x in x_train]
    a_train_full = [tf.reshape(tf.sparse.to_dense(a), (max_nodes, max_nodes)) for a in a_train]

    # Convert to numpy arrays
    x_train_full = np.array(x_train_full)
    a_train_full = np.array(a_train_full)

    # Predictions on the training set
    y_train_pred = gnn_model.predict([x_train_full, a_train_full])
    y_train_pred = (y_train_pred > 0.5).astype(int)

    # Classification report and confusion matrix for the test set
    print("Classification Report (Test):")
    print(classification_report(y_test, y_test_pred, zero_division=0))
    print("Confusion Matrix (Test):")
    print(confusion_matrix(y_test, y_test_pred))

    # Classification report and confusion matrix for the training set
    print("Classification Report (Train):")
    print(classification_report(y_train, y_train_pred, zero_division=0))
    print("Confusion Matrix (Train):")
    print(confusion_matrix(y_train, y_train_pred))

    # Loss and accuracy plots
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Over Epochs')

    plt.show()

if __name__ == "__main__":
    main()