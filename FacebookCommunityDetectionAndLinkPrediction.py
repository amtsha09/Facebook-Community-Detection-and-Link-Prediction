from collections import Counter, defaultdict, deque
import copy
import math
import networkx as nx

## Community Detection

def bfs(graph, root, max_depth):
    """
    Perform breadth-first search to compute the shortest paths from a root node to all
    other nodes in the graph. 

    Params:
      graph.......A networkx Graph
      root........The root node in the search graph (a string). We are computing
                  shortest paths from this node to all others.
      max_depth...An integer representing the maximum depth to search.

    Returns:
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    """
    dq = deque()
    node2distances = defaultdict(int)
    node2num_paths = defaultdict(int)
    node2parents = defaultdict(list)
    visited = []
    visited.append(root)
    dq.append(root)
    node2num_paths[root]=1
    while dq:
        parent = dq.popleft()
        children = graph.neighbors(parent)
        for child in children:
            if node2distances[parent] < max_depth:
                if child not in visited:
                    visited.append(child)
                    dq.append(child)
                    node2distances[child] = node2distances[parent] + 1
                if node2distances[child] == node2distances[parent] + 1:
                    node2parents[child].append(parent)
                    node2num_paths[child] = len(node2parents[child])

    #temp = sorted((node, sorted(parents)) for node, parents in node2parents.items())
    return node2distances, node2num_paths, node2parents


def bottom_up(root, node2distances, node2num_paths, node2parents):
    """
    Compute the final step of the Girvan-Newman algorithm.

    Params:
      root.............The root node in the search graph (a string). We are computing
                       shortest paths from this node to all others.
      node2distances...dict from each node to the length of the shortest path from
                       the root node
      node2num_paths...dict from each node to the number of shortest paths from the
                       root node that pass through this node.
      node2parents.....dict from each node to the list of its parents in the search
                       tree
    Returns:
      A dict mapping edges to credit value. 
      
    """
    node2dist = sorted(node2distances.items(), key=lambda x: -x[1])
    edgeOfNode = defaultdict(float)


    for i in node2parents:
        edgeOfNode[i] = 1/node2num_paths[i]

    for i,j in node2dist:
        for k in node2parents[i]:
            edgeOfNode[k]=edgeOfNode[k]+edgeOfNode[i]
    edgeOfNode.pop(root)

    edgeDict = {}
    for i,j in node2dist:
        for k in node2parents[i]:
            edgeDict[tuple(sorted([i,k]))] = edgeOfNode[i]

    return edgeDict


def approximate_betweenness(graph, max_depth):
    """
    Compute the approximate betweenness of each edge, using max_depth to reduce
    computation time in breadth-first search.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A dict mapping edges to betweenness. 
      
    """
    ApproxDict = defaultdict()
    for root in graph.nodes():
        node2distances, node2num_paths, node2parents = bfs(graph, root, max_depth )
        bottomDict = bottom_up(root, node2distances, node2num_paths, node2parents)
        for edge in bottomDict:
            if edge in ApproxDict:
                ApproxDict[edge] = ApproxDict[edge]+bottomDict[edge]
            else:
                ApproxDict[edge]= bottomDict[edge]

    for i in ApproxDict:
        ApproxDict[i] = ApproxDict[i]/2

    return ApproxDict


def partition_girvan_newman(graph, max_depth):
    """
    Use approximate_betweenness implementation to partition a graph.
    Compute the approximate betweenness of all edges, and remove
    them until multiple comonents are created.

    Params:
      graph.......A networkx Graph
      max_depth...An integer representing the maximum depth to search.

    Returns:
      A list of networkx Graph objects, one per partition.

    
    """
    count = 0
    components = [c for c in nx.connected_component_subgraphs(graph)]
    copied_graph = graph.copy()
    edge_to_remove = sorted(approximate_betweenness(copied_graph, max_depth).items(), key=lambda x: (-x[1],x[0]))
    while len(components) == 1:
        copied_graph.remove_edge(*edge_to_remove[count][0])
        components = [c for c in nx.connected_component_subgraphs(copied_graph)]
        count+=1

    return components


def get_subgraph(graph, min_degree):
    """Return a subgraph containing nodes whose degree is
    greater than or equal to min_degree.
    We'll use this in the main method to prune the original graph.

    Params:
      graph........a networkx graph
      min_degree...degree threshold
    Returns:
      a networkx graph, filtered as defined above.

    
    """
    subgraph = graph.copy()
    for node in subgraph.degree().items():
        if node[1] < min_degree:
            subgraph.remove_node(node[0])
    return subgraph


def volume(nodes, graph):
    """
    Compute the volume for a list of nodes, which
    is the number of edges in `graph` with at least one end in
    nodes.
    Params:
      nodes...a list of strings for the nodes to compute the volume of.
      graph...a networkx graph

    
    """
    count = 0
    for edge in graph.edges():
        if edge[0] in nodes:
            count +=1
        elif edge[1] in nodes:
            count +=1
    return count


def cut(S, T, graph):
    """
    Compute the cut-set of the cut (S,T), which is
    the set of edges that have one endpoint in S and
    the other in T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An int representing the cut-set.

    """
    count = 0
    for edge in graph.edges():
        if edge[0] in S and edge[1] in T or edge[0] in T and edge[1] in S:
            count += 1

    return count


def norm_cut(S, T, graph):
    """
    The normalized cut value for the cut S/T.
    Params:
      S.......set of nodes in first subset
      T.......set of nodes in second subset
      graph...networkx graph
    Returns:
      An float representing the normalized cut value

    """
    volS = volume(S, graph)
    volT = volume(T, graph)
    cutST = cut(S, T, graph)
    NCV = (cutST/volS) + (cutST/volT)
    return NCV


def score_max_depths(graph, max_depths):
    """
    In order to assess the quality of the approximate partitioning method
    we've developed, we will run it with different values for max_depth
    and see how it affects the norm_cut score of the resulting partitions.
    Recall that smaller norm_cut scores correspond to better partitions.

    Params:
      graph........a networkx Graph
      max_depths...a list of ints for the max_depth values to be passed
                   to calls to partition_girvan_newman

    Returns:
      A list of (int, float) tuples representing the max_depth and the
      norm_cut value obtained by the partitions returned by
      partition_girvan_newman. See Log.txt for an example.
    """
    smd = []
    for depth in max_depths:
        components = partition_girvan_newman(graph, depth)
        S = components[0]
        T = components[1]
        ncv = norm_cut(S, T, graph)
        smd.append((depth,ncv))
    return smd


## Link prediction


def make_training_graph(graph, test_node, n):
    """
    To make a training graph, we need to remove n edges from the graph.
    Remove the edges to the first n neighbors of
    test_node, where the neighbors are sorted alphabetically.
    
    Params:
      graph.......a networkx Graph
      test_node...a string representing one node in the graph whose
                  edges will be removed.
      n...........the number of edges to remove.

    Returns:
      A *new* networkx Graph with n edges removed.

   
    """
    oldgraph = graph.copy()
    li = sorted(graph.neighbors(test_node))
    for i in li[:n]:
        oldgraph.remove_edge(i,test_node)
    return oldgraph


def jaccard(graph, node, k):
    """
    Compute the k highest scoring edges to add to this node based on
    the Jaccard similarity measure.

    Params:
      graph....a networkx graph
      node.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.

    Returns:
      A list of tuples in descending order of score representing the
      recommended new edges. Ties are broken by
      alphabetical order of the terminal node in the edge.
   
    """
    neighbors = set(graph.neighbors(node))
    scores = []
    for n in graph.nodes():
        if node!=n and not graph.has_edge(node,n):
            neighbors2 = set(graph.neighbors(n))
            scores.append(((node,n), 1. * len(neighbors & neighbors2) / len(neighbors | neighbors2)))
    return sorted(scores, key=lambda x: (-x[1],x[0][1]))[:k]


def path_score(graph, root, k, beta):
    """
    Compute a new link prediction scoring function based on the shortest
    paths between two nodes.

    Params:
      graph....a networkx graph
      root.....a node in the graph (a string) to recommend links for.
      k........the number of links to recommend.
      beta.....the beta parameter in the equation above.

    Returns:
      A list of tuples in descending order of score. Ties are broken by
      alphabetical order of the terminal node in the edge.

    """
    node2distances, node2num_ppaths, node2parents = bfs(graph, root, math.inf)
    score = []
    for n in graph.nodes():
        if root!=n and not graph.has_edge(n,root):
            score.append(((root, n), (beta ** node2distances[n]) * node2num_ppaths[n]))
    return sorted(score, key = lambda x:(-x[1],x[0][1]))[:k]



def evaluate(predicted_edges, graph):
    """
    Return the fraction of the predicted edges that exist in the graph.

    Args:
      predicted_edges...a list of edges (tuples) that are predicted to
                        exist in this graph
      graph.............a networkx Graph

    Returns:
      The fraction of edges in predicted_edges that exist in the graph.

    """
    count = 0
    for edg in predicted_edges:
        if graph.has_edge(*edg):
            count+=1
    return count/len(predicted_edges)



def read_graph():
    """ Read 'edges.txt.gz' into a networkx **undirected** graph.
    Returns:
      A networkx undirected graph.
    """
    return nx.read_edgelist('edges.txt.gz', delimiter='\t')


def main():

    graph = read_graph()
    print('graph has %d nodes and %d edges' %
          (graph.order(), graph.number_of_edges()))
    subgraph = get_subgraph(graph, 2)
    print('subgraph has %d nodes and %d edges' %
          (subgraph.order(), subgraph.number_of_edges()))

    print('norm_cut scores by max_depth:')
    print(score_max_depths(subgraph, range(1,5)))

    clusters = partition_girvan_newman(subgraph, 3)
    print('first partition: cluster 1 has %d nodes and cluster 2 has %d nodes' %
          (clusters[0].order(), clusters[1].order()))
    print('cluster 2 nodes:')
    print(clusters[1].nodes())

    test_node = 'Bill Gates'
    train_graph = make_training_graph(subgraph, test_node, 5)
    print('train_graph has %d nodes and %d edges' %
          (train_graph.order(), train_graph.number_of_edges()))


    jaccard_scores = jaccard(train_graph, test_node, 5)
    print('\ntop jaccard scores for Bill Gates:')
    print(jaccard_scores)
    print('jaccard accuracy=%g' %
          evaluate([x[0] for x in jaccard_scores], subgraph))

    path_scores = path_score(train_graph, test_node, k=5, beta=.1)
    print('\ntop path scores for Bill Gates for beta=.1:')
    print(path_scores)
    print('path accuracy for beta .1=%g' %
          evaluate([x[0] for x in path_scores], subgraph))


if __name__ == '__main__':
    main()
