
# -*- coding: utf-8 -*-
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def spectral_reduction(G, 
                       weights_apply,
                       reduction_factor = 0.8 ,
                       threshold =0.1,
                       name_experiment = 'WL1'
                       ):
    for _edge_ in G.edges:  G[_edge_[0]][_edge_[1]]['weight'] = np.abs(weights_apply[_edge_[0] - 1])/2
    # for edge in G.edges(): G[edge[0]][edge[1]]['weight'] = np.random.randint(1, 10) # experiment L1 
    L = nx.laplacian_matrix(G, weight='weight').toarray()
    eigenvalues, eigenvectors = np.linalg.eigh(L) # Compute eigenvalues and eigenvectors
    
    # reduction_factor, threshold = 0.95 ,np.median(weights_apply),  # plane
    # reduction_factor, threshold = 0.8 ,.1,  # dam to jako gamma parameter zas ? 
    k =  int(len(eigenvalues)*reduction_factor) # Select a subset (for example, the first k eigenvalues)1 wl2
    
    # reduction_factor, threshold = 0.3 ,.1, # plane 
    # reduction_factor, threshold = 0.4 ,.3 # plane to neda
    # name_experiment, k = 'WL2', int(len(eigenvalues)*reduction_factor) # Select a subset (for example, the first k eigenvalues)1 wl2
    
    selected_eigenvalues = eigenvalues[:k]
    selected_eigenvectors = eigenvectors[:, :k]
    reduced_graph = nx.Graph() # Form a reduced graph using the selected eigenvectors
    reduced_graph.add_nodes_from(range(1, len(G.nodes) + 1)) # Add nodes to the reduced graph
    positions = {i + 1: (selected_eigenvectors[i, 0], selected_eigenvectors[i, 1]) for i in range(len(G.nodes))} 
    # threshold = .1 # dam to jako gamma parameter zas ? 
    
    top_indices = np.argsort(eigenvalues)[-k:]
    edges_to_keep = [(i, j) for i, j, data in G.edges(data=True) if data.get('weight', 1) > threshold]
    reduced_graph.add_nodes_from(positions)
    reduced_graph.add_edges_from(edges_to_keep)
    a_0    = nx.adjacency_matrix(G).toarray()
    a_red  = nx.adjacency_matrix(reduced_graph).toarray()
    graph = reduced_graph
    ltm = np.tril(np.random.rand(a_0.shape[0], a_0.shape[1]))
    s = ltm * a_red.T 
    I =  np.ones([1, a_0.shape[0]])
    degree_array = np.array(list(reduced_graph.degree))
    degree_array = degree_array[:,1] * I
    a = degree_array
    e0 = np.where(a < 1 )[1] + 1
    e1 = np.where(a < 2 )[1] + 1
    e_ = np.concatenate([e0,e1])
    eu = np.unique(e_)
    ea = list(np.array([eu,np.roll(eu,1)]).T)
    reduced_graph.add_edges_from(ea)
    return reduced_graph
class resco:
    def __init__(self, file_):
        self.dataset_name = file_.split('\\')[1].split('_')[0]   
        # self.df_ = df_
        d = np.load(file_,allow_pickle=True).item()
        self.df_ = pd.DataFrame.from_dict(d)
        pass
    
    def get_dfres_stacked(self, index_name = 'gcn'):
        '''
        
        Parameters
        ----------
        index_name : String for method selection
            DESCRIPTION. The default is 'gcn'.
            NEXT: 'sage'

        Returns
        -------
        df3 : TYPE
            DESCRIPTION.

        '''
        # stacked experiment by experiment
        # index_name = 'gcn' #|| 'sage'

        c3 = np.hstack(np.array(self.df_.loc[index_name]))
        ids = list(self.df_.loc[index_name].index)
        il = [ids]*self.df_.loc[index_name][0].shape[0]
        ia = np.array(il).T # pro zajimavos jak michat : # ia = np.array(il)
        ias = np.hstack(ia)
        df3 = pd.DataFrame(c3.astype('float64'), columns=['acc'])
        df3['mt'] = ias
        df3['framework'] = index_name
        self.df_stacked = df3
        
        return df3
    def to_res_array(self,):
        dfs = self.df_stacked
        gs = dfs.groupby('mt')
        gsd = gs.describe()
        wio = gsd['acc'][['mean','std']]
        wioT = wio.T
        # wio = dfg['acc'][['mean','std']].T*1e3
        # w = wioT.values
        w = wio.values
        # wr = np.reshape(w, [1,12])
        wh = np.hstack(w)
        self.df_grouped = gs
        self.df_res = pd.DataFrame(wh,columns=[self.dataset_name])
        return wh
    def get_dfres(self, index_name):
        # index_name = 'gcn' || 'sage'
        c2 = np.vstack(np.array(self.df_.loc[index_name]))
        df2 = pd.DataFrame( c2.astype('float64').T, columns=self.df_.loc[index_name].index)
        df2['framework'] = index_name
        return df2 
    def plot_results(file_,stacked_dataframe_compiled):
        dataset_name = file_.split('\\')[1].split('_')[0]   
        plt.figure()
        '''
        selector_y = 'steps' # sigma / eps
        '''
        ax = sns.boxplot(data=stacked_dataframe_compiled, x="mt", y='acc',
                    # hue="mt", 
                    hue="framework", 
                    # notch=True, 
                    showcaps=True,
                    medianprops={"color": "k", "linewidth": 2},
                    )
        plt.yscale('log',)
        plt.title(f'{dataset_name}')
        # plt.savefig(f'../graph_reduction/pics/{dataset}_hist_02.png')
def get_info(graph_):
    print(f'Nodes: {graph_.nodes.data()} \n')
    print(f'Edges: {graph_.edges.data()} \n')
def get_graph_weighted(graph_gfem, weights_apply):
    for _edge_ in graph_gfem.edges: 
        graph_gfem[_edge_[0]][_edge_[1]]['weight'] = np.abs(weights_apply[_edge_[0] - 1])/2
    return graph_gfem
def plotter(graph_original, graph_reduced, pos = None):
    '''
    chybi tu pozice pos

    Parameters
    ----------
    graph_original : TYPE
        DESCRIPTION.
    graph_reduced : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    node_size=400
    s1 = str(len(graph_original.edges.data()))
    s2 = str(len(graph_reduced.edges.data()))
    # Plot the original and reduced graphs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    nx.draw_networkx(graph_original, pos = pos,with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=font_size)
    plt.title('Original Fully Connected Graph')
    plt.text(x = 1, y = 1, s = s1 )
    # plt.text()
    
    plt.subplot(1, 2, 2)
    nx.draw_networkx(graph_reduced, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=font_size)
    plt.text(x = 1, y = 1, s = s2 )
    # plt.title(f'Reduced Graph (Laplacian, k={k})')
    # plt.title(f'Reduced Graph ()')
    print(f'fully conneccted: {nx.is_connected(graph_reduced)}')
    plt.show()   

def plotter3(graph_original, 
             graph_reduced, 
             graph_reduced_2, 
             pos = None):
    '''
    chybi tu pozice pos

    Parameters
    ----------
    graph_original : TYPE
        DESCRIPTION.
    graph_reduced : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    font_size=10
    node_size=400
    s1 = str(len(graph_original.edges.data()))
    s2 = str(len(graph_reduced.edges.data()))
    s3 = str(len(graph_reduced_2.edges.data()))
    # Plot the original and reduced graphs
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    nx.draw_networkx(graph_original, pos = pos,with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=font_size)
    plt.title('Original Fully Connected Graph')
    plt.text(x = .01, y = .01, s = s1 )
    # plt.text()
    
    plt.subplot(1, 3, 2)
    nx.draw_networkx(graph_reduced, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=font_size)
    plt.title(f'Graph {graph_reduced.name}')
    plt.text(x = .01, y = .01, s = s2 )
    
    plt.subplot(1, 3, 3)
    nx.draw_networkx(graph_reduced_2, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=font_size)
    plt.title(f'Graph {graph_reduced_2.name}')
    plt.text(x = .01, y = .01, s = s3 )
    # plt.title(f'Reduced Graph (Laplacian, k={k})')
    # plt.title(f'Reduced Graph ()')
    
    plt.show() 
def plotter6(graph_original, 
             graph_reduced_1, 
             graph_reduced_2, 
             graph_reduced_3, 
             graph_reduced_4, 
             pos = None):
    '''
    chybi tu pozice pos

    Parameters
    ----------
    graph_original : TYPE
        DESCRIPTION.
    graph_reduced : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    color_map = []
    # for node in graph_original:
        # if node < 10:
            # color_map.append('blue')
        # else: 
        # color_map.append('green')    
    # https://matplotlib.org/stable/users/explain/colors/colors.html        
    # color_map = len(graph_original.nodes) * ['green']
    # color_map = len(graph_original.nodes) * ['aqua']
    # color_map = len(graph_original.nodes) * ['blue']
    # color_map = len(graph_original.nodes) * ['azure']
    # color_map = len(graph_original.nodes) * ['lightblue']
    color_map = len(graph_original.nodes) * ['aquamarine']
    # nx.draw(graph_original, node_color=color_map, with_labels=True)
    font_size=10
    node_size=300
    s1 = str(len(graph_original.edges.data()))
    s2 = str(len(graph_reduced_1.edges.data()))
    s3 = str(len(graph_reduced_2.edges.data()))
    s4 = str(len(graph_reduced_3.edges.data()))
    s5 = str(len(graph_reduced_4.edges.data()))
    # Plot the original and reduced graphs
    # plt.figure(figsize=(12, 4))
    plt.figure(figsize=(12,7))
    
    plt.subplot(2, 2, 1)
    nx.draw_networkx(graph_original, pos = pos,with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=font_size,
                     node_color=color_map)
    
    plt.title(f'{graph_original.name}')
    #plt.text(x = .01, y = .01, s = s1 )
    # plt.text()
    
    plt.subplot(2, 2, 2)
    nx.draw_networkx(graph_reduced_1, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=font_size, 
                     node_color=color_map)
    
    
    plt.title(f'{graph_reduced_1.name}')
    #plt.text(x = .01, y = .01, s = s2 )
    
    plt.subplot(2, 2, 3)
    nx.draw_networkx(graph_reduced_2, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=font_size,
                     node_color=color_map)
    
    plt.title(f'{graph_reduced_2.name}')
    #plt.text(x = .01, y = .01, s = s3 )
    
    plt.subplot(2, 2, 4)
    nx.draw_networkx(graph_reduced_3, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=font_size,
                     node_color=color_map)
    
    plt.title(f'{graph_reduced_3.name}')
    #plt.text(x = .01, y = .01, s = s4 )
    
    # plt.subplot(1, 5, 5)
    # nx.draw_networkx(graph_reduced_4, pos=pos, with_labels=True, font_weight='bold', 
    #                  node_size=node_size, 
    #                  font_size=8, 
    #                  node_color=color_map)
    # plt.title(f'{graph_reduced_4.name}')
    #plt.text(x = .01, y = .01, s = s5 )
    plt.tight_layout()
    
    # plt.title(f'Reduced Graph (Laplacian, k={k})')
    # plt.title(f'Reduced Graph ()')
    
    plt.show() 
def plotter5(graph_original, 
             graph_reduced_1, 
             graph_reduced_2, 
             graph_reduced_3, 
             graph_reduced_4, 
             pos = None):
    '''
    chybi tu pozice pos

    Parameters
    ----------
    graph_original : TYPE
        DESCRIPTION.
    graph_reduced : TYPE
        DESCRIPTION.
     : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    
    color_map = []
    # for node in graph_original:
        # if node < 10:
            # color_map.append('blue')
        # else: 
        # color_map.append('green')    
    # https://matplotlib.org/stable/users/explain/colors/colors.html        
    # color_map = len(graph_original.nodes) * ['green']
    # color_map = len(graph_original.nodes) * ['aqua']
    # color_map = len(graph_original.nodes) * ['blue']
    # color_map = len(graph_original.nodes) * ['azure']
    # color_map = len(graph_original.nodes) * ['lightblue']
    color_map = len(graph_original.nodes) * ['aquamarine']
    # nx.draw(graph_original, node_color=color_map, with_labels=True)
    
    node_size=300
    s1 = str(len(graph_original.edges.data()))
    s2 = str(len(graph_reduced_1.edges.data()))
    s3 = str(len(graph_reduced_2.edges.data()))
    s4 = str(len(graph_reduced_3.edges.data()))
    s5 = str(len(graph_reduced_4.edges.data()))
    # Plot the original and reduced graphs
    # plt.figure(figsize=(12, 4))
    plt.figure(figsize=(12, 3.5))
    
    plt.subplot(1, 5, 1)
    nx.draw_networkx(graph_original, pos = pos,with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=8,
                     node_color=color_map)
    
    plt.title(f'{graph_original.name}')
    #plt.text(x = .01, y = .01, s = s1 )
    # plt.text()
    
    plt.subplot(1, 5, 2)
    nx.draw_networkx(graph_reduced_1, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, font_size=8, 
                     node_color=color_map)
    
    
    plt.title(f'{graph_reduced_1.name}')
    #plt.text(x = .01, y = .01, s = s2 )
    
    plt.subplot(1, 5, 3)
    nx.draw_networkx(graph_reduced_2, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=8,
                     node_color=color_map)
    
    plt.title(f'{graph_reduced_2.name}')
    #plt.text(x = .01, y = .01, s = s3 )
    
    plt.subplot(1, 5, 4)
    nx.draw_networkx(graph_reduced_3, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=8,
                     node_color=color_map)
    
    plt.title(f'{graph_reduced_3.name}')
    #plt.text(x = .01, y = .01, s = s4 )
    
    plt.subplot(1, 5, 5)
    nx.draw_networkx(graph_reduced_4, pos=pos, with_labels=True, font_weight='bold', 
                     node_size=node_size, 
                     font_size=8, 
                     node_color=color_map)
    plt.title(f'{graph_reduced_4.name}')
    #plt.text(x = .01, y = .01, s = s5 )
    plt.tight_layout()
    
    # plt.title(f'Reduced Graph (Laplacian, k={k})')
    # plt.title(f'Reduced Graph ()')
    
    plt.show() 
class gra:
    def __init__(self, G_orig):
        self.G = G_orig
        self.G_temp = nx.Graph()
        self.G_temp.add_nodes_from(self.G)
        # self.G_dummy = self.G_red
        # self.G_dummy.add_edges_from(self.G.edges)

        # pass
    def add_weights(self, weights_apply):
        for _edge_ in self.G.edges: 
            self.G[_edge_[0]][_edge_[1]]['weight'] = np.abs(weights_apply[_edge_[0] - 1])/2
        pass
    def get_graph_reduced(self, path_for_reduction):
        # G_red = self.G_temp   
        G_red = nx.Graph()
        G_red.add_nodes_from(self.G)
        a2 = np.array([path_for_reduction,np.roll(path_for_reduction,1)])
        a3 = a2[:,1:].T.tolist()
        G_red.add_edges_from(a3)
        return G_red
    def get_shortest_path(self, ):
        G_dummy = self.G_temp
        G_dummy.add_edges_from(self.G.edges)

        # self.G_dummy.add_edges_from(self.G.edges)
        # self.path_shortest = nx.shortest_path(self.G, source=1,  
        # nese sebou data o pos, =>  nejkratsi ceta je pak ta geometricky nejkratsi
        # self.path_shortest = nx.shortest_path(self.G_dummy, 
        self.path_shortest = nx.shortest_path(G_dummy, 
                              source=1, 
                              target = list(G_dummy.nodes)[-1],)
                                              # weight = 'weight') 
        self.G_red_ShP = self.get_graph_reduced(self.path_shortest)
        # return self.G_red_ShP
        return self.path_shortest
    def get_travel_salesman(self, graph_weighted):
        # self.G_dummy.add_weighted_edges_from(graph_weighted)
        G_dummy = self.G_temp
        G_dummy.add_edges_from(self.G.edges)
        self.path_salesman = nx.approximation.traveling_salesman_problem(G_dummy, 
                weight='weight', nodes=None, cycle=True, method=None)
        self.G_red_TrS  = self.get_graph_reduced(self.path_salesman)
        return self.path_shortest
    def laplacik(self,):
        path_ = somepather(andItsPars)
        pass
    def get_info(self,):

        print(f'Nodes: {self.G_red.nodes.data()} \n')
        print(f'Edges: {self.G_red.edges.data()} \n')
