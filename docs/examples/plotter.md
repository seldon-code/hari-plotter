# Plotter


```python
from hari_plotter import (
    Simulation,
    Interface,
    Plotter,
    Clustering,
)
import os
```


```python
os.makedirs("save_dir", exist_ok=True)
```


```python
Clustering.available_clustering_methods()
```




    ['Interval Clustering', 'K-Means Clustering']




```python
cl = {
    "clustering_method": "K-Means Clustering",
    "clustering_parameters": ["Opinion", "Neighbor mean opinion"],
    "scale": ["Tanh", "Tanh"],
    "n_clusters": 2,
}
```


```python
S = Simulation.from_dir("big_test")
S.group(num_intervals=3, interval_size=1)
```


```python
S
```




    Simulation(model=ActivityDrivenModel(params={'dt': 0.01, 'm': 10, 'eps': 0.01, 'gamma': 2.1, 'reciprocity': 0.5, 'homophily': 3.0, 'alpha': 3.0, 'K': 3}), parameters={'simulation': {'rng_seed': 12345678019}, 'model': {'max_iterations': 2000}, 'network': {'number_of_agents': 1000, 'connections_per_agent': 10}}, dynamics=<HariDynamics object with 3 LazyHariGraphs (0 initialized)>)




```python
interface = Interface.create_interface(S)
```


```python
interface.cluster_tracker.track_clusters(cl)
```




    [{0: {'Cluster 0': 'Cluster 1', 'Cluster 1': 'Cluster 0'},
      1: {'Cluster 0': 'Cluster 1', 'Cluster 1': 'Cluster 0'},
      2: {'Cluster 0': 'Cluster 0', 'Cluster 1': 'Cluster 1'}}]




```python
plotter = Plotter(interface)
print(plotter.available_plot_types_hint)
```

    Histogram: + 
    Hexbin: + 
    Scatter: + 
    Clustering: Centroids: + 
    Clustering: Fill: + 
    Clustering: Degree of Membership: + 
    Clustering: Density Plot: + 
    Static: Time line: + 
    Static: Node lines: + 
    Static: Graph line: + 
    Static: Graph Range: + 
    Static: Clustering Line: + 
    Static: Clustering Range: + 
    



```python
plotter.add_plot(
    "Static: Time line",
    {
        "parameters": ["Time", "Opinion"],
    },
    row=1,
    col=0,
)
plotter.add_plot(
    "Static: Time line",
    {
        "parameters": ["Time", "Opinion"],
    },
    row=2,
    col=1,
)

plotter.add_plot(
    "Static: Clustering Line",
    {
        "parameter": "Opinion",
        "scale": ["Linear", "Tanh"],
        "clustering_settings": cl,
        "show_legend": False,
        "color": {
            "mode": "Cluster Parameter Color",
            "settings": {
                "clustering_settings": cl,
                "parameter": "Opinion",
                "group_number": -1,
                "None Color": "",
            },
        },
    },
    row=1,
    col=0,
)

plotter.add_plot(
    "Static: Clustering Range",
    {
        "parameter": "Opinion",
        "range_parameter": "Opinion Standard Deviation",
        "clustering_settings": cl,
        "scale": ["Linear", "Tanh"],
        "color": {
            "mode": "Cluster Parameter Color",
            "settings": {
                "clustering_settings": cl,
                "parameter": "Opinion",
                "group_number": -1,
                "None Color": "",
            },
        },
    },
    row=1,
    col=0,
)

plotter.add_plot(
    "Histogram",
    {
        "parameter": "Opinion",
        "show_x_label": False,
        "scale": ["Tanh", "Linear"],
    },
    row=0,
    col=1,
)

plotter.add_plot(
    "Histogram",
    {
        "parameter": "Neighbor mean opinion",
        "rotated": True,
        "show_y_label": False,
        "scale": ["Linear", "Tanh"],
    },
    row=1,
    col=2,
)

plotter.add_plot(
    "Clustering: Fill",
    {
        "parameters": ["Opinion", "Neighbor mean opinion"],
        "clustering_settings": cl,
        "scale": ["Tanh", "Tanh"],
        "fill_color": {
            "mode": "Cluster Parameter Color",
            "settings": {"clustering_settings": cl, "parameter": "Opinion"},
        },
    },
    row=1,
    col=1,
)

plotter.add_plot(
    "Scatter",
    {
        "parameters": ["Opinion", "Neighbor mean opinion"],
        "scale": ["Tanh", "Tanh"],
        "color": {
            "mode": "Cluster Parameter Color",
            "settings": {"clustering_settings": cl, "parameter": "Opinion"},
        },
        "marker": {"mode": "Cluster Marker", "settings": {"clustering_settings": cl}},
    },
    row=1,
    col=1,
)

plotter.add_plot(
    "Static: Node lines",
    {
        "parameters": ["Time", "Opinion"],
        "scale": ["Linear", "Tanh"],
        "color": {
            "mode": "Cluster Parameter Color",
            "settings": {
                "clustering_settings": cl,
                "parameter": "Opinion",
                "group_number": -1,
                "None Color": "",
            },
        },
    },
    row=2,
    col=1,
)

plotter.add_plot(
    "Scatter",
    {
        "parameters": ["Opinion", "Activity"],
        # "marker": "x",
        "scale": ["Linear", "Linear"],
        "color": {
            "mode": "Parameter Colormap",
            "settings": {"parameter": "Opinion density", "scale": ("Linear",)},
        },
    },
    row=2,
    col=0,
)

plotter.size_ratios = [
    [4, 4, 1],
    [
        1,
        4,
        4,
    ],
]
# plotter.size_ratios = [[4, 1], [1, 4, 4]]

print("\n==========", plotter.info(), "\n==========")
```

    
    ========== 
    (1, 0) <hari_plotter.interface.SimulationInterface object at 0x718edf1c6e00>
    <hari_plotter.plot.plot_time_line object at 0x718ec87da680>
    <hari_plotter.plot.plot_clustering_line object at 0x718ec87da5c0>
    <hari_plotter.plot.plot_fill_between_clustering object at 0x718ec87da6e0>
    
    (2, 1) <hari_plotter.interface.SimulationInterface object at 0x718edf1c6e00>
    <hari_plotter.plot.plot_time_line object at 0x718ec87da620>
    <hari_plotter.plot.plot_node_lines object at 0x718ec87da800>
    
    (0, 1) <hari_plotter.interface.SimulationInterface object at 0x718edf1c6e00>
    <hari_plotter.plot.plot_histogram object at 0x718ec87da6b0>
    
    (1, 2) <hari_plotter.interface.SimulationInterface object at 0x718edf1c6e00>
    <hari_plotter.plot.plot_histogram object at 0x718ec87da740>
    
    (1, 1) <hari_plotter.interface.SimulationInterface object at 0x718edf1c6e00>
    <hari_plotter.plot.plot_clustering_fill object at 0x718ec87da7a0>
    <hari_plotter.plot.plot_scatter object at 0x718ec87da7d0>
    
    (2, 0) <hari_plotter.interface.SimulationInterface object at 0x718edf1c6e00>
    <hari_plotter.plot.plot_scatter object at 0x718ec87da830>
     
    ==========



```python
plotter.plot(1)
```

    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:912: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:914: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])





    
![png](plotter_files/plotter_11_1.png)
    




    
![png](plotter_files/plotter_11_2.png)
    



```python
plotter.plot_dynamics(
    mode=["show", "save", "gif", "mp4"],
    save_dir="save_dir",
    animation_path="save_dir/animation",
)
```

    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:912: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:914: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])



    
![png](plotter_files/plotter_12_1.png)
    


    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:912: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:914: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])



    
![png](plotter_files/plotter_12_3.png)
    


    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:912: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 0] = np.arctanh(mesh_points_scaled[:, 0])
    /home/ivan/Projects/hari-plotter/hari_plotter/plot.py:914: RuntimeWarning: divide by zero encountered in arctanh
      mesh_points_scaled[:, 1] = np.arctanh(mesh_points_scaled[:, 1])



    
![png](plotter_files/plotter_12_5.png)
    



    <Figure size 640x480 with 0 Axes>



```python

```
