# Graph

Import main classes


```python
from hari_plotter import Graph, Dynamics
```

Create Graph


```python
H = Graph.strongly_connected_components([10, 10], 10, 3)
```

Get Graph Info


```python
print(f"{H = }")
print(f"{H.mean_opinion = }")
```

    H = <HariGraph object at 126401107202480: 20 nodes, 190 edges>
    H.mean_opinion = 3.039989236795545


Get the information about parameters available for gathering 


```python
H.gatherer.node_parameters
```




    ['Opinion',
     'Opinion density',
     'Cluster size',
     'Importance',
     'Neighbor mean opinion',
     'Inner opinions',
     'Max opinion',
     'Min opinion',
     'Opinion Standard Deviation',
     'Label',
     'Type']



Get the node parameters


```python
H.gatherer.gather(["Cluster size", "Opinion"])
```




    {'Nodes': [(0,),
      (1,),
      (2,),
      (3,),
      (4,),
      (5,),
      (6,),
      (7,),
      (8,),
      (9,),
      (10,),
      (11,),
      (12,),
      (13,),
      (14,),
      (15,),
      (16,),
      (17,),
      (18,),
      (19,)],
     'Cluster size': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
     'Opinion': [1.874213762484462,
      1.9797692699253586,
      1.9864728476689875,
      2.0039220965041564,
      2.0122373057404292,
      2.0393409858053184,
      2.0595745641582326,
      2.101099281488391,
      2.1243752247086394,
      2.2051583723503527,
      3.8754472569677465,
      3.8934247681969056,
      3.9037741673809077,
      4.029159827368202,
      4.050855128109916,
      4.063263981030825,
      4.080563322149777,
      4.125622363739476,
      4.1870486600547645,
      4.204461550078053]}



Read the graph from Network


```python
H = Graph.read_network("../tests/network.txt", "../tests/opinions_0.txt")
print(f"{H = }")
```

    H = <HariGraph object at 126403507139872: 5 nodes, 6 edges>


Read Dynamics from Network


```python
HD = Dynamics.read_network(
    "../tests/5_ring/network.txt",
    [f"../tests/5_ring/opinions_{i}.txt" for i in range(3)],
)
print(f"{HD = }")
```

    HD = <hari_plotter.dynamics.Dynamics object at 0x72f60afe00d0>

