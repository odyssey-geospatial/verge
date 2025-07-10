# Verge: Vector-Mode Regional Geospatial Embeddings

This repository is for R&D on the subject of creating regional embeddings 
based on vector-mode geospatial entities.  

The general idea is to extract a set of vector-mode geospatial features from 
a source like OpenStreetMap, and put them through some transformation to yield 
an embedding that captures the essential spatial properties of the region.

The "vector mode" is the key element of this initiative.
One could do something like this by making a map as a raster-mode image,
or maybe using an overhead image of the region, 
and feeding it to something like a vision transformer. 
That actually works fairly well.
But there is a lot of potential to doing this using vector mode data --
WKT-like objects of type POINT, LINESTRING, POLYGON, etc.
That would make it easy to incorporate arbitrary metadata for geospatial entities, 
to filter and focus on certain types of entities, to include entities that are 
notional such as proposed development plans, possible travel paths, and so on. 

The methods here use Multi-Point Proximity (MPP) encoding
to incorporate vector-mode data into ML/AI models:

https://arxiv.org/abs/2506.05016

## Status: July 10, 2025
At this time, this repo inlcudes analysis notebooks for some exploratory data analysis,
and code implementing "Masked Geospatial Modeling", which is a first step towards 
regional embeddings. The content of these notebooks is described here:

https://odyssey-geospatial.ghost.io/masked-geospatial-modeling-towards-vector-mode-regional-geospatial-embeddings/


## Folders

* 01-feature-eda: Exploratory analysis on geographical features available from OSM
* 02-mgm: R&D on Masked Geospatial Modeling

