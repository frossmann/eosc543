
#%%
import rasterio
import numpy as np
import osmnx as ox
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

plt.style.use("ggplot")
plt.rcParams.update(
    {
        "font.size": 16,
        "axes.labelweight": "bold",
        "figure.figsize": (6, 6),
        "axes.edgecolor": "0.2",
    }
)

provinces = gpd.read_file("data-spatial/provinces")  # note that I point to the shapefile "directory" containg all the individual files
provinces = provinces.to_crs("EPSG:4326")    # I'll explain this later, I'm converting to a different coordinate reference system

provinces.plot(edgecolor="0.2", figsize=(10, 8))
plt.title("Canada Provinces and Territories");

province = "British Columbia"
bc = provinces.query("PRENAME == @province").copy()
bc
bc = (bc.loc[:, ["PRENAME", "geometry"]]
        .rename(columns={"PRNAME": "Province"})
        .reset_index(drop=True)
      )
bc
bc.plot(edgecolor="0.2")
plt.title("British Columbia");
# %%
plt.xlim((-125, -121.5))
plt.ylim((46.5 ))