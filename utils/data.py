from pathlib import Path
from astropy.io import fits
import pandas as pd
import numpy as np
import re

def prepare_data():
    # Read the tabular ds.
    ds_tabular = pd.read_csv("GCDetection/data/ACS_sources_original.csv", na_values=[""])

    # Create the FCC and VCC dfs.
    ds_fits_FCC = pd.DataFrame()
    ds_fits_VCC = pd.DataFrame()
    FCC_list = [] # Support list for FCC
    VCC_list = [] # ^ VCC

    # Insert (Galaxy, Frame, ImageData) tuples in FCC_list.
    [
        FCC_list.extend([(elem.name.split(sep="_")[0], i, entry) \
            # Iterate over frames.
            for i, entry in enumerate(np.array(fits.open(elem)[1].data))]) 
                # Iterate over FITS files and select "FCC|IC|NGC".
                for elem in Path("GCDetection/data/ImageData/").iterdir() \
                if (elem.is_file() and re.findall("FCC|IC|NGC", elem.name))
    ]

    # Repeat fo VCC_list.
    [
        VCC_list.extend([(elem.name.split(sep="_")[0], i, entry) \
            # Iterate over frames.
            for i, entry in enumerate(np.array(fits.open(elem)[1].data))]) 
                # Iterate over FITS files and select "VCC".
                for elem in Path("GCDetection/data/ImageData/").iterdir() \
                if (elem.is_file() and re.findall("VCC", elem.name))
    ]

    # Initialize ds_fits_FCC and merge it with ds_tabular.
    ds_fits_FCC = pd.DataFrame(FCC_list, columns = ["galaxy", "ID", "frame"]).set_index(["galaxy", "ID"])
    ds_fits_FCC = pd.merge(ds_fits_FCC, ds_tabular.set_index(["galaxy", "ID"]), on=["galaxy", "ID"], how="inner").drop("Unnamed: 0", axis=1)

    # Repeat for ds_fits_VCC.
    ds_fits_VCC = pd.DataFrame(VCC_list, columns = ["galaxy", "ID", "frame"]).set_index(["galaxy", "ID"])
    ds_fits_VCC = pd.merge(ds_fits_VCC, ds_tabular.set_index(["galaxy", "ID"]), on=["galaxy", "ID"], how="inner").drop("Unnamed: 0", axis=1)

    # Replicate NaN cleaning using the author's selected columns.
    feat = ['CI4_g', 'CI4_z', 'm4_g', 'm4_z', 'CI5_g', 'CI5_z', 'm5_g', 'm5_z', 'colour', 'm3_z', 'm3_g', 'CI_g', 'CI_z']
    ds_fits_FCC[feat] = ds_fits_FCC[feat].dropna()
    ds_fits_VCC[feat] = ds_fits_VCC[feat].dropna()

    # Add hard labels. 
    ds_fits_FCC["y"] = ds_fits_FCC["pGC"].apply(lambda x: x >= 0.5)
    ds_fits_VCC["y"] = ds_fits_VCC["pGC"].apply(lambda x: x >= 0.5)
    return ds_fits_FCC, ds_fits_VCC