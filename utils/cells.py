import awkward    as ak
import numpy      as np
import tensorflow as tf
import vector

metre = 1e3

def to_4momentum(cells, Et_key = 'cell_et'):
    position = to_3vector(cells)
    vectors  = vector.zip(
        {
            "m"  : ak.zeros_like(cells[Et_key]),
            "pt" : cells[Et_key],
            "eta": position.eta,
            "phi": pisition.phi,
        }
    )
    return vectors

def to_3vector(cells):
    vectors = vector.zip(
        {
            "x": cells.cell_x / metre,
            "y": cells.cell_y / metre,
            "z": cells.cell_z / metre,
        }
    )
    return vectors

def get_layer(sampling):
    layer_map = {
        0 : 0,  # PSB
        1 : 1,  # EMB1
        2 : 2,  # EMB2
        3 : 3,  # EMB3
        4 : 0,  # PSE
        5 : 1,  # EME1
        6 : 2,  # EME2
        7 : 3,  # EME3
        8 : 4,  # HEC0
        9 : 5,  # HEC1
        10: 6,  # HEC2
        11: 7,  # HEC3
        12: 4,  # TileBar0
        13: 5,  # TileBar1
        14: 6,  # TileBar2
        15: 5,  # TileGap1
        16: 6,  # TileGap2
        17: 7,  # TileGap3
        18: 4,  # TileExt0
        19: 5,  # TileExt1
        20: 6,  # TileExt2
        # things get weird in the fcal
        21: 1,  # FCAL0 (EM)
        22: 2,  # FCAL1 (Had)
        23: 3,  # FCAL1 (Had)
    }

    output = ak.copy(sampling)
    for k, v in layer_map.items():
        output = ak.where(output == k, v, output)

    return output

def remove_transition(cells):
    cell_abseta = np.abs(to_3vector(cells).eta)
    psb         = (cells.cell_sampling == 0 ) & (cell_abseta > 1.5)
    eme1        = (cells.cell_sampling == 5 ) & (cell_abseta < 1.5)
    eme2        = (cells.cell_sampling == 6 ) & (cell_abseta < 1.5)
    ext0        = (cells.cell_sampling == 18) & (cell_abseta > 1.5)
    mask        = (~psb) & (~eme1) & (~eme2) & (~ext0)
    return cells[mask]
