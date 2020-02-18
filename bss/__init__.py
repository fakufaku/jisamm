"""
This package contains the main algorithms for
overdetermined independent vector analysis
"""
from .overiva import (
    overiva,
    overiva_ip_param,
    overiva_ip2_param,
    overiva_demix_bg,
    overiva_ip_block,
    overiva_ip2_block,
    auxiva,
    auxiva2,
)
from .five import five
from .pca import pca
from .auxiva_pca import auxiva_pca
from .ogive import ogive, ogive_mix, ogive_demix, ogive_switch

algos = {
    "auxiva": auxiva,
    "auxiva2": auxiva2,
    "overiva": overiva,
    "overiva-ip": overiva_ip_param,
    "overiva-ip2": overiva_ip2_param,
    "overiva-ip-block": overiva_ip_block,
    "overiva-ip2-block": overiva_ip2_block,
    "overiva-demix-bg": overiva_demix_bg,
    "five": five,
    "ogive": ogive,
    "ogive-mix": ogive_mix,
    "ogive-demix": ogive_demix,
    "ogive-switch": ogive_switch,
    "auxiva_pca": auxiva_pca,
    "pca": pca,
}

is_single_source = {
    "auxiva": False,
    "auxiva2": False,
    "overiva": False,
    "overiva-ip": False,
    "overiva-ip2": False,
    "overiva-ip-block": False,
    "overiva-ip2-block": False,
    "overiva-demix-bg": False,
    "auxiva_pca": False,
    "pca": False,
    "five": True,
    "ogive": True,
    "ogive-mix": True,
    "ogive-demix": True,
    "ogive-switch": True,
}

# This is a list that indicates which algorithms
# can only work with two or more sources
is_dual_update = {
    "auxiva": False,
    "auxiva2": True,
    "overiva": False,
    "overiva-ip": False,
    "overiva-ip2": True,
    "overiva-ip-block": False,
    "overiva-ip2-block": True,
    "overiva-demix-bg": False,
    "auxiva_pca": True,
    "pca": False,
    "five": False,
    "ogive": False,
    "ogive-mix": False,
    "ogive-demix": False,
    "ogive-switch": False,
}

is_determined = {
    "auxiva": True,
    "auxiva2": True,
    "overiva": False,
    "overiva-ip": False,
    "overiva-ip2": False,
    "overiva-ip-block": False,
    "overiva-ip2-block": False,
    "overiva-demix-bg": False,
    "auxiva_pca": False,
    "pca": False,
    "five": False,
    "ogive": False,
    "ogive-mix": False,
    "ogive-demix": False,
    "ogive-switch": False,
}
