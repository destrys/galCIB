"""
This script rotates CIB maps for cross-correlation with
DESI.
"""

import numpy as np 
import healpy as hp 
import pymaster as nmt 
from time import time 

data_folder = '/Users/tkarim/research/galCIB/data/cib/cib-lenz19-data/'
nhi_cuts = ['1.5e+20_gp20',
    '1.8e+20_gp20',
    '2.0e+20_gp20',
    '2.5e+20_gp20',
    '3.0e+20_gp40',
    '4.0e+20_gp40']

healpy_data_path = '/Users/tkarim/research/galCIB/data/healpy/'

# rotator object
r = hp.Rotator(coord=['G','C'])
NSIDE = 1024

nu = '545'
nhi = nhi_cuts[2] #FIXME: read in proper nhi name

print(nu, nhi)

# read in relevant files
mask_apod = hp.read_map(f"{data_folder}/{nu}/{nhi}/mask_apod.hpx.fits")
mask_bool = hp.read_map(f"{data_folder}/{nu}/{nhi}/mask_bool.hpx.fits")
map_fm = hp.read_map(f"{data_folder}/{nu}/{nhi}/cib_fullmission.hpx.fits")

# prepare effective map 
map_eff = np.where(mask_bool, map_fm, 0.) * mask_apod

##--ROTATION STEPS--##

start = time() 
# convert map to alm
alm_map_eff_g_2048 = hp.map2alm(map_eff,pol=False,
                         use_pixel_weights=True,
                         datapath=healpy_data_path)

print(f'map2alm = {time()-start}')

t1 = time()
# rotate data alm
alm_map_eff_c_2048 = r.rotate_alm(alm_map_eff_g_2048)
print(f'rotate_alm = {time()-t1}')

t1 = time()
# convert alm to map 
map_eff_c_1024 = hp.alm2map(alm_map_eff_c_2048,
                            nside=NSIDE,pol=False)
print(f'alm2map = {time()-t1}')

t1 = time()
# rotate mask in pixel space 
mask_bool_pixel_rot = r.rotate_map_pixel(mask_bool)
print(f'rotate_map_pixel = {time()-t1}')

t1 = time()
# convert mask to NSIDE and rebinarize with threshold 0.95
mask_bool_pixel_rot_udgrade_1024 = hp.ud_grade(mask_bool_pixel_rot,
                                               NSIDE)
print(f'ud_grade = {time()-t1}')
mask_bool_pixel_rot_udgrade_1024 = np.where(mask_bool_pixel_rot_udgrade_1024>0.95,
                                            1,0)

# re-apodize mask 
# this apodization scheme is the same as Lenz+2019
t1 = time()
mask_apod_rot_1024 = nmt.mask_apodization(mask_bool_pixel_rot_udgrade_1024, 
                                          0.25, "C2")
print(f'mask_apodization = {time()-t1}')

# apply new schema to the rotated data 
map_eff_c_1024_re_apod = np.where(mask_bool_pixel_rot_udgrade_1024,
                                  map_eff_c_1024,
                                  0)*mask_apod_rot_1024

# for completeness set maske region to UNSEEN
map_eff_c_1024_re_apod[~mask_bool_pixel_rot_udgrade_1024.astype(bool)] = hp.UNSEEN

# save files 
np.save(f"{data_folder}/{nu}/{nhi}/cib_fullmission_c_1024_re_apod.npy", 
        map_eff_c_1024_re_apod) # re apodized data
np.save(f"{data_folder}/{nu}/{nhi}/mask_bool_c_1024.npy", 
        mask_bool_pixel_rot_udgrade_1024) # boolean mask 

print(f"Total time = {time()-start}")
