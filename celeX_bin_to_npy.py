import numpy as np
import os
from os import path, listdir
import sys

event_frame_time_sec = 0.03 
celeX_data_loc = 'celeX_files/bin'
npy_data_loc = 'celeX_files/npy'
currentDirectory = os.getcwd()

save_loc = path.join(currentDirectory, npy_data_loc)
files_loc = path.join(currentDirectory, celeX_data_loc)
files = listdir(files_loc)

lower6mask = ~(0xFF<<6)
ts_incremental = event_frame_time_sec / 4096.000000000
for file_name in files:
    r_list = []
    c_list = []
    t_list = []
    row = -1
    t_val = -1
    ts = 0.000000000000
    print(file_name)
    with open( path.join(files_loc,file_name), "rb") as f:
        # Ignore the Bin Header except the last 4bytes package_count
        f.read(8)
        # package_count
        pkg_cnt = int.from_bytes(f.read(4), byteorder='little')
        for pkg in range(pkg_cnt):
            # package_length
            pkg_len = int.from_bytes(f.read(4), byteorder='little')
            pkg_buffer = f.read(pkg_len)
            # ignore pkg timestamp
            f.read(8)
            imu_cnt = int.from_bytes(f.read(4), byteorder='little')
            # ignore all IMU data
            f.read(28*imu_cnt)
            pkg_buffer_list = [ pkg_buffer[0+i:7+i] for i in range(0, len(pkg_buffer), 7)]
            pkg_buffer_list = pkg_buffer_list[:-1]
            for item in pkg_buffer_list:
                d = []
                d.append( (item[4] & lower6mask) | (item[0]<<6) )
                d.append( ((item[4]>>6) | (item[5]<<2)) & lower6mask | (item[1]<<6) )
                d.append( ((item[5]>>4) | (item[6]<<4)) & lower6mask | (item[2]<<6) )
                d.append( (item[6]>>2) | (item[3]<<6) )
                for i in d:
                    i_bin = bin(i)[2:].zfill(14)
                    id = i_bin[-2:]
                    if id == '10':
                        row = 799 - int(i_bin[:-4], 2)
                    elif id=='01' and row!=-1:
                        r_list.append(row)
                        t_list.append(ts)
                        c_list.append( int(i_bin[:-3], 2))
                    elif id == '11':
                        t = int(i_bin[:-2], 2)
                        if t != t_val:
                            t_val = t
                            ts += ts_incremental
    r_np = np.asarray(r_list, dtype=np.float32)
    c_np = np.asarray(c_list, dtype=np.float32)
    t_np = np.asarray(t_list, dtype=np.float32)
    p_np = np.zeros_like(t_list)
    assert np.max(r_np)<800, "Error: row >= 800"
    assert np.max(c_np)<1280, "Error: col >= 1280"
    events = np.stack([c_np, r_np, t_np, p_np], axis=-1)    
    np.save( path.join(save_loc, file_name[:-4]), events)


            
