import sys
import os
import subprocess

str_bw_file = "./.STR_BW"

def get_storage_bandwidth(disk="/datadrive/mnt2"): 
    str_bw = strProfileExists() 
    if str_bw is not None:
        return str_bw
    else:
        paths = disk.split('/')
        print(paths)
        mnt_paths = [s for s in paths if s.startswith("mnt")]
        print(mnt_paths)
        disk = mnt_paths[0]
        dev_cmd = ['grep', disk, '/proc/mounts']  
        dev_cmd_cut = ['cut', '-d', ' ', '-f', '1'] 
        p = subprocess.Popen(dev_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = subprocess.check_output(dev_cmd_cut, stdin=p.stdout) 
        p.wait()
        print("Output = {}".format(output))
        if p.returncode != 0: 
            out, err = p.communicate()
            print("Error : {}".format(err.decode('utf-8'))) 
            return 0,0  
        device = output.decode('utf-8').rstrip()  
        print("Measuring bandwidth of storage dev  {}".format(device))
        dev_bw = ['hdparm', '-t', device] 
        #dev_bw = ['sudo', 'hdparm', '-t', device] 
        p = subprocess.Popen(dev_bw, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        out, err = p.communicate()  
        result = out.decode('utf-8') 
        print(result, err.decode('utf-8')) 
        str_bw = result.split()[-2]  
        os.environ['STR_BW'] = str_bw 
        with open(str_bw_file, 'w+') as wf: 
            wf.write(str_bw) 
        return str_bw

def strProfileExists():
    if 'STR_BW' in  os.environ:
        str_bw = os.environ['STR_BW']
        return float(str_bw)
    elif os.path.exists(str_bw_file):
        with open(str_bw_file, 'r') as rf:
            str_bw = rf.readline()
        return float(str_bw)
    else:
        return None
