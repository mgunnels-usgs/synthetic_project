#!/usr/bin/env python
import obspy
from obspy.clients.fdsn import Client
import matplotlib.pyplot as plt
import glob
import matplotlib as mpl
from scipy.signal import resample
from scipy import signal
import argparse
from obspy import read, Stream, read_events
import os

# Making the figures look nice
mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=False)
mpl.rc('font', size=18)


CONVERSION_M = (10**9)

def main():
    # Lets get the parser arguments
    parser_val = getargs()
    client = Client(parser_val.client)
    debug = parser_val.debug
    
    net = parser_val.network


    userminfre = 1.0/float(parser_val.filter[1])
    usermaxfre = 1.0/float(parser_val.filter[0])
    filtercornerpoles = int(parser_val.filter[2])
    
    if parser_val.sta:
        sta_list = parser_val.sta.replace(' ','')
        sta_list = sta_list.split(',')
        files = []
        for sta in sta_list:
            files += glob.glob(parser_val.syn + '/' + net + '*' + sta + '*MXZ*')
    else:
        files = glob.glob(parser_val.syn + '/' + net + '*MXZ*')

 

    #cmt = parser_val.cmt
    syn = parser_val.syn

    cwd = os.getcwd()
    #make_res_dir(res_dir)
    
    cat = obspy.read_events(parser_val.syn + '/CMTSOLUTION')

    for cur_file in files:
        cur_file = cur_file.replace('MXZ', 'MX*')
        try:
            st = read(cur_file)
        except:
            continue
        st.integrate()
        st.integrate()

        for tr in st:
            tr.data /= CONVERSION_M

        for tr in st:
            tr.data = resample(tr.data, int(tr.stats.endtime - tr.stats.starttime))
            tr.stats.sampling_rate = 1.
        st = stf(st, 8.)
        inv = client.get_stations(network=st[-1].stats.network,
                                  station=st[-1].stats.station,
                                  starttime=st[-1].stats.starttime,
                                  endtime=st[-1].stats.endtime,
                                  channel="LH*",
                                  level="response")
        try:
            st_data = client.get_waveforms(network=st[-1].stats.network,
                                           station=st[-1].stats.station,
                                           starttime=st[-1].stats.starttime,
                                           endtime=st[-1].stats.endtime,
                                           channel="LH*",
                                           location="*")
        except:
            continue

    # rotate and remove response
        st_data.detrend('constant')
        st_data.detrend('linear')
        st_data.remove_response(inv, output="DISP")
        st_data.rotate("->ZNE", inventory=inv)
        st += st_data
        st.filter('bandpass', freqmin=usermaxfre, freqmax=userminfre)
        st.taper(0.05)

        fig = plt.figure(1, figsize=(12, 12))
        for idx, chan in enumerate(['Z', 'N', 'E']):
            plt.subplot(3, 1, idx + 1)
            st_t = st.select(channel="*" + chan)
            for tr in st_t:
                plt.plot(tr.times(), tr.data * 1000., label=tr.id)
                plt.xlim((min(tr.times()), max(tr.times())))
            if idx == 1:
                filename = get_file_name(parser_val.res_dir, st_data)
                plt.ylabel('Amplitude (mm)')
            plt.legend(loc=1)
        plt.xlabel('Time (s)')
        plt.savefig(filename + '.PNG', format='PNG', dpi=200)
        plt.clf()
        plt.close()



def stf(st, hd):
    " Windows data"
    win = signal.hann(int(2*hd))
    for tr in st:
        tr.data = signal.convolve(tr.data, win, mode='same')/sum(win)
    return st

def make_res_dir(res_dir):
    " This function checks if result directory exists, if it does not, makes a results directory "
    if os.path.exists(res_dir):
        print('Path Exists')
    else:
        os.mkdir(res_dir)


def getargs():
    " Gets user arguments"
    parser = argparse.ArgumentParser(description = "Program to compare long-period event synthetics to data")

    parser.add_argument('-n', type=str, action="store",
                         dest = "network", required=True,
                         help="Network name Example: IU")

    parser.add_argument('-resDir', type=str, action="store",
                        dest="res_dir", required=True,
                        help="Result directory name Example: blah")

    parser.add_argument('-syn', type=str, action="store",
                         dest="syn", required=True,
                         help="Synthetics directory location Example: " +
                         "/SYNTHETICS/2014/C201401*")

    parser.add_argument('-sta', type=str, action="store",
                        dest="sta", required = False,
                        help="Stations to use Example with a comma (,) separator : TUC,ANMO")

    parser.add_argument('-client', type=str, action="store",
                        dest="client", required=False, default="IRIS",
                        help="Choose internet client, default is 'IRIS'")

    parser.add_argument('-debug', action="store_true", dest="debug",
                        default=False, help="Run in debug mode")

    parser.add_argument('-filter', action="store", nargs=3, dest="filter", required=False,
                        default=[300, 50, 4],
                        help="Filter parameters using minimum period maximum period "
                             "and number of corners Example: 50 300 4, "
                             "default is 300, 50, 4")

    parser_val = parser.parse_args()

    return parser_val

def get_file_name(res_dir,st_data):
    " Function that gets output figure filename"
    filename = '{res_dir}/{network}.{station}.{year}{julday}{hour}{min}'.format(
        res_dir=res_dir,
        network=st_data[0].stats.network,
        station=st_data[0].stats.station,
        year=str(st_data[0].stats.starttime.year),
        julday=str(st_data[0].stats.starttime.julday),
        hour=str(st_data[0].stats.starttime.hour),
        min=str(st_data[0].stats.starttime.minute)
    )
    return filename


if __name__ == "__main__":
    main()
