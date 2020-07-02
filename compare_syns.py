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
import numpy as np
from obspy.signal.cross_correlation import correlate, xcorr_max, xcorr
from obspy.geodetics.base import gps2dist_azimuth



# Making the figures look nice
mpl.rc('font', family='serif')
mpl.rc('font', serif='Times')
mpl.rc('text', usetex=False)
mpl.rc('font', size=18)

def setup_data(net, sta, stime, etime, client):
    try:
        inv = client.get_stations(network=net, station = sta, starttime=stime,
                              endtime = etime, channel = "LH*", level="response")
    except:
        print('Metadata problem for: ' + sta)
        return [], False
    try:
        # Wave form data
        st_data = client.get_waveforms(network=net, station=sta, starttime=stime,
                                   endtime = etime, channel = "LH*", location="*")
    except:
        print('Data problem for: ' + sta)
        return [], False
    st_data.detrend('constant')
    st_data.detrend('linear')
    st_data.remove_response(inv, output='DISP')
    st_data.rotate("->ZNE", inventory=inv)
    coors = inv.get_coordinates(st_data[0].id)
    return st_data, coors

def main():
    # Lets get the parser arguments
    parser_val = getargs()
    client = Client(parser_val.client)
    debug = parser_val.debug
    net = parser_val.network
    res_dir=parser_val.res_dir

    # Load in CMT Solution
    cat = obspy.read_events(parser_val.syn + '/CMTSOLUTION')

    # Made this a bit smaller
    if not os.path.exists(parser_val.res_dir):
        os.mkdir(parser_val.res_dir)
            # Open a file to write the correlation statistics
    evename=(str(cat.resource_id).split('/')[-2])
    statfile = open(os.getcwd() + '/' + res_dir + '/Results' + evename + net + '.csv' ,'w')
    statfile.write('net, sta, loc, chan, scalefac, lag, corr, time\n')

    # Get filter corners
    userminfre = 1.0/float(parser_val.filter[0])
    usermaxfre = 1.0/float(parser_val.filter[1])
    filtercornerpoles = int(parser_val.filter[2])

    if parser_val.sta:
        sta_list = parser_val.sta.replace(' ','')
        sta_list = sta_list.split(',')
        files = []
        for sta in sta_list:
            files += glob.glob(parser_val.syn + '/' + net + '*' + sta + '*MXZ*')
    else:
        files = glob.glob(parser_val.syn + '/' + net + '*MXZ*')

    # Synthetic Data
    syn = parser_val.syn


    # Get event information
    cmtlat = cat[0].origins[0].latitude
    cmtlon = cat[0].origins[0].longitude

    # Get source time function
    stf = cat[0].focal_mechanisms[0].moment_tensor.source_time_function
    # Apply a window
    win = signal.hann(int(2*stf['duration']))

    # Load in files
    for cur_file in files:
        cur_file = cur_file.replace('MXZ', 'MX*')
        try:
            st = read(cur_file)
        except:
            print('Directory Empty: No files to read in')
            continue
    # Loop over waveform data
        for tr in st:
            tr.data = resample(tr.data, int(tr.stats.endtime - tr.stats.starttime))
            tr.stats.sampling_rate = 1.
            tr.data = signal.convolve(tr.data,win, mode='same')/sum(win)


        st_data, coors = setup_data(net, st[-1].stats.station, st[-1].stats.starttime,
                                    st[-1].stats.endtime, client)
        try:
            st += st_data
        except:
            print('No Data for: ' + st[-1].stats.station)

            continue
        st.filter('bandpass', freqmin=usermaxfre, freqmax=userminfre)
        st.taper(0.05)


        # Open a file to write the correlation statistics
        # evename=(str(cat.resource_id).split('/')[-2])
        # statfile = open(os.getcwd() + '/' + res_dir + '/Results' + evename + net + '.csv' ,'w')
        # statfile.write('net, sta, loc, chan, scalefac, lag, corr, time\n')


        # Plot data, calculate correlation statistics, and save plot to directory
        fig = plt.figure(1, figsize=(12, 12))
        for idx, chan in enumerate(['Z', 'N', 'E']):
            if chan == 'Z':
                plt.subplot(3,1,1)
                title = st[0].stats.network + ' ' + \
                        st[0].stats.station + ' '
                starttime = st[0].stats.starttime
                stime = str(starttime.year) + ' ' + str(starttime.julday) + \
                        ' ' + str(starttime.hour) + ':' + \
                        str(starttime.minute) + ':' + \
                        str("{0:.2f}".format(starttime.second))
                title += stime + ' '
                lat= coors['latitude']
                lon = coors['longitude']
                dist = gps2dist_azimuth(float(cmtlat), float(cmtlon), lat, lon)
                bazi = "{0:.1f}".format(dist[2])
                dist = "{0:.1f}".format(0.0089932*dist[0]/1000.)
                title += 'Dist:' + str(dist)
                title += ' BAzi:' + str(bazi) + ' '
                title += str("{0:.0f}".format(1/userminfre)) + '-' + \
                         str("{0:.0f}".format(1/usermaxfre)) + ' s per.'
                plt.title(title, fontsize=24)

            # #Get Correlation Statistics
            # try:
            #     writestats(statfile, st, chan)
            # except:
            #     print('Problem with: ' + sta)

            plt.subplot(3, 1, idx + 1)
            st_t = st.select(channel="*" + chan)
            for tr in st_t:
                plt.plot(tr.times(), tr.data * 1000., label=tr.id)
                plt.xlim((min(tr.times()), max(tr.times())))
            # Only label subplot 2 y-axis
            if idx == 1:
                filename = get_file_name(parser_val.res_dir, st_data)
                plt.ylabel('Displacement (mm)')
            plt.legend(prop={'size': 8}, loc=2)

                #Get Correlation Statistics
            try:
                writestats(statfile, st, chan)
            except:
                print('Problem with: ' + sta)



        plt.xlabel('Time (s)')
        plt.savefig(filename + '.PNG', format='PNG', dpi=200)
        plt.clf()
        plt.close()
#    statfile.close()


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
                        default=[50, 150, 4],
                        help="Filter parameters using minimum period maximum period "
                             "and number of corners Example: 50 150 4, "
                             "default is 50,150, 4")

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



def writestats(statfile, st, chan):
    """
    calculate the correlation coefficient and lag time for the synthetic
    when compared to the observed data and write to a file.
    """
    try:

        syncomp = "MX" + chan
        datacomp = "LH" + chan
        syn = st.select(channel = syncomp)
        for tr in st.select(channel = datacomp):
            resi = "{0:.2f}".format(np.sum(tr.data*syn[0].data[:-1])/np.sum(np.square(syn[0].data[:-1])))
            cc=correlate(tr,syn[0],500)
            lag, corr = xcorr_max(cc)
            corr = "{0:.2f}".format(corr)
            #
            statfile.write(tr.stats.network + "," + tr.stats.station)
            statfile.write("," + tr.stats.location + "," + tr.stats.channel + "," +  str(resi))
            statfile.write("," + str(lag) + "," + str(corr) + ", ")
            statfile.write(str(tr.stats.starttime.month) + "/" + str(tr.stats.starttime.day) + \
                                "/" + str(tr.stats.starttime.year) + " " + str(tr.stats.starttime.hour) + ":" + \
                                str(tr.stats.starttime.minute) + ":" + str(tr.stats.starttime.second) + "\n")
    except:
    #    if debug:
            print('No residual for ' + tr.stats.station + ' ' + 'LH' + chan)
    return



if __name__ == "__main__":
    main()
