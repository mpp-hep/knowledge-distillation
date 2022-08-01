import ROOT
import h5py
import numpy as np
import argparse
import os
import shutil
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg") 
matplotlib.rcParams.update({'font.size': 18})

def compute_resolution(x, y, nbin,log_bins_x=True):
    """
    Returns the center of bins array, the mean of y for each bin and stand.dev.
    https://vmascagn.web.cern.ch/LABO_2020/profile_plot.html
    """
    # use of the 2d hist by numpy to avoid plotting
    if log_bins_x :
        nbin_x = np.logspace(0,3,nbin)
        nbin_y = np.linspace(-1,1,nbin)
        bins = [nbin_x,nbin_y]
    else :
        bins=nbin
    h, xe, ye = np.histogram2d(x,y,bins)

    # bin width
    xbinw = xe[1]-xe[0]

    x_array      = []
    x_slice_mean = []
    x_slice_mpv = []
    x_slice_res  = []
    for i in range(xe.size-1):
        yvals = y[ (x>xe[i]) & (x<=xe[i+1]) ]
        if yvals.size>0: # do not fill the quanties for empty slices
            x_array.append(xe[i]+ xbinw/2)
            x_slice_mean.append( yvals.mean())
            ye_slice,xe_slice = np.histogram(yvals, bins=np.linspace(0,3,100))
            xe_slice_centers = (xe_slice[:-1]+xe_slice[1:])/2
            x_slice_mpv.append(xe_slice[int(np.argmax(ye_slice))])
            x_slice_res.append( (np.quantile(yvals,0.84)-np.quantile(yvals,0.16))/2.)
    x_array = np.array(x_array)
    x_slice_mean = np.array(x_slice_mean)
    x_slice_res = np.array(x_slice_res)
    x_slice_mpv = np.array(x_slice_mpv)

    return x_array, x_slice_mean,x_slice_mpv, x_slice_res


def makeEffHist(name, refArr, corrArr, corrThr, xmax,nbins=20):
    not_matched = []
    eff_bins = np.linspace(0,xmax,nbins+1)
    ret = ROOT.TEfficiency(name+"_eff","",nbins,eff_bins)
    for i in range(0,len(refArr)):
        ret.Fill(corrArr[i]>corrThr,refArr[i])
    ret.SetStatisticOption(ret.kFCP)        
    return ret,eff_bins 

def calculateEff(name, refArr, corrArr, corrThr, xmax,nbins=20):
    ret,eff_bins = makeEffHist(name, refArr, corrArr, corrThr, xmax,nbins)
    eff_bins_centers = (eff_bins[1:]+eff_bins[:-1])/2.
    eff_bins_err = eff_bins_centers-eff_bins[:-1]
    yvals, yvals_up, yvals_down = [],[],[]
    for i in range(len(eff_bins_centers)):
        yvals.append(ret.GetEfficiency(i+1))
        yvals_up.append(ret.GetEfficiencyErrorUp(i+1))
        yvals_down.append(ret.GetEfficiencyErrorLow(i+1))
    return eff_bins_centers,eff_bins_err,np.array(yvals),np.vstack( (np.array(yvals_down),np.array(yvals_up)))
        
    
def makeRatePlot(refArr, corrThr):
    eff = [np.sum(refArr>thresh)/len(refArr) for thresh in corrThr]
    return eff

    