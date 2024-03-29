import ROOT
import h5py
import numpy as np
import argparse
import os
import shutil
from scipy.optimize import curve_fit
import matplotlib
import matplotlib.pyplot as plt
#matplotlib.use("Agg") 
import mplhep as hep
hep.style.use(hep.style.CMS) 

colors = ['#016c59', '#7a5195', '#ef5675', '#ffa600', '#67a9cf']
colors_reco_corr = ['black','#017517','#7a5195','#dede00','#f781bf']
markers = ['s','*','o','v','8']
linestyles=['solid','dashed','dotted','dashdot',(0, (1, 1))]


def gaus(x,a,mu,sigma):
    return a*np.exp(-(x-mu)**2/(2*sigma**2))


def compute_resolution(x, y, nbin,do_fit=True):
    percent_bins = np.linspace(0,100.,num=nbin+1)
    bins = np.percentile(x,percent_bins )
    h, xe, ye = np.histogram2d(x,y,bins)
    # bin width
    xbinw = xe[1]-xe[0]

    prop_dict = {}
    prop_dict['xval']  = []
    prop_dict['mean'] = []
    prop_dict['mpv'] = []
    prop_dict['quant_diff']  = []
    prop_dict['mu'] = []
    prop_dict['sigma']  = []
    for i in range(xe.size-1):
        yvals = y[ (x>xe[i]) & (x<=xe[i+1]) ]
        if yvals.size>0: # do not fill the quanties for empty slices
            prop_dict['xval'].append(xe[i]+ xbinw/2)
            prop_dict['mean'].append( yvals.mean())
            ye_slice,xe_slice = np.histogram(yvals, bins=np.linspace(0,3,100))
            xe_slice_centers = (xe_slice[:-1]+xe_slice[1:])/2
            prop_dict['mpv'].append(xe_slice[int(np.argmax(ye_slice))])
            prop_dict['quant_diff'].append( (np.quantile(yvals,0.84)-np.quantile(yvals,0.16))/2.)
            if do_fit:
                mask_width = 0.2
                mask = (xe_slice_centers>prop_dict['mpv'][i]*mask_width) & (xe_slice_centers<prop_dict['mpv'][i]*(1.+mask_width))
                popt,_ = curve_fit(gaus,xe_slice_centers[mask],ye_slice[mask])
                prop_dict['mu'].append(popt[1])
                prop_dict['sigma'].append(popt[2])
    for key in prop_dict.keys():
        prop_dict[key] = np.array(prop_dict[key])
    return prop_dict


def makeEffHist(refArr, corrArr, corrThr, xmax,nbins=20):
    not_matched = []
    eff_bins = np.linspace(0,xmax,nbins+1)
    ret = ROOT.TEfficiency("eff","",nbins,eff_bins)
    for i in range(0,len(refArr)):
        ret.Fill(corrArr[i]>corrThr,refArr[i])
    ret.SetStatisticOption(ret.kFCP)        
    return ret,eff_bins 

def calculateEff(refArr, corrArr, corrThr, xmax,nbins=20):
    ret,eff_bins = makeEffHist(refArr, corrArr, corrThr, xmax,nbins)
    prop_dict = {}
    prop_dict['bins_centers'] = (eff_bins[1:]+eff_bins[:-1])/2.
    prop_dict['bins_err'] = prop_dict['bins_centers']-eff_bins[:-1]
    prop_dict['yvals'] = []
    yvals_up, yvals_down = [],[]
    for i in range(len(prop_dict['bins_centers'])):
        prop_dict['yvals'].append(ret.GetEfficiency(i+1))
        yvals_up.append(ret.GetEfficiencyErrorUp(i+1))
        yvals_down.append(ret.GetEfficiencyErrorLow(i+1))
    prop_dict['yvals_down_up'] =np.vstack( (np.array(yvals_down),np.array(yvals_up)))
    for key in prop_dict.keys():
        prop_dict[key] = np.array(prop_dict[key])
    return prop_dict

def makeRate(values,max_x,nbins=20):
    x_values = np.linspace(0,max_x,nbins)
    eff = [np.sum(values>thresh)/len(values) for thresh in x_values]
    return np.array(eff),x_values



def get_threshold_from_rate(values,max_x,threshold):
    nbins=500 #we need many bins to do this
    eff,x_values = makeRate(values,max_x,nbins=nbins)
    idx = eff.size - np.searchsorted(eff[::-1], threshold, side = "right")
    return x_values[idx]
    
def get_rate_from_threshold(values,max_x,threshold):
    nbins=500 #we need many bins to do this
    eff,x_values = makeRate(values,max_x,nbins=nbins)
    rate = eff[np.searchsorted(x_values,threshold)]
    return rate

def plot_scatter(datas_x,datas_y,labels,xtitle,ytitle,semilogy=False,semilogx=False,output_dir='',plot_name='', title=''):
    for i in range(len(datas_x)):
        _ = plt.scatter(datas_x[i],datas_y[i],color=colors_reco_corr[i],marker=markers[i],s=100,label=labels[i])
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    plt.legend(frameon=False)
    if semilogy:
        plt.semilogy()
    if semilogx:  
        plt.semilogx()
    if output_dir=='' or plot_name=='':
        print('No output directory set, only showing the plot')
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, plot_name.replace('.pdf','.png')), bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_trigger_roc(datas_x,datas_y,labels,xtitle,ytitle,semilogy=False,semilogx=False,output_dir='',plot_name='', title='',thresholds=[]):
    fig, axs = plt.subplots(2,figsize=(12,15), gridspec_kw={'height_ratios': [3, 1]})
    for i in range(len(datas_x)):
        _ = axs[0].scatter(datas_x[i],datas_y[i],color=colors_reco_corr[i],marker=markers[i],s=100,label=labels[i])
    axs[0].set_title(title)
    axs[0].set_ylabel(ytitle)
    axs[0].grid(color='gray', linestyle='--', linewidth=0.3)
    axs[0].legend(frameon=False)
    axs[0].plot(np.linspace(0, 1),np.linspace(0, 1), ':', color='0.75', linewidth=2)
    for l in thresholds:
        axs[0].vlines(l, 0, 1, linestyles='--', color='#ef5675', linewidth=2)
    if semilogy:
        axs[0].semilogy()
    if semilogx:  
        axs[0].semilogx()

    for i in range(len(datas_y)-1):
        _ = axs[1].scatter(datas_x[i],np.array(datas_y[i+1])/np.array(datas_y[0])-1.,color=colors_reco_corr[i+1],marker=markers[i+1],s=100)
    if semilogx:  
        axs[0].semilogx()
    axs[1].set_ylabel(r'$\frac{\Delta}{\mathrm{%s}}$'%labels[0])
    axs[1].set_xlabel(xtitle)
    axs[1].grid(color='gray', linestyle='--', linewidth=0.3)
    #axs[1].set_ylim([0, 0.1])
    if output_dir=='' or plot_name=='':
        print('No output directory set, only showing the plot')
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, plot_name.replace('.pdf','.png')), bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_distibutions(datas, labels,xtitle,ytitle,bins=40,output_dir='',plot_name='', title='',semilogy=True, density=False):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    if isinstance(bins,int):
        min_x = np.min([np.min(data) for data in datas])
        max_x = np.max([np.quantile(data,0.999)*1.1 for data in datas])
        bins=np.linspace(min_x,max_x,bins)
    for i in range(len(datas)):
        _,_,_ = ax.hist(datas[i],bins=bins,density=density,histtype='step',linewidth=2,linestyle=linestyles[i],color=colors_reco_corr[i],
                    label=labels[i])
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    if semilogy:
        plt.semilogy()
    handles, labels = ax.get_legend_handles_labels()
    new_handles = [plt.Line2D([], [], c=h.get_edgecolor()) for h in handles]
    plt.legend(frameon=False, handles=new_handles, labels=labels)
    if output_dir=='' or plot_name=='':
        print('No output directory set, only showing the plot')
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, plot_name.replace('.pdf','.png')), bbox_inches="tight")
        plt.clf()
       # plt.close()



def plot_efficiency(datas_x,datas_y, thresholds,max_x,labels,xtitle,ytitle,nbins=40,output_dir='',plot_name='', title=''):
    for i in range(len(datas_x)):
        eff_dict = calculateEff(datas_x[i],datas_y[i],thresholds[i],max_x,nbins=nbins)
        _ = plt.errorbar(eff_dict['bins_centers'],eff_dict['yvals'],xerr=eff_dict['bins_err'],yerr=eff_dict['yvals_down_up'], 
            color=colors_reco_corr[i],marker=markers[i],ms=10,linestyle='',label=labels[i])

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    plt.legend(frameon=False,loc='lower right')
    if output_dir=='' or plot_name=='':
        print('No output directory set, only showing the plot')
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, plot_name.replace('.pdf','.png')), bbox_inches="tight")
        plt.clf()
        plt.close()



def plot_resolutions(datas_x,datas_y, labels,xtitle,ytitle,what_to_plot,do_fit=False,output_dir='',plot_name='', title=''):
    what_to_plot = what_to_plot.split('/')        
    for i in range(len(datas_x)):
        prop_dict = compute_resolution(datas_x[i],datas_y[i],nbin=20,do_fit=do_fit)
        if len(what_to_plot)==2:
            y = prop_dict[what_to_plot[0]]/prop_dict[what_to_plot[1]]
        else:
            y = prop_dict[what_to_plot[0]]
        plt.scatter(prop_dict['xval'], y, marker=markers[i],color=colors_reco_corr[i],s=100,label=labels[i])
        
    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    plt.legend(frameon=False,loc='upper right')
    if output_dir=='' or plot_name=='':
        print('No output directory set, only showing the plot')
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, plot_name.replace('.pdf','.png')), bbox_inches="tight")
        plt.clf()
        plt.close()


def plot_ratios(datas, labels,xtitle,ytitle,output_dir='',plot_name='', title='',semilogy=False):
    bins=np.linspace(0,4,200)
    bin_centers = (bins[1:]+bins[:-1])/2.
    for i,data in enumerate(datas):
        yval,_ = np.histogram(data, bins=bins,density=True)
        mean = np.mean(data)
        mpv = bin_centers[int(np.argmax(yval))]
        mask_width=0.2
        mask = (bin_centers>mpv*mask_width) & (bin_centers<mpv*(1.+mask_width))
        popt,pcov = curve_fit(gaus,bin_centers[mask],yval[mask])
        _ = plt.step(bin_centers,yval,linewidth=2,color=colors_reco_corr[i],where='mid',
                    label=labels[i]+f'\n mean={mean:.2f}, mpv={mpv:.2f}')
        plt.plot(bin_centers,gaus(bin_centers,*popt),linewidth=1, linestyle='--', color=colors_reco_corr[i],
                    label='fit gaus ' + f'\n $\mu$={popt[1]:.2f}, $\sigma$={popt[2]:.2f}')

    plt.title(title)
    plt.xlabel(xtitle)
    plt.ylabel(ytitle)
    plt.grid(color='gray', linestyle='--', linewidth=0.3)
    if semilogy:
        plt.semilogy()
    plt.legend(frameon=False,loc='upper right')
    if output_dir=='' or plot_name=='':
        print('No output directory set, only showing the plot')
        plt.show()
    else:
        plt.savefig(os.path.join(output_dir, plot_name), bbox_inches="tight")
        plt.savefig(os.path.join(output_dir, plot_name.replace('.pdf','.png')), bbox_inches="tight")
        plt.clf()
        plt.close()


        




    




    