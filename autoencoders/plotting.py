import h5py
import numpy as np
import tensorflow as tf
import math
import os
from sklearn.metrics import roc_curve, auc, confusion_matrix

BSM_SAMPLES = ['Leptoquark', 'A to 4 leptons', 'hChToTauNu', 'hToTauTau']
SAMPLES = ['QCD']+BSM_SAMPLES

PLOTTING_LABELS = ['Background', r'LQ $\rightarrow$ b$\tau$', r'A $\rightarrow$ 4$\ell$',
    r'$h^{\pm} \rightarrow \tau\nu$', r'$h^{0} \rightarrow \tau\tau$']
LABELS = {
    'Leptoquark': (r'LQ $\rightarrow$ b$\tau$', 'o', '#016c59'),
    'A to 4 leptons': (r'A $\rightarrow$ 4$\ell$', 'X', '#7a5195'),
    'hChToTauNu': (r'$h^{\pm} \rightarrow \tau\nu$', 'v', '#67a9cf'),
    'hToTauTau': (r'$h^{0} \rightarrow \tau\tau$', 'd', '#ffa600')}

def radius(mean, logvar):
    sigma = np.sqrt(np.exp(logvar))
    radius = mean*mean/sigma/sigma
    radius = np.sum(radius, axis=-1)

    radius = np.nan_to_num(radius)
    radius[radius==-np.inf] = 0
    radius[radius==np.inf] = 0
    radius[radius>=1E308] = 0
    return radius

def mse_loss(inputs, outputs):
    return np.mean(np.square(inputs-outputs), axis=-1)

def kl_loss(z_mean, z_log_var):
    kl = 1 + z_log_var - np.square(z_mean) - np.exp(z_log_var)
    kl = - 0.5 * np.mean(kl, axis=-1) # multiplying mse by N -> using sum (instead of mean) in kl loss (todo: try with averages)
    return kl

def reco_loss(inputs, outputs, dense=False):

    if dense:
        outputs = outputs.reshape(outputs.shape[0],19,3,1)
        inputs = inputs.reshape(inputs.shape[0],19,3,1)

    # trick on phi
    outputs_phi = math.pi*np.tanh(outputs)
    # trick on eta
    outputs_eta_egamma = 3.0*np.tanh(outputs)
    outputs_eta_muons = 2.1*np.tanh(outputs)
    outputs_eta_jets = 4.0*np.tanh(outputs)
    outputs_eta = np.concatenate([outputs[:,0:1,:,:], outputs_eta_egamma[:,1:5,:,:], outputs_eta_muons[:,5:9,:,:], outputs_eta_jets[:,9:19,:,:]], axis=1)
    outputs = np.concatenate([outputs[:,:,0,:], outputs_eta[:,:,1,:], outputs_phi[:,:,2,:]], axis=2)
    # change input shape
    inputs = np.squeeze(inputs, -1)
    # # calculate and apply mask
    mask = np.not_equal(inputs, 0)
    outputs = np.multiply(outputs, mask)

    # inputs = inputs[:,:,1:]
    # outputs = outputs[:,:,1:]

    reco_loss = mse_loss(inputs.reshape(inputs.shape[0],-1), outputs.reshape(outputs.shape[0],-1))
    return reco_loss

def make_plot_training_history(h5f, output_dir):
    loss = h5f['loss'][:]
    val_loss = h5f['val_loss'][:]

    plt.plot(loss)
    plt.plot(val_loss)
    plt.title('Training History')

    plt.semilogy()
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['training', 'validation'], loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_hist.pdf'))

def make_plot_loss_distribution(mse_data, loss_type, labels, output_dir):
    bin_size = 500
    plt.figure()
    for i, label in enumerate(labels):
        plt.hist(mse_data[i], bins=bin_size, label=label, histtype='step', fill=False, linewidth=1.5)
    plt.semilogy()
    plt.semilogx()
    plt.title(loss_type)
    plt.xlabel('Autoencoder Loss')
    plt.ylabel('Probability (a.u.)')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'{loss_type}_dist.pdf'))

def make_plot_roc_curves(plt, qcd, bsm, loss_type, color_id, alpha=1.0, line='-'):
    colors = ['#7a5195','#ef5675','#3690c0','#ffa600','#67a9cf','#014636', '#016c59']

    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.concatenate((bsm, qcd))

    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val)
    auc_loss = auc(fpr_loss, tpr_loss)

    plt.plot(fpr_loss, tpr_loss, '-',
        label=f'{loss_type} ROC (auc = %.1f%%)'%(auc_loss*100.),
        linewidth=2.0,
        linestyle=line,
        color=colors[color_id],
        alpha=alpha)
    plt.plot(np.linspace(0, 1),np.linspace(0, 1), '--', color='0.75')
    plt.vlines(1e-5, 0, 1, linestyles='--', color='lightcoral')

    plt.semilogx()
    plt.semilogy()
    plt.ylabel('True Positive Rate', fontsize=15)
    plt.xlabel('False Positive Rate', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.1, 1.05),frameon=False, fontsize=12)
    plt.tight_layout()

def read_loss_data(results_file, beta=None):

    with h5py.File(results_file, 'r') as data:

        vae = True if ('vae' in results_file) or ('VAE' in results_file) else False
        dense = True if 'epuljak' in results_file else False

        total_loss = []
        kl_data = []
        r_data = []
        mse_loss=[]

        X_test_scaled = data['QCD'][:]
        qcd_prediction = data['predicted_QCD'][:]
        #compute loss
        mse_loss.append(reco_loss(X_test_scaled, qcd_prediction.astype(np.float32), dense=dense))
        if vae:
            qcd_mean = data['encoded_mean_QCD'][:]
            qcd_logvar = data['encoded_logvar_QCD'][:]
            qcd_z = data['encoded_z_QCD'][:]
            kl_data.append(kl_loss(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32)))
            r_data.append(radius(qcd_mean.astype(np.float32), qcd_logvar.astype(np.float32)))

        #BSM
        for bsm in BSM_SAMPLES:
            bsm_target = data[bsm+'_scaled'][:]
            bsm_prediction = data['predicted_'+ bsm][:]
            mse_loss.append(reco_loss(bsm_target, bsm_prediction.astype(np.float32), dense=dense))
            if vae:
                bsm_mean = data['encoded_mean_'+bsm][:]
                bsm_logvar = data['encoded_logvar_'+bsm][:]
                bsm_z = data['encoded_z_'+bsm][:]
                kl_data.append(kl_loss(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32)))
                r_data.append(radius(bsm_mean.astype(np.float32), bsm_logvar.astype(np.float32)))
        if vae:
            total_loss = []
            for mse, kl in zip(mse_loss, kl_data):
                total_loss.append(np.add(mse, kl))
        else:
            total_loss = mse_loss.copy()

    if vae: del X_test_scaled, qcd_prediction, qcd_mean, qcd_logvar, qcd_z, bsm_target, bsm_prediction,\
                            bsm_mean, bsm_logvar, bsm_z
    else: del X_test_scaled, qcd_prediction, bsm_target, bsm_prediction

    return total_loss, mse_loss, kl_data, r_data

def add_logo(ax, fig, zoom, position='upper left', offsetx=10, offsety=10, figunits=False):
    # from https://github.com/thaarres/hls4ml_cnns/blob/master/plot.py
    # resize image and save to new file
    img = cv2.imread('plots/logo.jpeg', cv2.IMREAD_UNCHANGED)
    im_w = int(img.shape[1] * zoom )
    im_h = int(img.shape[0] * zoom )
    dim = (im_w, im_h)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    cv2.imwrite('plots/logo_resized.jpeg', resized );

    #read resized image
    im = cv2.imread('plots/logo_resized.jpeg')
    im_w = im.shape[1]
    im_h = im.shape[0]

    #get coordinates of corners in data units and compute an offset in pixel units depending on chosen position
    ax_xmin,ax_ymax = 0,0
    offsetX = 0
    offsetY = 0
    if position=='upper left':
        ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
        offsetX,offsetY = offsetx,-im_h-offsety
    elif position=='out left':
        ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[1]
        offsetX,offsetY = offsetx,offsety
    elif position=='upper right':
        ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
        offsetX,offsetY = -im_w-offsetx,-im_h-offsety
    elif position=='out right':
        ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[1]
        offsetX,offsetY = -im_w-offsetx,offsety
    elif position=='bottom left':
        ax_xmin,ax_ymax = ax.get_xlim()[0],ax.get_ylim()[0]
        offsetX,offsetY=offsetx,offsety
    elif position=='bottom right':
        ax_xmin,ax_ymax = ax.get_xlim()[1],ax.get_ylim()[0]
        offsetX,offsetY=-im_w-offsetx,offsety

    #transform axis limits in pixel units
    ax_xmin,ax_ymax = ax.transData.transform((ax_xmin,ax_ymax))
    #compute figure x,y of bottom left corner by adding offset to axis limits (pixel units)
    f_xmin,f_ymin = ax_xmin+offsetX,ax_ymax+offsetY

    #add image to the plot
    fig.figimage(im,f_xmin,f_ymin)

    #compute box x,y of bottom left corner (= figure x,y) in axis/figure units
    if figunits:
        b_xmin,b_ymin = fig.transFigure.inverted().transform((f_xmin,f_ymin))
    #print("figunits",b_xmin,b_ymin)
    else: b_xmin,b_ymin = ax.transAxes.inverted().transform((f_xmin,f_ymin))

    #compute figure width/height in axis/figure units
    if figunits: f_xmax,f_ymax = fig.transFigure.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to figure units the figure limits
    else: f_xmax,f_ymax = ax.transAxes.inverted().transform((f_xmin+im_w,f_ymin+im_h)) #transform to axis units the figure limits
    b_w = f_xmax-b_xmin
    b_h = f_ymax-b_ymin

    #set which units will be used for the box
    transformation = ax.transAxes
    if figunits: transformation=fig.transFigure

    rectangle = FancyBboxPatch((b_xmin,b_ymin),
                              b_w, b_h,
                  transform=transformation,
                  boxstyle='round,pad=0.004,rounding_size=0.01',
                  facecolor='w',
                  edgecolor='w',linewidth=0.8,clip_on=False)
    ax.add_patch(rectangle)