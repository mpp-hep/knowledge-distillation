import setGPU
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import shutil 
import tensorflow.keras.backend as K
import utils.data_processing as data_proc
import utils.plotting as plotting
from nn.models import GraphAttentionHyperModel
import nn.losses as nn_losses
import keras_tuner
from keras_tuner import HyperParameters
matplotlib.use("Agg") 
matplotlib.rcParams.update({'font.size': 18})





def main_analyze_results(data_file='',variable='',log_features=[],training_dir='',inclusive_thresholds=[],plot_dir=''):
    """
    Plots resolutions, rates, efficiencies for a given training 
    Arguments:
        data_file: str, path to the input file
        variable: str, variable on which to produce results : original_met, true_met, true_ht 
        log_features: list of str, which feature scale to be log
        plot_dir: str, plotting directory
        inclusive_thresholds: list of float, inclusive thresholds for MET
        training_dir: str, path to the training used TODO : to clean up
    """
    with h5py.File(data_file,'r') as open_file :
        reco_data = np.array(open_file['smeared_data'])
        reco_met = np.array(open_file['smeared_met'])
        reco_ht = np.array(open_file['smeared_ht'])
        true_data = np.array(open_file['true_data'])
        true_met = np.array(open_file['true_met'])
        original_met = np.array(open_file['original_met'])
        true_ht = np.array(open_file['true_ht'])
        ids = np.array(open_file['ids'])
        ids_names = np.array(open_file['ids_names'])

    if variable=='original_met':
        graph_data = data_proc.METGraphCreator(reco_data,reco_met,reco_ht, original_met,true_ht,ids,log_features=log_features)
    elif variable=='true_met':
        graph_data = data_proc.METGraphCreator(reco_data,reco_met,reco_ht, true_met,true_ht,ids,log_features=log_features)
    elif variable=='true_ht':
        graph_data = data_proc.HTGraphCreator(reco_data,reco_ht,true_ht,ids,log_features=log_features)
    if 'met' in variable:
        var_name = 'MET'
    elif 'ht' in variable:
        var_name = 'HT'

    num_filters = 1
    graph_conv_filters = graph_data.adjacency
    graph_conv_filters = K.constant(graph_conv_filters)
    loss_function = nn_losses.get_loss_func('huber_1.0')
    metric_thresholds = [0]
    metrics = [nn_losses.MseThesholdMetric(threshold=t) for t in metric_thresholds]
    hp = keras_tuner.HyperParameters()
    hypermodel = GraphAttentionHyperModel(features_input_shape=(graph_data.features.shape[1],graph_data.features.shape[2]), 
                                        adjancency_input_shape=(graph_data.adjacency.shape[1],graph_data.adjacency.shape[2]),
                                        filters_input_shape=(graph_conv_filters.shape[1],graph_conv_filters.shape[2]), 
                                        num_filters=num_filters, 
                                        loss_function=loss_function,
                                        metrics=metrics)

    tuner = keras_tuner.Hyperband(hypermodel = hypermodel,
                     objective = keras_tuner.Objective("val_loss", direction="min"),
                    executions_per_trial=1,
                    max_epochs=50,
                    overwrite=False,
                     directory=training_dir,
                     project_name='hyperband_tuner')
    model = tuner.get_best_models(num_models=1)[0] #Eventually the best model should be retrained
    dnn_correction = model.predict([graph_data.features, graph_data.adjacency,graph_conv_filters],batch_size=1024)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir) 

    for proc,proc_id in zip(['W + jets','QCD'],[0,1]):    
        proc_save_name = proc.replace(' ','').replace('+','')
        plot_subdir = plot_dir+'/'+proc_save_name
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)    
            
        proc_mask =  np.where(ids==proc_id)
        evaluator = data_proc.ResolutionEvaluator(graph_data.reco_data,graph_data.true_data,graph_data.process_ids,dnn_correction,mask=proc_mask)

        plotting.plot_distibutions([graph_data.labels[:,0][proc_mask],dnn_correction[proc_mask]], ['Target','Prediction'],
                                    xtitle='Target vs Prediction',ytitle='Normalized Yield',title=proc,nbins=100,semilogy=True,
                                    output_dir=plot_subdir,plot_name=f'plot_{var_name}_target_prediction_{proc_save_name}.pdf')

        plotting.plot_distibutions([evaluator.reco_data,evaluator.corr_data,evaluator.true_data],['Baseline','Corrected','True'],
                                xtitle=f'{var_name}',ytitle='Normalized Yield',nbins=100,semilogy=True,
                                output_dir=plot_subdir,plot_name=f'plot_{var_name}_{proc_save_name}.pdf')

        plotting.plot_ratios([evaluator.rot,evaluator.rot_corr],['Baseline','Corrected'] ,
                                xtitle=r'$\frac{%s^{reco}}{%s^{true}}$'%(var_name,var_name),ytitle='Normalized Yield',title=proc,semilogy=False,
                                    output_dir=plot_subdir,plot_name=f'plot_{var_name}_reco_o_over_{proc_save_name}.pdf')

        plotting.plot_ratios([evaluator.tor,evaluator.tor_corr],['Baseline','Corrected'] ,
                                xtitle=r'$\frac{%s^{true}}{%s^{reco}}$'%(var_name,var_name),ytitle='Normalized Yield',title=proc,semilogy=False,
                                    output_dir=plot_subdir,plot_name=f'plot_{var_name}_true_o_reco_{proc_save_name}.pdf')

        #Resolution plots
        for what_to_plot in ['quant_diff/mpv','quant_diff']:
            what_to_plot_name = what_to_plot.replace('/','_')
            plotting.plot_resolutions([evaluator.true_data,evaluator.true_data],[evaluator.tor,evaluator.tor_corr],
                                ['Baseline','Corrected'],xtitle=r'$%s^{true}$'%(var_name),ytitle=r'$\frac{%s^{true}}{%s^{reco}}$, '%(var_name,var_name)+what_to_plot,title=proc,
                                what_to_plot=what_to_plot,do_fit=False,
                                output_dir=plot_subdir,plot_name=f'plot_{var_name}_{what_to_plot_name}_true_o_reco_{proc_save_name}.pdf')

            plotting.plot_resolutions([evaluator.true_data,evaluator.true_data],[evaluator.rot,evaluator.rot_corr],
                                ['Baseline','Corrected'],xtitle=r'$%s^{true}$'%(var_name),ytitle=r'$\frac{%s^{reco}}{%s^{true}}$, '%(var_name,var_name)+what_to_plot,title=proc,
                                what_to_plot=what_to_plot,do_fit=False,
                                output_dir=plot_subdir,plot_name=f'plot_{var_name}_{what_to_plot_name}_reco_o_true_{proc_save_name}.pdf')


        #Inclusive efficiency plots
        for thresh in inclusive_thresholds:
            max_val = 420
            nbins=40
            plotting.plot_efficiency([evaluator.true_data,evaluator.true_data],[evaluator.reco_data,evaluator.corr_data], 
                                thresholds=[thresh,thresh],max_x=max_val,
                                labels=['Baseline','Corrected'],xtitle=r'$%s^{true}$'%(var_name),ytitle='Normalized Yield',title=proc,nbins=nbins,
                                output_dir=plot_subdir,plot_name=f'plot_inclusive_eff_{var_name}_{thresh}_{proc_save_name}.pdf')

        #Rate plots
        max_val = 600
        eff,thresholds = plotting.makeRate(evaluator.reco_data,max_val)
        eff_corr,_ = plotting.makeRate(evaluator.corr_data,max_val)
        plotting.plot_scatter([thresholds,thresholds],[eff,eff_corr],
                                ['Baseline','Corrected'],xtitle=r'$%s^{true}$'%(var_name),ytitle='Normalized Yield',title=proc,semilogy=True,
                                output_dir=plot_subdir,plot_name=f'plot_rates_{var_name}_{proc_save_name}.pdf')

        #Efficiency plots for the same rates
        for reco_thresh in inclusive_thresholds:
            max_val = 600
            rate = plotting.get_rate_from_threshold(evaluator.reco_data,max_val,reco_thresh)
            corr_thresh = plotting.get_threshold_from_rate(evaluator.corr_data,max_val,rate)
            max_val_plot = 420
            nbins=40
            plotting.plot_efficiency([evaluator.true_data,evaluator.true_data],[evaluator.reco_data,evaluator.corr_data], 
                                thresholds=[reco_thresh,corr_thresh],max_x = max_val_plot,
                                labels=[f'Baseline, {var_name}>{reco_thresh:.1f} GeV',f'Corrected, {var_name}>{corr_thresh:.1f} GeV'],
                                xtitle=r'$%s^{true}$'%(var_name),ytitle='Normalized Yield',title=proc+f', Rate={rate:.3f}',nbins=nbins,
                                output_dir=plot_subdir,plot_name=f'plot_eff_{var_name}_{reco_thresh}_const_rate_{proc_save_name}.pdf')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Path to the input file ')
    parser.add_argument('--log_features', type=str, help='Which features scale to be log')
    parser.add_argument('--variable', type=str, help='Variable to analyze : MET or HT ')
    parser.add_argument('--training_dir', type=str, help='Where the training is')
    parser.add_argument('--inclusive_thresholds', type=str, help='Inclusive thresholds for MET or HT')
    parser.add_argument('--plot_dir', type=str, help='Plotting dir')
    args = parser.parse_args()
    args.inclusive_thresholds = [float(f) for f in args.inclusive_thresholds.replace(' ','').split(',')]
    if args.log_features!='':
        args.log_features = [str(f) for f in args.log_features.replace(' ','').split(',')]
    else :
        args.log_features=[]
    main_analyze_results(**vars(args))




