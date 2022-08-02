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





def main_analyze_results(data_file='',training_dir='',inclusive_thresholds=[],plot_dir=''):
    """
    Plots resolutions, rates, efficiencies for a given training 
    Arguments:
        data_file: str, path to the input file 
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
        true_ht = np.array(open_file['true_ht'])
        ids = np.array(open_file['ids'])
        ids_names = np.array(open_file['ids_names'])

    graph_data = data_proc.GraphCreator(reco_data,reco_met,reco_ht, true_met,true_ht,ids,log_features=['pt'])
    num_filters = 1
    graph_conv_filters = graph_data.adjacency
    graph_conv_filters = K.constant(graph_conv_filters)
    #TO DO :loss and metrics should not be part of the model, fix
    loss_function = nn_losses.get_loss_func('huber_1.0')
    metric_thresholds = [0,50,100]
    metrics = [nn_losses.MseThesholdMetric(threshold=t) for t in metric_thresholds]
    hp = keras_tuner.HyperParameters()
    hypermodel = GraphAttentionHyperModel(features_input_shape=(graph_data.features.shape[1],graph_data.features.shape[2]), 
                                        adjancency_input_shape=(graph_data.adjacency.shape[1],graph_data.adjacency.shape[2]),
                                        filters_input_shape=(graph_conv_filters.shape[1],graph_conv_filters.shape[2]), 
                                        num_filters=num_filters, 
                                        loss_function=loss_function,
                                        metrics=metrics)

    training_dir = "/eos/user/n/nchernya/MLHEP/AnomalyDetection/knowledge-distillation/L1regression/output2/"
    tuner = keras_tuner.Hyperband(hypermodel = hypermodel,
                     objective = keras_tuner.Objective("val_loss", direction="min"),
                    executions_per_trial=1,
                    max_epochs=50,
                    overwrite=False,
                     directory=training_dir,
                     project_name='hyperband_tuner')
    model = tuner.get_best_models(num_models=1)[0]
    dnn_correction = model.predict([graph_data.features, graph_data.adjacency,graph_conv_filters],batch_size=1024)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir) 

    for proc,proc_id in zip(['W + jets','QCD'],[0,1]):    
        proc_save_name = proc.replace(' ','').replace('+','')
        plot_subdir = plot_dir+'/'+proc_save_name
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)    
            
        proc_mask =  np.where(ids==proc_id)
        evaluator = data_proc.MetResolutionEvaluator(graph_data.reco_met,graph_data.true_met,graph_data.process_ids,dnn_correction,mask=proc_mask)

        plotting.plot_distibutions([graph_data.labels[:,0][proc_mask],dnn_correction[proc_mask]], ['Target','Prediction'],
                                    xtitle='Target vs Prediction',ytitle='Normalized Yield',title=proc,nbins=100,semilogy=True,
                                    output_dir=plot_subdir,plot_name=f'plot_target_prediction_{proc_save_name}.pdf')

        plotting.plot_distibutions([evaluator.reco_met,evaluator.corr_met,evaluator.true_met],['Baseline','Corrected','True'],
                                xtitle='MET',ytitle='Normalized Yield',nbins=100,semilogy=True,
                                output_dir=plot_subdir,plot_name=f'plot_met_{proc_save_name}.pdf')

        plotting.plot_met_ratios([evaluator.met_rot,evaluator.met_rot_corr],['Baseline','Corrected'] ,
                                xtitle=r'$\frac{MET^{reco}}{MET^{true}}$',ytitle='Normalized Yield',title=proc,semilogy=False,
                                    output_dir=plot_subdir,plot_name=f'plot_reco_o_over_{proc_save_name}.pdf')

        plotting.plot_met_ratios([evaluator.met_tor,evaluator.met_tor_corr],['Baseline','Corrected'] ,
                                xtitle=r'$\frac{MET^{true}}{MET^{reco}}$',ytitle='Normalized Yield',title=proc,semilogy=False,
                                    output_dir=plot_subdir,plot_name=f'plot_true_o_reco_{proc_save_name}.pdf')

        #Resolution plots
        for what_to_plot in ['quant_diff/mpv','quant_diff']:
            what_to_plot_name = what_to_plot.replace('/','_')
            plotting.plot_resolutions([evaluator.true_met,evaluator.true_met],[evaluator.met_tor,evaluator.met_tor_corr],
                                ['Baseline','Corrected'],xtitle=r'$MET^{true}$',ytitle=r'$\frac{MET^{true}}{MET^{reco}}$, '+what_to_plot,title=proc,
                                what_to_plot=what_to_plot,do_fit=False,
                                output_dir=plot_subdir,plot_name=f'plot_{what_to_plot_name}_true_o_reco_{proc_save_name}.pdf')

            plotting.plot_resolutions([evaluator.true_met,evaluator.true_met],[evaluator.met_rot,evaluator.met_rot_corr],
                                ['Baseline','Corrected'],xtitle=r'$MET^{true}$',ytitle=r'$\frac{MET^{reco}}{MET^{true}}$, '+what_to_plot,title=proc,
                                what_to_plot=what_to_plot,do_fit=False,
                                output_dir=plot_subdir,plot_name=f'plot_{what_to_plot_name}_reco_o_true_{proc_save_name}.pdf')


        #Inclusive efficiency plots
        for thresh in inclusive_thresholds:
            max_met = 420
            nbins=40
            plotting.plot_efficiency([evaluator.true_met,evaluator.true_met],[evaluator.reco_met,evaluator.corr_met], 
                                thresholds=[thresh,thresh],max_x=max_met,
                                labels=['Baseline','Corrected'],xtitle=r'$MET^{true}$',ytitle='Normalized Yield',title=proc,nbins=nbins,
                                output_dir=plot_subdir,plot_name=f'plot_inclusive_eff_met_{thresh}_{proc_save_name}.pdf')

        #Rate plots
        max_met = 600
        eff,thresholds = plotting.makeRate(evaluator.reco_met,max_met)
        eff_corr,_ = plotting.makeRate(evaluator.corr_met,max_met)
        plotting.plot_scatter([thresholds,thresholds],[eff,eff_corr],
                                ['Baseline','Corrected'],xtitle=r'$MET^{true}$',ytitle='Normalized Yield',title=proc,
                                output_dir=plot_subdir,plot_name=f'plot_rates_{proc_save_name}.pdf')

        #Efficiency plots for the same rates
        for reco_thresh in inclusive_thresholds:
            max_met = 600
            rate = plotting.get_rate_from_threshold(evaluator.reco_met,max_met,reco_thresh)
            corr_thresh = plotting.get_threshold_from_rate(evaluator.corr_met,max_met,rate)
            max_met_plot = 420
            nbins=40
            plotting.plot_efficiency([evaluator.true_met,evaluator.true_met],[evaluator.reco_met,evaluator.corr_met], 
                                thresholds=[reco_thresh,corr_thresh],max_x = max_met_plot,
                                labels=[f'Baseline, MET>{reco_thresh:.1f} GeV',f'Corrected, MET>{corr_thresh:.1f} GeV'],
                                xtitle=r'$MET^{true}$',ytitle='Normalized Yield',title=proc+f', Rate={rate:.3f}',nbins=nbins,
                                output_dir=plot_subdir,plot_name=f'plot_eff_met_{reco_thresh}_const_rate_{proc_save_name}.pdf')
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Path to the input file ')
    parser.add_argument('--training_dir', type=str, help='Where the training is')
    parser.add_argument('--inclusive_thresholds', type=str, help='Inclusive thresholds for MET')
    parser.add_argument('--plot_dir', type=str, help='Plotting dir')
    args = parser.parse_args()
    args.inclusive_thresholds = [float(f) for f in args.inclusive_thresholds.replace(' ','').split(',')]
    main_analyze_results(**vars(args))




