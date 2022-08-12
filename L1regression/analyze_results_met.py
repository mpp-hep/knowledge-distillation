import setGPU
import h5py
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import shutil 
import tensorflow as tf
import tensorflow.keras.backend as K
import keras 
import utils.data_processing as data_proc
import utils.plotting as plotting
from nn.models import GraphAttentionHyperModel
import nn.losses as nn_losses
import keras_tuner
from keras_tuner import HyperParameters
matplotlib.use("Agg") 

fixed_seed = 2021
tf.keras.utils.set_random_seed(fixed_seed)


def save_improvement_data(txt_outfile,bg_eff, sig_eff_reco, sig_eff_corr):
    with open(txt_outfile, "w") as f:
        for b_e,s_e,s_e_corr in zip(bg_eff,sig_eff_reco,sig_eff_corr):
            summary =f"@ BG eff {b_e:.5f} :  Sig Eff RECO {s_e:.5f}, Sig Eff CORR {s_e_corr:.5f}, improvement in %: {(s_e_corr/s_e-1.)*100.:.2f}"
            print(summary, file=f)
            print(summary)



def main_analyze_results(data_file='',signal_names='',bg_names='',variable='',log_features=[],training_dir='',loss_function='',inclusive_thresholds=[],plot_dir='',use_generator=''):
    """
    Plots resolutions, rates, efficiencies for a given training 
    Arguments:
        data_file: str, path to the input file
        signal_names: list of str, signals to consider
        bg_names: list of str, bg to consider
        loss_function: str or loss function object
        variable: str, variable on which to produce results : original_met, true_met, true_ht 
        log_features: list of str, which feature scale to be log
        plot_dir: str, plotting directory
        inclusive_thresholds: list of float, inclusive thresholds for MET
        training_dir: str, path to the training used 
        use_generator: bool, whether to use data generator or not
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

    ids_names_dict = data_proc.get_process_id_dict(ids_names)

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
    custom_objects = {loss_function.__name__:loss_function,
                    "MseThesholdMetric":nn_losses.MseThesholdMetric}
    model = keras.models.load_model(training_dir+'/best_model',custom_objects = custom_objects)

    if use_generator:
        print('Using generator')
        data_generator = data_proc.DataGenerator(graph_data.features, graph_data.adjacency,graph_conv_filters,graph_data.labels, batch_size=2048, shuffle=False)
        dnn_correction = model.predict(data_generator)
    else :
        dnn_correction = model.predict([graph_data.features, graph_data.adjacency,graph_conv_filters],batch_size=2048)

    if not os.path.exists(plot_dir):
        os.makedirs(plot_dir) 


    for proc in signal_names+bg_names:    
        proc_id = ids_names_dict[proc]
        proc_save_name = proc.replace(' ','').replace('+','')
        plot_subdir = plot_dir+'/'+proc_save_name
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)    
            
        proc_mask =  np.where(graph_data.process_ids==proc_id)
        if 'met' in variable:
            evaluator = data_proc.ResolutionEvaluator(graph_data.reco_met,graph_data.true_met,graph_data.process_ids,dnn_correction,mask=proc_mask)
        elif 'ht' in variable:
            evaluator = data_proc.ResolutionEvaluator(graph_data.reco_ht,graph_data.true_ht,graph_data.process_ids,dnn_correction,mask=proc_mask)

        plotting.plot_distibutions([graph_data.labels[:,0][proc_mask],dnn_correction[proc_mask]], ['Target','Prediction'],
                                    xtitle='Target vs Prediction',ytitle='Yield',title=proc,bins=100,semilogy=True,density=False,
                                    output_dir=plot_subdir,plot_name=f'plot_{var_name}_target_prediction_{proc_save_name}.pdf')

        plotting.plot_distibutions([evaluator.reco_data,evaluator.corr_data,evaluator.true_data],['Baseline','Corrected','True'],
                                xtitle=f'{var_name}',ytitle='Normalized Yield',bins=100,semilogy=True,density=True,
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


        #Eficiency curves
        for thresh in inclusive_thresholds:
            max_val = np.max(inclusive_thresholds)*3
            nbins=40
            plotting.plot_efficiency([evaluator.true_data,evaluator.true_data],[evaluator.reco_data,evaluator.corr_data], 
                                thresholds=[thresh,thresh],max_x=max_val,
                                labels=['Baseline','Corrected'],xtitle=r'$%s^{true}$'%(var_name),ytitle='Efficiency',title=proc,nbins=nbins,
                                output_dir=plot_subdir,plot_name=f'plot_inclusive_eff_{var_name}_{thresh}_{proc_save_name}.pdf')

        #Normalized Rate/Total eff plots
        plot_subdir = plot_dir+'/trigger_rates/'
        if not os.path.exists(plot_subdir):
            os.makedirs(plot_subdir)      
        max_val = np.max(inclusive_thresholds)*2
        eff,thresholds = plotting.makeRate(evaluator.reco_data,max_val)
        eff_corr,_ = plotting.makeRate(evaluator.corr_data,max_val)
        plotting.plot_scatter([thresholds,thresholds],[eff,eff_corr],
                                ['Baseline','Corrected'],xtitle=r'L1 %s threshold'%(var_name),ytitle='Efficiency above L1 threshold',title=proc,semilogy=True,
                                output_dir=plot_subdir,plot_name=f'plot_rates_{var_name}_{proc_save_name}.pdf')

    del evaluator
    
    #Computing and plotting ROCs
    bg_selection_str = ''
    for bg in bg_names:
        bg_selection_str+='(graph_data.process_ids==%f)|'%ids_names_dict[bg]
    bg_selection_str=bg_selection_str[:-1] # remove the last "|"
    proc_mask_bg = np.where(eval(bg_selection_str))
    if 'met' in variable:
        evaluator_bg = data_proc.ResolutionEvaluator(graph_data.reco_met,graph_data.true_met,graph_data.process_ids,dnn_correction,mask=proc_mask_bg)
    elif 'ht' in variable:
        evaluator_bg = data_proc.ResolutionEvaluator(graph_data.reco_ht,graph_data.true_ht,graph_data.process_ids,dnn_correction,mask=proc_mask_bg)


    for proc_sig in signal_names:
        proc_id_sig = ids_names_dict[proc_sig]
        proc_mask_sig =  np.where(graph_data.process_ids==proc_id_sig)

        if 'met' in variable:
            evaluator_sig = data_proc.ResolutionEvaluator(graph_data.reco_met,graph_data.true_met,graph_data.process_ids,dnn_correction,mask=proc_mask_sig)
        elif 'ht' in variable:
            evaluator_sig = data_proc.ResolutionEvaluator(graph_data.reco_ht,graph_data.true_ht,graph_data.process_ids,dnn_correction,mask=proc_mask_sig)


        bg_efficiencies = [0.95,0.5,0.2,0.1,0.08,0.05,0.03,0.01,0.005]
        bg_eff_thresholds = [0.1,0.01]
        bg_thresholds_reco = []
        bg_thresholds_corr = []
        sig_eff_reco = []
        sig_eff_corr = []
        max_val = np.max(inclusive_thresholds)*4
        for i,bg_eff in enumerate(bg_efficiencies):
            bg_thresholds_reco.append(np.quantile(evaluator_bg.reco_data,1.-bg_eff))
            bg_thresholds_corr.append(np.quantile(evaluator_bg.corr_data,1.-bg_eff))
            sig_eff_reco.append(np.sum(evaluator_sig.reco_data>bg_thresholds_reco[i])/evaluator_sig.reco_data.shape[0])
            sig_eff_corr.append(np.sum(evaluator_sig.corr_data>bg_thresholds_corr[i])/evaluator_sig.corr_data.shape[0])      

        bg_names_title = ','.join(bg_names)
        plotting.plot_trigger_roc([bg_efficiencies,bg_efficiencies],[sig_eff_reco,sig_eff_corr],
                            ['Baseline','Corrected'],xtitle='Background Efficiency',ytitle='Signal Efficiency',title=f'Signal - {proc_sig}, Background - {bg_names_title}',semilogy=True, semilogx=True,
                            output_dir=plot_subdir,plot_name=f'plot_ROC_{proc_sig}_{var_name}.pdf',thresholds=bg_eff_thresholds)

        txt_outfile = plot_subdir+f'/improvement_summary_{proc_sig}.txt'
        save_improvement_data(txt_outfile,bg_efficiencies, sig_eff_reco, sig_eff_corr)

            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str, help='Path to the input file ')
    parser.add_argument('--loss_function', type=str, help='Which loss function to use')
    parser.add_argument('--log_features', type=str, default='',help='Which features scale to be log')
    parser.add_argument('--variable', type=str, help='Variable to analyze : MET or HT ')
    parser.add_argument('--training_dir', type=str, help='Where the training is')
    parser.add_argument('--inclusive_thresholds', type=str, help='Inclusive thresholds for MET or HT')
    parser.add_argument('--signal_names', type=str, default='W,hChToTauNu,hToTauTau',help='Which signals to consider')
    parser.add_argument('--bg_names', type=str, default='QCD',help='Which bg to consider')
    parser.add_argument('--plot_dir', type=str, help='Plotting dir')
    parser.add_argument('--use_generator', type=int, default=0, help='True/False to use generator')
    args = parser.parse_args()
    args.loss_function = nn_losses.get_loss_func(args.loss_function)
    args.inclusive_thresholds = [float(f) for f in args.inclusive_thresholds.replace(' ','').split(',')]
    args.signal_names = [f for f in args.signal_names.replace(' ','').split(',')]
    args.bg_names = [f for f in args.bg_names.replace(' ','').split(',')]
    args.use_generator = bool(args.use_generator)
    if args.log_features!='':
        args.log_features = [str(f) for f in args.log_features.replace(' ','').split(',')]
    else :
        args.log_features=[]
    main_analyze_results(**vars(args))




