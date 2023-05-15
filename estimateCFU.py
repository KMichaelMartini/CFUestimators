import numpy as np
import gradio as gr
import os
import pandas as pd

import scipy.special as sc
from scipy.stats import poisson
import scipy.integrate as integrate
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize


### Find the maximum likelihood estimator for CFUs (MPN method)
### samples - a numpy array of colony counts
### dil - a numpy array of dilutions
### V - the Volume
### N - Max number of Colonies
def findMLE(samples, dil, V, N):
    f = lambda x: np.sum(samples*dil/(N*(1-np.exp(-x*dil*V/N))))-np.sum(dil)
    if any(N<samples):
        return [np.inf, np.inf]
    sol = optimize.root_scalar(f, bracket=[0, 100000000], method='brentq')
    r_mle=sol.root
    p0=np.exp(-r_mle*dil*V/N)
    invVar=np.sum(dil*dil*V*V*samples*p0/(N*N*(1-p0)**2))
    estVar=1/invVar
    r_mle_std=np.sqrt(estVar)
    return [r_mle,r_mle_std]


### Find the Poisson estimator for CFUs
### samples - a numpy array of colony counts
### dil - a numpy array of dilutions
### V - the Volume
def findNaivePoisson(samples, dil, V):
    r_p=np.sum(samples)/(np.sum(dil)*V)
    r_p_std=np.sqrt(r_p*r_p/np.sum(samples))
    return [r_p, r_p_std]


### Find the Poisson estimator with a cutoff
### samples - a numpy array of colony counts
### dil - a numpy array of dilutions
### V - the Volume
### N - Cutoff above which there are crowding effects
def findPoissonCutoff(samples, dil, V, N):
    mask=samples<N
    r_p=np.sum(samples[mask])/(np.sum(dil[mask])*V)
    r_p_std=np.sqrt(r_p*r_p/np.sum(samples[mask]))
    return [r_p, r_p_std]


### common gui string function
def printCommon(samples,dil,M,N):
    tmp=findMLE(samples=samples,dil=dil,V=1.0,N=N)
    outputText=f"MPN MLE estimator: {tmp[0]:.4f}±{tmp[1]:.4f}\n"
    tmp=findNaivePoisson(samples=samples,dil=dil,V=1.0)
    outputText+=f"Naive Poisson: {tmp[0]:.4f}±{tmp[1]:.4f}\n"
    tmp=findPoissonCutoff(samples=samples,dil=dil,V=1.0,N=M)
    outputText+=f"Poisson with Cutoff: {tmp[0]:.4f}±{tmp[1]:.4f}\n"
    return outputText


### gui function for the enter data tab
def printAll(df, N, M):
    df = df.drop(df.index[df.isnull().all(axis = 1)]).reset_index(drop = True)
    df = df.replace(r'^\s*$', np.NaN, regex=True)
    
    samples=df["Count"].to_numpy().astype(float)
    dil=df["Dilution"].to_numpy().astype(float)

    mask= ~(np.isnan(samples) | np.isnan(dil))
    samples=samples[mask]
    dil=dil[mask]

    return printCommon(samples=samples, dil=dil, M=M, N=N)


### gui function for the paste data tab
def printAll2(data1,data2,N, M):
    samples=np.array(data1.split(','),float)
    dil=np.array(data2.split(','),float)
    
    mask= ~(np.isnan(samples) | np.isnan(dil))
    samples=samples[mask]
    dil=dil[mask]

    return printCommon(samples=samples, dil=dil, M=M, N=N)
    


### gui function for the upload file tab
def printAll3(fileName, N, M):

    # find the file extension
    ext = os.path.splitext(fileName.name)[1]

    if ext == '.psv':
        df = pd.read_psv(fileName.name,header=None)
    elif ext == '.csv':
        df = pd.read_csv(fileName.name,header=None)
    elif ext == '.xls':
        df = pd.read_excel(fileName.name,header=None)
    elif ext == '.xlsx':
        df = pd.read_excel(fileName.name,header=None)
    else:
        raise RuntimeError('File extension not recognized')

    arr=df.to_numpy()
    samples=arr[:,1]
    dil=arr[:,0]
    
    mask= ~(np.isnan(samples) | np.isnan(dil))
    samples=samples[mask]
    dil=dil[mask]
    
    return printCommon(samples=samples, dil=dil, M=M, N=N)


### the gui
with gr.Blocks(title="CFU estimator") as demo:
    gr.Markdown(    
"""
# Calculate Colony Forming Units (CFUs) Demo
This demo calculates several different estimators for CFUs. Dilutions should be entered as fractional volumes of the original sample. When entered this way the CFU will be measured in terms of the original volume and its units. For example, if a sample is originally 0.2ml then the resulting CFU will be measured be per 0.2ml. A ten-fold dilution corresponds to a volume fraction 0.1, a hundred-fold dilution corresponds to a volume fraction of 0.01. In the above example 0.1 and 0.01 should be entered in the dilution column.  

There are several ways to enter data in this demo. You may enter data directly in Enter Data tab, paste data in the paste data tab, or upload a csv file or excel file containing dilutions and counts. Uploaded files should be in column format with the first column corresponding to the dilution (volume fraction) and the second column corresponding to the colony counts.

To use the MPN estimator a max number of colonies should be entered in the first text box. A good starting estimate is the ratio of the plated area to the average colony size. To use the Poisson estimator with a cutoff, enter a cutoff number, M, into the second textbox.
"""
    )
    max_N = gr.Number(value=5000,label="Estimated Max Colony Number (N)",info="A starting estimate is the ratio of the area of the plate to the average colony size. Used for calculating MPN")
    max_M = gr.Number(value=300,label="Cutoff (M)",info="The cutoff number used in the Poisson with a cutoff. Should be a number less than when crowding starts to be important.")
    with gr.Tab("Enter Data"):
        with gr.Row():
            data_input = gr.Dataframe(headers=["Dilution", "Count"], datatype=["number","number"], row_count=1, col_count=(2,"fixed"))
            data_output = gr.Textbox(label="Calculated CFU")
        image_button = gr.Button("Calculate")
    with gr.Tab("Paste Data"):
        with gr.Row():
            with gr.Column():
                data_dil = gr.Textbox(label="Dilutions",info="A comma seperated list of dilutions")
                data_count = gr.Textbox(label="Counts",info="A comma seperated list of colony counts")
            data_output2 = gr.Textbox(label="Calculated CFU")
        paste_button = gr.Button("Calculate")
    with gr.Tab("Upload file"):
        with gr.Row():
            text_input = gr.File(file_count="single", file_types=["text", ".csv", ".xlsx", ".psv"])
            text_output = gr.Textbox(label="Calculated CFU")
        text_button = gr.Button("Calculate")

    text_button.click(printAll3, inputs=[text_input,max_N,max_M], outputs=text_output)
    image_button.click(printAll, inputs=[data_input,max_N,max_M], outputs=data_output)
    paste_button.click(printAll2, inputs=[data_count,data_dil,max_N,max_M], outputs=data_output2)

demo.launch()