from scipy.stats import pearsonr
import pandas as pd
import numpy as np
import seaborn as sns

def sem(x):
    std = np.std(x)
    n = np.size(x)
    sem = std/np.sqrt(n)
    sem = pd.DataFrame({'sem':sem})
    return sem


def sem(x):
    """Calculates the standard error of the mean for a given Series."""
    std = np.std(x)
    n = np.size(x)
    if n > 0:
        sem_value = std / np.sqrt(n)
    else:
        sem_value = np.nan
    return sem_value


def RT_filter(x, sd): 
    # this function was used to filter those trials with a RT that are above X standard deviation from the mean
    rt_mean = x['RT'].mean()
    up_lim = rt_mean + sd * x['RT'].std()
    x = x.loc[(x['RT'] < up_lim) & (x['RT'] > 0), : ]
    return x #the same dataframe with RT filtered

def log_reg_fit(x, f): # this function was used to calculate the log linear regression between two vectors
    reg = smf.glm(formula = f, data = x, family=sm.families.Binomial()).fit()
    params = reg.params
    PSE = - reg.params.Intercept/reg.params[1]
    # concatenating parameters
    out = pd.DataFrame({'intercept':[params[0]] , 'weight':[params[1]],'PSE': PSE})
    return out #intercept + weight

def z_RT(x):
    # this function was used to calculate the z score of the RT values from a dataframe 
    rt_mean = x['RT'].mean()
    rt_sd   = x['RT'].std()
    x['RTz'] = (x['RT']- rt_mean)/rt_sd
    x['1/RTz']= 1/x['RT']
    rt_mean1  = x['1/RTz'].mean()
    rt_sd1    = x['1/RTz'].std()
    x['1/RTz']=(x['1/RTz']- rt_mean1)/rt_sd1
    return x

def z_Confi(x):
    # this function was used to calculate the z score of the DV values from a dataframe 
    dv_mean = x['confi'].mean()
    dv_sd   = x['confi'].std()
    x['confiz'] = (x['confi']- dv_mean)/dv_sd
    return x

def z_DV(x):
    # this function was used to calculate the z score of the DV values from a dataframe 
    dv_mean = x['DV'].mean()
    dv_sd   = x['DV'].std()
    x['DVz'] = (x['DV']- dv_mean)/dv_sd
    return x

def cartesian(arrays, out=None):
    """
    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    #m = n / arrays[0].size
    m = int(n / arrays[0].size) 
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m, 1:])
        for j in range(1, arrays[0].size):
        #for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m, 1:] = out[0:m, 1:]
    return out

def sdt(x):
    dprime = x.dprime()
    c = x.criterion()
    out = pd.DataFrame([[dprime, c]], columns=['dprime', 'c'])   
    return out

# This function calculates the minimum difference in circular space between two angles
def circ_angles_diff(x,ref):
    x = np.rad2deg(x)
    ref = np.rad2deg(ref)
    diffs = x - ref
    diffs = (diffs + 180) % 360 - 180
    return diffs
#circ_angles_diff(np.array([5, 20, 30]),200)    

# function that I used to plot multiple data
def barsplot(data, dx, dy, hue, col, row, pal, size, yaxis, axislabels, sizepoint, dodge):    
    sns.set(font_scale = 1.5, style = 'ticks')         
    ort = "v"; pal = pal; sigma = .5
    g = sns.FacetGrid(data ,  row = row, col = col, height= size['height'], aspect=size['aspect'], margin_titles=True) # col="nrep",    
    if sizepoint == None:
        sizepoint = 6
    if yaxis != None:
        g.set(yaxis['ylim'], yaxis['yticks'])   

    g.map_dataframe(sns.stripplot, x = dx, y = dy, palette = pal, hue=hue, size = sizepoint, edgecolor = "white",
                    linewidth = 0.6, jitter = 0.2, orient = ort,alpha = 0.5, dodge=dodge)
    g.map_dataframe(sns.barplot, x = dx, y = dy, palette = pal, hue=hue,  linewidth = 0.6, orient = ort, dodge=dodge)
    
    #g.map_dataframe(sns.violinplot, x = dx, y = dy,  palette = pal,bw = .5, cut = 0.,scale = "area", width = .6, inner = None, orient = ort, linewidth = 0, zorder = 2)
    
    g.add_legend()

    sns.despine(offset = .5,  trim=True);
    # Set x-axis and y-axis labels
    g.set_axis_labels( axislabels['xlabel'] , axislabels['ylabel'], fontsize = 15 )
    #g.tight_layout()
    return g

def organize_data_betas(x, nreps, repeat):

    # non repeat
    # bias prevC rep 0
    estim = x.loc['(Intercept)'][['Estimate','2.5_ci','97.5_ci']]
    estim = pd.DataFrame(np.array(estim).reshape(1,3), columns = ['Estimate','2.5_ci','97.5_ci']); estim['nrep'] = 0; estim['parameter'] = 'bias'; estim['prev_resp'] = 'C'; estim['repeat'] = 'nonrepeat'
    b0c =  np.array(estim['Estimate'])
    
    # bias prevD rep 0
    aux = np.array(x.loc['response1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0c+ aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D';aux['repeat'] = 'nonrepeat'; 
    b0d =  np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)


    # weight prevC rep 0
    aux = x.loc['rDV'][['Estimate','2.5_ci','97.5_ci']] 
    aux = pd.DataFrame(np.array(aux).reshape(1,3), columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C';aux['repeat'] = 'nonrepeat'
    w0c = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    # weight prevD rep 0
    aux = np.array(x.loc['rDV:response1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0c + aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D';aux['repeat'] = 'nonrepeat'; 
    w0d = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    ## repeated
    if repeat:
        # bias prevC rep 0
        aux = x.loc['trial_typerepeat'][['Estimate','2.5_ci','97.5_ci']]; aux[0] = np.array(b0c+ aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
        aux = pd.DataFrame(np.array(aux).reshape(1,3), columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'bias'; aux['repeat'] = 'repeat'; aux['prev_resp'] = 'C'
        b0cr =  np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)
        
        # bias prevD rep 0
        aux = np.array(x.loc['response1:trial_typerepeat'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0cr+ aux[0])[0]; aux[1:3] = b0cr + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D';aux['repeat'] = 'repeat'
        b0dr =  np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevC rep 0
        aux = x.loc['rDV:trial_typerepeat'][['Estimate','2.5_ci','97.5_ci']]; aux[0] = np.array(w0c+ aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
        aux = pd.DataFrame(np.array(aux).reshape(1,3), columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C';aux['repeat'] = 'repeat'
        w0cr = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        # weight prevD rep 0
        aux = np.array(x.loc['rDV:response1:trial_typerepeat'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0cr + aux[0])[0]; aux[1:3] = w0cr + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D';aux['repeat'] = 'repeat'
        w0dr = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

    if nreps > 0:

        # Non repeat rep 1

        # bias prevC rep 1
        aux = np.array(x.loc['nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0c + aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'bias'; aux['prev_resp'] = 'C';aux['repeat'] = 'nonrepeat';
        b1c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        # bias prevD rep 1
        aux = np.array(x.loc['response1:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b1c + aux[0])[0]; aux[1:3] = b1c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D';aux['repeat'] = 'nonrepeat'
        b1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevC rep 1

        aux = np.array(x.loc['rDV:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0c + aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C';aux['repeat'] = 'nonrepeat'
        w1c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevD rep 1
        aux = np.array(x.loc['rDV:response1:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w1c  + aux[0])[0]; aux[1:3] = w1c  + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D';aux['repeat'] = 'nonrepeat'
        w1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        if repeat:
            ## Repeat

            # bias prevC nonrep 1
            aux = np.array(x.loc['trial_typerepeat:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b1c + aux[0])[0]; aux[1:3] = b1c + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'bias'; aux['prev_resp'] = 'C';aux['repeat'] = 'repeat'
            b1cr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)



            # bias prevD nonrep 1
            aux = np.array(x.loc['response1:trial_typerepeat:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b1d + aux[0])[0]; aux[1:3] = b1d + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D';aux['repeat'] = 'repeat'
            b1dr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)


            # weight prevC nonrep 1

            aux = np.array(x.loc['rDV:trial_typerepeat:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w1c + aux[0])[0]; aux[1:3] = w1c + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C';aux['repeat'] = 'repeat'
            w1cr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)


            # weight prevD nonrep 1
            aux = np.array(x.loc['dv:response1:trial_typerepeat:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w1d  + aux[0])[0]; aux[1:3] = w1d  + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D';aux['repeat'] = 'repeat'
            w1dr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)

    if nreps > 1:

        # non repeat
        # bias prevC rep 2
        aux = np.array(x.loc['nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0c + aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'bias'; aux['prev_resp'] = 'C';aux['repeat'] = 'nonrepeat'
        b2c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)



        # bias prevD rep 2
        aux = np.array(x.loc['response1:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b2c + aux[0])[0]; aux[1:3] = b2c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D';aux['repeat'] = 'nonrepeat'
        b1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevC rep 2

        aux = np.array(x.loc['rDV:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0c + aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C';aux['repeat'] = 'nonrepeat'
        w2c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevD rep 2
        aux = np.array(x.loc['rDV:response1:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w2c  + aux[0])[0]; aux[1:3] = w2c  + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D';aux['repeat'] = 'nonrepeat'
        w1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        if repeat:
            # repeat

            # bias prevC nonrep 2
            aux = np.array(x.loc['trial_typerepeat:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b2c + aux[0])[0]; aux[1:3] = b2c + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'bias'; aux['prev_resp'] = 'C';aux['repeat'] = 'repeat'
            b2cr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)



            # bias prevD nonrep 2
            aux = np.array(x.loc['response1:trial_typerepeat:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b2d + aux[0])[0]; aux[1:3] = b2d + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D';aux['repeat'] = 'repeat'
            b2dr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)


            # weight prevC nonrep 2

            aux = np.array(x.loc['rDV:trial_typerepeat:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w2c + aux[0])[0]; aux[1:3] = w2c + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C';aux['repeat'] = 'repeat'
            w2cr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)


            # weight prevD nonrep 2
            aux = np.array(x.loc['rDV:response1:trial_typerepeat:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w2d  + aux[0])[0]; aux[1:3] = w2d  + aux[1:3]; 
            aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D';aux['repeat'] = 'repeat'
            w2dr = np.array(aux['Estimate'])
            estim = pd.concat([estim,aux], ignore_index=True)



    estim['prev_resp'] = estim['prev_resp'].astype('str')
    estim['nrep'] = estim['nrep'].astype('str')
    return estim 


def organize_data_param(x, nreps, var):
    # bias prevC rep 0
    estim = x.loc['(Intercept)'][['Estimate','2.5_ci','97.5_ci']]
    estim = pd.DataFrame(np.array(estim).reshape(1,3), columns = ['Estimate','2.5_ci','97.5_ci']); estim['nrep'] = 0; estim['parameter'] = 'bias';estim['prev_resp'] = 'C'
    b0c =  np.array(estim['Estimate'])
    
    # bias prevD rep 0
    aux = np.array(x.loc['response1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0c+ aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D'
    b0d =  np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)


    # weight prevC rep 0
    aux = x.loc['rDV'][['Estimate','2.5_ci','97.5_ci']] 
    aux = pd.DataFrame(np.array(aux).reshape(1,3), columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C'
    w0c = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    # weight prevD rep 0
    aux = np.array(x.loc['rDV:response1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0c + aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D'
    w0d = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)


    # param effects interactions

    # with intercept
    aux = np.array(x.loc[var][['Estimate','2.5_ci','97.5_ci']]);
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = var; aux['prev_resp'] = 'nan'
    cf0 = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    # with dv
    aux = np.array(x.loc['rDV:'+var][['Estimate','2.5_ci','97.5_ci']]);
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = var + 'Xdv'; aux['prev_resp'] = 'nan'
    cfdv0 = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    # with prev response
    aux = np.array(x.loc['response1:'+var][['Estimate','2.5_ci','97.5_ci']]);
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = var + 'Xprev_resp'; aux['prev_resp'] = 'nan'
    cfresp0 = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    # with dv and prev response

    aux = np.array(x.loc['rDV:response1:'+var][['Estimate','2.5_ci','97.5_ci']]);
    aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 0; aux['parameter'] = var + 'XdvXprev_resp'; aux['prev_resp'] = 'nan'
    cfdvresp0 = np.array(aux['Estimate'])
    estim = pd.concat([estim,aux], ignore_index=True)

    if nreps > 0:

        # bias prevC rep 1
        aux = np.array(x.loc['nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0c + aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'bias'; aux['prev_resp'] = 'C'
        b1c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)



        # bias prevD rep 1
        aux = np.array(x.loc['response1:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b1c + aux[0])[0]; aux[1:3] = b1c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D'
        b1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevC rep 1

        aux = np.array(x.loc['rDV:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0c + aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C'
        w1c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevD rep 1
        aux = np.array(x.loc['rDV:response1:nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w1c  + aux[0])[0]; aux[1:3] = w1c  + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D'
        w1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        # param rep 1
        aux = np.array(x.loc[var +':nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(cf0 + aux[0])[0]; aux[1:3] = cf0 + aux[1:3];
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = var; aux['prev_resp'] = 'nan'
        cf1 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        aux = np.array(x.loc['rDV:' + var +':nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(cfdv0 + aux[0])[0]; aux[1:3] = cfdv0 + aux[1:3];
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = var + 'Xdv'; aux['prev_resp'] = 'nan'
        cfdv1 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        aux = np.array(x.loc['response1:' + var + ':nrep1'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(cfresp0 + aux[0])[0]; aux[1:3] = cfresp0 + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = var + 'Xprev_resp'; aux['prev_resp'] = 'nan'
        cfresp1 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        aux = np.array(x.loc['rDV:response1:'+ var+ ':nrep1'][['Estimate','2.5_ci','97.5_ci']]);  aux[0] = np.array(cfdvresp0 + aux[0])[0]; aux[1:3] = cfdvresp0 + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 1; aux['parameter'] = var + 'XdvXprev_resp'; aux['prev_resp'] = 'nan'
        cfdvresp1 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

    if nreps > 1:

        # bias prevC rep 2
        aux = np.array(x.loc['nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b0c + aux[0])[0]; aux[1:3] = b0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'bias'; aux['prev_resp'] = 'C'
        b2c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # bias prevD rep 2
        aux = np.array(x.loc['response1:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(b2c + aux[0])[0]; aux[1:3] = b2c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'bias'; aux['prev_resp'] = 'D'
        b1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        # weight prevC rep 2

        aux = np.array(x.loc['rDV:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w0c + aux[0])[0]; aux[1:3] = w0c + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'weight'; aux['prev_resp'] = 'C'
        w2c = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


        # weight prevD rep 2
        aux = np.array(x.loc['rDV:response1:nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(w2c  + aux[0])[0]; aux[1:3] = w2c  + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = 'weight'; aux['prev_resp'] = 'D'
        w1d = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        # param rep 2
        aux = np.array(x.loc[var +':nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(cf1 + aux[0])[0]; aux[1:3] = cf1 + aux[1:3];
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = var; aux['prev_resp'] = 'nan'
        cf2 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        aux = np.array(x.loc['rDV:' + var +':nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(cfdv1 + aux[0])[0]; aux[1:3] = cfdv1 + aux[1:3];
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = var + 'Xdv'; aux['prev_resp'] = 'nan'
        cfdv2 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        aux = np.array(x.loc['response1:' + var + ':nrep2'][['Estimate','2.5_ci','97.5_ci']]); aux[0] = np.array(cfresp1 + aux[0])[0]; aux[1:3] = cfresp1 + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = var + 'Xprev_resp'; aux['prev_resp'] = 'nan'
        cfresp2 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)

        aux = np.array(x.loc['rDV:response1:'+ var+ ':nrep2'][['Estimate','2.5_ci','97.5_ci']]);  aux[0] = np.array(cfdvresp1 + aux[0])[0]; aux[1:3] = cfdvresp1 + aux[1:3]; 
        aux = pd.DataFrame(aux.reshape(1,3),columns = ['Estimate','2.5_ci','97.5_ci']); aux['nrep'] = 2; aux['parameter'] = var + 'XdvXprev_resp'; aux['prev_resp'] = 'nan'
        cfdvresp2 = np.array(aux['Estimate'])
        estim = pd.concat([estim,aux], ignore_index=True)


    estim['prev_resp'] = estim['prev_resp'].astype('str')
    estim['nrep'] = estim['nrep'].astype('str')
    return estim 




def perm_test_corr(x, y, n_perm=1000, alpha=0.05):
    """
    Perform a permutation test to assess the significance of the correlation between two variables.
    
    Args:
        x (array-like): First variable.
        y (array-like): Second variable.
        n_perm (int): Number of permutations (default: 1000).
        alpha (float): Significance level (default: 0.05).
    
    Returns:
        tuple: (correlation coefficient, p-value, 95% confidence interval, is_significant)
    """
    # Compute the observed correlation coefficient
    obs_corr, _ = pearsonr(x, y)
    
    # Permute one of the variables and compute correlation coefficients
    corr_perm = []
    for i in range(n_perm):
        perm_y = np.random.permutation(y)
        corr, _ = pearsonr(x, perm_y)
        corr_perm.append(corr)
    
    # Compute the p-value and confidence interval
    corr_perm = np.array(corr_perm)
    p_value = np.sum(np.abs(corr_perm) >= np.abs(obs_corr)) / n_perm
    ci_lower = np.percentile(corr_perm, 2.5)
    ci_upper = np.percentile(corr_perm, 97.5)
    
    # Determine if the correlation is significant
    if p_value < alpha:
        is_significant = True
    else:
        is_significant = False
    
    return obs_corr, p_value, (ci_lower, ci_upper), is_significant