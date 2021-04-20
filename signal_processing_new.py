
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.integrate import simps
import time
import requests


tlow, thigh = 4, 8

def compute_periodogram(eegdata):
  #Find the power spectral density of the data
  p_freqs, p_psd = signal.periodogram(eegdata, srate)
  #Frequency resolution (bin size)
  p_freq_res = p_freqs[1] - p_freqs[0]
  #Find intersecting values in frequency vector
  p_idx_theta = np.logical_and(p_freqs >= tlow, p_freqs <= thigh)
  
  # Compute the absolute power by approximating the area under the curve
  p_theta_power = simps(p_psd[p_idx_theta], dx=p_freq_res)

    # Relative power (expressed as a percentage of total power)
  p_total_power = simps(p_psd, dx=p_freq_res)

  p_theta_rel_power = p_theta_power / p_total_power
  
  return(pd.DataFrame([p_theta_rel_power]))


#find theta powers using Welch's method
def compute_Welch(data):
  #Length of Fourier Transform
  winlength = 1 * srate 
  #noverlap is number of points of overlap
  nOverlap = np.round(srate/2)
  #Calculated power spectral density
  w_freqs, w_psd = signal.welch(data, srate, nperseg=winlength, noverlap=nOverlap)
  #if not specified, noverlap = 50%

  # Find intersecting values in frequency vector
  w_idx_theta = np.logical_and(w_freqs >= tlow, w_freqs <= thigh)

  #Frequency resolution
  w_freq_res = w_freqs[1] - w_freqs[0]

  # Compute the absolute power by approximating the area under the curve
  w_theta_power = simps(w_psd[w_idx_theta], dx=w_freq_res)

  # Relative delta power (expressed as a percentage of total power)
  w_total_power = simps(w_psd, dx=w_freq_res)

  w_theta_rel_power = w_theta_power / w_total_power

  return(pd.DataFrame([w_theta_rel_power]))


#obtain data
data = pd.read_csv('/Users/willstonehouse/Desktop/CruXProject/baseline.txt', sep = ',',header = 4, usecols = [" EXG Channel 0", " EXG Channel 1"])
data = data.dropna()
srate = 200

#Calculating Welch's method over first 1 minute to get baseline theta values
baseline_eeg_right = data.iloc[0:srate*60," EXG Channel 0"]
baseline_eeg_left = data.iloc[0:srate*60," EXG Channel 1"]

w_rel_power_right = pd.DataFrame.transpose(compute_Welch(baseline_eeg_right))
w_rel_power_left = pd.DataFrame.transpose(compute_Welch(baseline_eeg_left))
  
w_rel_power_right.columns = ["Theta_Rel"]
w_rel_power_left.columns = ["Theta_Rel"]

powers_eeg_right= pd.DataFrame()
powers_eeg_left= pd.DataFrame() 

'''
#right = data[' EXG Channel 0']
#left = data[' EXG Channel 1']
[left, right] = compute_Welch(left, right)
'''
#iterate such that 50% overlap, so start at 200 data points after the previous starting point (i.e. 1 second after)
#look at 2nd minute's worth of data, starting at 60 seconds and 50% overlap till last interval
#we could find length of time in seconds: len(data) / srate -> number of iterations for the for loop  (seconds - 1)

for starting_sec in range(60,[len(data)-1]) :
  start = starting_sec * srate #start of each 2 seconds worth of data
  end = (starting_sec * srate) + (srate * 2) #end, srate*2 = 2 seconds worth of data
  
  #in case we don't get enough data points for the last iteration
  if(end > len(data)):
    binaural_eeg_right = data.iloc[start:len(data), " EXG Channel 0"]
    binaural_eeg_left = data.iloc[start:len(data), " EXG Channel 1"]
  else:
    binaural_eeg_right = data.iloc[start:end, " EXG Channel 0"]
    binaural_eeg_left = data.iloc[start:end, " EXG Channel 1"]
  
  #get the powers from periodogram
  binaural_right_powers = pd.DataFrame.transpose(compute_periodogram(binaural_eeg_right))
  binaural_left_powers = pd.DataFrame.transpose(compute_periodogram(binaural_eeg_left))
  
  #add the values to dataframe for future calculations/reference
  powers_eeg_right = pd.concat([powers_eeg_right,binaural_right_powers], ignore_index = True)
  powers_eeg_left = pd.concat([powers_eeg_left,binaural_left_powers], ignore_index = True)

#relative theta powers separated by electrode
powers_eeg_right.columns = ["Theta_Rel"]
powers_eeg_left.columns = ["Theta_Rel"]

#combined relative theta powers
theta = (powers_eeg_right.iloc[:, 0].values + powers_eeg_left.iloc[:, 0].values)/2

right_theta = w_rel_power_right.iat[0,0]
left_theta = w_rel_power_left.iat[0,0]

#average baseline theta powers
theta_total = (right_theta + left_theta)/2

baselined_theta = pd.DataFrame(((theta - theta_total)/theta_total)*100)

#To simulate streaming, we update our database periodically with the values
#given in the sequences
request_url = "http://localhost:8010/proxy/test/"
 
theta_percent_change = baselined_theta.values[:, 0]

for i in range(len(theta_percent_change)): 
    requests.put(request_url + "theta", json={'value': round(theta_percent_change[i], 3)})
    time.sleep(2)

