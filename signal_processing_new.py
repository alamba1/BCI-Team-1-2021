
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
  global p_freqs
  global p_psd
  global p_freq_res
  global p_theta_power
  global p_total_power
  global p_theta_rel_power

  p_freqs, p_psd = signal.periodogram(eegdata, srate)
  if p_freqs.size > 0:
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


srate = 200
filename = '/Users/Ashley/Desktop/CruX/OpenBCI-RAW-2021-05-10_19-10-40.txt'

data = np.loadtxt(filename, delimiter=',', skiprows = 6, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8))

index = data[:, 0]
exg_0 = data[:, 1]
exg_1 = data[:, 2]
exg_2 = data[:, 3]
exg_3 = data[:, 4]
accel_0 = data[:, 5]
accel_1 = data[:, 6]
accel_2 = data[:, 7]

print(data[0])

#plot graph
colOfInterest = exg_0

fs = 200
colOfInterest = colOfInterest - np.mean(colOfInterest)
col_fft = np.abs(np.fft.fft(colOfInterest))
col_freq = np.fft.fftfreq(len(col_fft), 1/fs)
plt.plot(col_freq, col_fft)
plt.xlim([5, 10])
plt.ylim(0, 16000)

#Calculating Welch's method over first 1 minute to get baseline theta values
baseline_eeg_midline = data[0:srate*30, 1]
print("Baseline EEG Midline")
print(baseline_eeg_midline)

w_rel_power_midline = pd.DataFrame.transpose(compute_Welch(baseline_eeg_midline))
  
w_rel_power_midline.columns = ["Theta_Rel"]

powers_eeg_midline= pd.DataFrame()


#iterate such that 50% overlap, so start at 200 data points after the previous starting point (i.e. 1 second after)
#look at 2nd minute's worth of data, starting at 60 seconds and 50% overlap till last interval
#we could find length of time in seconds: len(data) / srate -> number of iterations for the for loop  (seconds - 1)
lengthData = len(data)//srate
myList = range(40, lengthData)

for starting_sec in myList:
  start = starting_sec * srate #start of each 2 seconds worth of data
  end = (starting_sec * srate) + (srate * 2) #end, srate*2 = 2 seconds worth of data
  
  #in case we don't get enough data points for the last iteration
  if(end > len(data)):
    binaural_eeg_midline = data[start:len(data), 1]
 
  else:
    binaural_eeg_midline = data[start:end, 1]
  
  #get the powers from periodogram
  binaural_midline_powers = pd.DataFrame.transpose(compute_periodogram(binaural_eeg_midline))
  
  #add the values to dataframe for future calculations/reference
  powers_eeg_midline = pd.concat([powers_eeg_midline,binaural_midline_powers], ignore_index = True)

#relative theta powers separated by electrode
powers_eeg_midline.columns = ["Theta_Rel"]
#print("Powers EEG Midline")
#print(powers_eeg_midline)

#combined relative theta powers
theta_midline = powers_eeg_midline.iloc[:, 0].values


midline_theta_total = w_rel_power_midline.iat[0,0]

baselined_theta = pd.DataFrame(((theta_midline - midline_theta_total)/midline_theta_total*100))

print(baselined_theta)

final_value = baselined_theta.iloc[:, 0:1].sum()/len(baselined_theta)
print ("Average Percent Difference")
print (final_value)
