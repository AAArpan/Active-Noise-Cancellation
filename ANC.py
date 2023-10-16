import numpy as np
import soundfile as sf
import sounddevice as sd
import matplotlib.pyplot as plt

# Load signals
x, fs = sf.read(r'C:\Users\arpan\Downloads\sunflower-street-drumloop-85bpm-163900.mp3')
noise, fs = sf.read(r'C:\Users\arpan\Downloads\mixkit-airport-departures-hall-357.wav') 
start_row = 0
end_row = len(x)
noise = noise[start_row:end_row, :]
x = x[:, 0]
noise = noise[:, 0]

def lms(x_sig, d_sig):
    '''
    x_sig = noise + clean signal hai from primary path
    d_sig = signal from secondary path   
    
    '''
    def adapt_filt(w, x_sig, filt_ord, sig_len):
        filt_out = np.zeros_like(x_sig)
        for i in range(sig_len):
            for j in range(filt_ord):
                temp = 0
                pos = i - j
                if pos >= 0:
                    temp += w[j] * x_sig[pos]
            filt_out[i] = temp

        return filt_out
    
    filt_ord = 64
    # Ask user for flter order
    filt_ord = np.uint16(input("Filter order (default: 64) ---->> "))

    while filt_ord < 1 and filt_ord%2 != 0:
        print("Value too low or order number is odd.")
        filt_ord = np.uint16(input("Filter order ----->> "))
    
    mu = 0.0001
    w = np.zeros(filt_ord, dtype=np.float32)
    sig_len = len(x_sig)

    # est_size = sig_len % filt_ord
    # est = np.zeros(np.uint8(sig_len / filt_ord) + est_size, dtype=np.float32)

    e = np.zeros((sig_len), dtype=np.float64)
    
    for i in range(filt_ord, sig_len):

        window_sig = x_sig[i:i-filt_ord:-1]
        est = np.dot(w, window_sig)
        
        e[i:i-filt_ord:-1] = d_sig[i:i-filt_ord:-1] - est
        w = w + 2 * mu * window_sig * e[i-filt_ord]
    
    filt_out = adapt_filt(w, x_sig, filt_ord, sig_len)
    # filt_out = np.convolve(x_sig, w)
    plot_flg = 'y'
    plot_flg = input("Visualize learning rate? [Y/n] ---->> ").lower()

    if plot_flg != 'y' and plot_flg != 'n':
        plot_flg = input("Visualize learning rate? [Y/n] ----->> ").lower()

    if plot_flg == 'y':
        plt.plot(e)
        plt.xlabel('Epoch')
        plt.ylabel('Error rate')
        plt.show()

    return filt_out

N = len(x)
M = 100  # Filter length/order

# Primary path - response
primary_path = x + noise

# Secondary path - response
secondary_path = 0.1*np.random.randn(N)

mu_primary = 0.001  
mu_secondary = 0.001
w_primary = np.zeros(M)  
w_secondary = np.zeros(M)

d = primary_path + secondary_path

sd.play(primary_path, fs)
sd.wait()

err = []
filtered_signal = np.zeros(N)
for z in range(10):
    for n in range(M, N):
        x_n = x[n:n - M:-1]  
        y_primary = np.dot(w_primary, x_n)  
        y_secondary = np.dot(w_secondary, x_n)  
        
        e = d[n] - y_primary - y_secondary  
        err.append(e)
        
        w_primary = w_primary + 2 * mu_primary * e * x_n
        w_secondary = w_secondary + 2 * mu_secondary * e * x_n

        y = y_primary + y_secondary
        filtered_signal[n] = y

plot_flg = 'y'
plot_flg = input("Visualize learning rate? [Y/n] ---->> ").lower()

if plot_flg != 'y' and plot_flg != 'n':
    plot_flg = input("Visualize learning rate? [Y/n] ----->> ").lower()

if plot_flg == 'y':
    plt.plot(err)
    plt.xlabel('Epoch')
    plt.ylabel('Error rate')
    plt.show()

# Plot the original and filtered signals
plt.figure()
plt.plot(d, label='Desired Signal', color='b')
plt.plot(filtered_signal, label='Filtered Signal', color='r')
plt.legend()
plt.grid()
plt.show()

sd.play(filtered_signal, fs)
sd.wait()