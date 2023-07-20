import librosa
import matplotlib.pyplot as plt
import numpy as np
import sys, csv, pdb

# h5_path = sys.argv[1]
# data=h5py.File(h5_path,'r')

# filename = sys.argv[2]


sr = 44100
audio_path = sys.argv[1]
audio, y = librosa.load(audio_path, sr=sr)

x_values=[]
y_values=[]

contour_path = sys.argv[2]
with open(contour_path, 'r') as contourCsv:
	reader = csv.reader(contourCsv)
	for row in reader:
		x_values.append(float(row[0]))
		y_values.append(int(float(row[1])))
		print(float(row[0]), int(float(row[1])))

total_entries =len(audio)
total_time = total_entries/sr
time_axis = np.linspace(0, total_time, total_entries)

fig, ax1 = plt.subplots()

color = 'tab:blue'
ax1.set_xlabel('time (s)')
ax1.set_ylabel('sample', color=color)
ax1.plot(time_axis, audio, color=color)
ax1.tick_params(axis='y', labelcolor=color)

ax2 = ax1.twinx()

color = 'tab:orange'
ax2.set_ylabel('pitch', color=color)
ax2.plot(x_values, y_values, color=color)
ax2.tick_params(axis='y', labelcolor=color)

fig.tight_layout()
plt.show()
