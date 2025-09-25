import numpy as np
import matplotlib.gridspec as gridspec
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import pandas as pd

#Data 
#--------------------------------
x0=0;xf=1
y0=0;yf=1
points=80

x, y= np.meshgrid(np.linspace(x0, xf, points,endpoint=False), np.linspace(x0, xf, points,endpoint=False), indexing='ij')



#--------------------------------                                   
a3_02_1= 1.5026822555281243;a6_02_1= 0.8752194027900603
z02_1=(np.sin((x * a3_02_1)) * (a6_02_1 ** (y ** 2)))
#--------------------------------
#--------------------------------
a3=0.49591911278429224
z04_0=(np.sin(((a3 * x) + x)) * np.cos((a3 * y)))
#--------------------------------
#--------------------------------
a0_08_2= 2.097276938985649;a6_08_2= 1.4975925185416339
z08_2=(np.sin((x * a6_08_2)) * np.cos((y / a0_08_2)))
#--------------------------------
#--------------------------------
a5_12_0=0.5039583161559179
z12_0=np.sin(((a5_12_0 * (x * np.cos(y))) + x))
#--------------------------------
#--------------------------------                                   
a1_12_2=2.0661451810006404
z12_2=np.sin((x + (x / ((y ** 2) + a1_12_2))))
#--------------------------------                                   
#--------------------------------
a1_12_1=2.0661451810006404
z12_2=np.sin((x + (x / ((y ** 2) + a1_12_1))))
#--------------------------------
#--------------------------------
a1_16_1=1.5198873727806672
z16_1=np.sin(((a1_16_1 ** np.cos(y)) * x))
#--------------------------------
#--------------------------------
a1_18_0=-2.4827756986867495;a4_18_0=1.558120652445591
z18_0=(((x / (y + a1_18_0)) + a4_18_0) * x)
#--------------------------------

z_to_plot=z12_2

#Define figure size in cm
cm = 1/2.54 #convert inch to cm
width = 8*cm; height=4*cm


df = pd.DataFrame({
    'x': x.ravel(order='C'),  # 'C' means row-major: y changes first with indexing='ij'
    'y': y.ravel(order='C'),
    'z': z_to_plot.ravel(order='C')
})

df.to_csv('synthetic_test.csv')
print(df)

#Figure settings                                                     
#--------------------------------                                    
output_path='figures/'
name_fig='example_fig'
extensions=['.svg','.png','.pdf']     #Extensions to save figure     


limits={1:{'x':[-1.1,1.1], 'y':[-3.25, 8]}, 5:{'x':[-1.2,1.2], 'y':[-0.8,-0.2]},
	7:{'x':[-0.1,2.1], 'y': [0.4,3.1]}, 8:{'x':[-0.1,4.2], 'y':[-0.1, 2.5]},
	10:{'x':[-0.1,1.1], 'y':[-0.1,1.1], 'z':[-0.1,1.1] } }

ticks={1:{'x':[0, 0.5, 1], 'y':[ -3, 0, 3, 6.0]}, 5:{'x':[-1, 0, 1], 'y':[-0.75,-0.5, -0.25]},
       7:{'x':[0,1,2,], 'y': [1,2,3]}, 8:{'x':[0,2,4], 'y':[0, 1, 2]},
       10:{'x':[0,0.5,1], 'y':[0,0.5,1], 'z':[0,0.5,1] } }

n=10


#Define figure size                                                  
cm = 1/2.54 #convert inch to cm                                      
width = 8*cm; height=4*cm #8x4cm for each figure in panel

#Fonts and sizes                                                     
size_axis=7;size_ticks=6;size_title=5
line_w=1;marker_s=1
#--------------------------------

#Plots                                                               
#--------------------------------
fig=figure(figsize=(width,height), dpi=300)
ax = fig.add_subplot(111, projection='3d')

ax.plot(x, y, z_to_plot, '.', markersize=marker_s)


#Labels                                                              
plt.xlabel('x',fontsize=size_axis);plt.ylabel('y',fontsize=size_axis)
plt.xlim(limits[n]['x'][0], limits[n]['x'][1]);plt.ylim(limits[n]['y'][0], limits[n]['y'][1])

ax.set_xticks(ticks[n]['x']);ax.set_yticks(ticks[n]['y']);ax.set_zticks(ticks[n]['z'])

#legend                                                              
plt.legend(loc='best',fontsize=size_ticks,frameon=False)

plt.show()
#-------------------------------- 
