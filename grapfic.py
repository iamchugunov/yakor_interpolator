import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os

directory_folder = os.getcwd()

file_name = '3poi-rszo'
f = open('logs/' + file_name + '.txt', 'r')

# передавать количетво точек? где разрывы и где нет
points = []

for line in f:

    a = line.split()
    poit = {}
    poit["execTime"] = float(a[1])
    poit["Beta"] = 0.
    poit["sBeta"] = 0.
    poit["Epsilon"] = float(a[6]) #6 - rszo
    poit["sEpsilon"] = 0.
    poit["R"] = float(a[3])
    poit["sR"] = 0.
    poit["Vr"] = abs(float(a[4]))
    poit["sVr"] = 0.
    poit["Amp"] = float(a[2])
    points.append(poit)

f.close()

meas_dict = {}
meas_dict["points"] = points

t = []
R = []
Vr = []
E = []


for i in range(len(meas_dict["points"])):
    t.append(meas_dict["points"][i]["execTime"])
    R.append(meas_dict["points"][i]["R"])
    Vr.append(meas_dict["points"][i]["Vr"])
    E.append(meas_dict["points"][i]["Epsilon"])

A = []
for i in range(len(Vr) - 1):
    A.append((Vr[i+1] - Vr[i])/(t[i+1] - t[i]))

time_in = 0

for i in range(len(A)):
    if A[i] < 100:
        time_in = i
        break

print(time_in)
print(A)

# trace_R = go.Scatter(x=t, y=R, name='расстояние', mode='markers')
# trace_Vr = go.Scatter(x=t, y=Vr, name='радиальная скорость относительно локатора', mode='markers')
# trace_E = go.Scatter(x=t, y=E, name='угол', mode='markers')
#
# fig = make_subplots(rows=3, cols=1)
#
# fig.add_trace(trace_R, row=1, col=1)
# fig.add_trace(trace_Vr, row=2, col=1)
# fig.add_trace(trace_E, row=3, col=1)
#
# fig.update_layout(title_text="Графики файла " + file_name)
#
# fig.update_traces(hoverinfo="all", hovertemplate="x: %{x}<br>y: %{y}")
#
# fig.write_html('result/' + file_name + '.html')
# добавлять в название


