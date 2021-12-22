from plotly.subplots import make_subplots
import dataclasses
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st


"""
## Belief state visualization

There is a given initial distribution of agent position on a line depicted by a bar chart below. 
Initially the agent is in the middle of the board with probability 1.0.
You can either ask the agent to perform one of the move commands (`Go left` or `Go right`)
or use the sensor.

It is important to notice that in the case of using the sensor here we control what was 
the result of the sensory reading. 
The belief distribution is updated using Bayes theorem relying on the assumptions listed below.


### Probabilistic motion model

* For the case of `Go left` the agent with probability 0.9 will indeed go left 
(unless it is already at the very left in which case it will stay) and 
with 0.1 probability it will remain in its current position.
* Analogously for `Go right` the agent with probability 0.9 will indeed go right 
(unless it is already at the very right in which case it will stay) and 
with 0.1 probability it will remain in its current position.

### Probabilistic sensory model

* The sensor with probability 0.7 returns a correct reading and with probability 0.3 returns a random color from {R, G, B}.
* Q: what is the probability of `P(z=RED | x=RED)`?

"""

COLORS = 'RRRRRRRGGGGGGGRRBBBRRRRRRRRRRRRBB'
INITIAL_DISTRIBUTION = np.zeros(len(COLORS), dtype=np.float32)
INITIAL_DISTRIBUTION[len(COLORS)//2] = 1.0

state = st.session_state

fig = make_subplots(rows=2, cols=1)

"""
## Command the agent
"""

if 'p' not in state or st.button("Init"):
    state.p = np.copy(INITIAL_DISTRIBUTION)
    state.sequence = ''

col1, col2 = st.columns([0.2, 1])

if col1.button("Go left"):
  newp = state.p * 0.1

  newp[0] += state.p[0] * 0.9
  newp[:-1] += state.p[1:] * 0.9

  state.p = newp
  state.sequence += 'l'

if col2.button("Go right"):
  newp = state.p * 0.2

  newp[-1] += state.p[-1] * 0.8
  newp[1:] += state.p[:-1] * 0.8

  state.p = newp
  state.sequence += 'r'

col1, col2, col3 = st.columns([0.25, 0.25, 1])

sensor = False

if col1.button("Red"):
  state.sequence += 'R'
  sensor = True

if col2.button("Green"):
  state.sequence += 'G'
  sensor = True

if col3.button("Blue"):
  state.sequence += 'B'
  sensor = True

if sensor:
  c = state.sequence[-1]
  newp = np.copy(state.p)
  print(np.array(list(COLORS)) == c) 
  likelihood = (np.array(list(COLORS)) == c) * 0.8 + (np.array(list(COLORS)) != c) * 0.1
  print(likelihood)
  newp *= likelihood
  print(newp)
  newp /= np.sum(newp)
  print(newp)
  state.p = newp

"""
## Belief 

At the top there is a bar chart showing the current belief of the robot states (distribution over the states). 
At the bottom you can see colors of each of the cells of the grid.
"""

x = list(range(len(COLORS)))
fig.add_trace(go.Bar(x=x, y=state.p), row=1, col=1)

img_rgb = np.array([[[255, 0, 0] if c == 'R' else [0, 255, 0] if c == 'G' else [0, 0, 255] for c in COLORS]], dtype=np.uint8)

fig.add_trace(px.imshow(img_rgb).data[0], row=2, col=1)
fig.update_layout(height=350, width=600)

fig.update_layout(
  #  xaxis=dict(showticklabels=False, range=[0,len(COLORS)]),
    xaxis=dict(showticklabels=False),
    yaxis2=dict(showticklabels=False)
)
st.plotly_chart(fig)
st.write('Sequence of commands: ', state.sequence)
