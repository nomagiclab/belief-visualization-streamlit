from plotly.subplots import make_subplots
import dataclasses
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import streamlit as st


"""
## Belief state visualization

There is a given initial distribution of agent position on a line depicted by a bar chart below. 
You can either ask the agent to perform one of the move commands (`Go left` or `Go right`)
or use the sensor.

It is important to notice that in the case of using the sensor here we control what was 
the result of the sensory reading. 
The belief distribution is updated using Bayes theorem relying on the assumptions listed below.


### Probabilities for moves

* For the case of `Go left` the agent with probability 0.9 will indeed go left 
(unless it is already at the very left in which case it will stay) and 
with 0.1 probability it will remain in its current position.
* For the case of `Go right` the agent with probability 0.8 will indeed go right 
(unless it is already at the very right in which case it will stay) and 
with 0.2 probability it will remain in its current position.

### Probabilities for sensory readings

* The sensor with probability 0.7 returns a correct reading and with probability 0.3 returns a random color (R, G, B).

"""

INITIAL_DISTRIBUTION = np.array([0.1, 0.2, 0.5, 0.05, 0.05, 0.1])
COLORS = 'RGBRGB'


@dataclasses.dataclass
class GameState:
    p = np.copy(INITIAL_DISTRIBUTION)
    sequence = ''

@st.cache(allow_output_mutation=True)
def persistent_game_state() -> GameState:
    return GameState()

state = persistent_game_state()

fig = make_subplots(rows=2, cols=1)

"""
## Command the agent
"""

if st.button("Init"):
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
"""

x = list(range(6))
fig.add_trace(go.Bar(x=x, y=state.p), row=1, col=1)

img_rgb = np.array([[[255, 0, 0] if c == 'R' else [0, 255, 0] if c == 'G' else [0, 0, 255] for c in COLORS]], dtype=np.uint8)

fig.add_trace(px.imshow(img_rgb).data[0], row=2, col=1)
fig.update_layout(height=350, width=600)

fig.update_layout(
    xaxis=dict(showticklabels=False),
    yaxis2=dict(showticklabels=False)
)
st.plotly_chart(fig)
st.write('Sequence of commands: ', state.sequence)
