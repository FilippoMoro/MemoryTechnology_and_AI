import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from utils import fitness_score, normalize_weights, compute_fitness_scores
from plotting import plot_fitness_table, plot_weights

# Streamlit Docs https://docs.streamlit.io/

st.title("Matching Memory Technology to AI workloads")

st.write(
    '''
    This app supports a poster presented by F. Moro, S. Liu, S. Billaudelle, A. Liu and M. Payvand at the ISMC conference. Find details about the conference here: https://newevent.bg/en/

    The title of the poster is "Matching Memory Technology to AI workloads" and the work proposes to analyze Memory Technology and Neural Networks in different applications to find the best fit between all the combinations. 

    First, we analyze the landscape of Memory Technology by evaluating different memory types on a set of metrics. These metrics include Density, Leakage Power, Write and Read latency, Write and Read energy and Endurance.

    Second, we consider a wide range of AI applications domain, and - in particular - the Neural Networks employed in these domains. These NNs are also evalueted on a set of metrics that concern Memory utilization during inference. Among such metrics are Parameter Size, Ratio between Static and Dynamic (activations) memory, number of Multipy-and-Accumulate (MAC) operations per inference, inference rate per second.

    #### --> **Fitness Score**
    These two sets of metrics are combined in the Fitness Score, which quantifies the match between the Memory characteristics and the NN's requirements.
    
    ------------
    The Fitness score is composed of 9 sub-scores:
    1. **Footprint Area Memory Score (FAM)**: Smaller memory footprint (bytes/mm²) → higher score    
    2. **Write Intensity**
    3. **Read Intensity**
    4. **Latency Score**: Based on required vs actual throughput
    5. **Volatility Score**: Bonus if memory is non-volatile proportional on static weight percentage on total NN footprint
    6. **Static Power Score**: Proportional to leakage power and 
    7. **CIM Score**: Bonus if memory supports Compute In Memory
    8. **CMOS Score**: Penalizes larger CMOS node sizes
    9. **Cost Score**: Proportional to the cost of the memory per GB

    These sub-scores are weighted by the following 9 weights:
    Weights (which are automatically normalized to sum to 1) balance the relevance of 9 sub-scores in a given AI application domain:
    1. w1: footprint
    2. w2: write intensity
    3. w3: read intensity
    4. w4: latency
    5. w5: volatility
    6. w6: static power
    7. w7: cim
    8. w8: cmos node
    9. w9: cost score

    > the Fitness Score is normalized in [0-100], where 100 represent a *perfect fit* between Memory Technology and Neural Network architecture and application requirements.

    We evaluate the Fitness score for common Neural Network archiectures in two macro-domains in AI: Edge and Cloud-based applications.
    The Edge AI domain is futher subdivided into Vision, Speech and Biomedical application domains.
    The Cloud AI domain is split into LLMs and WebApp domains.

    *Which is the best Memory Technology for all of those application domain?*
    > Let's find out!
    '''
)

# --------------------------------
#.1 Import the memory table
path_spreadsheet = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv&id={}&gid={}'
memory_id = '18oZaJpiCprey9iLsKH61v3uwfOBHGfqtkrmbZhll80E'
memory_gid = '109329253'
memory_path = path_spreadsheet.format(memory_id, memory_id, memory_gid)

st.header('Memory Technology')
st.divider()
st.write('Here we load data for the main Memory Technology. You can take a look at the table pressing the button below.')
memory_table = pd.read_csv(memory_path)
if st.button("Show Memory Table"):
    st.dataframe( memory_table )

# --------------------------------
#.2 Import the NN data
vision_id, vision_gid = '1yS2G0FW1GcVzydrEhpHzcmPqy9Q-Hxl-v9gCDDBO6Es', '1375486251'
nn_vision_table = pd.read_csv(path_spreadsheet.format(vision_id, vision_id, vision_gid))

speech_id, speech_gid = '1xH_Ff4KeCdwFRUXqAfGzS_7v4wDkqn_kyKgTB9lm-UY', '1452321839'
nn_speech_table = pd.read_csv(path_spreadsheet.format(speech_id, speech_id, speech_gid))

biomed_id, biomed_gid = '1LwTbJ-AA6E11IyJXbP0RI176upNIMd__TgvQrkJSWrM', '1378057546'
nn_biomed_table = pd.read_csv(path_spreadsheet.format(speech_id, biomed_id, biomed_gid))

llm_id, llm_gid = '1zL808Rsxim7G1-Lb4lmBseFW8MHrImuEporNb18Ge10', '350385473'
nn_llm_table = pd.read_csv(path_spreadsheet.format(llm_id, llm_id, llm_gid))

web_id, web_gid = '1QMl1bliK1w4dHQoUIXCBpM6k6AweriEqkDtMqkfEky0', '935249215'
nn_web_table = pd.read_csv(path_spreadsheet.format(web_id, web_id, web_gid))


# --------------------------------
# Edge AI: Vision
st.header('Edge AI: Vision')
st.divider()

if st.button("Show Vision NNs Table"):
    st.dataframe( nn_vision_table )

st.write("Let's compute the Fitness Score for our Memory Technology types and several popular Neural Networks in the Edge AI domain. First, we have to set the *weights* for the Fitness Score:")

# weight scores
w1 = st.slider("Footprint Area", 0, 100, 14, 1)
w2 = st.slider("Write Intensity", 0, 100, 11, 1)
w3 = st.slider("Read Intensity", 0, 100, 14, 1)
w4 = st.slider("Latency", 0, 100, 14, 1)
w5 = st.slider("Volatility", 0, 100, 7, 1)
w6 = st.slider("Static Power", 0, 100, 18, 1)
w7 = st.slider("Compute-in-Memory", 0, 100, 7, 1)
w8 = st.slider("CMOS compatibility", 0, 100, 7, 1)
w9 = st.slider("Cost score", 0, 100, 7, 1)

weights_vision = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
weights_vision = normalize_weights( weights_vision )

# visualize weights
# plot_weights( weights_vision, filename = 'Table_Fitness_Edge_Vision_weights', plot_values = True, color='orange' )


area_norm = 5 # mm^2
latency_norm = 1/60 # s (60 fps inference frequency)
read_norm = 10 # pJ * s, proportional to read_energy * read_latency
cmos_norm = 22 # nm
cost_norm = 1 # USD per asic
write_norm = 10 # pJ * s, proportional to write_energy * write_latency
leakage_norm = 1e-6 # W the target standby power for the memory
norm_const_vision = [area_norm, latency_norm, read_norm, 
                     cmos_norm, cost_norm, write_norm, leakage_norm]

vision_fitness = compute_fitness_scores( memory_table, nn_vision_table, 
                                        norm_const_vision, weights_vision, verbose=False )

fig_vision = plot_fitness_table( vision_fitness, memory_table, nn_vision_table, 
                    xlabel_size=12, ylabel_size=12,
                    quantile_col = 60,
                    vminmax = [20, 80],
                    filename = 'Table_Fitness_Edge_Vision' )
st.pyplot(fig_vision)

# --------------------------------
# Edge AI: Speech
st.header('Edge AI: Speech')
st.divider()

if st.button("Show Speech NNs Table"):
    st.dataframe( nn_speech_table )

# weight scores
weights_speech = [1]*9
weights_speech[0] = st.slider("Speech: Footprint Area", 0, 100, 14, 1)
weights_speech[1] = st.slider("Speech: Write Intensity", 0, 100, 11, 1)
weights_speech[2] = st.slider("Speech: Read Intensity", 0, 100, 14, 1)
weights_speech[3] = st.slider("Speech: Latency", 0, 100, 14, 1)
weights_speech[4] = st.slider("Speech: Volatility", 0, 100, 7, 1)
weights_speech[5] = st.slider("Speech: Static Power", 0, 100, 18, 1)
weights_speech[6] = st.slider("Speech: Compute-in-Memory", 0, 100, 7, 1)
weights_speech[7] = st.slider("Speech: CMOS compatibility", 0, 100, 7, 1)
weights_speech[8] = st.slider("Speech: Cost score", 0, 100, 7, 1)

# weights_speech = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
weights_speech = normalize_weights( weights_speech )
# visualize weights
# plot_weights( weights_speech, filename = 'Table_Fitness_Edge_Speech_weights', plot_values=True, color='orange' )

area_norm = 2.5 # mm^2
latency_norm = 1/100 # s (60 fps inference frequency)
energy_norm = 1 # pJ, referred to read energy for memory
cmos_norm = 22 # nm
cost_norm = 1 # USD
write_norm = 10 # pJ * s, proportional to write_energy * write_latency
leakage_norm = 1e-6 # W the target standby power for the memory
norm_const_speech = [area_norm, latency_norm, read_norm, 
                     cmos_norm, cost_norm, write_norm, leakage_norm]

speech_fitness = compute_fitness_scores( memory_table, nn_speech_table, 
                                        norm_const_speech, weights_speech, verbose=False )

fig_speech = plot_fitness_table( speech_fitness, memory_table, nn_speech_table, 
                    xlabel_size=12, ylabel_size=12, 
                    quantile_col=65,
                    filename = 'Table_Fitness_Edge_Speech',
                    vminmax = [20, 80],
                    no_ylabel= False)
st.pyplot(fig_speech)

# --------------------------------
# Edge AI: Biomedical
st.header('Edge AI: Biomedical')
st.divider()

if st.button("Show Biomedical NNs Table"):
    st.dataframe( nn_biomed_table )

# weight scores
w1 = st.slider("Biomed: Footprint Area", 0, 100, 14, 1)
w2 = st.slider("Biomed: Write Intensity", 0, 100, 11, 1)
w3 = st.slider("Biomed: Read Intensity", 0, 100, 14, 1)
w4 = st.slider("Biomed: Latency", 0, 100, 14, 1)
w5 = st.slider("Biomed: Volatility", 0, 100, 7, 1)
w6 = st.slider("Biomed: Static Power", 0, 100, 18, 1)
w7 = st.slider("Biomed: Compute-in-Memory", 0, 100, 7, 1)
w8 = st.slider("Biomed: CMOS compatibility", 0, 100, 7, 1)
w9 = st.slider("Biomed: Cost score", 0, 100, 7, 1)

weights_biomed = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
weights_biomed = normalize_weights( weights_biomed )
# visualize weights
# plot_weights( weights_biomed, filename = 'Table_Fitness_Edge_BioMed_weights', plot_values=True, color='orange' )

area_norm = 2 # mm^2
latency_norm = 1 # s (0.33 fps inference frequency)
energy_norm = 1 # pJ, referred to read energy for memory
cmos_norm = 22 # nm
cost_norm = 1
write_norm = 10 # pJ * s, proportional to write_energy * write_latency
leakage_norm = 1e-6 # W the target standby power for the memory
norm_const_biomed = [area_norm, latency_norm, read_norm, 
                     cmos_norm, cost_norm, write_norm, leakage_norm]

biomed_fitness = compute_fitness_scores( memory_table, nn_biomed_table, 
                                        norm_const_biomed, weights_biomed )

fig_biomed = plot_fitness_table( biomed_fitness, memory_table, nn_biomed_table, 
                    xlabel_size=12, ylabel_size=12, 
                    quantile_col=65,
                    filename = 'Table_Fitness_Edge_BioMed',
                    vminmax = [20, 80],
                    no_ylabel=False)
st.pyplot(fig_biomed)


# --------------------------------
# Cloud AI: LLMs
st.header('Cloud AI: LLMs')
st.divider()

if st.button("Show LLM NNs Table"):
    st.dataframe( nn_llm_table )

# weight scores
w1 = st.slider("LLM: Footprint Area", 0, 100, 7, 1)
w2 = st.slider("LLM: Write Intensity", 0, 100, 14, 1)
w3 = st.slider("LLM: Read Intensity", 0, 100, 7, 1)
w4 = st.slider("LLM: Latency", 0, 100, 7, 1)
w5 = st.slider("LLM: Volatility", 0, 100, 1, 1)
w6 = st.slider("LLM: Static Power", 0, 100, 0, 1)
w7 = st.slider("LLM: Compute-in-Memory", 0, 100, 2, 1)
w8 = st.slider("LLM: CMOS compatibility", 0, 100, 28, 1)
w9 = st.slider("LLM: Cost score", 0, 100, 35, 1)
weights_llm = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
weights_llm = normalize_weights( weights_llm )

area_norm = 1e6 # mm^2
latency_norm = 1/100 # s (60 fps inference frequency)
energy_norm = 1 # pJ, referred to read energy for memory
cmos_norm = 12
cost_norm = 10
write_norm = 100 # pJ * s, proportional to write_energy * write_latency
leakage_norm = 1e3 # W the target standby power for the memory
norm_const_llm = [area_norm, latency_norm, energy_norm, cmos_norm, cost_norm, write_norm, leakage_norm]

llm_fitness = compute_fitness_scores( memory_table, nn_llm_table, 
                                        norm_const_llm, weights_llm, verbose=False )

fig_llm = plot_fitness_table( llm_fitness, memory_table, nn_llm_table, 
                    xlabel_size=12, ylabel_size=12, cmap='Purples',
                    vminmax = [20, 80],
                    quantile_col= 65,
                    filename = 'Table_Fitness_Could_LLM' )
st.pyplot(fig_llm)

# --------------------------------
# Cloud AI: Webapps
st.header('Cloud AI: Webapps')
st.divider()

if st.button("Show Webapps NNs Table"):
    st.dataframe( nn_web_table )

# weight scores
w1 = st.slider("WebApps: Footprint Area", 0, 100, 7, 1)
w2 = st.slider("WebApps: Write Intensity", 0, 100, 14, 1)
w3 = st.slider("WebApps: Read Intensity", 0, 100, 7, 1)
w4 = st.slider("WebApps: Latency", 0, 100, 7, 1)
w5 = st.slider("WebApps: Volatility", 0, 100, 1, 1)
w6 = st.slider("WebApps: Static Power", 0, 100, 0, 1)
w7 = st.slider("WebApps: Compute-in-Memory", 0, 100, 2, 1)
w8 = st.slider("WebApps: CMOS compatibility", 0, 100, 28, 1)
w9 = st.slider("WebApps: Cost score", 0, 100, 35, 1)
weights_web = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
weights_web = normalize_weights( weights_web )
# visualize weights
plot_weights( weights_web, filename = 'Table_Fitness_Could_Web_weights', plot_values = True, color='orange' )

area_norm = 1e4 # mm^2
latency_norm = 1/10 # s (60 fps inference frequency)
energy_norm = 1 # pJ, referred to read energy for memory
cmos_norm = 12
cost_norm = 10
write_norm = 100 # pJ * s, proportional to write_energy * write_latency
leakage_norm = 1e3 # W the target standby power for the memory
norm_const_web = [area_norm, latency_norm, energy_norm, cmos_norm, cost_norm, write_norm, leakage_norm]

vision_fitness = compute_fitness_scores( memory_table, nn_web_table, 
                                        norm_const_web, weights_web, verbose=False )

fig_web = plot_fitness_table( vision_fitness, memory_table, nn_web_table, 
                    xlabel_size=12, ylabel_size=12, cmap='Purples',
                    vminmax = [20, 80],
                    quantile_col= 64,
                    filename = 'Table_Fitness_Could_Web', no_ylabel=False )
st.pyplot(fig_web)