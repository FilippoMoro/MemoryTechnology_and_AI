import streamlit as st
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import fitness_score, normalize_weights, compute_fitness_scores
from plotting import plot_fitness_table, plot_weights

st.title("Matching Memory Technology to AI workloads")
# st.write(
#     "Let's start building! For help and inspiration, head over to [docs.streamlit.io](https://docs.streamlit.io/)."
# )
st.write(
    '''Description:
    ------------
    The **Fitness Score** computes a weighted fitness score quantifying the suitability of a memory device for a given neural network workload. The score reflects how well the memory device matches the model's performance and energy requirements, considering multiple hardware and workload-specific metrics.

    Parameters:
    -----------
    - memory_data (list):
    A list containing memory-related characteristics:
    [memory_type, density, leakage, latency_write, latency_read, energy_read,
    bits_per_read, endurance, memory_is_nonvolatile, cmos_node, cim_compatible, ...]

    - nn_data (list):
    A list describing the neural network's workload and memory behavior:
    [model_type, param_size, activation_size, static_dynamic_ratio,
    macs_per_inference, inference_rate, access_pattern, reuse_factor,
    peak_bw, write_intensity, memory_notes]

    - norm_const (list, optional):
    Normalization constants for scaling raw values before scoring:
    [area_norm, latency_norm, energy_norm, cmos_norm] = [5, 1/60, 1, 22]

    - weights (list, optional):
    Weights (must sum to 1) to balance the 8 sub-scores:
        w1: footprint
        w2: write intensity
        w3: read intensity
        w4: latency
        w5: volatility
        w6: static power
        w7: cim
        w8: cmos node
        w9: cost score

    - verbose (bool, optional):
    If True, prints the 8 sub-scores.

    Returns:
    --------
    - total_score (float): The final weighted fitness score.
    - wscores (np.ndarray): The individual weighted scores for the 8 components.

    Scoring Components:
    -------------------
    1. **Footprint Area Memory Score**: Smaller memory footprint (bytes/mm²) → higher score
    2. **Write Intensity**
    3. **Read Intensity**
    4. **Latency Score**: Based on required vs actual throughput
    5. **Volatility Score**: Bonus if memory is non-volatile proportional on static weight percentage on total NN footprint
    6. **Static Power Score**: Proportional to leakage power and 
    7. **CIM Score**: Bonus if memory supports Compute In Memory
    8. **CMOS Score**: Penalizes larger CMOS node sizes
    9. **Cost Score**: Proportional to the cost of the memory per GB

    Example:
    --------
    score, breakdown = fitness_score(memory_data, nn_data)
    '''
)

#.1 Import the memory table
path_spreadsheet = 'https://docs.google.com/spreadsheets/d/{}/export?format=csv&id={}&gid={}'
memory_id = '18oZaJpiCprey9iLsKH61v3uwfOBHGfqtkrmbZhll80E'
memory_gid = '109329253'
memory_path = path_spreadsheet.format(memory_id, memory_id, memory_gid)

memory_table = pd.read_csv(memory_path)
memory_table.head(5)


#.2 Import the NN data
vision_id, vision_gid = '1yS2G0FW1GcVzydrEhpHzcmPqy9Q-Hxl-v9gCDDBO6Es', '1375486251'
nn_vision_table = pd.read_csv(path_spreadsheet.format(vision_id, vision_id, vision_gid))

speech_id, speech_gid = '1xH_Ff4KeCdwFRUXqAfGzS_7v4wDkqn_kyKgTB9lm-UY', '1452321839'
nn_speech_table = pd.read_csv(path_spreadsheet.format(speech_id, speech_id, speech_gid))

biomed_id, biomed_gid = '1LwTbJ-AA6E11IyJXbP0RI176upNIMd__TgvQrkJSWrM', '1378057546'
nn_biomed_table = pd.read_csv(path_spreadsheet.format(speech_id, biomed_id, biomed_gid))


# --------------------------------
# Edge AI: Vision
st.header('Edge AI: Vision')
st.divider()

st.write("Let's compute the Fitness Score for our Memory Technology types and several popular Neural Networks in the Edge AI domain. First, we have to set the *weights* for the Fitness Score:")

    # w1: footprint
    # w2: write intensity
    # w3: read intensity
    # w4: latency
    # w5: volatility
    # w6: static power
    # w7: cim
    # w8: cmos node
    # w9: cost score

# weight scores
w1 = st.slider("Footprint Area", 0, 1.5, 1, 0.05)
w2 = st.slider("Write Intensity", 0, 1.5, 0.75, 0.05)
w3 = st.slider("Read Intensity", 0, 1.5, 1, 0.05)
w4 = st.slider("Latency", 0, 1.5, 1, 0.05)
w5 = st.slider("Volatility", 0, 1.5, 0.5, 0.05)
w6 = st.slider("Static Power", 0, 1.5, 1.25, 0.05)
w7 = st.slider("Compute-in-Memory", 0, 1.5, 0.5, 0.05)
w8 = st.slider("CMOS compatibility", 0, 1.5, 0.5, 0.05)
w9 = st.slider("Cost score", 0, 1.5, 0.5, 0.05)

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

plot_fitness_table( vision_fitness, memory_table, nn_vision_table, 
                    xlabel_size=12, ylabel_size=12,
                    quantile_col = 60,
                    vminmax = [20, 80],
                    filename = 'Table_Fitness_Edge_Vision' )

# --------------------------------
# Edge AI: Speech
st.header('Edge AI: Speech')
st.divider()

# weight scores
w1 = st.slider("Footprint Area", 0, 1.5, 1, 0.05)
w2 = st.slider("Write Intensity", 0, 1.5, 0.75, 0.05)
w3 = st.slider("Read Intensity", 0, 1.5, 1, 0.05)
w4 = st.slider("Latency", 0, 1.5, 1, 0.05)
w5 = st.slider("Volatility", 0, 1.5, 0.5, 0.05)
w6 = st.slider("Static Power", 0, 1.5, 1.5, 0.05)
w7 = st.slider("Compute-in-Memory", 0, 1.5, 0.5, 0.05)
w8 = st.slider("CMOS compatibility", 0, 1.5, 0.5, 0.05)
w9 = st.slider("Cost score", 0, 1.5, 0.5, 0.05)

weights_speech = [w1,w2,w3,w4,w5,w6,w7,w8,w9]
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

plot_fitness_table( speech_fitness, memory_table, nn_speech_table, 
                    xlabel_size=12, ylabel_size=12, 
                    quantile_col=65,
                    filename = 'Table_Fitness_Edge_Speech',
                    vminmax = [20, 80],
                    no_ylabel= True)

# --------------------------------
# Edge AI: Biomedical
st.header('Edge AI: Biomedical')
st.divider()

# weight scores
w1 = st.slider("Footprint Area", 0, 1.5, 1, 0.05)
w2 = st.slider("Write Intensity", 0, 1.5, 0.75, 0.05)
w3 = st.slider("Read Intensity", 0, 1.5, 1, 0.05)
w4 = st.slider("Latency", 0, 1.5, 1, 0.05)
w5 = st.slider("Volatility", 0, 1.5, 0.5, 0.05)
w6 = st.slider("Static Power", 0, 1.5, 1.5, 0.05)
w7 = st.slider("Compute-in-Memory", 0, 1.5, 0.5, 0.05)
w8 = st.slider("CMOS compatibility", 0, 1.5, 0.5, 0.05)
w9 = st.slider("Cost score", 0, 1.5, 0.5, 0.05)

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

plot_fitness_table( biomed_fitness, memory_table, nn_biomed_table, 
                    xlabel_size=12, ylabel_size=12, 
                    quantile_col=65,
                    filename = 'Table_Fitness_Edge_BioMed',
                    vminmax = [20, 80],
                    no_ylabel=True)


# --------------------------------
# Cloud AI: LLMs
st.header('Cloud AI: LLMs')
st.divider()

#.3 Import the NN data
llm_id, llm_gid = '1zL808Rsxim7G1-Lb4lmBseFW8MHrImuEporNb18Ge10', '350385473'
nn_llm_table = pd.read_csv(path_spreadsheet.format(llm_id, llm_id, llm_gid))

web_id, web_gid = '1QMl1bliK1w4dHQoUIXCBpM6k6AweriEqkDtMqkfEky0', '935249215'
nn_web_table = pd.read_csv(path_spreadsheet.format(web_id, web_id, web_gid))

# just as a test
nn_llm_table.head(5)


# weight scores
w1 = st.slider("Footprint Area", 0, 5, 1, 0.05)
w2 = st.slider("Write Intensity", 0, 5, 2, 0.05)
w3 = st.slider("Read Intensity", 0, 5, 0.1, 0.05)
w4 = st.slider("Latency", 0, 5, 1, 0.05)
w5 = st.slider("Volatility", 0, 5, 0.1, 0.05)
w6 = st.slider("Static Power", 0, 5, 0.0, 0.05)
w7 = st.slider("Compute-in-Memory", 0, 5, 0.5, 0.05)
w8 = st.slider("CMOS compatibility", 0, 5, 4, 0.05)
w9 = st.slider("Cost score", 0, 5, 5, 0.05)

weights_llm = [w1,w2,w3,w4,w5,w6,w7,w8,w9]

weights_llm = normalize_weights( weights_llm )
# visualize weights
plot_weights( weights_llm, filename = 'Table_Fitness_Could_LLM_weights', plot_values=True, color='orange' )

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

plot_fitness_table( llm_fitness, memory_table, nn_llm_table, 
                    xlabel_size=12, ylabel_size=12, cmap='Purples',
                    vminmax = [20, 80],
                    quantile_col= 65,
                    filename = 'Table_Fitness_Could_LLM' )

# Cloud AI: Webapps

# weight scores
w1 = st.slider("Footprint Area", 0, 5, 1, 0.05)
w2 = st.slider("Write Intensity", 0, 5, 2, 0.05)
w3 = st.slider("Read Intensity", 0, 5, 0.1, 0.05)
w4 = st.slider("Latency", 0, 5, 1, 0.05)
w5 = st.slider("Volatility", 0, 5, 0.1, 0.05)
w6 = st.slider("Static Power", 0, 5, 0.05, 0.05)
w7 = st.slider("Compute-in-Memory", 0, 5, 0.25, 0.05)
w8 = st.slider("CMOS compatibility", 0, 5, 4, 0.05)
w9 = st.slider("Cost score", 0, 5, 5, 0.05)

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

plot_fitness_table( vision_fitness, memory_table, nn_web_table, 
                    xlabel_size=12, ylabel_size=12, cmap='Purples',
                    vminmax = [20, 80],
                    quantile_col= 64,
                    filename = 'Table_Fitness_Could_Web', no_ylabel=True )