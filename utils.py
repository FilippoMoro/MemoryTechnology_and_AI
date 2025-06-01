import numpy as np

# final fitness score
def fitness_score(memory_data, # data about the memory
                  nn_data, # data about the Neural Network
                  norm_const :list = [5, 1/60, 10, 22, 1, 40, 1e-3], 
                  weights :list = [1/9]*9,
                  verbose :bool = False):

    # weight scores
    # w1: footprint
    # w2: write intensity
    # w3: read intensity
    # w4: latency
    # w5: volatility
    # w6: static power
    # w7: cim
    # w8: cmos node
    # w9: cost score
    w1, w2, w3, w4, w5, w6, w7, w8, w9 = weights
    assert np.round( sum(weights), 2 ) == 1., "ERROR: score weights not normalized"


    # memory data
    memory_type, density, leakage, latency_write, energy_write, latency_read, energy_read, bits_per_read, \
        endurance, memory_is_nonvolatile, cmos_node, cim_compatible, cost_per_gb, _ = memory_data

    # NN data
    model_type, param_size, activation_size, static_dynamic_ratio, \
      macs_per_inference, inference_rate, access_pattern, reuse_factor, \
        peak_bw, write_intensity, memory_notes = nn_data

    # normalization constants
    area_norm, latency_norm, read_norm, cmos_norm, cost_norm, write_norm, leakage_norm = norm_const


    # ==========================================================================

    # 1. FA score: Footprint Area Score , 8 is for bytes
    footprint = (param_size) / ( (density * bits_per_read) / 8)
    FA_score = 1 / (1 + ( (footprint) / area_norm )**1 )

    # 2. WI: Write Intensity score
    WI_score = ( ( 1- static_dynamic_ratio / (1+static_dynamic_ratio) )**0.1 ) * ( 1/ ( 1 + ((energy_write*latency_write*np.log10(endurance))/write_norm)**1 ) )

    # 3. Read Intensity score TODO: add the throughput dependency
    RI_score = 1 / (1 + ( (energy_read*latency_read) / read_norm )**2 )

    # 4. Lat: Latency score
    mac_per_sec = inference_rate * macs_per_inference
    required_read_throughput = mac_per_sec / bits_per_read
    # TODO: evaluate the number of layers in the networks by the reuse factor of memory
    Lat_score = 1 / (1 + ( (latency_read) / required_read_throughput )**1 )

    # 5. Vol: Volatility bonus
    Vol_score = static_dynamic_ratio / (1+static_dynamic_ratio) if (memory_is_nonvolatile == 'Non-volatile') else 0

    # 6. SP: Static Power score
    SP_score = sigmoid( -np.log10( (param_size*1E6) * (leakage*1E-12) / leakage_norm ) ) 

    # 7. Compute in Memory score
    CIM_score = 1.0 if cim_compatible and access_pattern in ['sequential', 'regular'] else 0.5

    # 8. CMOS compatibility score
    CMOS_score = 1 / (1 + ( (cmos_node) / cmos_norm )**1 )

    # 9. Cost score
    Cost_score = 1 / (1 + ( ( cost_per_gb  / cost_norm ) )**1 )

    # ==========================================================================

    # Scores
    scores = np.array( [FA_score, WI_score, RI_score, Lat_score, Vol_score, 
                        SP_score, CIM_score, CMOS_score, Cost_score] )

    # Weighted Scores
    wscores = scores * np.array( weights )
    
    # some printing
    if verbose:
        print( f'{memory_type}, {model_type}: {scores}' )
        # print( f'{memory_type}, {model_type}: {Cost_score}, {cost_per_gb}' )

    return np.sum(wscores), wscores


def sigmoid( x ):
  return 1 / (1 + np.exp(-x))


def compute_fitness_scores( memory_table, nn_table, norm_const, weights, verbose=False ):
    fitness_scores = np.zeros( (len( memory_table.values ), len(nn_table.values)) )
    for m, memory_data in enumerate( memory_table.values):
        for n, nn_data in enumerate( nn_table.values ):
            fitness_scores[m, n], wscores = fitness_score(memory_data, nn_data, norm_const, weights, verbose)
    return fitness_scores


def normalize_weights( weights ):
    assert isinstance( weights, list ), "ERROR: weights not passed as a list"
    return list( weights / np.sum(weights) )  
