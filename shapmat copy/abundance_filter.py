
def set_abundance_to_zero(data,abundance_threshold=1e-15):
    """
    Set abundance that are < threshold to 0
    """
    low_abundance = [] 
    for species in data.columns:
        for i,v in enumerate(data[species]):
            if v != 0 and v < abundance_threshold:
                low_abundance.append([species,i,v])

    ab_filtered_1 = data.copy()
    ab_filtered_1[data<abundance_threshold] = 0.0
    return ab_filtered_1

def ab_filter(data,abundance_threshold=1e-15,prevalence_threshold=0.9):
    """
    1. Set abundance that are lower than abundance_threshold to 0
    2. Remove low frequency features. Features with number of zeros > percent_threshold are removed
    Returns:
        DataFrame
    """
    ab_filtered_1 = set_abundance_to_zero(data,abundance_threshold)
    species_list = data.columns

    selected_species = []
    for species in species_list:
        species_ab = list(ab_filtered_1[species])
        n_zero = species_ab.count(0) 
        percent_zeros = n_zero/len(species_ab)
        
        if percent_zeros < prevalence_threshold:
            selected_species.append(species)

    filtered_data = ab_filtered_1[selected_species]

    return filtered_data