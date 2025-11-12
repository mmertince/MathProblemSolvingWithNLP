import pandas as pd
import itertools
import numpy as np

### In the following docstrings, the values are given just to exemplify the format of the output, not to give the actual "real" output

steps = [1, 2, 3, 4]

def get_missing_steps(row, steps):
    """
    Get missing steps as a tuple for each row.
    
    Example:
        Input: row = {'step1_missing': 1, 'step2_missing': 0, 'step3_missing': 1, 'step4_missing': 0},
               steps = [1, 2, 3, 4]
        Output: (2, 4)
    """
    missing_steps = []
    ##############################################################################
    ### STUB: INSERT CODE HERE: Get missing steps as a tuple for each row.###
    ##############################################################################

    #check for mising steps with f string
    for step in steps:
        if row[f"step{step}_missing"] == 0:
                missing_steps.append(step)

    #return missing steps as increasing order sorted 
    return tuple(sorted(missing_steps))

def generate_all_subsets(steps):
    """
    Generate all possible subsets for the missing steps.

    Example:
        Input: steps = [1, 2, 3]
        Output: [(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    """
    all_subsets_missing = []
    for r in range(len(steps) + 1):
        subsets_r = list(itertools.combinations(steps, r))
        all_subsets_missing.extend(subsets_r)
    return all_subsets_missing

def compute_v_S(df, all_subsets_missing):
    """
    Compute v(S) for all subsets of missing steps.

    Example:
        Input: 
            df = pd.DataFrame({'missing_steps': [(), (1,), (2,), (1, 2)], 'is_correct': [0.8, 0.7, 0.6, 0.5]})
            all_subsets_missing = [(), (1,), (2,), (1, 2)]
        Output: 
            v_S = {(): 0.8, (1,): 0.7, (2,): 0.6, (1, 2): 0.5}
    """
    ##################################################################################
    ### STUB: INSERT CODE HERE: Compute v(S) for all subsets of missing steps.###
    ##################################################################################

    v_S = {}

    for subset in all_subsets_missing:
        #filter dataframe to get missing steps == subset
        matching_rows = df[df['missing_steps'] == subset]
        #get matching_rows scores
        scores = matching_rows['is_correct']
        #calculate its mean
        mean_score = scores.mean()
        v_S[subset] = mean_score
    return v_S

def compute_marginal_contributions(steps, v_S):
    """
    Compute the marginal contributions for each step.

    Example:
        Input:
            steps = [1, 2]
            v_S = {(): 0.85, (1,): 0.67, (2,): 0.72, (1, 2): 0.75}
        Output:
            Delta_sum = {1: 0.08, 2: 0.12}, valid_permutations_count = 2
    """
    permutations = list(itertools.permutations(steps))
    Delta_sum = {i: 0.0 for i in steps}
    valid_permutations_count = 0

    total_steps_set = set(steps)

    for pi in permutations:
        valid_permutation = True
        for i in steps:
            idx_i = pi.index(i)
            #############################################################################################
            ### STUB: INSERT CODE HERE: Retrieve S_i, S_i_union_i, S_i_sorted, S_i_union_i_sorted###
            #############################################################################################

            #steps that come before i
            S_i = set(pi[0:idx_i])
            #steps that come before i plus i
            S_i_union_i = set(pi[0:idx_i + 1])

            #convert sets to sorted tuples
            missing_S_i_sorted = tuple(sorted(list(S_i)))
            missing_S_i_union_i_sorted = tuple(sorted(list(S_i_union_i)))

            #look for S_i in dictionary
            v_S_i = v_S.get(missing_S_i_sorted, np.nan)
            #look for S_i + i in dictionary
            v_S_i_union_i = v_S.get(missing_S_i_union_i_sorted, np.nan)

            #invalid permutation if one of them or both nan
            if np.isnan(v_S_i) or np.isnan(v_S_i_union_i):
                valid_permutation = False
                break
            else:
                ###############################################################################
                ### STUB: INSERT CODE HERE: Compute the marginal contribution of step i###
                ###############################################################################

                #compute marginal contribution and delta sum
                marginal_contribution = v_S_i_union_i - v_S_i
                Delta_sum[i] += marginal_contribution

        if valid_permutation:
            valid_permutations_count += 1
    return Delta_sum, valid_permutations_count

def compute_shapley_values(Delta_sum, valid_permutations_count, steps):
    """
    Compute the Shapley values for each step.

    Example:
        Input: 
            Delta_sum = {1: 0.08, 2: 0.12}
            valid_permutations_count = 2
            steps = [1, 2]
        Output: 
            Shapley_values = {1: 0.04, 2: 0.06}
    """
    ##############################################################
    ### STUB: INSERT CODE HERE: Compute the Shapley values###
    ##############################################################

    shapley_values = {}

    #safety caution
    if valid_permutations_count == 0:
        return {i: 0 for i in steps}

    #compute shapley value for each step
    for i in steps:
        shapley_values[i] = Delta_sum[i] / valid_permutations_count

    return shapley_values

def main():
    df = pd.read_csv('evaluation_with_steps.csv')

    df['missing_steps'] = df.apply(get_missing_steps, axis=1, args=(steps,))

    ######################################################################################
    # STUB: INSERT CODE HERE: Generate all possible subsets for the missing steps #       (1 line of code)
    ######################################################################################

    #get all_subsets_missing using generate_all_subsets()
    all_subsets_missing = generate_all_subsets(steps)

    ###############################################
    # STUB: INSERT CODE HERE: Compute v(S) #    (1 line of code)
    ###############################################

    #get v_S using compute_v_S
    v_S = compute_v_S(df, all_subsets_missing)

    #############################################################
    # STUB: INSERT CODE HERE: Compute the Shapley values and print for each step #
    #############################################################
    
    #getting Delta_sum  and valid_permutations_count using compute_marginal_contributions
    Delta_sum, valid_permutations_count = compute_marginal_contributions(steps, v_S)
    #getting shapley_values using compute_shapley_values
    shapley_values = compute_shapley_values(Delta_sum, valid_permutations_count, steps)

    #printing results
    print("Shapley Values for evaluation_with_steps.csv")

    if valid_permutations_count > 0:
        for step, value in shapley_values.items():
            print(f"Step {step}: {value:.4f}")
    else:
        print("No valid permutations found")

    

    

if __name__ == "__main__":
    main()
