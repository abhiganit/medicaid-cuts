import model
import numpy as np
import pandas as pd
from itertools import product

# create input criteria for scenarios:
## Choose age_upper_bound:
age_upper_bound = 64# 55 or 64

## Functions
def get_age_based_inputs(age_upper_bound):
    if age_upper_bound == 55:
        # For 19--55 age-group
        agerange = [19, 55]
        ages = ['19 to 25 years', '26 to 34 years', '35 to 44 years', '45 to 54 years']
        agesD = ['15-24', '25-34', '35-44', '45-54']
    else:
        # For 19--64 age-group
        agerange = [19,65]
        ages = ['19 to 25 years', '26 to 34 years', '35 to 44 years', '45 to 54 years', '55 to 64 years']
        agesD = ['15-24','25-34','35-44','45-54','55-64']
    return agerange, ages, agesD


def get_morbidity_inputs(disease='diabetes', age_upper_bound=64):
    pop55 = 158334581; pop65 = 199563303
    if age_upper_bound == 55:
        pop = 158334581
    else:
        pop = 199563303
    if disease == 'diabetes':
        prev55, prev65 = 0.048, 0.189
        uprev55, uprev65 = 0.305, 0.298
        eru = 1.855
        if age_upper_bound == 55:
            dp = prev55
            ud = uprev55
        elif age_upper_bound == 64:
            dp = (prev55 * pop55 + prev65 * (pop65 - pop55)) / pop65
            ud = (uprev55 * prev55 * pop55 + uprev65 * prev65 * (pop65 - pop55)) / (dp * pop65)
        else:
            raise ValueError("Unsupported age upper bound (only 55 or 64 supported)")
        return pop, dp, ud, eru

    elif disease == 'hypertension':
        eru = 0.055
        if age_upper_bound == 55:
            dp =0.35
            ud = 0.9
        elif age_upper_bound == 64:
            dp = 0.39
            ud = 0.92
        else:
            raise ValueError("Unsupported age upper bound (only 55 or 64 supported)")
        return pop, dp, ud, eru
    elif disease == 'cholesterol':
        eru = 0.050
        if age_upper_bound == 55:
            dp = 0.10
            ud = 0.55
        elif age_upper_bound == 64:
            dp = 0.12
            ud = 0.63
        else:
            raise ValueError("Unsupported age upper bound (only 55 or 64 supported)")
        return pop, dp, ud, eru
    else:
         raise ValueError("Unsupported disease type")




def get_morbidity_results(pop, prev, current_coverage, updated_coverage, uncontrolled_rate,
                              relative_risk=None, adjusted_difference=None):
    if (relative_risk is None and adjusted_difference is None) or (relative_risk is not None and adjusted_difference is not None):
        raise ValueError("Please provide either relative_risk or adjusted_difference, but not both.")

    if uncontrolled_rate is None:
        raise ValueError("uncontrolled_rate must be provided.")

    current_uncontrolled = uncontrolled_rate * prev * pop

    if relative_risk is not None:
        multiplier = ((updated_coverage + relative_risk * (1 - updated_coverage)) /
                      (current_coverage + relative_risk * (1 - current_coverage)))
        projected_uncontrolled = multiplier * current_uncontrolled
    else:
        delta_coverage = current_coverage - updated_coverage
        excess_uncontrolled = pop * prev * delta_coverage * adjusted_difference
        projected_uncontrolled = current_uncontrolled + excess_uncontrolled

    return current_uncontrolled, projected_uncontrolled, projected_uncontrolled - current_uncontrolled


agerange0, ages0, agesD0 = get_age_based_inputs(age_upper_bound)
pop_model = model.PopulationModel('data/np2023.csv',agerange0)
ins_model = model.InsuranceModel('data/ACS ST 5Y 2023 Data.xlsx', ages0)
death_model = model.DeathRateModel('data/Mortality Data 2021-2022.csv',agesD0)

population = pop_model.get_population()
pci = ins_model.compute_insurance_ratio()
death_rate = death_model.compute_death_rate()

sim = model.HealthImpactSimulator(1.4, [1.06, 1.84], population, pci,death_rate)

li = 13.7*1e6
pui = sim.simulate_uninsurance(li)
excess = sim.compute_excess_deaths(pui, True)
print(np.quantile(excess,[0.5,0.025,0.975]))





'''
Age-related calculations: It is possible for me to do, age-distributed calculations within 19-64.
1. population code is just set up to work
2. insurance one as well 
3. death rates
Not sure how to go about calculation that extends beyond adult population.
'''
