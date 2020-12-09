def mapper_chest_loc(x):
    return map_chest_loc[x]


def mapper_acquisition(x):
    return map_acquisition[x]


def mapper_rec_equipment(x):
    return map_rec_equipment[x]


def mapper_diagnosis(x):
    return map_diagnosis[x]


def mapper_diagnosis_reverse(x):
    return map_diagnosis_reverse[x]


def mapper_sex(x):
    return map_sex[x]


map_chest_loc = {
    'Tc': 0,
    'Al': 1,
    'Ar': 2,
    'Pl': 3,
    'Pr': 4,
    'Ll': 5,
    'Lr': 6
}
map_acquisition = {
    'sc': 0,
    'mc': 1
}
map_rec_equipment = {
    'AKGC417L': 0,
    'LittC2SE': 1,
    'Litt3200': 2,
    'Meditron': 3
}
map_diagnosis = {
    'URTI': 0,
    'COPD': 1,
    'Bronchiectasis': 2,
    'Pneumonia': 3,
    'Bronchiolitis': 4,
    'LRTI': 5,
    'Healthy': 6,
    'Asthma': 7,
}

map_sex = {
    'F': 0,
    'M': 1
}

map_diagnosis_reverse = {
    '0': 'URTI',
    '1': 'COPD',
    '2': 'Bronchiectasis',
    '3': 'Pneumonia',
    '4': 'Bronchiolitis',
    '5': 'LRTI',
    '6': 'Healthy',
    '7': 'Asthma',
}
