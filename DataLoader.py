import csv



"""Data structure for storing information extracted from CSV file"""
class Data():
    def __init__(self):
        self.year = []
        self.id = []
        self.label = []
        self.feature = []
        self.serial_fraud = [] # identifier for whether this is a serial fraud case


def DataLoader(filepath, year_start, year_end):
    with open(filepath) as f:
        f_csv = csv.reader(f)
        headers = next(f_csv)
        data = Data()
        num_pos = 0 # to count how many positive cases
        num_neg = 0 # to count how many negative cases
        for row in f_csv:
            if int(row[0]) >= year_start and int(row[0]) <= year_end:
                data.year.append(row[0])
                data.id.append(row[1])
                data.serial_fraud.append(row[7])
                data.label.append(row[8])
                data.feature.append(row[9:37])
                if row[8] == '0': # to count how many positive and negative cases
                    num_neg = num_neg + 1
                else:
                    num_pos = num_pos + 1

    """convert string columns to float and int format"""
    data.year = [int(x) for x in data.year]
    data.id = [int(x) for x in data.id]
    temp = data.serial_fraud[:]
    data.serial_fraud = []
    for x in temp:
        if x != '':
            data.serial_fraud.append(int(x))
        else:
            data.serial_fraud.append(None)
    data.label = [int(x) for x in data.label]
    data.feature = [[float(y) for y in x] for x in data.feature]


    print(f"Data Loaded: {filepath}, {len(data.feature[0])} features, {len(data.label)} observations ({num_pos} pos, {num_neg} neg)")
    return data


