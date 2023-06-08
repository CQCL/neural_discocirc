import os
import pickle


def run():
    file_path = '../data/task_vocab_dicts/'
    datasets = {}
    for filename in sorted(os.listdir(file_path)):
        if '9' not in filename:
            continue
        with open(file_path + filename, 'rb') as f:
            dataset = pickle.load(f)

        dataset = dataset[0]

        with open(file_path + "test", "wb") as f:
            pickle.dump(dataset, f)

        break

        print('dataset')



if __name__ == "__main__":
    run()