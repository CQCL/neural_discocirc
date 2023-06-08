import os
import pickle


def run():
    #%%
    # read the file
    base_path = os.path.abspath('../../data/')
    data_path = base_path + "/task_vocab_dicts/"

    vocab = []
    for filename in sorted(os.listdir(data_path)):
        print(data_path + filename)
        with open(data_path + filename,
                  "rb") as f:
            # dataset is a tuple (context_circuit,(question_word_index, answer_word_index))
            file_vocab = pickle.load(f)
            for v in file_vocab:
                if v not in vocab:
                    vocab.append(v)

    pickle.dump(vocab, open(base_path+"/task_vocab_dicts/all_vocabs.p", "wb"))



if __name__ == "__main__":
    run()