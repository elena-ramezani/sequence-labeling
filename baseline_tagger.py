import hw2_corpus_tool as tool
import pdb
import sys
import pycrfsuite
import os
import glob
#import numpy as np
#from sklearn.metrics import precision_recall_fscore_support

def main():
    input_dir = sys.argv[1]
    test_dir = sys.argv[2]
    output_file = sys.argv[3]
    input = read(input_dir)
    input_test = read(test_dir)



    aa = get_feature(input)
    feature_train = aa[0]
    tag_train = aa[1]

    train_object = pycrfsuite.Trainer(verbose = True)
    train_object.append(feature_train,tag_train)
    train_object.set_params({
        'c1': 1.0,
        'c2': 1e-3,
        'max_iterations': 50,


        'feature.possible_transitions': True
    })

    train_object.train("baseline_model_crfsuite")


    bb = get_feature(input_test)
    feature_test = bb[0]
    tag_test = bb[1]
    file_len = bb[2]
    test_object = pycrfsuite.Tagger()
    test_object.open("baseline_model_crfsuite")

    predicted = test_object.tag(feature_test)

    #predicted_list = np.array(predicted)
    #true_value = np.array(tag_test)
    #precision_recall_fscore_support(true_value, predicted_list, average='macro')

    mismatch = 0
    match = 0
    for i,item  in enumerate(predicted):
        if item == tag_test[i]:
            match+=1
        else:
            mismatch+=1
    accuracy = match*100/(mismatch + match)
    pdb.set_trace()
    write_file(output_file,predicted,file_len)
    #pdb.set_trace()


def get_feature(input):
    tag_list =[]
    feature_list =[]
    file_len=[]
    for item in input:
        prev_speaker =""
        firs_uterence = "1"
        count = 0
        for line in range(len(item)):
            act_tag = item[line][0]
            if act_tag != None:
                count+=1

                tag_list.append(act_tag)
                feature = []
                feature.append(firs_uterence)
                firs_uterence = "0"
                current_speaker = item[line][1]
                if prev_speaker == "":
                    feature.append("same_speaker")
                    prev_speaker = current_speaker

                elif current_speaker == prev_speaker:
                    feature.append("same_speaker")
                else:
                    feature.append("change_speaker")


                part_of_speaching = item[line][2]

                #if (part_of_speaching != None):
                try:
                    for pos in part_of_speaching:
                        feature.append("TOKEN_"+pos[0])
                        feature.append("POS_"+ pos[1])

                except TypeError:
                    feature.append("TOKEN_" + pos[0])
                    feature.append("POS_" + pos[1])


                feature_list.append(feature)
        file_len.append(count)

    return feature_list , tag_list ,file_len


def write_file(output_file ,predicted ,file_len):
    with open(output_file, "w") as output:

        all_filename = get_filename(sys.argv[2])


        t = 0

        for i in range(len(all_filename)):

            for j in range(file_len[i]):
                output.write(predicted[t])

                t+=1
                output.write("\n")
            output.write("\n")

    return

def read(path):

    data = tool.get_data(path)
    return data

def get_filename(directory):
	all_filenames = glob.glob(os.path.join(directory, "*.csv"))

	return all_filenames



if __name__ == "__main__": main()
