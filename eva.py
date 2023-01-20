## Finds recall and precision for a chosen label at the end of each line
## It assumes that both system output and answer key are the same length
import os


def get_line_numbers_for_arg(infile, argument, NNP_lists=False):
    argument = argument.lower()
    with open(infile) as instream:
        num = 0
        outlist = []
        nnp_lines = []
        nnp_match = False
        for line in instream:
            line = line.lower()
            line = line.strip('\n \t\r')
            ## print(line)
            line_list = line.split('\t')
            if len(line_list) > 1:
                pos = line_list[1]
            else:
                pos = ''
            if NNP_lists and (pos in ['nnp', 'nnps']):
                nnp_lines.append(num)
                if line.endswith(argument):
                    nnp_match = True
            elif (len(nnp_lines) > 0):
                if nnp_match:
                    outlist.append(nnp_lines)
                nnp_lines = []
                nnp_match = False
            if ((not pos in ['nnp', 'nnps']) or NNP_lists) and line.endswith(argument):
                outlist.append(num)
            ## make two item list: list of NNP line numbers, or just one number
            num = 1 + num
    return (outlist)


def make_percent(number):
    return (round(number * 100, 2))


def check_list_item_against_answers(item, answers):
    check = False
    for item2 in item:
        if item2 in answers:
            check = True
    return (check)


def score_file(answer_file, system_file, argument):
    ## first argument is answer key, e.g., %-dev or %-test
    ## second argument is sytem output. The relevant file ends with
    ## the argument we are scoring, e.g., ARG1
    ## the third argument is the argument we are scoring, e.g., ARG1
    answers = get_line_numbers_for_arg(answer_file, argument)
    system = get_line_numbers_for_arg(system_file, argument, NNP_list=True)
    correct = 0
    print('System', system)
    print('Answers', answers)
    for item in system:
        if (type(item) == list):
            if check_list_item_against_answers(item, answers):
                correct += 1
        elif item in answers:
            correct += 1
    print('correct', correct)
    print('lengths', len(system), len(answers))
    ## prevent dividing by zero
    if correct == 0:
        precision = 0
        recall = 0
        f = 0
    else:
        prec_raw = correct / len(system)
        precision = make_percent(prec_raw)
        recall_raw = correct / len(answers)
        recall = make_percent(recall_raw)
        f = make_percent(2 / (1 / prec_raw + 1 / recall_raw))
    print('precision = ', precision)
    print('recall = ', recall)
    print('f-measure = ', f)


def member_match(choices, list_to_match):
    match = False
    for item in choices:
        if item in list_to_match:
            match = True
    return (match)


def score_file_with_NNP_adjustment(answer_file, system_file, argument, score_file=False):
    ## first argument is answer key, e.g., %-dev or %-test
    ## second argument is sytem output. The relevant file ends with
    ## the argument we are scoring, e.g., ARG1
    ## the third argument is the argument we are scoring, e.g., ARG1
    answers = get_line_numbers_for_arg(answer_file, argument, NNP_lists=True)
    system = get_line_numbers_for_arg(system_file, argument, NNP_lists=True)

    system_new = list()
    skip = False
    for i in range(len(system)):
        if skip:
            skip = False
        else:
            if (i < len(system) - 1 and type(system[i + 1]) == list):
                if (system[i] in system[i + 1]):
                    system_new.append(system[i + 1])
                    skip = True
                else:
                    system_new.append(system[i])
            else:
                system_new.append(system[i])

    answers_new = list()
    skip = False
    for i in range(len(answers)):
        if skip:
            skip = False
        else:
            if (i < len(answers) - 1 and type(answers[i + 1]) == list):
                if (answers[i] in answers[i + 1]):
                    answers_new.append(answers[i + 1])
                    skip = True
                else:
                    answers_new.append(answers[i])
            else:
                answers_new.append(answers[i])

    flatMapSystem = list()
    flatMapAnswers = list()

    for entry in system_new:
        if (type(entry)) == list:
            for sub_entry in entry:
                flatMapSystem.append(sub_entry)
        else:
            flatMapSystem.append(entry)

    for entry in answers_new:
        if (type(entry)) == list:
            for sub_entry in entry:
                flatMapAnswers.append(sub_entry)
        else:
            flatMapAnswers.append(entry)

    answers_new = flatMapAnswers
    system_new = flatMapSystem

    correct = 0
    if score_file:
        outstream = open(score_file, 'w')
    else:
        outstream = False
    if outstream:
        outstream.write('System ' + str(system_new) + '\n')
        outstream.write('Answer ' + str(answers_new) + '\n')
    else:
        print('System', system_new)
        print('Answers', answers_new)

    for item in system_new:
        if type(item) == list:
            if member_match(item, answers_new):
                correct += 1
        elif type(item) == tuple:
            if (item in answers_new):
                correct += 1
        elif item in answers_new:
            correct += 1

    if outstream:
        outstream.write('correct ' + str(correct) + '\n')
        outstream.write('lengths ' + str(system_new) + ' ' + str(answers) + '\n')
    else:
        print('correct', correct)
        print('lengths', len(system_new), len(answers_new))
    ## prevent dividing by zero
    if correct == 0:
        precision = 0
        recall = 0
        f = 0
    else:
        prec_raw = correct / len(system_new)
        precision = make_percent(prec_raw)
        recall_raw = correct / len(answers_new)
        recall = make_percent(recall_raw)
        f = make_percent(2 / (1 / prec_raw + 1 / recall_raw))
    if outstream:
        outstream.write('precision = ' + str(precision) + '\n')
        outstream.write('recall = ' + str(recall) + '\n')
        outstream.write('f-measure = ' + str(f) + '\n')
        # close(outstream)
    else:
        print('precision = ', precision)
        print('recall = ', recall)
        print('f-measure = ', f)

train_data_path1 = "data/%_nombank.clean.train"
valid_data_path1 = "data/%_nombank.clean.dev"
test_data_path1 = "data/%_nombank.clean.test"

train_data_path = "data/partitive_group_nombank.clean.train"
valid_data_path = "data/partitive_group_nombank.clean.dev"
test_data_path = "data/partitive_group_nombank.clean.test"

score_file_with_NNP_adjustment("output.txt", test_data_path, "arg1")

score_file_with_NNP_adjustment("result/percent.txt",test_data_path1,"arg1")
#
# score_file_with_NNP_adjustment("result/bert1/percent.txt",test_data_path1,"arg1")

# score_file_with_NNP_adjustment("output.txt",test_data_path1,"arg1")

# score_file("result/partitive.txt",test_data_path,"arg1")