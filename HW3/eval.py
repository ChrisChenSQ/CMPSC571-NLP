import time

from pyrouge import Rouge155
import os

def rouge(name):
    r = Rouge155()
    r.system_dir = 'System_Summaries/{}'.format(name)
    r.model_dir = 'Human_Summaries/eval'
    r.system_filename_pattern = 'd(\d+)t.{}'.format(name)
    r.model_filename_pattern = 'D#ID#.M.100.T.[A-Z]'

    output = r.convert_and_evaluate()
    output_dict = r.output_to_dict(output)
    return output_dict

def printResult(outDic):
    linestr = "|"
    for s in ["1","2","l"]:
        for i in [fName.format(s) for fName in ["rouge_{}_recall", "rouge_{}_precision","rouge_{}_f_score"]]:
            linestr += "{0:10}\t".format(outDic[i])
        linestr += "|"
    return linestr
finalstr = "|-----------------------------------------------------------------------------------------------------------------------|\n"\
           "|          |              ROUGE-1                |            ROUGE-2                |             ROUGE-L              |\n"\
           "|==========|============================================================================================================|\n"\
           "|System    |   P(%)         R(%)        F(%)     |   P(%)       R(%)        F(%)     |   P(%)       R(%)        F(%)    |\n"\
           "|==========|============================================================================================================|\n"
time_str = ""
for i in os.listdir("System_Summaries"):
    start = time.time()
    if i == ".DS_Store":
        continue
    finalstr += "|{0:10}".format(i)
    finalstr += "{}\n".format(printResult(rouge(i)))
    end = time.time()
    time_str += "{}: {}\n".format(i, end-start)
print(finalstr)
print(time_str)