import os
import re
import time
from pathlib import Path
import spacy
import json


def writefromFile(path, type, label):
    fileList = os.listdir(path)
    nlp = spacy.load("en_core_web_sm")
    with open("{}.{}.review".format(label, type), "w") as file:
        file.truncate(0)
        for fileRead in fileList:
            print("Start processing file {}/{}...".format(path, fileRead))
            try:
                txt = Path("{}/{}".format(path, fileRead)).read_text()
            except:
                continue
            doc = nlp(txt)
            wordDic = {}
            writestr = ""
            for token in doc:
                if not token.is_stop and token.pos_ not in ["PUNCT", "X", "_SP"] and "@" not in token.text:
                    word = re.sub(r"[^a-zA-Z0-9]", "", token.lemma_)
                    if word != "":
                        if word not in wordDic:
                            wordDic[word] = 1
                        else:
                            wordDic[word] += 1

            for word in wordDic:
                writestr += "{}:{} ".format(word, wordDic[word])
            writestr += "#label#:{}\n".format(label)
            file.write(writestr)
            print("Write file {} complete!".format(fileRead))


start = time.time()
writefromFile("20news-bydate/20news-bydate-train/rec.autos", "train", "positive")
writefromFile("20news-bydate/20news-bydate-train/comp.sys.mac.hardware", "train", "negative")

writefromFile("20news-bydate/20news-bydate-test/rec.autos", "test", "positive")
writefromFile("20news-bydate/20news-bydate-test/comp.sys.mac.hardware", "test", "negative")

end = time.time()
print("The total runtime is: {}".format(end-start))