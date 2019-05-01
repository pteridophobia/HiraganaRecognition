import numpy as np
import struct
from PIL import Image, ImageEnhance
from PIL import ImageOps, ImageMath
from matplotlib import pyplot as plt
import bitstring
from sklearn.neural_network import MLPClassifier
import random
from sklearn.ensemble import VotingClassifier as voteC
from sklearn.model_selection import cross_validate

def load_k49():
    data = np.load('k49-test-imgs.npz')
    #lst = data.files
    i = 0
    output = open("out.txt", 'w')
    for item in data:
        #print(item)
        #print(data[item])
        for s in data[item]:
            for n in s:
               #print(n)
                if n > 0:
                    output.write(11)
                else:
                    output.write("")
                for k in n:
                    #print(n)
                    if k > 0:
                        output.write("1")
                    else:
                        output.write(" ")
                    i += 1
                output.write("\n")
            output.write("\n")
    print(i, " is i")
    output.close()


t56s = '0123456789[#@:>? ABCDEFGHI&.](<  JKLMNOPQR-$*);\'|/STUVWXYZ ,%="!'


def read_record_ETL4(f, pos):
    f = bitstring.ConstBitStream(filename=f)
    f.bytepos = pos * 2952
    r = f.readlist('2*uint:36,uint:8,pad:28,uint:8,pad:28,4*uint:6,pad:12,15*uint:36,pad:1008,bytes:21888')
    """
    print('Serial Data Number:', r[0])
    print('Serial Sheet Number:', r[1])
    print('JIS Code:', r[2])
    print('EBCDIC Code:', r[3])
    print('4 Character Code:', ''.join([t56s[c] for c in r[4:8]]))
    print('Evaluation of Individual Character Image:', r[8])
    print('Evaluation of Character Group:', r[9])
    print('Sample Position Y on Sheet:', r[10])
    print('Sample Position X on Sheet:', r[11])
    print('Male-Female Code:', r[12])
    print('Age of Writer:', r[13])
    print('Industry Classification Code:', r[14])
    print('Occupation Classifiaction Code:', r[15])
    print('Sheet Gatherring Date:', r[16])
    print('Scanning Date:', r[17])
    print('Number of X-Axis Sampling Points:', r[18])
    print('Number of Y-Axis Sampling Points:', r[19])
    print('Number of Levels of Pixel:', r[20])
    print('Magnification of Scanning Lens:', r[21])
    print('Serial Data Number (old):', r[22])
    """
    return r


filename = '/home/phobia/Desktop/Hiraganarecognition/ETL4C'
#filename = 'C:\\Users\\Pteridophobia\\Downloads\\k49-test-imgs.npz\\ETL4C'  # specify the ETL4 filename here
A = [([], )]
I = [([], )]
U = [([], )]
E = [([], )]
O = [([], )]
KA = [([], )]
KI = [([], )]
KU = [([], )]
KE = [([], )]
KO = [([], )]
SA = [([], )]
SHI = [([], )]
SU = [([], )]
SE = [([], )]
SO = [([], )]
TA = [([], )]
CHI = [([], )]
TSU = [([], )]
TE = [([], )]
TO = [([], )]
NA = [([], )]
NI = [([], )]
NU = [([], )]
NE = [([], )]
NO = [([], )]
HA = [([], )]
HI = [([], )]
FU = [([], )]
HE = [([], )]
HO = [([], )]
MA = [([], )]
MI = [([], )]
MU = [([], )]
ME = [([], )]
MO = [([], )]
YA = [([], )]
YU = [([], )]
YO = [([], )]
RA = [([], )]
RI = [([], )]
RU = [([], )]
RE = [([], )]
RO = [([], )]
WA = [([], )]
WI = [([], )]
WO = [([], )]
NN = [([], )]
MasterData = [([], )]
iter = 0
r = read_record_ETL4(filename, iter)
total = 0
while total < 6120 - 51:
    r = read_record_ETL4(filename, total)
    iF = Image.frombytes('F', (r[18], r[19]), r[-1], 'bit', 4)
    # iP = iF.convert('P')
    # enhancer = ImageEnhance.Brightness(iP)
    # iE = enhancer.enhance(16.0)
    data = np.asarray(iF, dtype="int32")
    plt.imshow(iF)
    #plt.show()
    binary_img = []
    for arr in data:
        new = []
        for pixel in arr:

            if pixel > 5:
                new.append(1.0)
            else:
                new.append(0.0)
       # print(new)
        binary_img.append(new)
    if iter == 0:
        A.append((binary_img, "A"))
        MasterData.append((binary_img, "A"))
    elif iter == 1:
        I.append((binary_img, "I"))
        MasterData.append((binary_img, "I"))
    elif iter == 2:
        U.append((binary_img, "U"))
        MasterData.append((binary_img, "U"))
    elif iter == 3:
        E.append((binary_img, "E"))
        MasterData.append((binary_img, "E"))
    elif iter == 4:
        O.append((binary_img, "O"))
        MasterData.append((binary_img, "O"))
    elif iter == 5:
        KA.append((binary_img, "KA"))
        MasterData.append((binary_img, "KA"))
    elif iter == 6:
        KI.append((binary_img, "KI"))
        MasterData.append((binary_img, "KI"))
    elif iter == 7:
        KU.append((binary_img, "KU"))
        MasterData.append((binary_img, "KU"))
    elif iter == 8:
        KE.append((binary_img, "KE"))
        MasterData.append((binary_img, "KE"))
    elif iter == 9:
        KO.append((binary_img, "KO"))
        MasterData.append((binary_img, "KO"))
    elif iter == 10:
        SA.append((binary_img, "SA"))
        MasterData.append((binary_img, "SA"))
    elif iter == 11:
        SHI.append((binary_img, "SHI"))
        MasterData.append((binary_img, "SHI"))
    elif iter == 12:
        SU.append((binary_img, "SU"))
        MasterData.append((binary_img, "SU"))
    elif iter == 13:
        SE.append((binary_img, "SE"))
        MasterData.append((binary_img, "SE"))
    elif iter == 14:
        SO.append((binary_img, "SO"))
        MasterData.append((binary_img, "SO"))
    elif iter == 15:
        TA.append((binary_img, "TA"))
        MasterData.append((binary_img, "TA"))
    elif iter == 16:
        CHI.append((binary_img, "CHI"))
        MasterData.append((binary_img, "CHI"))
    elif iter == 17:
        TSU.append((binary_img, "TSU"))
        MasterData.append((binary_img, "TSU"))
    elif iter == 18:
        TE.append((binary_img, "TE"))
        MasterData.append((binary_img, "TE"))
    elif iter == 19:
        TO.append((binary_img, "TO"))
        MasterData.append((binary_img, "TO"))
    elif iter == 20:
        NA.append((binary_img, "NA"))
        MasterData.append((binary_img, "NA"))
    elif iter == 21:
        NI.append((binary_img, "NI"))
        MasterData.append((binary_img, "NI"))
    elif iter == 22:
        NU.append((binary_img, "NU"))
        MasterData.append((binary_img, "NU"))
    elif iter == 23:
        NE.append((binary_img, "NE"))
        MasterData.append((binary_img, "NE"))
    elif iter == 24:
        NO.append((binary_img, "NO"))
        MasterData.append((binary_img, "NO"))
    elif iter == 25:
        HA.append((binary_img, "HA"))
        MasterData.append((binary_img, "HA"))
    elif iter == 26:
        HI.append((binary_img, "HI"))
        MasterData.append((binary_img, "HI"))
    elif iter == 27:
        FU.append((binary_img, "FU"))
        MasterData.append((binary_img, "FU"))
    elif iter == 28:
        HE.append((binary_img, "HE"))
        MasterData.append((binary_img, "HE"))
    elif iter == 29:
        HO.append((binary_img, "HO"))
        MasterData.append((binary_img, "HO"))
    elif iter == 30:
        MA.append((binary_img, "MA"))
        MasterData.append((binary_img, "MA"))
    elif iter == 31:
        MI.append((binary_img, "MI"))
        MasterData.append((binary_img, "MI"))
    elif iter == 32:
        MU.append((binary_img, "MU"))
        MasterData.append((binary_img, "MU"))
    elif iter == 33:
        ME.append((binary_img, "ME"))
        MasterData.append((binary_img, "ME"))
    elif iter == 34:
        MO.append((binary_img, "MO"))
        MasterData.append((binary_img, "MO"))
    elif iter == 35:
        YA.append((binary_img, "YA"))
        MasterData.append((binary_img, "YA"))
    elif iter == 36:
        I.append((binary_img, "I"))
        MasterData.append((binary_img, "I"))
    elif iter == 37:
        YU.append((binary_img, "YU"))
        MasterData.append((binary_img, "YU"))
    elif iter == 38:
        E.append((binary_img, "E"))
        MasterData.append((binary_img, "E"))
    elif iter == 39:
        YO.append((binary_img, "YO"))
        MasterData.append((binary_img, "YO"))
    elif iter == 40:
        RA.append((binary_img, "RA"))
        MasterData.append((binary_img, "RA"))
    elif iter == 41:
        RI.append((binary_img, "RI"))
        MasterData.append((binary_img, "RI"))
    elif iter == 42:
        RU.append((binary_img, "RU"))
        MasterData.append((binary_img, "RU"))
    elif iter == 43:
        RE.append((binary_img, "RE"))
        MasterData.append((binary_img, "RE"))
    elif iter == 44:
        RO.append((binary_img, "RO"))
        MasterData.append((binary_img, "RO"))
    elif iter == 45:
        WA.append((binary_img, "WA"))
        MasterData.append((binary_img, "WA"))
    elif iter == 47:
        U.append((binary_img, "U"))
        MasterData.append((binary_img, "U"))
    elif iter == 49:
        WO.append((binary_img, "WO"))
        MasterData.append((binary_img, "WO"))
    elif iter == 50:
        NN.append((binary_img, "NN"))
        MasterData.append((binary_img, "NN"))
    iter += 1
    total += 1
    if iter == 51:
        iter = 0
    if total == 6120:
        iter = 51


char = ""
MasterData = MasterData[1:]
#for tup in MasterData:
#    print(tup[1])
#    for row in tup[0]:
#       print(row)



"""
for item in training:
    train_class.append(item[1])
    temp = []
    for row in item[0]:
        #temp.append(row)
        for pix in row:
            temp.append(pix)
            #train_features.append(pix)
    train_features.append(temp)

for item in testing:
    test_class.append(item[1])
    temp = []
    for row in item[0]:
        for pix in row:
            temp.append(pix)
            #test_features.append(pix)
        #temp.append(row)
    test_features.append(temp)
"""


def predict_for_char(char, MasterData):
    random.shuffle(MasterData)
    training = MasterData[:int(float(len(MasterData) * .66666666667))]
    testing = MasterData[int(float(len(MasterData) * .66666666667)):]
    train_features = []
    train_class = []
    test_features = []
    test_class = []
    for item in training:
        if item[1] == char:
            train_class.append(1.0)
        else:
            train_class.append(0.0)
        temp = []
        for row in item[0]:
            for pix in row:
                temp.append(pix)
        train_features.append(temp)

    for item in testing:
        if item[1] == char:
            test_class.append(1.0)
        else:
            test_class.append(0.0)
        temp = []
        for row in item[0]:
            for pix in row:
                temp.append(pix)
        test_features.append(temp)

    print("Training")
    clf = MLPClassifier(solver='adam', alpha=1e-05, hidden_layer_sizes=(5,2), max_iter=333,  random_state=3)
    clf.fit(train_features, train_class)
    clf.predict(test_features)
    acc = clf.score(test_features, test_class)
    print("Accuracy for predicting if the character is: ",char, " :", acc)
    return acc, clf


accuracies = []
CLFs = []

app = 0
app, clfA = predict_for_char("A", MasterData)
accuracies.append(app)
CLFs.append(clfA)
app, clfI = predict_for_char("I", MasterData)
accuracies.append(app)
CLFs.append(clfI)
app, clfU = predict_for_char("U", MasterData)
accuracies.append(app)
CLFs.append(clfU)
app, clfE = predict_for_char("E", MasterData)
accuracies.append(app)
CLFs.append(clfE)
app, clfO = predict_for_char("O", MasterData)
accuracies.append(app)
CLFs.append(clfO)
app, clfKA = predict_for_char("KA", MasterData)
accuracies.append(app)
CLFs.append(clfKA)
app, clfKI = predict_for_char("KI", MasterData)
accuracies.append(app)
CLFs.append(clfKI)
app, clfKU = predict_for_char("KU", MasterData)
accuracies.append(app)
CLFs.append(clfKU)
app, clfKE = predict_for_char("KE", MasterData)
accuracies.append(app)
CLFs.append(clfKE)
app, clfKO = predict_for_char("KO", MasterData)
accuracies.append(app)
CLFs.append(clfKO)
app, clfSA = predict_for_char("SA", MasterData)
accuracies.append(app)
CLFs.append(clfSA)
app, clfSHI = predict_for_char("SHI", MasterData)
accuracies.append(app)
CLFs.append(clfSHI)
app, clfSU = predict_for_char("SU", MasterData)
accuracies.append(app)
CLFs.append(clfSU)
app, clfSE = predict_for_char("SE", MasterData)
accuracies.append(app)
CLFs.append(clfSE)
app, clfSO = predict_for_char("SO", MasterData)
accuracies.append(app)
CLFs.append(clfSO)
app, clfTA = predict_for_char("TA", MasterData)
accuracies.append(app)
CLFs.append(clfTA)
app, clfCHI = predict_for_char("CHI", MasterData)
accuracies.append(app)
CLFs.append(clfCHI)
app, clfTSU = predict_for_char("TSU", MasterData)
accuracies.append(app)
CLFs.append(clfTSU)
app, clfTE = predict_for_char("TE", MasterData)
accuracies.append(app)
CLFs.append(clfTE)
app, clfTO = predict_for_char("TO", MasterData)
accuracies.append(app)
CLFs.append(clfTO)
app, clfNA = predict_for_char("NA", MasterData)
accuracies.append(app)
CLFs.append(clfNA)
app, clfNI = predict_for_char("NI", MasterData)
accuracies.append(app)
CLFs.append(clfNI)
app, clfNU = predict_for_char("NU", MasterData)
accuracies.append(app)
CLFs.append(clfNU)
app, clfNE = predict_for_char("NE", MasterData)
accuracies.append(app)
CLFs.append(clfNE)
app, clfNO = predict_for_char("NO", MasterData)
accuracies.append(app)
CLFs.append(clfNO)
app, clfHA = predict_for_char("HA", MasterData)
accuracies.append(app)
CLFs.append(clfHA)
app, clfHI = predict_for_char("HI", MasterData)
accuracies.append(app)
CLFs.append(clfHI)
app, clfFU = predict_for_char("FU", MasterData)
accuracies.append(app)
CLFs.append(clfFU)
app, clfHE = predict_for_char("HE", MasterData)
accuracies.append(app)
CLFs.append(clfHE)
app, clfHO = predict_for_char("HO", MasterData)
accuracies.append(app)
CLFs.append(clfHO)
app, clfMA = predict_for_char("MA", MasterData)
accuracies.append(app)
CLFs.append(clfMA)
app, clfMI = predict_for_char("MI", MasterData)
accuracies.append(app)
CLFs.append(clfMI)
app, clfMU = predict_for_char("MU", MasterData)
accuracies.append(app)
CLFs.append(clfMU)
app, clfME = predict_for_char("ME", MasterData)
accuracies.append(app)
CLFs.append(clfME)
app, clfMO = predict_for_char("MO", MasterData)
accuracies.append(app)
CLFs.append(clfMO)
app, clfYA = predict_for_char("YA", MasterData)
accuracies.append(app)
CLFs.append(clfYA)
app, clfYU = predict_for_char("YU", MasterData)
accuracies.append(app)
CLFs.append(clfYU)
app, clfYO = predict_for_char("YO", MasterData)
accuracies.append(app)
CLFs.append(clfYO)
app, clfRA = predict_for_char("RA", MasterData)
accuracies.append(app)
CLFs.append(clfRA)
app, clfRI = predict_for_char("RI", MasterData)
accuracies.append(app)
CLFs.append(clfRI)
app, clfRU = predict_for_char("RU", MasterData)
accuracies.append(app)
CLFs.append(clfRU)
app, clfRE = predict_for_char("RE", MasterData)
accuracies.append(app)
CLFs.append(clfRE)
app, clfRO = predict_for_char("RO", MasterData)
accuracies.append(app)
CLFs.append(clfRO)
app, clfWA = predict_for_char("WA", MasterData)
accuracies.append(app)
CLFs.append(clfWA)
app, clfWO = predict_for_char("WO", MasterData)
accuracies.append(app)
CLFs.append(clfWO)
app, clfNN = predict_for_char("NN", MasterData)
accuracies.append(app)
CLFs.append(clfNN)


def predict_char(chars, CLFs):
    char_dict = {1: 'A',
                 2: 'I',
                 3: 'U' ,
                 4: 'E' ,
                 5: 'O' ,
                 6: 'KA',
                 7: 'KI',
                 8: 'KU',
                 9: 'KE',
                 10: 'KO',
                 11: 'SA',
                 12: 'SHI',
                 13: 'SU',
                 14: 'SE',
                 15: 'S0',
                 16: 'TA',
                 17: 'CHI',
                 18: 'TSU',
                 19: 'TE',
                 20: 'TO',
                 21: 'NA',
                 22: 'NI',
                 23: 'NU',
                 24: 'NE',
                 25: 'NO',
                 26: 'HA',
                 27: 'HI',
                 28: 'FU',
                 29: 'HE',
                 30: 'HO',
                 31: 'MA',
                 32: 'MI',
                 33: 'MU',
                 34: 'ME',
                 35: 'MO',
                 36: 'YA',
                 37: 'YU',
                 38: 'YO',
                 39: 'RA',
                 40: 'RI',
                 41: 'RU',
                 42: 'RE',
                 43: 'RO',
                 44: 'WA',
                 45: 'WO',
                 46: 'NN'
                 }
    ret = []
    corr = 0
    tot = 0
    for classif in CLFs:
        for cha in chars:
            ret.append(classif.predict(cha))
    results = zip(ret, chars)
    for result, char in results:
        matches = 0
        i = 1
        for value in result:
            if value == 1.0:
                matches += 1
            i +=1
        if matches == 1:
            print("The character was predicted correctly as:", char[1])
            corr += 1
            tot += 1
        elif matches > 1:
            print("There were ", matches ," different predicted characters when real character was ", char[1])
            tot += 1
        else:
            print("There were no recognized characters")
            tot += 1
    return matches/tot


def voting(__clf, x_data, y_label):
    vote = voteC(__clf)
    acc = cross_validate(vote, x_data, y_label, cv=3)['test_score'].tolist()
    sum_score = 0
    i = 1
    for score in acc:
        print("Fold num: ", i, " = ", score)
        sum_score += score
        i += 1
    print("Voting Average: ", sum_acc(acc)/len(acc))
    return sum(acc)/len(acc)



sum_acc = sum(accuracies)
avg_acc = sum_acc / len(accuracies)
print("Average accuracy is:", avg_acc)

print("predicting 50 random characters:\n")
j = 0
num_correct = 0
random.shuffle(MasterData)
chars_to_predict = []


for item in MasterData:
    while j < 50:
        temp = []
        for row in item[0]:
            for pix in row:
                temp.append(pix)
        chars_to_predict.append(temp)
        j +=1
#_50_acc = predict_char(chars_to_predict, CLFs)
#print("The accuracy for predicting 50 random samples is: ", _50_acc)
x_d = []
y_d = []
for item__ in MasterData:
    x_d.append(item__[0])
    y_d.append(item__[1])
voting(CLFs, x_d, y_d)

print("DONE")