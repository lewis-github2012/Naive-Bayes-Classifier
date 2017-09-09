# -*- coding: utf-8 -*-

from numpy import *
import feedparser
import sys
default_encoding = 'utf-8'
if sys.getdefaultencoding() != default_encoding:
    reload(sys)
    sys.setdefaultencoding(default_encoding)

# 4-1
# 词表到向量的转换函数
def loadDataSet():
    postingList = [['my','dog','has','flea','problems','help','please'],
                   ['maybe','not','take','him','to','dog','park','stupid'],
                   ['my','dalmation','is','so','cute','I','love','him'],
                   ['stop','posting','stupid','worthless','garbage'],
                   ['mr','licks','ate','my','steak','how','to','stop','him'],
                   ['quit','buying','worthless','dog','food','stupid']
                   ]
    classVec = [0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet = set([]) #set([])为“集合”类型，它是无序无重复元素的集合
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def setOfWords2Vec(vocabList,inputSet):
    returnVec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary" % word
    return returnVec
'''        
## 测试转换函数             
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
print myVocabList      
print setOfWords2Vec(myVocabList,listOPosts[0])
print setOfWords2Vec(myVocabList,listOPosts[3])
'''

# 4-2
# 朴素贝叶斯分类器训练函数
def trainNB0(trainMatrix,trainCategory):
    #trainMatrix为文档向量化的训练样本列表；trainCategory为类别向量
    numTrainDocs = len(trainMatrix)  #训练集中文本个数
    numWords = len(trainMatrix[0])   #词个数。因为所有文本都向量化为等长的向量了
    pAbusive = sum(trainCategory)/float(numTrainDocs) #分类为1的文本所占比例，也就是概率
    # sum为向量求和，trainCategory为类向量，并且类别只有1和0
    # 所以求和就计算出了分类为1的样本数
    # 再除以样本总数就得到相应的比例，也就是概率
    p0Num = ones(numWords)  # 属于类0的所有词的统计向量的初始化
    p1Num = ones(numWords)  # 属于类1的所有词的统计向量的初始化
    p0Denom = 2.0 # 类0的词总数初始化
    p1Denom = 2.0 # 类1的词总数初始化
    for i in range(numTrainDocs):     #遍历原始样本集
        if trainCategory[i] == 1:     #如果为类1
            p1Num += trainMatrix[i]   #则该文本向量中出现的所有词的统计量都相应加1
            p1Denom +=sum(trainMatrix[i])  #相应的词总数也增加
        else:
            p0Num += trainMatrix[i]
            p0Denom +=sum(trainMatrix[i])
    p1Vect = log(p1Num/p1Denom)  #类1各词的出现比例，即概率
    p0Vect = log(p0Num/p0Denom)  #类0各词的出现比例，即概率
    return p0Vect,p1Vect,pAbusive

'''
## 测试trainNB0()函数 
listOPosts,listClasses = loadDataSet()
myVocabList = createVocabList(listOPosts)
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(setOfWords2Vec(myVocabList,postinDoc))
p0V,p1V,pAb = trainNB0(trainMat,listClasses)
print pAb
print p0V
print p1V    
'''
# 4-3
# 朴素贝叶斯分类函数
def classifyNB(vec2Classify,p0Vec,p1Vec,pClass1):
    #vec2Classify待分类向量,p0Vec,p1Vec,pClass1三个先验概率
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

###   
def testingNB():
    listOPost,listClasses = loadDataSet()
    #listOPost训练集（原始数据集），listClasses对应的类向量
    myVocablist = createVocabList(listOPost) #由训练集创建包含所有词的词汇表
    trainMat = []
    for postinDoc in listOPost:  #遍历训练集
        trainMat.append(setOfWords2Vec(myVocablist,postinDoc))
        #先通过setOfWords2Vec（）将单一文本样本转换为向量
        #trainMat为由向量作为元素的列表
    p0V,p1V,pAb = trainNB0(array(trainMat),array(listClasses))
    #计算先验概率
    testEntry = ['love','my','dalmation'] #测试集
    thisDoc = array(setOfWords2Vec(myVocablist,testEntry)) #将测试集转换成向量
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
    testEntry = ['stupid','garbage']
    thisDoc = array(setOfWords2Vec(myVocabList,testEntry))
    print testEntry,'classified as:',classifyNB(thisDoc,p0V,p1V,pAb)
                
# 4-4
# 朴素贝叶斯词袋模型
def bagOfWords2VecMN(vocabList,inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
	
'''
######   使用朴素贝叶斯过滤垃圾邮件   ######
'''

# 4-5
# 文本解析及完整的垃圾邮件测试函数

def textParse(bigString):#分词处理
    import re
    listOfTokens = re.split(r'\W*',bigString) #分词
    return [tok.lower() for tok in listOfTokens if len(tok) > 2] #去空格，变换大小写

def spamTest():
    docList = []
    classList = []
    fullText = []
    for i in range(1,26):
        wordList = textParse(open('email/spam/%d.txt' % i).read())#spam文件夹里的都是垃圾邮件
        docList.append(wordList)  #以一个完整的wordList作为元素添加到docList中
        fullText.extend(wordList) #把wordList中的元素添加到fullText列表后
        classList.append(1)       #添加对应的类标号
        wordList = textParse(open('email/ham/%d.txt' % i).read())#ham文件夹中的邮件都是正常有家
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)  #创建词向量
    trainingSet = range(50)  
    testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet))) #从原始数据集中随机选择序号，作为测试集
        testSet.append(trainingSet[randIndex]) #测试集序号
        del(trainingSet[randIndex])            #从原始数据集中删除作为测试集的序号，剩下的都是训练集
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet: #遍历训练集
        trainMat.append(setOfWords2Vec(vocabList,docList[docIndex])) #将训练集中的文本转换为向量，并添加到训练集的列表中
        trainClasses.append(classList[docIndex])  #训练集对应的类标号
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses)) #计算先验概率
    errorCount = 0  #错误计数
    for docIndex in testSet:  #遍历测试集
        wordVector = setOfWords2Vec(vocabList,docList[docIndex])  #转换为词向量
        if classifyNB(array(wordVector),p0V,p1V,pSpam) != classList[docIndex]: #进行分类，并判断分类是否正确
            errorCount += 1
            print docList[docIndex]
    print 'the error rate is: ',float(errorCount)/len(testSet)

#spamTest()




'''
############     使用朴素贝叶斯从个人广告中获取区域倾向            #########
'''
# 4-6
# RSS源分类器及高频词去除函数
def calcMostFreq(vocabList,fullText):
    import operator
    freqDict = {}
    for token in vocabList:                      #遍历vocabList
        freqDict[token] = fullText.count(token)  #统计各词出现次数
    sortedFreq = sorted(freqDict.iteritems(),key=operator.itemgetter(1),reverse=True)  #排序
    return sortedFreq[:30]



def localWords(feed1,feed0): #feed1,feed0是RSS源数据的地址
    import feedparser
    docList=[]
    classList=[]
    fullText=[]
    minLen = min(len(feed1['entries']),len(feed0['entries']))
    for i in range(minLen):
        wordList = textParse(feed1['entries'][i]['summary']) #调用textParse（）进行分词
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList = textParse(feed0['entries'][i]['summary']) #调用textParse（）进行分词
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)          #转换成向量
    top30Words = calcMostFreq(vocabList,fullText) #获取高频词
    for pairW in top30Words:           #去除高频词
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet = range(2*minLen)
    testSet = []
    for i in range(20):  #形成测试集
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat = []
    trainClasses = []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(array(trainMat),array(trainClasses))  #计算先验概率
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(array(wordVector),p0V,p1V,pSpam)  != classList[docIndex]:
            errorCount += 1
    print 'the error rate is: ',float(errorCount)/len(testSet)
    return vocabList,p0V,p1V

'''
## 测试localWords()函数  
ny = feedparser.parse('http://newyork.craigslist.org/stp/index.rss')
sf = feedparser.parse('http://sfbay.craigslist.org/stp/index.rss')
vocabList,pSF,pNY = localWords(ny,sf)
'''
    
# 4-7
# 最具表征性的词汇显示函数

def getTopWords(ny,sf):
    import operator
    vocabList,p0V,p1V = localWords(ny,sf)
    topNY = []
    topSF = []
    for i in range(len(p0V)):
        if p0V[i] > -4.5:
            topSF.append((vocabList[i],p0V[i]))
        if p1V[i] > -4.5:
            topNY.append((vocabList[i],p1V[i]))
    sortedSF = sorted(topSF,key=lambda pair:pair[1],reverse=True)
    print "SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF**SF"
    for item in sortedSF:
        print item[0]
    sortedNY = sorted(topNY,key=lambda pair:pair[1],reverse=True)
    print "NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY**NY"
    for item in sortedNY:
        print item[0]

'''
##测试getTopWords()函数 
getTopWords(ny,sf)
'''
    
    
    
        
        
        
        
        
        