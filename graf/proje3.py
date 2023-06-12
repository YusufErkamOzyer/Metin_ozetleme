from PyQt5.QtWidgets import *
from interface import Ui_MainWindow
from PyQt5.QtGui import QPainter, QBrush, QPen,QFont
from PyQt5.QtCore import Qt
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import spacy
from math import log10
import string
import networkx as nx
from functools import partial
from sentence_transformers import SentenceTransformer, util
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from sklearn.metrics.pairwise import cosine_similarity
from nltk.util import ngrams

sentences=""
pos=0
text1=0
text2=0
text=0
x1=0;y1=0;x2=0;y2=0;x=0;y=0
pen = QPen(Qt.black, 3, Qt.SolidLine)
pen_purple=QPen(Qt.magenta, 3, Qt.SolidLine)
brush_yellow = QBrush(Qt.yellow, Qt.SolidPattern)
brush = QBrush(Qt.gray, Qt.SolidPattern)
brush_white = QBrush(Qt.white, Qt.SolidPattern)
agirlik=0
font = QFont('Arial', 15)
rect1x=[]
rect1y=[]
rect2x=[]
rect2y=[]
ellipsex=[]
ellipsey=[]
linex1=[]
linex2=[]
liney1=[]
liney2=[]
cumle_skorlari1=[]
G = nx.Graph()
nltk.download('stopwords')
noktalama_isaretleri = set(string.punctuation)
tema_kelimeleri=[]
cumle_skorlari=[]
def noktalama_kaldir(text):
    return ''.join(char for char in text if char not in noktalama_isaretleri)

def calculate_tf(key,cumleler,cumle):
    TF=0
    count=0
    for sentence in cumleler:
        if sentence==cumle:
            cumle=word_tokenize(cumle)
            for kelime in cumle:
                if key.lower()==kelime.lower():
                    count=count+1
    TF=float(count/len(cumle))
    return TF   
def calculate_idf(key,cumleler):
    IDF=0
    DF=0
    count=0
    for cumle in cumleler:
        kelimeler=word_tokenize(cumle)
        for kelime in kelimeler:
            if kelime.lower()==key.lower():
                count=count+1
                break
    DF=float(len(cumleler)/count)
    IDF=log10(DF)
    return IDF             
nlp = spacy.load("en_core_web_sm")

def is_proper_noun(word):
    doc = nlp(word)
    for token in doc:
        if token.pos_ != "PROPN":
            return False
    return True

def p1(sentence):
    
    words = word_tokenize(sentence)
    uzunluk=len(words)
    counter=0
    for word in words:
        if is_proper_noun(word):
            counter=counter+1
    return float(counter/uzunluk)
    
def p2(sentence):
    uzunluk=len(sentence)
    words = word_tokenize(sentence)
    counter=0
    for word in words:
        if word.isdigit():
            counter=counter+1
    return float(counter/uzunluk)

    
def p4(sentence,up):
    words=word_tokenize(sentence)
    baslik=word_tokenize(up)
    uzunluk=len(sentence)
    counter=0
    for i in baslik:
        for word in words:
            if i.lower()==word.lower():
                counter=counter+1
    return float(counter/uzunluk)
def p5(sentence):
    uzunluk=len(sentence)
    words = word_tokenize(sentence)
    counter=0
    for word in words:
        if word in tema_kelimeleri:
            counter=counter+1
    return float(counter/uzunluk)


def calculate_rouge_score(system_output, reference):
    system_tokens = system_output.split()
    reference_tokens = reference.split()
    
    system_grams = set(ngrams(system_tokens, 1))
    reference_grams = set(ngrams(reference_tokens, 1))
    
    intersection = system_grams.intersection(reference_grams)
    precision = len(intersection) / len(system_grams)
    recall = len(intersection) / len(reference_grams)
    
    if precision + recall > 0:
        f1_score = (2 * precision * recall) / (precision + recall)
    else:
        f1_score = 0.0
    
    return f1_score   



def calculat_TF_IDF(sentences):
    TF_IDF=[]
    for sentence in sentences:
        kelimeler=word_tokenize(sentence)
        for kelime in kelimeler:
            TF=calculate_tf(kelime,sentences,sentence)
            IDF=calculate_idf(kelime,sentences)
            TF_IDF.append(TF*IDF)
    return TF_IDF
def similarity(cumle1,cumle2):
        # Önceden eğitilmiş bir modeli yükle
        model_name = 'bert-base-nli-mean-tokens'
        model = SentenceTransformer(model_name)

        # İki cümleyi oluştur
        sentence1 = cumle1
        sentence2 = cumle2

        # Cümleleri vektörlere dönüştür
        embedding1 = model.encode(sentence1, convert_to_tensor=True)
        embedding2 = model.encode(sentence2, convert_to_tensor=True)

        # İki cümle arasındaki benzerlik skorunu hesapla
        bert_score = util.pytorch_cos_sim(embedding1, embedding2).item()
        #print("Cümleler arasındaki benzerlik skoru:", cosine_score.item())
        
        
        # Cümleleri oluştur
        sentence1 = cumle1
        sentence2 = cumle2

        # Kelimeleri ayır ve küçük harflere dönüştür
        tokens1 = simple_preprocess(sentence1)
        tokens2 = simple_preprocess(sentence2)

        # Kelime vektör modelini oluştur
        model = Word2Vec([tokens1, tokens2], min_count=1, vector_size=500)

        # Kelime vektörlerini al
        vectors1 = [model.wv[word] for word in tokens1]
        vectors2 = [model.wv[word] for word in tokens2]

        # Cümleler arasındaki benzerlik skorunu hesapla
        embedding_score = cosine_similarity(vectors1, vectors2).mean()
        #print("Cümleler arasındaki benzerlik skoru:", similarity_score)
        similarity_score=float((bert_score+embedding_score)/2)
        return similarity_score

                       
            
p3_values=[]
class main(QMainWindow):
    def __init__(self,G) -> None:
        super().__init__()
        self.G=G
        self.arayuz=Ui_MainWindow()
        self.arayuz.setupUi(self)
        scene = QGraphicsScene()
        self.similar_button_count=0
        self.summery_sentences_list=[]
        view = QGraphicsView(scene,self)
        view.setGeometry(800,100, 824, 800)
        self.arayuz.pushButton.clicked.connect(partial(self.upload_folder,scene,view,self.G))
        self.arayuz.pushButton_2.clicked.connect(partial(self.similar_button,scene,view,self.G))
        self.arayuz.pushButton_3.clicked.connect(partial(self.sentence_score,scene,view,self.G))
        self.arayuz.ozet_button.clicked.connect(self.calculating_rouge_score)
    cumle_skorlari=[]
    
    def reset(self,scene,view,G):
        scene.clear()
        viewport = view.viewport()  # QGraphicsView'in görünüm alanını al
        items = viewport.findChildren(QGraphicsItem)  # Görünüm alanındaki tüm öğeleri bul
        for item in items:
            scene.removeItem(item)  # Öğeleri kaldır

        for index, edge in enumerate(G.edges()):
            scene.addLine(linex1[index],liney1[index],linex2[index],liney2[index],pen)
            weight = G.edges[edge]['weight']
            agirlik=scene.addText(str(weight)[:4],font)
            agirlik.setPos((linex1[index]+linex2[index])/2,(liney1[index]+liney2[index])/2)
            
        for index, node in enumerate(G.nodes()):
            scene.addRect(rect1x[index],rect1y[index],40,40,pen,brush_white)
            scene.addRect(rect2x[index],rect2y[index],40,40,pen,brush_white)
            scene.addEllipse(ellipsex[index],ellipsey[index],40,40,pen,brush)
            text = scene.addText(str(node), font)
            text.setPos(ellipsex[index],ellipsey[index])
            
        
    def upload_folder(self,scene,view,G):
        fname=QFileDialog.getOpenFileName(self,'Open File','/home')
        if fname[0]:
            with open(fname[0],'r') as f:
                data=f.read()
                #cümlelere ayırma
                sentences = sent_tokenize(data)
                #başlığı belirleme
                baslik=sentences[0].split("\n")
                baslik.remove('')
                sentences[0]=baslik[1]
                self.summery_sentences_list=sentences 
            
                sentences = [noktalama_kaldir(text) for text in sentences]
                baslik.remove(baslik[1])
                baslik=baslik[0]
                for i in sentences:
                    cumle_skorlari.append(float(0))
                j=0
                for sentence in sentences:
                    cumle_skorlari[j]= cumle_skorlari[j]+p1(sentence)
                    cumle_skorlari[j]= cumle_skorlari[j]+p2(sentence)
                    j=j+1
                
                new_sentences=[]
                for sentence in sentences:
                    words = nltk.word_tokenize(sentence)
                    stop_words = set(stopwords.words('english'))
                    filtered_words = [word for word in words if word.casefold() not in stop_words]
                    filtered_string = ' '.join(filtered_words)
                    new_sentences.append(filtered_string)
                sentences=new_sentences
                j=0
                for sentence in sentences:
                    cumle_skorlari[j]= cumle_skorlari[j]+p4(sentence,baslik)
                    j=j+1
                TF_IDF=calculat_TF_IDF(sentences)
                kelime_sayisi=len(TF_IDF)
                tema_ids=[]
                all_words=[]
                for sentence in sentences:
                    words=word_tokenize(sentence)
                    for word in words:
                        all_words.append(word)
                indexes=[]
                for x in range(kelime_sayisi):
                    indexes.append(x)
                for i in range(kelime_sayisi):
                    for j in range(kelime_sayisi):
                        if TF_IDF[i]>TF_IDF[j]:
                            temp=TF_IDF[i]
                            TF_IDF[i]=TF_IDF[j]
                            TF_IDF[j]=temp
                        
                            temp=indexes[i]
                            indexes[i]=indexes[j]
                            indexes[j]=temp
                for i in range(kelime_sayisi):
                    tema_kelimeleri.append(all_words[indexes[i]])
                j=0
                for sentence in sentences:
                    cumle_skorlari[j]=cumle_skorlari[j]+p5(sentence)
                    j=j+1
                #CUMLE SKORLAMA------------------------------------------------------
                # GRAF DUGUMLERİ OLUŞTU
                for i in range(len(sentences)):
                    G.add_node(i, cumle=sentences[i])
                #GRAF KENARLARI OLUŞTUR
                dugum_sayisi=G.number_of_nodes()
                for i in range(0,dugum_sayisi):
                    for j in range(1,dugum_sayisi):
                        if not(G.has_edge(i,j)) and i!=j:
                            G.add_edge(i,j)      
                # GRAFI ÇİZ----------------------------------------
                # QGraphicsScene ve QGraphicsView oluştur
                

                # Düğümleri ve kenarları çiz
                pos = nx.spring_layout(G)
                #GRAFLARIN KENAR BÜYÜKLÜĞÜNÜ YAZ

                for node1 in G.nodes():
                    for node2 in G.nodes():
                        if G.has_edge(node1,node2):
                            sentence1=G.nodes[node1]['cumle']
                            sentence2=G.nodes[node2]['cumle']
                            G[node1][node2]['weight']=similarity(sentence1,sentence2)
                for edge in G.edges():
                    x1, y1 = pos[edge[0]]
                    x2, y2 = pos[edge[1]]
                    scene.addLine(x1 * 400, y1 * 400, x2 * 400, y2 * 400, pen)
                    linex1.append(x1 * 400)
                    liney1.append(y1 * 400)
                    linex2.append(x2 * 400)
                    liney2.append(y2 * 400)
                    weight = G.edges[edge]['weight']
                    agirlik=scene.addText(str(weight)[:4],font)
                    agirlik.setPos((x1 * 400+x2 * 400)/2,(y1 * 400+y2 * 400)/2)
                #GRAFLARIN NODE LARINI YAZ
                for node in G.nodes():
                    x, y = pos[node]
                    # İlk kutu
                    rect1 = scene.addRect(x * 400 - 50, y * 400 -100, 40, 40, pen)
                    rect1x.append(x * 400 - 50)
                    rect1y.append(y * 400 -100)
                    # İkinci kutu
                    rect2 = scene.addRect(x*400+50, y*400 -100, 40, 40, pen)
                    rect2x.append(x*400+50)
                    rect2y.append(y*400 -100)
                    
                    ellipsex.append(x * 400 - 20)
                    ellipsey.append(y * 400 - 20)
                    scene.addEllipse(x * 400 - 20, y * 400 - 20, 40, 40, pen, brush)
                    text = scene.addText(str(node), font)
                    text.setPos(x * 400 - 20, y * 400 - 20)
                
                #-------------------------------------------------------------------
                #ÖZET YAZMA İŞLEMLERİ
                #-------------------------------------------------------------------
                
                
                
                
                
    def similar_button(self,scene,view,G):
        print("buttona bastın")
        self.reset(scene,view,G)
        benzerlik=float(self.arayuz.textEdit.toPlainText())
        benzerlik_asan=0
        
        for index,node in enumerate(G.nodes()):
            for node2 in G.nodes():
                if G.has_edge(node,node2):
                    if G[node][node2]['weight']>=benzerlik:
                        benzerlik_asan=benzerlik_asan+1
            text = scene.addText(str(benzerlik_asan), font)
            text.setPos(rect1x[index],rect1y[index])
            if self.similar_button_count==0:
                p3_values.append(float(benzerlik_asan/G.degree[node]))
            else:
                p3_values[index]=float(benzerlik_asan/G.degree[node])
            benzerlik_asan=0
        
        for i in cumle_skorlari:
            cumle_skorlari1.append(0)
        for index,node in enumerate(G.nodes()):
            cumle_skorlari1[index]=cumle_skorlari[index]+p3_values[index]
        for index,node in enumerate(G.nodes()):
            scene.addRect(rect2x[index],rect2y[index],40,40,pen,brush_white)
            text = scene.addText(str(cumle_skorlari[index]+p3_values[index])[:4], font)
            print(index,"   ",cumle_skorlari[index],"     ",p3_values[index])
            text.setPos(rect2x[index],rect2y[index])
            
        for index,edge in enumerate(G.edges()):
            if G.edges[edge]['weight']>=benzerlik:
                scene.addLine(linex1[index],liney1[index],linex2[index],liney2[index],pen_purple)
        for index,node in enumerate(G.nodes()):
            scene.addEllipse(ellipsex[index],ellipsey[index], 40, 40,pen,brush)
            text = scene.addText(str(node), font)
            text.setPos(ellipsex[index],ellipsey[index]) 
        self.similar_button_count=self.similar_button_count+1   
            
    def sentence_score(self,scene,view,G):
        input_score=float(self.arayuz.textEdit_2.toPlainText())
        indexler=[]
        summary_text=""
        for index,node in enumerate(G.nodes()):
            if cumle_skorlari1[index]>=input_score:
                scene.addRect(rect2x[index],rect2y[index],40,40,pen,brush_yellow)
                text = scene.addText(str(cumle_skorlari1[index])[:4], font)
                text.setPos(rect2x[index],rect2y[index])
                
            else:
                scene.addRect(rect2x[index],rect2y[index],40,40,pen,brush_white)
                text = scene.addText(str(cumle_skorlari1[index])[:4], font)
                text.setPos(rect2x[index],rect2y[index])
        for index,scores in enumerate(cumle_skorlari1):
            if scores>=input_score:
                indexler.append(index)
        for index in indexler:
            summary_text=summary_text+self.summery_sentences_list[index]
        self.arayuz.textBrowser.clear()
        self.arayuz.textBrowser.append(summary_text)
        
    def calculating_rouge_score(self):
        fname=QFileDialog.getOpenFileName(self,'Open File','/home')
        if fname[0]:
            with open(fname[0],'r') as f:
                reference=f.read()
                system_output=self.arayuz.textBrowser.toPlainText()
                rouge_score=calculate_rouge_score(system_output,reference)
                self.arayuz.skor.setText(str(rouge_score)[:4])
                
                
                
        
        
        
        
        
        
        
                
            
            
        
               
                
                    
                           
                            
                
                    
                
                
                
                
                        
                    
                
                    
                






app=QApplication([])
pencere=main(G)
pencere.show()
app.exec_()