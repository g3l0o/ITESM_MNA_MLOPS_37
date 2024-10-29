from v1 import preproccesing, transformation,data_loader
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np


class Pipeline():
    def __init__(self):
        self.df = None
        self.modelo = None
        self.preprocess_df = None
        self.train_embedds,self.test_embedds,self.y_train,self.y_test = None,None,None,None

    def load_data(self,path):
        self.df = data_loader.load_data(path)
    
    def preprocess_data(self):
        self.preprocess_df = preproccesing.preprocces(self.df)
        return self.preprocess_df
        

    def transform_data(self,df,split_percentage=0.20,random_state=2):
        train_vocab, test_vocab,self.y_train,self.y_test = transformation.obtain_vocabs(df,split_percentage,random_state)
        self.train_embedds = transformation.embedd_sentences(train_vocab,'train')
        self.test_embedds = transformation.embedd_sentences(test_vocab,'test')
        return {'train_vocab':train_vocab,
                'test_vocab':test_vocab,
                'y_train':self.y_train,
                'y_test':self.y_test,
                'train_embedds':self.train_embedds,
                'test_embedds':self.test_embedds}


    def EDA(self,vocab,classes):
        transformation.wordClouds(vocab,classes)

    def train(self,model=KMeans()):
        self.modelo = model
        self.modelo.fit(X=self.train_embedds)

    def predict(self,input_data):
        self.predicts = self.modelo.predict(input_data)
        return self.predicts

    def evalute_model(self,in_train=True):
        if in_train:
            y = self.y_train
            x = self.train_embedds
        else:
            y = self.y_test
            x = self.test_embedds

        predicts = self.modelo.predict(x)
        acc = accuracy_score(predicts,y)
        print(acc)
        cm = confusion_matrix(predicts,y)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
        disp.plot()
        plt.show()
        # Calcular métricas
        tp = np.sum(np.diag(cm))  # Verdaderos positivos
        fp = np.sum(cm, axis=0) - np.diag(cm)  # Falsos positivos
        fn = np.sum(cm, axis=1) - np.diag(cm)  # Falsos negativos
        tn = np.sum(cm) - (fp + fn + tp)  # Verdaderos negativos

        return tp,fp,fn,tn
    
    def run_pipeline(self,path,modelo):
        df = self.load_data(path)
        pre_df = self.preprocess_data()
        transformed_data = self.transform_data(pre_df)
        self.train(modelo)
        self.evalute_model()