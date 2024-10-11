from v1 import preproccesing, transformation,data_loader
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt

class Pipeline():
    def load_data(self,path):
        self.df = data_loader.load_standarize_data(path)
    
    def preprocess_data(self):
        self.df = preproccesing.preprocces(self.df)

    def transform_data(self,split_percentage=0.20,random_state=2):
        self.train_vocab, self.test_vocab,self.y_train,self.y_test = transformation.obtain_vocabs(self.df,split_percentage,random_state)
        self.train_embedds = transformation.embedd_sentences(self.train_vocab,'train')
        self.test_embedds = transformation.embedd_sentences(self.test_vocab,'test')

    def EDA(self,vocab,classes):
        transformation.wordClouds(vocab,classes)

    def model(self,model):
        self.modelo = model
        self.modelo.fit(X=self.train_embedds)
        self.predicts = self.modelo.predict(self.train_embedds)
        acc = accuracy_score(self.y_train,self.predicts)
        print(acc)
        cm = confusion_matrix(self.y_train,self.predicts)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm,)
        disp.plot()
        plt.show()