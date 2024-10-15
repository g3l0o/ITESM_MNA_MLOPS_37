from v1 import preproccesing, transformation,data_loader
from sklearn.metrics import accuracy_score,confusion_matrix,ConfusionMatrixDisplay
import matplotlib.pyplot as plt


class Pipeline():
    def load_data(self,path):
        self.df = data_loader.load_data(path)
    
    def preprocess_data(self):
        return preproccesing.preprocces(self.df)
        

    def transform_data(self,df,split_percentage=0.20,random_state=2):
        train_vocab, test_vocab,y_train,y_test = transformation.obtain_vocabs(df,split_percentage,random_state)
        train_embedds = transformation.embedd_sentences(train_vocab,'train')
        test_embedds = transformation.embedd_sentences(test_vocab,'test')
        return {'train_vocab':train_vocab,
                'test_vocab':test_vocab,
                'y_train':y_train,
                'y_test':y_test,
                'train_embedds':train_embedds,
                'test_embedds':test_embedds}


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