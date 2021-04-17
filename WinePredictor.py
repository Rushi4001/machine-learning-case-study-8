from sklearn import metrics
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

def WinePredictor():
    #load dataset
    wine=datasets.load_wine()
    
    #print the name of the features
    print(wine.feature_names)
    
    #print label species(class_0 , class_1,class_2)
    print(wine.target_names)
    
    #print the wine data top5
    print(wine.data[0:5])
    
    #print the wine labels(class_0,class_1,class_2)
    print(wine.target)
    
    #split dataset into training set and test set
    x_train,x_test,y_train,y_test=train_test_split(wine.data,wine.target,test_size=0.3)#70% training and 30% test
   
    #create KNN Classifier
    knn=KNeighborsClassifier(n_neighbors=3)
    
    #train the model using the training sets
    knn.fit(x_train,y_train)
    
    #predict the responce for the test dataset
    y_pred=knn.predict(x_test)
    
    #model accuracy, how often is the classifier correct?
    print("Accuracy;",metrics.accuracy_score(y_test,y_pred))
    
def main():
    
    print("_______machine learning application_______")
    print("wine predictor application using K nearest knighbor algorithm")
    WinePredictor()
    
if __name__=="__main__":
    main()