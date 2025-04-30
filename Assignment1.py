'''
Name : Darshita Guckhool
Student ID : 1709250
Course: CMPUT 175
Assignment 1
'''

'''
Imports
'''
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from carousel import Carousel, DLinkedListNode
import matplotlib.pyplot as plt


def open_read(filename):
    '''
    Opens the csv file in read mode
    returns the count of rows
    '''
    file = open(filename,"r")
    count = 0
    for row in file:
        count +=1
    file.close()
    return count

def convert_list(filename):
    '''
    Converts the contents of the CSV file into a nested list
    calls the open_read(filename) function to get the number of rows 
    returns nested list of all of contents
    '''
    count = open_read(filename)
    all_rows=[]
    file= open(filename, "r")
    for i in range(count):
        rows = file.readline()
        rows_list = rows.strip().split(",") #splitting the contents to place in the list
        all_rows.append(rows_list)
    file.close()    
    return all_rows

def remove_empty(all_rows):
    '''
    Function checks which columns have missing data
    Prints the columns names and the number of missing values in the column
    Removes any missing attribute 
    Returns filtered nested list where missing attributes have been removed
    ''' 
    filtered_data=[] 
    count = [0]*len(all_rows[0]) # creating count variables for each of the columns
    
    for i in range(len(all_rows)):
        flag = False
        for j in range(len(all_rows[i])):
            
            if  all_rows[i][j]=='':
                count[j] +=1 #counting number of empty values
                flag = True # flag that changes if there is a missing value
        
        if flag == False:
            filtered_data.append(all_rows[i]) #keeping non empty items
                     
    for j in range(len(all_rows[0])):
        if count[j] > 0:
            print(f"{all_rows[0][j]} : {count[j]} values missing")
    
    print(f"Remaining number of rows: {len(filtered_data)-1}") # '-1' to account for the header
    return filtered_data

def remove_borrowers(filtered_data):
    '''
    Function that removes borrowers aged 90 and above
    Returns nested list of the data without borrowers aged 90 and above
    '''
    count = 0
    for i in range(len(filtered_data)-1,0,-1): # inversely looping to not affect the indexes
        if int(filtered_data[i][0]) >= 90:
            count +=1
            filtered_data.remove(filtered_data[i])
    print(f"\nNumber of records with age > 90: {count}")
    print(f"Remaining number of rows: {len(filtered_data)-1}")  # '-1' to account for the header
    return filtered_data

def plot_loan_distribution(filtered_data):
    '''
    Function utilizes matplotlib to display histograms
    Uses for loop to loop through nested list of data
    Stores the default values and non- default values in seperate lists
    plots the default and non default histograms
    '''
    default = []
    not_default=[]
  
    for i in range(1,len(filtered_data)): #seperating defaults and not defaults
        if filtered_data[i][8] == '1':
            default.append(int(filtered_data[i][0]))
        elif filtered_data[i][8] == '0':
            not_default.append(int(filtered_data[i][0]))
            
    # Histogram for Loans in Default
    plt.hist(default, bins=[0,10,20,30,40,50,60,70,80,90,100], edgecolor='black') # defining bins and histogram edge color
    plt.xlabel("Age (in years)")
    plt.ylabel("No. of borrowers")
    plt.title("Loans in Default")
    plt.show()
    
   # Histogram for loans not in default         
    plt.hist(not_default, bins=[0,10,20,30,40,50,60,70,80,90,100], edgecolor = 'black') # defining bins and histogram edge color
    plt.xlabel("Age (in years)")
    plt.ylabel("No. of borrowers")
    plt.title("Loans not in Default")
    plt.show()
    
def plot_piechart(filtered_data):
    '''
    Loops through the nested list to find home owners
    Checks if homeowners default or not
    Updates counts and uses matplotlib to print pie chart
    
    '''
    default = 0
    not_default = 0
    for i in range(len(filtered_data)):
        if filtered_data[i][2] == "OWN": 
            if filtered_data[i][8] == '1':
                default += 1
            elif filtered_data[i][8] == '0':
                not_default +=1
                
    pie_list =[default, not_default] #creating pie list
    labels = 'Defaulted','Not Defaulted'
    fig, ax=plt.subplots() 
    ax.pie(pie_list, labels=labels, autopct = '%1.1f%%') #plotting pie chart
    plt.title("Homeowners: Default vs. Not Default")
    plt.show()
    
def class_imbalance(filtered_data):
    ''' 
    Function counts the number of defaults and non-defaults in the data
    returns list where first value is defaulters and second is non defaulters

    '''
    default_count = 0
    not_default_count= 0
    
    for i in range(len(filtered_data)):
        if filtered_data[i][8] == '1':
            default_count += 1
        elif filtered_data[i][8] == '0':
            not_default_count += 1
            
    count_list=[default_count, not_default_count]
    return count_list

def undersampling (filtered_data,count_list):
    '''
    Functions accounst for undersampling in the data
    loops and removes not in default rows to be the same number as defaults
    Returns nested list of the newly filtered data
    '''
    difference = count_list[1] - count_list[0] # checks difference between the two numbers
    count = 0
    for i in range(len(filtered_data)-1,-1,-1): #inversely loops to ensure index isnt affected when removing
        if filtered_data[i][8] == '0' and count < difference: 
            filtered_data.remove(filtered_data[i]) #removing the data
            count += 1
   
    return filtered_data
    
def feature_scaling(filtered_data):
    '''
    Function uses standard scaler to scale loan amount and personal income data
    collects all the loan amounts and corresponding personal incomes into a list
    Utilizes StandardScaler() to scale the data
    Returns a nested list of scaled_data
    '''
    unscaled_data=[]
    for i in range(1,len(filtered_data)):
        for j in range(len(filtered_data[0])): # loops through the row that contains headings
            if filtered_data[0][j]== "loan_amnt":
                k=j #stores the index number
            if filtered_data[0][j] == "person_income":
                l=j
        pair_list=[int(filtered_data[i][k]),int(filtered_data[i][l])] 
        unscaled_data.append(pair_list)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(unscaled_data) #scaling data
    
    return scaled_data

def decision_tree_classifier(scaled_data,filtered_data):
    '''
    Uses scaled data retrieved from the feature_scaling(filtered_data) function
    Function creates a training model to train decision tree model
    stores the scaled amounts and credit history in x_train and y_train lists
    Uses DecisionTreeClassifier to train model 
    Returns the trained model
    '''
    x_train=[]
    y_train = []
    for i in range(1,len(filtered_data)):
        for j in range(len(filtered_data[0])):
            if filtered_data[0][j] == "cb_person_cred_hist_length":
                k=j
            if filtered_data[0][j] == "loan_status":
                l=j
    filtered_data.pop(0)
    for i in range(len(scaled_data)):
        x_value=[scaled_data[i][0],scaled_data[i][1],int(filtered_data[i][k])]
        x_train.append(x_value) #placing scaled loan amount and scaled income in x_train
        y_train.append(int(filtered_data[i][l])) #placing credit history in y_traine
    
    clf=DecisionTreeClassifier(random_state=42)
    return clf.fit(x_train,y_train) #creating trained model

def predict_model(filename, clf):
    '''
    Function creates prediction
    calls the convert_list, class_imbalance and feature_scaling functions to modify the data into a balanced and scaled data set
    uses .predict to output the prediciton
    returns prediction
    '''
    data=convert_list(filename)
    imbalance_counter=class_imbalance(data)
    balanced_data=undersampling(data,imbalance_counter)
    scaled_data=feature_scaling(balanced_data)
    
    x_test=[]
    for i in range(1,len(data)): #index starts at 1 to exclude headers
        for j in range(len(data[0])):
            if data[0][j] == "cb_person_cred_hist_length":
                k=j
    data.pop(0)
    for i in range(len(scaled_data)):
        x_value=[scaled_data[i][0],scaled_data[i][1],int(data[i][k])] #creates x_test list
        x_test.append(x_value)
        
    y_pred = clf.predict(x_test) #creates prediction
    return y_pred

def accuracy(y_pred, filename):
    '''
    Evaluates the accuracy of the prediction against a test data set
    calls the convert_list, class_imbalance and feature_scaling functions to modify the data into a balanced and scaled data set
    outputs test accuracy, confusion matrix and classification report, using the prediction calculates in predict_model function
    '''
    data=convert_list(filename)
    imbalance_counter=class_imbalance(data)
    balanced_data=undersampling(data,imbalance_counter)
    scaled_data=feature_scaling(balanced_data)
    y_test =[]
    for i in range(1,len(data)):
        for j in range(len(data[0])):
            if data[0][j] == "loan_status":
                l=j 
        y_test.append(int(data[i][l])) #creating the y-test list of credit history from the test data
        
    # evaluating predictions
    print(f"\nTest Accuracy: {accuracy_score(y_test,y_pred):.2f}") 
    print("classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred)) 
    
def building_carousel(filename,prediction):
    '''
    Calls carousel function from carousel.py
    creates list of dictionaries where key is the header of the column and values are the corresponding data.
    Adds a prediction key in the dictionary and the value is the predicted values calculated in the predict_model function
    returns list of dictionaries containg all the data
    '''
    vertical_carousel = Carousel()
    loan_data =convert_list(filename)
    key = loan_data[0] #getting header values
    key.append("prediction") 
    value_data = loan_data[1:] #getting data
    
    for i in range(len(value_data)):
        value_data[i].append(prediction[i])
        loan_dict={}
        for j in range(len(key)):
            loan_dict[key[j]] = value_data[i][j]
        vertical_carousel.add(loan_dict) #using add method from carousel.py to add the dictionary to the carousel
        
        if i == 0:
            first_node = vertical_carousel.current 
    vertical_carousel.current = first_node #sets the first node to be the first dictionary entered in the carouse
    return vertical_carousel 

def display(current_borrower):
    '''
    Display function that print the borrower information
    Displays whether they will default or not and the recommendation
    '''
    
    print("--------------------------------------------------")
    print(f"Borrower: {current_borrower['borrower']}")
    print(f"\nAge: {current_borrower['person_age']}")
    print(f"\nIncome: ${current_borrower['person_income']}")
    print(f"\nHome_ownership: {current_borrower['person_home_ownership']}")
    print(f"\nEmployment: {current_borrower['person_emp_length']}")
    print(f"\nLoan intent: {current_borrower['loan_intent']}")
    print(f"\nLoan grade: {current_borrower['loan_grade']}")
    print(f"\nAmount: ${current_borrower['loan_amnt']}")
    print(f"\nInterest Rate: {current_borrower['loan_int_rate']}")
    print(f"\nLoan percent income: {current_borrower['loan_percent_income']}")
    
    if current_borrower['cb_person_default_on_file'] == 'N':
        answer = 'No'
    else:
        answer = 'Yes'
    print(f"\nHistorical Defaults: {answer}")
    
    print(f"\nCredit History: {current_borrower['cb_person_cred_hist_length']} years")
    print("\n--------------------------------------------------")
    
    if current_borrower['prediction'] == 0:
        default = 'Will not default'
        recommend = 'Accept'
    else:
        default = 'Will default'
        recommend = 'Reject'
        
    print(f"\nPredicted loan_status: {default}")
    print(f"Recommend: {recommend}")
    print("\n--------------------------------------------------")
    
def main():
    # 5.1.1
    filename = "credit_risk_train.csv"
    count = open_read(filename)
    print(f"Initial number of rows: {count - 1}") # "-1" to account for header
    all_rows = convert_list(filename)
    data=remove_empty(all_rows)
    
    #5.1.2
    filtered_data=remove_borrowers(data)
    
    #5.1.3
    plot_loan_distribution(filtered_data)
    
    #5.1.4
    plot_piechart(filtered_data)
    
    #5.1.5
    imbalance_counter = class_imbalance(filtered_data)
    print(f"\nNumber of borrowers who defaulted: {imbalance_counter[0]}")
    print(f"Number of borrowers who did not default: {imbalance_counter[1]}")    
    
    balanced_data=undersampling(filtered_data,imbalance_counter)
    
    #5.2.1
    scaled_data=feature_scaling(balanced_data)
    
    #5.2.2 
    fit =decision_tree_classifier(scaled_data,filtered_data)
    file_name="credit_risk_test.csv"
    y_pred =predict_model(file_name,fit)
    accuracy(y_pred, file_name)
    
    #5.3.1
    f = "loan_requests.csv"
    prediction = predict_model(f,fit)
    print(f"\nPredictions{prediction}")

    
    #5.3.2
    loan_carousel = building_carousel(f,prediction) 
    loan_carousel.current = loan_carousel.head #setting the head of the carousel to be the current node
     
    #5.3.3
    choice_str = """Choose one of the following options:
                 1. Go Next
                 2. Go Back
                 0. Quit
                 Enter the choice (eg. 2)
                 """
    input("\nPress Enter to display information")
    exit= False
    while exit == False: #while loop to control whether the software runs or ends
        
        current_borrower = loan_carousel.getCurrentData()
        display(current_borrower)
        print(choice_str)
        choice = input(">>>")
        while choice not in ['1','2','0']: #input validation
            print("Invalid Input")
            choice = input("Enter valid input >>>")
            
        if choice == '1':
            loan_carousel.moveNext() #from carousel.py moves next
        elif choice == '2':
            loan_carousel.movePrevious() #from carousel.py moves back
        elif choice == '0':  #ends game
            print("Good Bye")
            exit = True
            
main()