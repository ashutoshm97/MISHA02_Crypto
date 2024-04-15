"""

    Created By: Ashutosh Mishra | Parisha Desai

    Date: 14 April 2024

    Enhancement Project [Crypto]
    This script allows to utilize pretrained models and finetunes a standard model to predict if a comment from hugging face IMDB Dataset is having a positive or negative sentiment. There is also an option for GPT-2 and Roberta to perform with and without adversarial attacks. 
    Also, the distil-bert is fine tuned with and without adversarial attacks.

    
"""
import evaluate
import numpy as np
import torch
from datasets import load_dataset, DatasetDict, Dataset 
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DistilBertTokenizer, DataCollatorWithPadding, TFAutoModelForSequenceClassification , create_optimizer
from transformers.keras_callbacks import KerasMetricCallback
import random

def create_pertubations(texts, tokenizer):

    print("Size is " + str(len(texts)))
    moded_text = []

    #operations = [1, -1, 2, -2, 3, -3]     #To randomly change values

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    for sentence in inputs['input_ids']:
        
        
        sentenceLength = int(len(sentence))
        
        #Create 5 modified texts
        for times in range(5):
            temp = sentence
            for val in range(3):
                targetLocation = random.randint(3, 35)
                if temp[targetLocation] > 1:
                    temp[targetLocation] -= 1 
            moded_text.append(tokenizer.decode(temp, skip_special_tokens=True))
    print("New Size is " + str(len(moded_text)))
    return moded_text


# Function to perform sentiment analysis with batching
def batch_predict_sentiment(texts, tokenizer, model, mode, batch_size=10):

    if mode == True:
        moded_values = create_pertubations(texts, tokenizer)

        total_texts = len(moded_values)            #Calculate total Length
        predicted_sentiments = list()           #Define an empty list

        for i in range(0, total_texts, batch_size):   #Perform action in Batches

            texts_batch = moded_values[i:i + batch_size]    #Select Batch
            inputs = tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True)  #Using tokenizer gernerate tensors for torch

            with torch.no_grad():       #Use model to predict
                outputs = model(**inputs)   #Save Results

            predicted_classes = torch.argmax(outputs.logits, dim=1).tolist()   #Get Max Vals as correct predictions
            predicted_sentiments.extend(predicted_classes)  #Append results to list
            print("Predicted " + str(i) + "elements")
    else:
        total_texts = len(texts)            #Calculate total Length
        predicted_sentiments = list()           #Define an empty list

        for i in range(0, total_texts, batch_size):   #Perform action in Batches

            texts_batch = texts[i:i + batch_size]    #Select Batch
            inputs = tokenizer(texts_batch, return_tensors="pt", padding=True, truncation=True)  #Using tokenizer gernerate tensors for torch

            with torch.no_grad():       #Use model to predict
                outputs = model(**inputs)   #Save Results

            predicted_classes = torch.argmax(outputs.logits, dim=1).tolist()   #Get Max Vals as correct predictions
            predicted_sentiments.extend(predicted_classes)  #Append results to list
            print("Predicted " + str(i) + "elements")

    return predicted_sentiments      #Return list of values

def calculate_accuracy(predicted_sentiments, actual_labels):

    correct_predictions = sum(1 for pred, label in zip(predicted_sentiments, actual_labels) if pred == label)    #Sum correct predictions
    total_predictions = len(actual_labels)    #Get total predictions
    accuracy = correct_predictions / total_predictions * 100  #Calculate Accuracy
    return accuracy    #Return accuracy


def predict_gpt2(test_data, mode):

    print("Loading Tokenizer and Model")
    tokenizer = AutoTokenizer.from_pretrained("mnoukhov/gpt2-imdb-sentiment-classifier")  #Load the tokenizer for gpt2 trained on IMDB Dataset
    model = AutoModelForSequenceClassification.from_pretrained("mnoukhov/gpt2-imdb-sentiment-classifier") #Load the pre-trained hugging face model for gpt2

    print("Predicting in batches")
    batch_predicted_sentiments = batch_predict_sentiment(test_data['text'], tokenizer, model, mode)   #Run the predictions in batch mode to reduce load on Memory
    actual_labels = test_data['label']        #Define True Labels

    print("Calculating Accuracies")
    accuracy = calculate_accuracy(batch_predicted_sentiments, actual_labels)    #Calculate the Accuracy achieved
    print(f"The accuracy achieved for predictions is {accuracy:.2f}%")

def predict_roberta(test_data, mode):

    print("Loading Tokenizer and Model")
    tokenizer = AutoTokenizer.from_pretrained("abhishek/autonlp-imdb-roberta-base-3662644")  #Load the tokenizer for gpt2 trained on IMDB Dataset
    model = AutoModelForSequenceClassification.from_pretrained("abhishek/autonlp-imdb-roberta-base-3662644") #Load the pre-trained hugging face model for gpt2

    print("Predicting in batches")
    batch_predicted_sentiments = batch_predict_sentiment(test_data['text'], tokenizer, model, mode)   #Run the predictions in batch mode to reduce load on Memory
    actual_labels = test_data['label']        #Define True Labels

    print("Calculating Accuracies")
    accuracy = calculate_accuracy(batch_predicted_sentiments, actual_labels)    #Calculate the Accuracy achieved
    print(f"The accuracy achieved for predictions is {accuracy:.2f}%")

def create_pertubations_finetuned(data, tokenizer):
    moded_text = []

    texts = []
    labels = []

    for row in data:
        texts.append(row['text'])
        labels.append(row['label'])

    labels = [label for label in labels for _ in range(5)]

    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    
    for sentence in inputs['input_ids']:
        for times in range(5):
            temp = sentence
            for val in range(3):
                targetLocation = random.randint(3, 35)
                if temp[targetLocation] > 1:
                    temp[targetLocation] -= 1 
            moded_text.append(tokenizer.decode(temp, skip_special_tokens=True))


    data_dict = {"text": moded_text, "label": labels}

    dataset = Dataset.from_dict(data_dict)
    return dataset


def finetuned_predict_distilbert(imdb, mode):

    print("Loading Tokenizer and Model")

    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")     #Load the tokenizer for distilbert trained on IMDB Dataset
    model = TFAutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)      #Load the pre-trained hugging face model for distilbert

    small_train_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(3000))])
    small_val_dataset = imdb["train"].shuffle(seed=42).select([i for i in list(range(300))])
    small_test_dataset = imdb["test"].shuffle(seed=42).select([i for i in list(range(300))])

    if mode == True:
        moded_train_dataset = create_pertubations_finetuned(small_train_dataset, tokenizer)
        moded_val_dataset = create_pertubations_finetuned(small_val_dataset, tokenizer)
        moded_test_dataset = create_pertubations_finetuned(small_test_dataset, tokenizer)

        small_dataset_dict = DatasetDict({
                    'train': moded_train_dataset,
                    'validation': moded_val_dataset,
                    'test': moded_test_dataset
                })
    else:
        small_dataset_dict = DatasetDict({
                    'train': small_train_dataset,
                    'validation': small_val_dataset,
                    'test': small_test_dataset
                })

    def preprocess_function(examples):
        return tokenizer(examples["text"], truncation=True)             #Using a function to map the whole dataset across train, test and validation

    tokenized_imdb = small_dataset_dict.map(preprocess_function, batched=True)        #Mapping the data with the tokenizer
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="tf")    #Using data collator to capture the fields
    accuracy = evaluate.load("accuracy")    #Using accuracy for computing metrics

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)

    #Defining important hyper parameters
    batch_size = 16
    num_epochs = 5
    batches_per_epoch = len(tokenized_imdb["train"]) // batch_size
    total_train_steps = int(batches_per_epoch * num_epochs)
    optimizer, schedule = create_optimizer(init_lr=2e-5, num_warmup_steps=0, num_train_steps=total_train_steps)

    #Converting to tensor-flow sets for input requirement
    tf_train_set = model.prepare_tf_dataset(
        tokenized_imdb["train"],
        shuffle=True,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_validation_set = model.prepare_tf_dataset(
        tokenized_imdb["validation"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    tf_test_set = model.prepare_tf_dataset(
        tokenized_imdb["test"],
        shuffle=False,
        batch_size=16,
        collate_fn=data_collator,
    )

    #Compiling model based on previous additions
    model.compile(optimizer=optimizer)

    #Using Keras Metrics for Call Backs and traing the models for 5 epochs
    metric_callback = KerasMetricCallback(metric_fn=compute_metrics, eval_dataset=tf_validation_set)
    model.fit(x=tf_train_set, validation_data=tf_validation_set, epochs=5, verbose = True)

    #predicting the results
    results = model.predict(tf_test_set)

    #Fetching max values for correct labels
    predicted_labels = np.argmax(results['logits'], axis=1)

    #Get the accuracy
    accuracy = np.mean(predicted_labels == small_dataset_dict['test']['label'])
    print(f"The accuracy achieved for predictions is {accuracy:.2f}%")


def main():

    """
        This is the main function that controls the execution of the script. The user is prompted to choose from a menu to either select a function or exit script

    """

    #Printing a welcome screen Message
    print('\t\t\t\t\t\t!! Welcome !!')
    print('#####################################################################################################################')
    print('This python program will use pretrained models and predict if a comment from test dataset is positive or negative ')
    print('#####################################################################################################################\n')


    #Instancing the variables
    blRun = True                                                                                        #Creating a boolean Flag to control the while loop execution. By default the value is set as True
    nInst = 1                                                                                           #Creating a integer type variable to count the number of times the while loop was executed

    while(blRun):                                                                                       #While the boolean condition is true run the loop else exit
        print('\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\tInstance --> ' + str(nInst))
        print('\nPlease select one of the following options : -')

        print('\t\t\tFor using the predict_gpt2 function please type 1')
        print('\t\t\tFor using the predict_roberta function please type 2')
        print('\t\t\tFor using the finetuned_predict_distilbert function please type 3')
        print('\t\t\tFor quitting the program please type q or Q')

        choice = input('\nPlease enter your input --> ')

        if choice == '1':
            print('\nPlease select one of the following options : -')
            print('\t\t\t\tFor using the predict_gpt2 function please type 1')
            print('\t\t\t\tFor using the predict_gpt2 with adversarial attack function please type 2')
            choice = input('\nPlease enter your input --> ')

            if choice == '1':
              imdb = load_dataset("imdb")
              small_test_dataset = imdb["test"].shuffle(seed=42).select(range(200))
              predict_gpt2(small_test_dataset, False)
            elif choice == '2':
              imdb = load_dataset("imdb")
              small_test_dataset = imdb["test"].shuffle(seed=42).select(range(200))
              predict_gpt2(small_test_dataset, True)
            else:
              print("\n!!!!!!!!!!!!!!!Invalid Input!!!!!!!!!!!!!!!\n Please check the input choice.\n")
            

        elif choice == '2':
            print('\nPlease select one of the following options : -')
            print('\t\t\t\tFor using the predict_roberta function please type 1')
            print('\t\t\t\tFor using the predict_roberta with adversarial attack function please type 2')
            choice = input('\nPlease enter your input --> ')

            if choice == '1':
              imdb = load_dataset("imdb")
              small_test_dataset = imdb["test"].shuffle(seed=42).select(range(200))
              predict_roberta(small_test_dataset, False)
            elif choice == '2':
              imdb = load_dataset("imdb")
              small_test_dataset = imdb["test"].shuffle(seed=42).select(range(200))
              predict_roberta(small_test_dataset, True)
            else:
              print("\n!!!!!!!!!!!!!!!Invalid Input!!!!!!!!!!!!!!!\n Please check the input choice.\n")


        elif choice == '3':
            print('\nPlease select one of the following options : -')
            print('\t\t\t\tFor using the finetuned_predict_distilbert function please type 1')
            print('\t\t\t\tFor using the finetuned_predict_distilbert with adversarial attack function please type 2')
            choice = input('\nPlease enter your input --> ')

            if choice == '1':
              imdb = load_dataset("imdb")
              finetuned_predict_distilbert(imdb, False)
            elif choice == '2':
              imdb = load_dataset("imdb")
              finetuned_predict_distilbert(imdb, True)
            else:
              print("\n!!!!!!!!!!!!!!!Invalid Input!!!!!!!!!!!!!!!\n Please check the input choice.\n")

        elif (choice == 'q' or choice == 'Q'):
            print("Thankyou for trying out the program.\n You have tried out this program for " + str(nInst) + " time/s. \nHave a nice day!!\n")
            blRun = False

        else:
          print("\n!!!!!!!!!!!!!!!Invalid Input!!!!!!!!!!!!!!!\n Please check the input choice.\n")

        nInst += 1




if __name__ == '__main__':
    main()
