Open source question Answering project
--------------------------------------------
if you are directly running this file, it will throw error because I have removed all other datasets and files to make this entire folder less than 20MB

1. 'data_scripting.py' - is the first file that need to be run.
2. 'wiki.py'           - for wikipedia retrieveral (2nd step)
3. 'chunking.py'       - cleaning of the wikipedia dataseta and make them into chunks (3rd step)
4. 'embeddings.py'     -  after chunking comes the (4st step)
5. 'llm.py'            - where the retrieval is being passed and the dataset is being ready for eval.
6. 'main.py'           - consists of evalution, it runs the whole evalution on the entire given dataset 




*********************************
for furthur analysis 
'catgeory_result.py' - just to visulaise the how well the question is being categorised
'chunking_analysis.py' - for chunking visualistion
'result.py'           - for the final visualisation that how my both promts have performed to check the accuracy.

*********************************

Read Technical Report for furthur understanding.
