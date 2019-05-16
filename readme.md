Steps to Run the Project

1. Dataset
a : Run the Dataset/wikicategory_extractor.ipynb to extract pages from different categories. 
b : Run the Datasetr/folderdocs2csv_all.py to create the CSV file for term recognizer ML model. 
c : Run the Terms_extractor/folderdocs2csv_domain.py to create domain wise CSV files for document labelling ML model.

2: Rule based model
2.1 Run the DomainTermsExtractor.py to extract all the unique domain terms from the wikipedia documents from Dataset folder. 
	It creates the dump files of the terms in Terms folder
2.2 Run the DocumentLabelling.py to label the test documents based on the terms extracted above.
	TestDocuments folder should contain the test documents to which labelling has to be done.

3: ML model
3.1: Domain Identification
	3.1.1: Train the models provided in phy_model, math_model and chem_model folder, folder location: ML_Model/Document_labelling. 
	3.1.2: After training run the ML_Model/Document_labelling/model_test_label_all.py to test the labelling. 
3.2: Terms Extraction:
	3.2.1: Train the ML_model/Domain_Terms/ner_crf_wikiall_train.py and save the weights.
	3.3.2: Test the model with ML_model/Domain_Terms/model_test_terms.py


Required python Pakages:
-----------------

1- keras
2- Tensorflow
3: scikit-learn
4: pandas
5: keras_contrib (for CRF)
