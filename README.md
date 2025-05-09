**OVERVIEW**
These projects were part of the Master's in AI that I pursued at the University of St. Thomas from the Fall of 2023 to the Fall of 2025. Each class included a hallmark project and here are a few of them. As these are raw ipynb files, please read the table of contents as well as comments within code blocks to understand every module within a project's source code.
Ipynb files can be uploaded into applications that support python coding environments. 

**Movie Review Summarizer**
- Project for SEIS 767 Conversational AI
- The goal of this project was to create an intelligent movie review summarizer that takes in multiple reviews of a movie and generates a singular review that captures the critic semantics of all the inputted reviews
- This project utilized Bert for extractive summarization and T-5 Flan for absractive summarization to accomplish its goals
- The dataset used is from Kaggle, consisting of movie reviews on 17,000 movies. https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset?select=rotten_tomatoes_critic_reviews.csv

**Table Of Contents:**
- Import Statements: Import neccessary modules
- Data Preprocessing: Load in and clean data including a normalization function for the review score and grouping by movie name
-  Prepare Dataset For Model: Filter out stopwords, embed words
- Stop Words: List out stop word list and define functions to filter our stop words and labeling words with embeddings
- MiniLM Embedding: Load in MiniLM and apply embedding-based word labeling
- Top Word Extraction: Function to pull out top words by co-sine similarity
- Flan Summary Generation: Load in Flan-T5 model and define function to abstratively generate summary from a prompt including key ideas from the top words
- Results & Similarity Score: Generate summaries and evaluate similarity score on a sample of the validation set
- Citations

**Real Fake Image Classifier**
- Project for SEIS 766 Vision AI
- The goal of this project was to distinguish between real and ai-generated images by taking in images, reconstructing them in an unsupervised fashion using an autoencoder architecture, and classifying those images based on underlying semantic information in a supervised fashion
- This project utilized a transfered encoder from Resnet 18, a custom decoder, and a classifier head to accomplish its goals
- The dataset used is from Kaggle, consisting of 60,000 total images. 30,000 are real, 30,000 are ai-generated through DALL-E, stable difussion, etc. https://www.kaggle.com/datasets/tristanzhang32/ai-generated-images-vs-real-images/data
  
**Table Of Contents:**
- Import Library Modules: Import necessary modules
- Unnormalization: Images will be normalized for use by Resnet 18. This function unnormalize images for visualization
- Dataset Processing: Image processing included resizing, normalization and creating a combined dataset
- Show Samples: Show samples from the images
- Autoencoder Classifier: Define the autoencoder classifier including a transfered encoder, custom decoder, and classifier head
- Threshold (From Previous Training): Predefining the treshold for classification based on the Youden Statistic of previous runs
- Visualization Of Reconstruction: Function to visualize reconstructions
- Hybrid Loss: Define hybrid loss function for reconstructions including L1, LPIPS, and SSIM
- Focal Loss: Define focal loss function for classification
- Save Checkpoints: Define saving and loading model from directory
- Training Model: Define a custom training loop from a saved checkpoint using the hybrid and focal loss as well as progressive weighting and freeze epochs on classification loss
- Setup Optuna Training Parameters (From Previous Run): Setup custom training parameters based on a previous Optuna optimization test
- Optuna Optimization Test: Run optuna optiization test on smaller sample
- Setup Best Training Parameters: Re-initialize training parameters based on best results from Optuna Test
- Run Training: Run training on the training set with the last 3 layers of the transferred encoder as unfrozen
- Architecture Diagram: Visualize the architecture of the model
- Evaluate Model on Test Set: Evaluate model on test set using Youden Treshold. Show classification report and ROC curve.
- Confidence Distribution: Plot out confidence distribution of predictions
- Youden Treshold: Plot out F-1 score and threshold for future runs
- Visualize Misclasified Images: Visualize misclassfied images and show confidence score
- Save Model: Save model for future use 
