# TallAI
TallAI is a simple hobby ML-project corncered with the problem of counting the amount of tallies on a tally sheet. For this task a CNN-based model architecture is proposed. 
As the training domain is rather comprehensive, with tally-counts spanning the range N = 0,1...50; the model uses a regressional approach rather than the more typical 
vector-based classification. As no appropriate dataset was found, this project "daringly" attempts to utilize a generated dataset (running the **test_gen.py** will create two folders: **data** containing trainingdata and **testdata** containing validationdata). 
To combat the many downsides of such an approach, the data is subjected to extensive preprocessing. The preprocessing, model architecture, performance tests, hyperparamter-tuner settup and data-visualization are all contained in the **notebook**. ONGOING...
