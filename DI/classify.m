dirname = 'C:\Users\Elango\Documents\MATLAB\food';
Dataset = imageDatastore(dirname,...
    'IncludeSubfolders',true,...
    'LabelSource','foldernames');

[Training_Dataset,Validation_Dataset] = splitEachLabel(Dataset,0.7,'randomized');

Number_of_Classes = numel(categories(Training_Dataset.Labels));

net = googlenet;
analyzeNetwork(net);

Input_Layer_Size = net.Layers(1).InputSize(1:2);

Resized_Training_Image= augmentedImageDatastore(Input_Layer_Size,Training_Dataset,"ColorPreprocessing","gray2rgb");


Resized_Validation_Image=augmentedImageDatastore(Input_Layer_Size,Validation_Dataset,"ColorPreprocessing","gray2rgb");

Feature_Learner = net.Layers(142);

Output_Classifier = net.Layers(144);

New_Feature_Learner = fullyConnectedLayer(Number_of_Classes,...
'Name','Food Feature Learner',...
'WeightLearnRateFactor',10,...
'BiasLearnRateFactor',10);

New_Classifier_Layer = classificationLayer('Name','Food Items Classifier');


Layer_Graph = layerGraph(net);

New_Layer_Graph= replaceLayer(Layer_Graph,Feature_Learner.Name,New_Feature_Learner); 


New_Layer_Graph = replaceLayer(New_Layer_Graph,Output_Classifier.Name,New_Classifier_Layer);

analyzeNetwork(New_Layer_Graph);

Size_of_Minibatch = 5;

Validation_Frequency = floor(numel(Resized_Training_Image.Files)/Size_of_Minibatch);

Training_Options = trainingOptions('sgdm',...
    'MiniBatchSize',Size_of_Minibatch,...
    'MaxEpochs',20,...
    'InitialLearnRate',3e-4,...
    'Shuffle','every-epoch',...
    'ValidationData',Resized_Validation_Image,...
    'ValidationFrequency',Validation_Frequency,...
   'Verbose',false,...
   'Plots','training-progress');



net2 = trainNetwork(Resized_Training_Image,New_Layer_Graph,Training_Options);



YPred = classify(net2,Resized_Validation_Image);
YTest = Validation_Dataset.  Labels;
accuracy = sum(YPred == YTest)/numel(YTest)


save('classify_20.mat',"net2");
