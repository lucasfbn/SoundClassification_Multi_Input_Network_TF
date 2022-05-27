# Environmental sound classification using a multi-input network

This work studies the feasibility of a multi-input network for environmental sound classification. It uses a dataset comprised of sound excerpts of urban environments with 10 classes. 

The network utilises two commonly used digital audio representations and combines the extracted features in a shared classification layer. A 1D-CNN is used to process the raw waveform of each audio file, and a 2D-CNN is used to process the respective mel spectrogram. The network attempts to capture distinct patterns from the same input and effectively provide the model with the option to `choose' between the pattern that might be most relevant to each of the different classes of the dataset.