## To do

- Implement the feature extraction in
  `enhancedFeatureExtractorDigit` function in the file
  `dataClassifier.py`. 

- Implement the functions: `trainAndTune` and
  `calculateLogJoinProbabilities` in `naiveBayes.py`.


## Notes

- What are features?

- Basic features for digit? There is a `util.Counter` that I should
  use. The basic feature methodtells whether each pixel is white (0)
  or gray/black(1). features[(x,y)] = 1 when the pixel at (x,y) is
  gray/black, 0 when it is white. 

- Basic features for face: tells whether each pixel in the provided
  datum is an edge (1) or no edge (0).

- We can enhance feature extractor ourselves. More details are in
  section "Feature Design"

- Example: for the digit data, consider the number of separate,
  connected regions of white pixels, which varies by digit
  type. 1,2,3,5,7 tend to have one contiguous region of white space
  while the loops in 6,8,9 create more. 

- What is `kgrid`? It is the list of possible k values for trying
  Laplace smoothing. 

- In `trainAndTune`, estimate conditional probabilities from the
  training data for each possible values of k given in the list
  `kgrid`. 

- `self.features`: list of all possible features.

- `self.legalLabels`: list of all possible labels

- `trainingData`: see `dataClassifier.py`, the function
  `runClassifier`. For digits, there are 100 datums in the
  trainingData. 

- `trainingLabels`: There are 100 labels here. 

- You can add code to the `analysis` method in `dataClassifier.py` to
  explore the mistakes that your classifier is making (optional)

- 
